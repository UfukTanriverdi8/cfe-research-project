import os
import csv
import json
import time
import sys, re, unicodedata
from collections import OrderedDict
from typing import List, Dict, Any, Tuple

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===
INPUT_CSV = "./data/results/100q_results_filtered.csv"               # your 2500-row file
KEEP_CSV = "./data/results/100q_results_filteredssss.csv"           # non-ambiguous output (append)
AMBIG_CSV = "./data/results/100q_results_ambigiousss.csv"                      # ambiguous output (append)
STATE_JSON = ".filter_progress.json"             # tracks processed question keys
MODEL = "deepseek/deepseek-chat-v3.1"                          # change if you like
TEMPERATURE = 0.2

# IMPORTANT: Leave the SYSTEM PROMPT blank; you will fill it yourself.
SYSTEM_PROMPT = """
You are an expert question evaluator.
Your task is to determine if a question has one objectively correct answer.
We have a question dataset and we'll show you each question, its gold (correct) answer, and a list of plausible but incorrect candidate answers.
Keep in mind that the gold answer and candidate answers may not be perfect, but your goal is to assess the question itself. Gold answer and candidates are given for context.
Subjective questions, questions with multiple valid answers, time-sensitive questions are some examples of ambiguous questions.
Output Format:
Explanation: <your explanation here>
Ambiguous: <Yes or No>
"""

# Use OpenRouter with the OpenAI client
# Set your API key in the environment:  export OPENROUTER_API_KEY=...
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: Please set OPENROUTER_API_KEY in your environment.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# === Helpers ===

def normalize_q(s: str) -> str:
    s = unicodedata.normalize('NFKC', str(s)).strip()
    s = s.replace('\x85', '…')
    s = re.sub(r'\s+', ' ', s)
    return s

def load_state() -> Dict[str, bool]:
    if os.path.exists(STATE_JSON):
        try:
            return json.load(open(STATE_JSON, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state: Dict[str, bool]) -> None:
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def parse_candidates_for_prompt(val) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    s = str(val)
    # try JSON list
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # fallbacks
    if "||" in s:
        return [x.strip() for x in s.split("||") if x.strip()]
    if "\n" in s:
        return [x.strip() for x in s.split("\n") if x.strip()]
    return [s.strip()]

def pick_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = {c.lower(): c for c in df.columns}
    q_col = None
    for name in ["question", "q_text", "prompt_question", "input_question"]:
        if name in cols:
            q_col = cols[name]; break
    gold_col = None
    for name in ["gold_answer", "answer_gold", "reference_answer", "exact_answer", "gt_answer"]:
        if name in cols:
            gold_col = cols[name]; break
    cand_col = None
    for name in ["candidate_answers", "candidates", "plausible_candidates", "distractors"]:
        if name in cols:
            cand_col = cols[name]; break
    missing = [n for n, v in [("question", q_col), ("gold_answer", gold_col), ("candidate_answers", cand_col)] if v is None]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Found columns: {list(df.columns)}")
    return q_col, gold_col, cand_col

def build_messages(question: str, gold_answer: str, candidates_list: List[str]) -> list:
    user_content = (
        "You will see a question, its gold (correct) answer, and a list of plausible but incorrect candidate answers.\n"
        "Provide your explanation/analysis now.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold Answer:\n{gold_answer}\n\n"
        "Candidate Answers (incorrect but plausible):\n"
        + "\n".join(f"- {c}" for c in candidates_list)
    ).strip()
    messages = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
    messages.append({"role": "user", "content": user_content})
    return messages

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
def call_llm(messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()

def append_df(path: str, df_chunk: pd.DataFrame) -> None:
    header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
    df_chunk.to_csv(path, mode="a", header=header_needed, index=False, encoding="utf-8")

# === Main ===

def main():
    df = pd.read_csv(INPUT_CSV)
    q_col, gold_col, cand_col = pick_columns(df)

    # Build a stable key per question (normalized question + gold answer)
    df["_q_norm"] = df[q_col].map(normalize_q)
    df["_qkey"] = df["_q_norm"].astype(str) + " || " + df[gold_col].astype(str).str.strip()

    # Representative rows per unique question key (for prompting)
    reps = df.groupby("_qkey").first().reset_index()

    # Progress state
    state = load_state()

    total_unique = reps["_qkey"].nunique()
    print(f"Total unique questions: {total_unique}")
    print("Controls: [k] keep (non-ambiguous), [a] ambiguous, [s] skip, [q] quit")
    print("=" * 80)

    for _, rep in reps.iterrows():
        qkey = rep["_qkey"]
        if state.get(qkey):
            continue

        # Gather full block (all rows for this question) for saving later
        block = df[df["_qkey"] == qkey].copy()

        # Use representative fields for the prompt preview
        question = str(rep[q_col])
        gold = str(rep[gold_col])
        cands_raw = rep[cand_col]
        cands_list = parse_candidates_for_prompt(cands_raw)

        # Call LLM for explanation (not saved)
        try:
            explanation = call_llm(build_messages(question, gold, cands_list))
        except Exception as e:
            print(f"\nLLM error: {e}\nSkipping for now…")
            time.sleep(1)
            continue

        # Show to you
        print("\n" + "-" * 80)
        print(f"QUESTION:\n{question}\n")
        print(f"GOLD ANSWER:\n{gold}\n")
        print("CANDIDATES:")
        for c in cands_list:
            print(f"- {c}")
        print("\nLLM EXPLANATION:\n" + explanation)
        print("-" * 80)

        choice = input("Your decision [k/a/s/q]: ").strip().lower()
        if choice == "q":
            save_state(state)
            print("Exiting. Progress saved.")
            return
        elif choice == "s":
            print("Skipped; will reappear next run.")
            continue
        elif choice == "a":
            append_df(AMBIG_CSV, block.drop(columns=["_q_norm", "_qkey"], errors="ignore"))
            state[qkey] = True
            save_state(state)
            print(f"→ Appended {len(block)} row(s) to {AMBIG_CSV}")
        elif choice == "k":
            append_df(KEEP_CSV, block.drop(columns=["_q_norm", "_qkey"], errors="ignore"))
            state[qkey] = True
            save_state(state)
            print(f"→ Appended {len(block)} row(s) to {KEEP_CSV}")
        else:
            print("Invalid choice. Skipping (will reappear next run).")

    print("\nAll unique questions processed. 🎉")

if __name__ == "__main__":
    main()