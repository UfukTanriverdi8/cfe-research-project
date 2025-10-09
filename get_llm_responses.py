"""
Run a verified QA dataset benchmark against 5 models and 3 different system-prompt approaches.

Written for the OpenAI-compatible Python SDK.  Requires:
  pip install openai python-dotenv pandas tqdm
"""

import os, json, time, csv, sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, OpenAIError
from json.decoder import JSONDecodeError


# ─────────────────────────────  CONFIG  ──────────────────────────────
current_dir = Path(__file__).parent.resolve()
QUESTIONS_JSON   = os.path.join(current_dir, "./data/questions/100_questions.json")          # input
RESULT_CSV       = os.path.join(current_dir, "./data/results/100q_results_new_cfe.csv")       # output
RETRIES          = 3                               # per request
TEMPERATURE      = 0.0                             # deterministic

# ❶── system-prompt variants (edit to taste) ─────────────────────────
SYSTEM_PROMPTS = {
    "baseline": (
        "You are an expert QA assistant. Provide an answer to the user’s question."
    ),
    "clarify-doubts":(
        "You are an expert QA assistant. "
        "Provide an answer to the user’s question. Clarify doubts only if it’s needed."
    ),
    
    "cfe": (
        "You are a Counterfactual Explanatory QA assistant. "
        "For each question, state the correct answer and then if there are plausible but incorrect alternative answers to the question, "
        "provide clarifications, explaining why they are not correct. If not needed, do not provide clarifications and simply give the correct answer. "
        "Never reveal your reasoning or mention that you judged the question. "
    ),
    "cfe-unrestricted": (
        "You are a Counterfactual Explanatory QA assistant."
        "For each question, state the correct answer and then provide clarifications "
        "that address plausible but incorrect alternative answers to the question, explaining why they are not correct. "
    ),
    "clarify-doubts-unrestricted": (
        "You are an expert QA assistant. "
        "Provide an answer to the user’s question. Clarify doubts."
    )
}


# ❷── model roster & routing info ────────────────────────────────────
#
# Each entry: (friendly_name, model_id, base_url_env_var, api_key_env_var)
MODELS = [
    ("qwen-2.5-7b",
    "qwen/qwen-2.5-7b-instruct",
    "OPENROUTER_BASE_URL","OPENROUTER_API_KEY"),       # base_url set below

    ("llama-3.1-8b",
     "meta-llama/llama-3.1-8b-instruct",
     "OPENROUTER_BASE_URL","OPENROUTER_API_KEY"),


    ("qwen-2.5-72b",
     "qwen/qwen-2.5-72b-instruct",
     "OPENROUTER_BASE_URL","OPENROUTER_API_KEY"),

    ("llama-3.3-70b",
     "meta-llama/llama-3.3-70b-instruct",
     "OPENROUTER_BASE_URL","OPENROUTER_API_KEY"),

    # gold-standard – OpenAI
    ("gpt-4o",
     "gpt-4o",
     "OPENAI_DEFAULT",  "OPENAI_API_KEY"),
]

# default URLs for each provider
BASE_URLS = {
    "OPENAI_DEFAULT"     : "https://api.openai.com/v1",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    "TOGETHER_BASE_URL"  : "https://api.together.xyz/v1",  # :contentReference[oaicite:0]{index=0}
}

# Optional OpenRouter ranking headers
EXTRA_HEADERS = {
    "HTTP-Referer": os.getenv("HTTP_REFERER", "<YOUR_SITE_URL>"),
    "X-Title"    : os.getenv("X_TITLE",     "<YOUR_SITE_NAME>"),
}

# ───────────────────────── END CONFIG ───────────────────────────────

load_dotenv()  # read API keys from .env

def make_client(base_url_key: str, api_key_env: str) -> OpenAI:
    """Return an OpenAI-compatible client for the given provider."""
    return OpenAI(
        base_url = BASE_URLS[base_url_key],
        api_key  = os.getenv(api_key_env)
    )

def call_model(client: OpenAI, model_id: str,
               sys_prompt: str, user_prompt: str,
               base_url: str) -> str:
    """
    Robust call with simple retry.
    Only injects EXTRA_HEADERS when base_url is the OpenRouter endpoint.
    Logs any exception for visibility.
    """
    for attempt in range(RETRIES):
        if attempt > 0:
            tqdm.write(f"[RETRY] {model_id} at {base_url} (attempt {attempt + 1})")
        try:
            params = {
                "model": model_id,
                "messages": [
                    {"role": "system",  "content": sys_prompt},
                    {"role": "user",    "content": user_prompt},
                ],
                "temperature": TEMPERATURE,
            }
            # only OpenRouter needs these ranking headers
            if base_url.startswith("https://openrouter.ai"):
                params["extra_headers"] = EXTRA_HEADERS

            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content.strip()
        except JSONDecodeError as e:
            tqdm.write(f"[DEBUG] JSON decode error (attempt {attempt + 1}) for {model_id}: {e}")

        except OpenAIError as e:
            # print out what the service actually returned
            tqdm.write(f"[ERROR] OpenAIError for {model_id} at {base_url}: {e}")
            # if error is a rate limit error, we can retry
            if "rate limit" in str(e).lower():
                tqdm.write(f"[INFO] Rate limit error for {model_id} at {base_url}, retrying priced model instead -> {model_id.split(":")[0]}")
                # drop the free from the model name
                call_model(client, model_id[:-6],  # remove ":free" suffix
                        sys_prompt, user_prompt, base_url)
        except Exception as e:
            # catch anything else
            tqdm.write(f"[ERROR] Exception for {model_id} at {base_url}: {e}")

        time.sleep(1 + attempt * 0.5)  # back off a bit longer each time
    return "[ERROR]"


def main():
    # load the 100 questions
    data = json.loads(Path(QUESTIONS_JSON).read_text(encoding="utf-8"))
    rows = []

    # pre-create one client per provider to reuse connections
    clients = {
        base_key : make_client(base_key, api_key_env)
        for _, _, base_key, api_key_env in MODELS
    }

    for q in tqdm(data, desc="questions"):
        question_text   = q["question"]
        correct_answer  = q["answer"]
        cand_dict = q.get("candidate_answers", {})
        # sort answer texts by their listwise score descending
        candidate_ans = sorted(
            cand_dict.keys(),
            key=lambda ans: cand_dict[ans].get("listwise", 0),
            reverse=True
        )
        # add listwise scores to the candidates
        candidate_ans = [
            f"{ans} ({cand_dict[ans].get('listwise', 0)})"
            for ans in candidate_ans
        ]

        user_prompt = (
            f"{question_text}"
        )

        for prompt_label, system_prompt in SYSTEM_PROMPTS.items():
            if prompt_label in ["baseline", "clarify-doubts", "cfe-unrestricted", "clarify-doubts-unrestricted"]:
                continue  # skip these prompts for now
            for name, model_id, base_key, _ in MODELS:

                client = clients[base_key]
                base_url = BASE_URLS[base_key]
                response = call_model(
                    client,
                    model_id,
                    system_prompt,
                    user_prompt,
                    base_url
                )

                # --- NEW: print progress to terminal ---
                debug_block = "\n".join([
                    "=" * 80,
                    f"Model:           {name}",
                    f"Prompt Variant:  {prompt_label}",
                    f"Question:        {question_text}",
                    f"Gold Answer:     {correct_answer}",
                    f"Candidates:      {', '.join(candidate_ans)}",
                    f"LLM Response:    {response}",
                    f"Response Length: {len(response)} chars",
                    "=" * 80,
                ])
                tqdm.write(debug_block)   # <-- safe print
                
                rows.append(
                    {
                        "id"               : q["id"],
                        "prompt_variant"   : prompt_label,
                        "model"            : name,
                        "question"         : question_text,
                        "gold_answer"      : correct_answer,
                        "candidate_answers": "|".join(candidate_ans),
                        "llm_response"     : response,
                        "response_length"  : len(response),
                        "score"            : ""
                    }
                )
                time.sleep(0.5)
                # save intermediate results to avoid losing data
        """ tqdm.write(f"Saving intermediate results to {RESULT_CSV}...")
        df = pd.DataFrame(rows)
        df.to_csv(RESULT_CSV, index=False)
        tqdm.write(f"Saved {len(df)} rows → {RESULT_CSV}") """

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False)
    tqdm.write(f"Saved {len(df)} rows → {RESULT_CSV}")

if __name__ == "__main__":
    if not Path(QUESTIONS_JSON).exists():
        sys.exit(f"❌  File not found: {QUESTIONS_JSON}")
    main()