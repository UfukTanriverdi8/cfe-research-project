import pandas as pd
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import time
load_dotenv()

# --- EVAL_CLIENT setup ---
EVAL_CLIENT = OpenAI(
    #base_url=os.getenv("OPENROUTER_BASE_URL"),
    base_url = None,
    api_key=os.getenv("OPENAI_API_KEY")
)
EVAL_MODEL  = "gpt-4o"
EVAL_TIMEOUT = 3  # retries

# Placeholder prompt for the evaluator LLM (you can edit this)
EVAL_SYSTEM_PROMPT = (
    "You are an expert Question Evaluator. Your task is to evaluate the question and responses of a dataset. There is exact one correct answer for each question and you'll be given that answer. "
    "Other than the correct answer, you'll see the responses given by different language models."
    "First you'll decide whether the question is confusing or not. By 'confusing' we mean that the questions that can have other plausible sounding but incorrect answers."
    "You will represent your confusability decision as a score from 1 to 5, where 1 means the question does not have any plausible other answer and 5 means the question is very confusing and has many plausible sounding but incorrect answers."
    "After deciding on that, you will then score the model's response on a scale of 1 to 5, where 1 is a very poor response and 5 is an excellent response."
    "Your evaluation process will be changed on the question type, whether how confusing it is or not."
    "If you've decided the question is not a confusing one, you can just evaluate the model's response based on the accuracy and the quality of the response. Response should not talk about other plausible answers, having a correct answer is enough. If the question is not confusing, the conciseness is the key."
    "If you've decided the question is confusing, the model's response MUST give the correct answer and then it MUST also explain why the other plausible answers are incorrect by clarifying potential doubts. So you will evaluate the model's response based on these criteria."
    "You MUST provide clear explanations for your confusability score and the model's response score."
    "Remember that if the question is confusing, the given response does not have to clarify every possible confusion, but it should address the most common ones."
    "Think about your evaluation process as a human evaluator would, consider other plausible answers for the question, decide on the confusability and based on that evaluate how well the model's response addresses those."
)


current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(current_dir, "test_results.csv")

def prompt_confusing():
    while True:
        val = input("Is this question confusing? [y/n] (blank to quit): ").strip().lower()
        if val == "" or val in ("q","quit"):
            return None
        if val in ("y","yes"):
            return "yes"
        if val in ("n","no"):
            return "no"
        print("  ⚠️  Please enter 'y' or 'n', or blank to quit.")

def prompt_score():
    while True:
        val = input("Enter score [1–5] (blank to quit): ").strip()
        if val == "" or val.lower() in ("q","quit"):
            return None
        if val.isdigit() and 1 <= int(val) <= 5:
            return val
        print("  ⚠️  Invalid—please enter a number from 1 to 5, or blank to quit.")

def call_evaluator(question, gold, candidates, response):
    """Call evaluator to get an automatic evaluation."""
    user_msg = (
        f"Question: {question}\n"
        f"Correct Answer: {gold}\n"
        #f"Candidates: {', '.join(candidates)}\n"
        f"Model Response: {response}\n\n"
        "Please evaluate the clarity of the question and adequacy of the response based on the confusability of the question."
    )
    for _ in range(EVAL_TIMEOUT):
        try:
            resp = EVAL_CLIENT.chat.completions.create(
                model=EVAL_MODEL,
                extra_headers={
                    "HTTP-Referer": os.getenv("HTTP_REFERER", ""),
                    "X-Title":      os.getenv("X_TITLE",     ""),
                },
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=250,
            )
            return resp.choices[0].message.content.strip()
        except OpenAIError as e:
            print(f"❌  OpenAI API error: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"❌  Error calling evaluator: {e}")
            print("Please check your API key and network connection.")
            sys.exit(1)
    return "[EVAL_ERROR]"


def main():
    # Load CSV
    try:
        df = pd.read_csv(CSV_PATH, dtype=str, na_filter=False)
    except FileNotFoundError:
        print(f"❌  Could not find {CSV_PATH}")
        sys.exit(1)

    # Ensure columns exist
    if "confusing" not in df.columns:
        df["confusing"] = ""
    if "score" not in df.columns:
        df["score"] = ""
    df.to_csv(CSV_PATH, index=False)  # write back if we added columns

    # Iterate rows
    for idx, row in df.iterrows():
        done_confusing = str(row["confusing"]).strip() != ""
        done_score     = str(row["score"]).strip()     != ""
        if done_confusing and done_score:
            continue

        eval_text = call_evaluator(
            row["question"],
            row["gold_answer"],
            row["candidate_answers"].split("|"),
            row["llm_response"]
        )


        print("\n" + "="*80)
        print(f"ROW #{idx}")
        print(f"Model:            {row['model']}")
        print(f"Prompt Variant:   {row['prompt_variant']}")
        print(f"Question:         {row['question']}")
        print(f"Gold Answer:      {row['gold_answer']}")
        print(f"Candidates:       {row['candidate_answers']}")
        print(f"LLM Response:\n{row['llm_response']}")
        print("-"*40)
        print(f"Evaluation Result:\n{eval_text}")
        print("="*80)
        # 1) confusing?
        if not done_confusing:
            conf = prompt_confusing()
            if conf is None:
                print("🔌  Exiting early. Progress saved.")
                break
     
            # propagate to all rows with the same question
            df.loc[df["question"] == row["question"], "confusing"] = conf
            df.to_csv(CSV_PATH, index=False)
            print(f"✅  Recorded confusing={conf} for all rows of this question")

        # 2) score?
        if not done_score:
            sc = prompt_score()
            if sc is None:
                print("🔌  Exiting early. Progress saved.")
                break
            df.at[idx, "score"] = sc
            df.to_csv(CSV_PATH, index=False)
            print(f"✅  Recorded score={sc}")

    else:
        print("\n🎉  All rows labeled and scored!")

if __name__ == "__main__":
    main()