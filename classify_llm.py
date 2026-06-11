"""
classify_llm.py — LLM Classification Pipeline
===============================================
Runs the balanced test set (balanced_test_sample.csv) through the Anthropic API
and saves two result files:

  llm_results_zero_shot.csv  — classified with no examples
  llm_results_few_shot.csv   — classified with 6 labeled examples in the prompt

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python classify_llm.py

  # Or pass key inline:
  ANTHROPIC_API_KEY="sk-ant-..." python classify_llm.py

  # Dry run (first 10 tweets only, to test before spending API credits):
  python classify_llm.py --dry-run

Output columns added to the original test CSV:
  pred_leaning   — "Democrat" or "Republican"
  pred_stance    — "Populist" or "Establishment"
  confidence     — integer 0–100
  reasoning      — one-sentence explanation from the model
  raw_response   — full raw text returned by API (for debugging)
  parse_error    — True if the JSON response couldn't be parsed
"""
import pandas as pd
import os
import json
import time
import argparse
import anthropic

from prompts import SYSTEM_PROMPT, build_zero_shot_prompt, build_few_shot_prompt

# ---------------------------------------------------------------------------
# CONFIG — adjust these if needed
# ---------------------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 300       # enough for the JSON response, not wasteful
RETRY_LIMIT    = 3         # how many times to retry a failed API call
RETRY_DELAY    = 5         # seconds to wait between retries
DRY_RUN_LIMIT  = 10        # number of tweets used in --dry-run mode

TEST_CSV       = "balanced_test_sample.csv"
TRAINING_PARQ  = "cleaned_all_data.parquet"

OUTPUT_ZERO    = "llm_results_zero_shot.csv"
OUTPUT_FEW     = "llm_results_few_shot.csv"
# ---------------------------------------------------------------------------


def call_api(client: anthropic.Anthropic, user_message: str) -> dict:
    """
    Sends one tweet to the API and returns the parsed JSON result.
    Retries up to RETRY_LIMIT times on failure.

    Returns a dict with keys: pred_leaning, pred_stance, confidence, reasoning,
    raw_response, parse_error
    """
    raw_text = ""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = response.content[0].text.strip()

            # Strip markdown code fences if the model adds them (it shouldn't, but safety)
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            parsed = json.loads(raw_text)

            return {
                "pred_leaning":  parsed.get("leaning", "PARSE_ERROR"),
                "pred_stance":   parsed.get("stance",  "PARSE_ERROR"),
                "confidence":    parsed.get("confidence", -1),
                "reasoning":     parsed.get("reasoning", ""),
                "raw_response":  raw_text,
                "parse_error":   False,
            }

        except json.JSONDecodeError:
            # Model returned something we can't parse — save it and flag it
            return {
                "pred_leaning": "PARSE_ERROR",
                "pred_stance":  "PARSE_ERROR",
                "confidence":   -1,
                "reasoning":    "",
                "raw_response": raw_text,
                "parse_error":  True,
            }

        except anthropic.RateLimitError:
            print(f"  Rate limit hit. Waiting {RETRY_DELAY * attempt}s before retry {attempt}/{RETRY_LIMIT}...")
            time.sleep(RETRY_DELAY * attempt)

        except anthropic.APIError as e:
            print(f"  API error on attempt {attempt}: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    # All retries exhausted
    return {
        "pred_leaning": "API_ERROR",
        "pred_stance":  "API_ERROR",
        "confidence":   -1,
        "reasoning":    "All retries failed",
        "raw_response": raw_text,
        "parse_error":  True,
    }


def run_classification(
    test_df: pd.DataFrame,
    client: anthropic.Anthropic,
    mode: str,          # "zero_shot" or "few_shot"
    training_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Loops through every tweet in test_df, calls the API, and appends result columns.

    Args:
        test_df:     The 250-tweet evaluation set
        client:      Authenticated Anthropic client
        mode:        "zero_shot" or "few_shot"
        training_df: Required for few-shot mode; ignored for zero-shot

    Returns:
        A copy of test_df with prediction columns added
    """
    results = []
    total = len(test_df)

    print(f"\n{'='*60}")
    print(f"Running {mode.replace('_', '-')} classification on {total} tweets...")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(test_df.iterrows()):
        tweet_text = row["text"]

        # Build the appropriate prompt
        if mode == "zero_shot":
            user_message = build_zero_shot_prompt(tweet_text)
        else:
            user_message = build_few_shot_prompt(tweet_text, training_df)

        result = call_api(client, user_message)
        results.append(result)

        # Progress indicator every 10 tweets
        if (i + 1) % 10 == 0 or (i + 1) == total:
            correct_so_far = sum(
                1 for r, (_, row) in zip(results, test_df.iterrows())
                if r["pred_leaning"] == row["party"] and not r["parse_error"]
            )
            print(f"  [{i+1}/{total}] Running accuracy: {correct_so_far}/{i+1} = {correct_so_far/(i+1):.1%}")

        # Small delay to stay well under rate limits (free tier: ~50 req/min)
        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    return pd.concat([test_df.reset_index(drop=True), results_df], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Run LLM classification pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Only classify first {DRY_RUN_LIMIT} tweets (for testing before spending credits)"
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot", "both"],
        default="both",
        help="Which classification mode to run (default: both)"
    )
    args = parser.parse_args()

    # --- API key check ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # --- Load data ---
    print(f"Loading test set from {TEST_CSV}...")
    test_df = pd.read_csv(TEST_CSV)

    if args.dry_run:
        print(f"DRY RUN: using first {DRY_RUN_LIMIT} tweets only")
        test_df = test_df.head(DRY_RUN_LIMIT)

    print(f"  Loaded {len(test_df)} tweets")
    print(f"  Party distribution:\n{test_df['party'].value_counts().to_string()}")
    print(f"  Edge cases: {test_df['is_edge_case'].sum()}")

    training_df = None
    if args.mode in ("few_shot", "both"):
        print(f"\nLoading training pool from {TRAINING_PARQ}...")
        training_df = pd.read_parquet(TRAINING_PARQ)
        print(f"  Loaded {len(training_df)} tweets for few-shot sampling")

    # --- Run classification ---
    if args.mode in ("zero_shot", "both"):
        zero_results = run_classification(test_df, client, "zero_shot")
        zero_results.to_csv(OUTPUT_ZERO, index=False)
        print(f"\nSaved zero-shot results to {OUTPUT_ZERO}")

    if args.mode in ("few_shot", "both"):
        few_results = run_classification(test_df, client, "few_shot", training_df)
        few_results.to_csv(OUTPUT_FEW, index=False)
        print(f"\nSaved few-shot results to {OUTPUT_FEW}")

    print("\nDone. Run `python evaluate.py` to see accuracy breakdowns.")


if __name__ == "__main__":
    main()
