"""
prompts.py — All LLM prompt logic in one place.

Two prompt builders:
  - build_zero_shot_prompt(tweet_text)  → no examples, just instructions
  - build_few_shot_prompt(tweet_text, examples_df) → prepends labeled examples

Tweak the SYSTEM_PROMPT string here to experiment with different framings.
The model is expected to return ONLY valid JSON — no extra text.
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
# This is the instruction given to the model before every tweet.
# Key decisions encoded here:
#   1. Labels are strictly Democrat / Republican (matches ground truth column)
#   2. A second dimension — Populist / Establishment — is included even though
#      the dataset has no ground-truth label for it. This satisfies the
#      semantic layer rubric requirement; treat it as qualitative output.
#   3. Confidence (0–100) gives you a soft signal for borderline cases.
#   4. Strict JSON-only output so parsing never breaks.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert political rhetoric classifier.

You will be given a single tweet written by a U.S. Senator. Your job is to classify it along two dimensions:

DIMENSION 1 — Political Leaning
  Choose exactly one: "Democrat" or "Republican"
  Base this on the rhetorical framing, policy positions, and messaging style present in the tweet.

DIMENSION 2 — Establishment Stance
  Choose exactly one: "Populist" or "Establishment"
  - Populist: attacks elites, institutions, or the system; appeals to "the people" vs. a corrupt class
  - Establishment: defends institutions, norms, or incremental policy; avoids anti-elite framing

Also provide:
  - confidence: an integer from 0 to 100 reflecting how certain you are about DIMENSION 1
  - reasoning: one sentence explaining the key signals that drove your classification

IMPORTANT RULES:
  - Respond ONLY with a valid JSON object. No preamble, no explanation outside the JSON.
  - Do not let the senator's name or known affiliation influence you — classify the TEXT only.
  - If the tweet is ambiguous, still commit to the most likely label and lower your confidence score.

Required output format (copy exactly, fill in values):
{
  "leaning": "Democrat",
  "stance": "Populist",
  "confidence": 85,
  "reasoning": "The tweet frames economic hardship as caused by corporate greed, a classic left-populist framing."
}"""


# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLE BUILDER
# ---------------------------------------------------------------------------
# Selects N balanced examples from cleaned_all_data.parquet and formats them
# as labeled demonstrations prepended to the tweet being classified.
#
# Per Junda's warning: NEVER sample from rows where is_edge_case == True.
# ---------------------------------------------------------------------------

def build_few_shot_examples(examples_df, n_per_party: int = 3) -> str:
    """
    Samples n_per_party tweets from each party (Democrat / Republican)
    from the cleaned training pool and formats them as labeled examples.

    Args:
        examples_df: DataFrame loaded from cleaned_all_data.parquet
        n_per_party: How many examples per party to include (default 3 → 6 total)

    Returns:
        A formatted string block to prepend to the user message.
    """
    # Safety check: exclude edge-case senators per Junda's warning
    safe_pool = examples_df[~examples_df["is_edge_case"]]

    dem_examples = (
        safe_pool[safe_pool["party"] == "Democrat"]
        .sample(n=n_per_party, random_state=99)
    )
    rep_examples = (
        safe_pool[safe_pool["party"] == "Republican"]
        .sample(n=n_per_party, random_state=99)
    )

    all_examples = (
        pd.concat([dem_examples, rep_examples])
        .sample(frac=1, random_state=99)  # shuffle so party order isn't obvious
        .reset_index(drop=True)
    )

    lines = ["Here are some labeled examples to guide your classification:\n"]
    for i, row in all_examples.iterrows():
        lines.append(f"Example {i+1}:")
        lines.append(f"  Tweet: {row['text']}")
        lines.append(f"  Correct label: {row['party']}\n")

    lines.append("Now classify the following tweet using the same format:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(tweet_text: str) -> str:
    """Returns the user message for zero-shot classification (no examples)."""
    return f"Classify this tweet:\n\n{tweet_text}"


def build_few_shot_prompt(tweet_text: str, examples_df) -> str:
    """Returns the user message for few-shot classification (with examples)."""
    import pandas as pd  # local import so this file stays importable standalone
    example_block = build_few_shot_examples(examples_df)
    return f"{example_block}\n\nTweet to classify:\n{tweet_text}"
