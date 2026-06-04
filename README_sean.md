# Sean's LLM Classification Pipeline

This is the LLM classification portion of the project. It reads Junda's
cleaned data and runs it through the Anthropic API, then evaluates the results.

---

## File Overview

| File | What it does |
|---|---|
| `prompts.py` | All prompt logic — edit this to experiment with different framings |
| `classify_llm.py` | Main pipeline — reads the CSV, calls the API, saves results |
| `evaluate.py` | Reads results, prints accuracy breakdowns and failure analysis |
| `requirements_sean.txt` | Just `anthropic` — everything else is in Junda's requirements.txt |

---

## Setup (one time)

```bash
# 1. Install Junda's requirements first
pip install -r requirements.txt

# 2. Install the Anthropic SDK
pip install -r requirements_sean.txt

# 3. Set your API key (get one at console.anthropic.com)
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Running the Pipeline

### Step 1 — Run Junda's data pipeline first (if you haven't)
```bash
python clean_data.py
# produces: balanced_test_sample.csv and cleaned_all_data.parquet
```

### Step 2 — Dry run (test 10 tweets, costs ~$0.01)
```bash
python classify_llm.py --dry-run
```
Check that the output looks right before running all 250.

### Step 3 — Full run (all 250 tweets, both modes)
```bash
python classify_llm.py
# produces: llm_results_zero_shot.csv and llm_results_few_shot.csv
```

### Step 4 — Evaluate
```bash
python evaluate.py
# prints full accuracy breakdown, saves failure_analysis.csv
```

---

## Output Files

After running, you'll have:

- `llm_results_zero_shot.csv` — test set with zero-shot predictions added
- `llm_results_few_shot.csv` — test set with few-shot predictions added
- `failure_analysis.csv` — all misclassified tweets for the report

Each results CSV has all of Junda's original columns plus:

| Column | Description |
|---|---|
| `pred_leaning` | Model's prediction: Democrat or Republican |
| `pred_stance` | Model's populism label: Populist or Establishment |
| `confidence` | Model's self-reported confidence (0–100) |
| `reasoning` | One-sentence explanation from the model |
| `parse_error` | True if the API response couldn't be parsed |

---

## Estimated API Cost

| Run | Tweets | Approx. cost |
|---|---|---|
| Dry run | 10 | ~$0.01 |
| Zero-shot full | 250 | ~$0.25–0.50 |
| Few-shot full | 250 | ~$0.40–0.80 |
| **Total** | | **~$1–2** |

---

## Experimenting with the Prompt

All prompt logic lives in `prompts.py`. The most impactful things to try:

1. **Change the label descriptions** in `SYSTEM_PROMPT` — how you define
   "Populist" vs. "Establishment" will shift results significantly.

2. **Add context about the time period** — telling the model "it's 2021,
   post-Trump" may improve era-specific accuracy.

3. **Change n_per_party in few-shot** — try 2, 3, or 5 examples per party
   and see if more examples help or hurt edge cases.

Run `--mode zero_shot` or `--mode few_shot` to test one at a time:
```bash
python classify_llm.py --mode zero_shot --dry-run
```

---

## Mapping to Rubric Requirements

| Rubric item | Where it's covered |
|---|---|
| AI Method | `classify_llm.py` — LLM prompting pipeline |
| Semantic Layer | `prompts.py` — populism dimension + reasoning |
| Evaluation Results | `evaluate.py` — overall + subgroup accuracy |
| Failure Analysis | `evaluate.py` sections 4–7 + `failure_analysis.csv` |
| Responsible Use | Report section (use failure analysis findings as evidence) |
