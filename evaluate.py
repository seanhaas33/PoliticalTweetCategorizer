"""
evaluate.py — Evaluation & Failure Analysis
=============================================
Reads the output CSVs from classify_llm.py and prints a full breakdown of
model performance. Also saves a failure analysis CSV for your report.

Usage:
  python evaluate.py                          # evaluates both result files
  python evaluate.py --file llm_results_zero_shot.csv   # one file only

What this produces:
  1. Overall accuracy (zero-shot vs. few-shot comparison)
  2. Accuracy by party (Democrat vs. Republican)
  3. Accuracy by era (trump-era-early / trump-era-late / biden-era)
  4. Accuracy: normal cases vs. edge cases
  5. Accuracy by individual edge-case senator
  6. Confidence calibration — is high confidence actually more accurate?
  7. failure_analysis.csv — all misclassified tweets for your report
"""

import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def accuracy(df: pd.DataFrame) -> float:
    """Overall accuracy, ignoring parse errors."""
    valid = df[~df["parse_error"]]
    if len(valid) == 0:
        return 0.0
    correct = (valid["pred_leaning"] == valid["party"]).sum()
    return correct / len(valid)


def accuracy_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Returns a DataFrame with accuracy and sample size per group."""
    valid = df[~df["parse_error"]]
    rows = []
    for group, subset in valid.groupby(group_col):
        n = len(subset)
        n_correct = (subset["pred_leaning"] == subset["party"]).sum()
        rows.append({
            group_col:   group,
            "n_tweets":  n,
            "n_correct": n_correct,
            "accuracy":  n_correct / n if n > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Runs the full evaluation suite on one results DataFrame.
    Returns a DataFrame of misclassified rows for failure export.
    """
    print(f"\n{'='*60}")
    print(f"  RESULTS: {label}")
    print(f"{'='*60}")

    parse_errors = df["parse_error"].sum()
    if parse_errors > 0:
        print(f"  ⚠ Parse errors (excluded from accuracy): {parse_errors}")

    valid = df[~df["parse_error"]]

    # ------------------------------------------------------------------
    # 1. Overall accuracy
    # ------------------------------------------------------------------
    print_section("1. Overall Accuracy")
    overall = accuracy(df)
    print(f"  {overall:.1%}  ({(valid['pred_leaning'] == valid['party']).sum()} / {len(valid)} correct)")

    # ------------------------------------------------------------------
    # 2. By party
    # ------------------------------------------------------------------
    print_section("2. Accuracy by Party")
    party_acc = accuracy_by_group(valid, "party")
    print(party_acc.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. By era
    # ------------------------------------------------------------------
    print_section("3. Accuracy by Political Era")
    era_acc = accuracy_by_group(valid, "era")
    print(era_acc.to_string(index=False))
    print("\n  NOTE: biden-era dominates (~99% of data). Cross-era numbers")
    print("  have low statistical power — flag this in the report.")

    # ------------------------------------------------------------------
    # 4. Normal vs. edge cases
    # ------------------------------------------------------------------
    print_section("4. Normal Cases vs. Edge Cases")
    normal = valid[~valid["is_edge_case"]]
    edge   = valid[ valid["is_edge_case"]]

    def _acc_line(subset, name):
        n = len(subset)
        c = (subset["pred_leaning"] == subset["party"]).sum()
        pct = c/n if n > 0 else 0.0
        return f"  {name:<20} {pct:.1%}  ({c}/{n})"

    print(_acc_line(normal, "Normal cases"))
    print(_acc_line(edge,   "Edge cases"))

    # ------------------------------------------------------------------
    # 5. Per edge-case senator
    # ------------------------------------------------------------------
    if "username" in valid.columns:
        print_section("5. Accuracy per Edge-Case Senator")
        edge_senators = valid[valid["is_edge_case"]]
        if len(edge_senators) > 0:
            senator_acc = accuracy_by_group(edge_senators, "username")
            print(senator_acc.to_string(index=False))
        else:
            print("  No edge-case rows found in this file.")

    # ------------------------------------------------------------------
    # 6. Confidence calibration
    # ------------------------------------------------------------------
    print_section("6. Confidence Calibration")
    print("  (Are high-confidence predictions actually more accurate?)\n")

    bins = [(0, 50), (50, 70), (70, 85), (85, 95), (95, 101)]
    for lo, hi in bins:
        bucket = valid[(valid["confidence"] >= lo) & (valid["confidence"] < hi)]
        if len(bucket) == 0:
            continue
        n = len(bucket)
        c = (bucket["pred_leaning"] == bucket["party"]).sum()
        print(f"  Confidence {lo:>3}–{hi-1:<3}: {c/n:.1%}  ({c}/{n})")

    # ------------------------------------------------------------------
    # 7. Failures
    # ------------------------------------------------------------------
    print_section("7. Sample Failures (first 5)")
    failures = valid[valid["pred_leaning"] != valid["party"]]
    print(f"  Total misclassified: {len(failures)} / {len(valid)}\n")

    for _, row in failures.head(5).iterrows():
        print(f"  USERNAME:   {row.get('username', 'N/A')}")
        print(f"  TWEET:      {row['text'][:120]}...")
        print(f"  TRUE LABEL: {row['party']}")
        print(f"  PREDICTED:  {row['pred_leaning']}  (confidence: {row['confidence']})")
        print(f"  REASONING:  {row['reasoning']}")
        print()

    return failures


# ---------------------------------------------------------------------------
# COMPARISON TABLE (zero-shot vs. few-shot)
# ---------------------------------------------------------------------------

def comparison_table(results: dict):
    """Prints a side-by-side summary if multiple result files are loaded."""
    print(f"\n{'='*60}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*60}")

    rows = []
    for label, df in results.items():
        valid = df[~df["parse_error"]]
        n = len(valid)
        edge = valid[valid["is_edge_case"]]
        normal = valid[~valid["is_edge_case"]]

        rows.append({
            "Mode":            label,
            "Overall":         f"{accuracy(df):.1%}",
            "Normal cases":    f"{(normal['pred_leaning'] == normal['party']).sum() / len(normal):.1%}" if len(normal) else "—",
            "Edge cases":      f"{(edge['pred_leaning'] == edge['party']).sum() / len(edge):.1%}" if len(edge) else "—",
            "Parse errors":    df["parse_error"].sum(),
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM classification results")
    parser.add_argument("--file", type=str, default=None,
                        help="Evaluate a single result file instead of both defaults")
    args = parser.parse_args()

    if args.file:
        files = {args.file: args.file}
    else:
        files = {
            "zero-shot": "llm_results_zero_shot.csv",
            "few-shot":  "llm_results_few_shot.csv",
        }

    loaded = {}
    all_failures = []

    for label, path in files.items():
        try:
            df = pd.read_csv(path)
            # Ensure parse_error column exists (older runs may not have it)
            if "parse_error" not in df.columns:
                df["parse_error"] = False
            df["parse_error"] = df["parse_error"].fillna(False).astype(bool)
            loaded[label] = df
        except FileNotFoundError:
            print(f"  ⚠ File not found: {path} — skipping")

    if not loaded:
        print("No result files found. Run classify_llm.py first.")
        return

    for label, df in loaded.items():
        failures = analyze(df, label)
        failures["mode"] = label
        all_failures.append(failures)

    if len(loaded) > 1:
        comparison_table(loaded)

    # Save all failures to one CSV for the report
    if all_failures:
        failure_df = pd.concat(all_failures, ignore_index=True)
        failure_df.to_csv("failure_analysis.csv", index=False)
        print(f"\n  Failure analysis saved to failure_analysis.csv ({len(failure_df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
