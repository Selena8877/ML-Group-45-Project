"""
Error analysis of the best model (Milestone 5, Person 3).

1. Picks the best model: highest mean test Macro-F1 in shared_evaluation_results.csv
   (or pass a model name: python src/error_analysis_best.py "Logistic Regression").
2. Finds that model's test predictions in any results/*_predictions.csv file.
3. Counts errors per language, compares them with KNN (which errors are fixed or new),
   and saves example posts and simple error patterns (length, keyword presence).

Run from the repository root:
    python src/error_analysis_best.py

Outputs (results/):
    best_model_error_summary.csv   TP/TN/FP/FN per language, plus errors fixed/new vs KNN
    best_model_error_patterns.csv  median post length and keyword rate by outcome
    best_model_error_examples.csv  the misclassified test posts with their text
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_pipeline import load_splits  # noqa: E402

RESULTS = ROOT / "results"
LANGUAGES = ["Arabic", "Chinese", "Portuguese", "Thai"]
KEYWORDS = {"Arabic": "اكتئاب", "Chinese": "抑郁", "Portuguese": "suicid", "Thai": "ซึมเศร้า"}
KNN_NAME = "KNN (CountVectorizer)"


def load_all_predictions():
    """Collect test predictions from every *_predictions.csv into one table."""
    frames = []
    for path in RESULTS.glob("*_predictions.csv"):
        # Skip files that hold only a subset of predictions (errors only).
        if path.name.startswith(("best_model", "misclassified")):
            continue
        df = pd.read_csv(path)
        if not {"id", "y_true", "y_pred"}.issubset(df.columns) or "model" not in df.columns:
            continue
        if "split" in df.columns:
            df = df[df["split"] == "test"]
        if "variant" in df.columns:
            df = df[df["variant"].isna() | (df["variant"] == "original")]
        df["language"] = df["language"].str.capitalize()
        frames.append(df[["model", "language", "id", "y_true", "y_pred"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["model", "id"])


def pick_best_model():
    if len(sys.argv) > 1:
        return sys.argv[1]
    df = pd.read_csv(RESULTS / "shared_evaluation_results.csv")
    df = df[df["split"] == "test"]
    if "variant" in df.columns:
        df = df[df["variant"].isna() | (df["variant"] == "original")]
    return df.groupby("model")["f1_macro"].mean().idxmax()


def outcome(row):
    if row.y_true == row.y_pred:
        return "TP" if row.y_true == 1 else "TN"
    return "FP" if row.y_pred == 1 else "FN"


def main():
    preds = load_all_predictions()
    best = pick_best_model()
    print(f"Best model: {best}")

    texts = pd.concat([load_splits(lang.lower())[2][["id", "text"]] for lang in LANGUAGES])
    best_df = preds[preds["model"] == best].merge(texts, on="id", how="left")
    if best_df.empty:
        sys.exit(f"No test predictions found for '{best}' in results/*_predictions.csv")
    best_df["outcome"] = best_df.apply(outcome, axis=1)
    best_df["length"] = best_df["text"].str.len()
    best_df["has_keyword"] = [
        KEYWORDS[l].lower() in str(t).lower() for l, t in zip(best_df["language"], best_df["text"])
    ]

    # Compare with KNN: which posts the bst model fixes, and which it newly gets wrong.
    knn = preds[preds["model"] == KNN_NAME][["id", "y_pred"]].rename(columns={"y_pred": "knn_pred"})
    best_df = best_df.merge(knn, on="id", how="left")
    best_correct = best_df["y_true"] == best_df["y_pred"]
    knn_correct = best_df["y_true"] == best_df["knn_pred"]

    summary = []
    for lang in LANGUAGES:
        m = best_df["language"] == lang
        counts = best_df[m]["outcome"].value_counts()
        summary.append({
            "model": best, "language": lang,
            **{k: int(counts.get(k, 0)) for k in ["TP", "TN", "FP", "FN"]},
            "fixed_vs_knn": int((m & best_correct & ~knn_correct).sum()),
            "new_errors_vs_knn": int((m & ~best_correct & knn_correct).sum()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "best_model_error_summary.csv", index=False)

    patterns = (best_df.groupby(["language", "outcome"])
                .agg(n=("id", "size"), median_length=("length", "median"),
                     keyword_rate=("has_keyword", "mean"))
                .round(2).reset_index())
    patterns.to_csv(RESULTS / "best_model_error_patterns.csv", index=False)

    errors = best_df[best_df["outcome"].isin(["FP", "FN"])]
    errors = errors[["model", "language", "id", "outcome", "y_true", "y_pred", "knn_pred", "has_keyword", "text"]]
    errors.sort_values(["language", "outcome"]).to_csv(
        RESULTS / "best_model_error_examples.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print()
    print(patterns.to_string(index=False))
    print("\nSaved best_model_error_summary.csv, best_model_error_patterns.csv and best_model_error_examples.csv in results/")


if __name__ == "__main__":
    main()