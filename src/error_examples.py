
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"

LANGUAGES = ["arabic", "chinese", "portuguese", "thai"]


def main():
    errors = pd.read_csv(
        RESULTS / "misclassified_predictions.csv"
    )

    datasets = []

    for language in LANGUAGES:
        test = pd.read_csv(DATA / f"{language}_test.csv")
        datasets.append(
            test[["id", "language", "label", "text"]]
        )

    posts = pd.concat(datasets, ignore_index=True)

    merged = errors.merge(
        posts,
        on=["id", "language"],
        how="left",
        validate="many_to_one"
    )

    if merged["text"].isna().any():
        raise ValueError("Some posts could not be matched")

    if not (merged["y_true"] == merged["label"]).all():
        raise ValueError("Actual labels do not match")

    merged = merged.drop(columns=["label"])

    output = RESULTS / "error_examples.csv"
    merged.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Matched {len(merged)} error records")
    print(f"Saved results to: {output}")

    print("\nKNN Original Errors by Language\n")

    knn = merged[
        (merged["model"].str.startswith("KNN"))
        & (merged["variant"] == "original")
    ]

    summary = pd.crosstab(
        knn["language"],
        knn["error_type"]
    )

    print(summary.to_string())


if __name__ == "__main__":
    main()
