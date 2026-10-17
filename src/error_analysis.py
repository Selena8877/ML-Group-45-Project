
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def classify_error(actual, predicted):
    if actual == 1 and predicted == 1:
        return "TP"
    if actual == 0 and predicted == 0:
        return "TN"
    if actual == 0 and predicted == 1:
        return "FP"
    return "FN"


def analyse_predictions(data):
    data = data.copy()

    data["error_type"] = [
        classify_error(actual, predicted)
        for actual, predicted in zip(data["y_true"], data["y_pred"])
    ]

    summary = []

    for (model, language, variant), group in data.groupby(
        ["model", "language", "variant"]
    ):
        counts = group["error_type"].value_counts()

        summary.append({
            "model": model,
            "language": language,
            "variant": variant,
            "TP": int(counts.get("TP", 0)),
            "TN": int(counts.get("TN", 0)),
            "FP": int(counts.get("FP", 0)),
            "FN": int(counts.get("FN", 0)),
            "total_errors": int(
                counts.get("FP", 0) + counts.get("FN", 0)
            ),
            "total_samples": len(group)
        })

    return pd.DataFrame(summary), data


def compare_masking(knn):
    original = knn[knn["variant"] == "original"]
    masked = knn[knn["variant"] == "masked"]

    paired = original.merge(
        masked,
        on=["id", "language", "split"],
        suffixes=("_original", "_masked"),
        validate="one_to_one"
    )

    if not (
        paired["y_true_original"] == paired["y_true_masked"]
    ).all():
        raise ValueError("Actual labels do not match")

    paired["original_correct"] = (
        paired["y_true_original"] == paired["y_pred_original"]
    )
    paired["masked_correct"] = (
        paired["y_true_masked"] == paired["y_pred_masked"]
    )

    paired["masking_effect"] = "unchanged"
    paired.loc[
        paired["original_correct"] & ~paired["masked_correct"],
        "masking_effect"
    ] = "became_incorrect"

    paired.loc[
        ~paired["original_correct"] & paired["masked_correct"],
        "masking_effect"
    ] = "became_correct"

    changes = paired.groupby(
        ["language", "masking_effect"]
    ).size().unstack(fill_value=0)

    return paired, changes.reset_index()


def main():
    naive = pd.read_csv(
        RESULTS / "naive_benchmark_predictions.csv"
    )
    knn = pd.read_csv(
        RESULTS / "baseline_model_predictions.csv"
    )

    naive = naive[naive["split"] == "test"].copy()
    knn = knn[knn["split"] == "test"].copy()

    naive["model"] = "Naive"
    naive["variant"] = "original"

    naive = naive[
        ["model", "language", "variant",
         "split", "id", "y_true", "y_pred"]
    ]

    knn = knn[
        ["model", "language", "variant",
         "split", "id", "y_true", "y_pred"]
    ]

    combined = pd.concat([naive, knn], ignore_index=True)

    summary, predictions = analyse_predictions(combined)

    errors = predictions[
        predictions["error_type"].isin(["FP", "FN"])
    ].copy()

    paired, masking_summary = compare_masking(knn)

    summary.to_csv(
        RESULTS / "error_summary.csv", index=False
    )
    errors.to_csv(
        RESULTS / "misclassified_predictions.csv", index=False
    )
    paired.to_csv(
        RESULTS / "masking_comparison.csv", index=False
    )
    masking_summary.to_csv(
        RESULTS / "masking_summary.csv", index=False
    )

    print("\nError Analysis (Test Set)\n")
    print(summary.to_string(index=False))

    print("\nMasking Comparison\n")
    print(masking_summary.to_string(index=False))

    print("\nSaved four analysis files to results/")


if __name__ == "__main__":
    main()
