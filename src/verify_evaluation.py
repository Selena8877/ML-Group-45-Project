from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (
    evaluate_model,
    bootstrap_confidence_intervals,
)

RESULTS = ROOT / "results"

PREDICTION_FILES = [
    "naive_benchmark_predictions.csv",
    "baseline_model_predictions.csv",
    "naive_bayes_predictions.csv",
]

MODEL_NAMES = {
    "naive_benchmark_predictions.csv": "Naive (majority class)",
    "baseline_model_predictions.csv": "KNN (CountVectorizer)",
    "naive_bayes_predictions.csv": "Multinomial Naive Bayes",
}


def main():
    rows = []

    for filename in PREDICTION_FILES:
        path = RESULTS / filename

        if not path.exists():
            print(f"Skipping missing file: {filename}")
            continue

        data = pd.read_csv(path)

        if "model" not in data.columns:
            data["model"] = MODEL_NAMES[filename]

        group_cols = ["model", "language", "split"]

        if "variant" in data.columns:
            group_cols.append("variant")

        for keys, group in data.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)

            info = dict(zip(group_cols, keys))

            if info["split"] != "test":
                continue

            if len(group) != 100:
                raise ValueError(
                    f"{info}: Expected 100 test predictions, "
                    f"found {len(group)}"
                )

            if group["id"].duplicated().any():
                raise ValueError(
                    f"{info}: Duplicate test IDs found"
                )

            metrics = evaluate_model(
                group["y_true"],
                group["y_pred"],
            )

            intervals = bootstrap_confidence_intervals(
                group["y_true"],
                group["y_pred"],
            )

            rows.append({
                **info,
                **metrics,
                "macro_f1_ci_lower": intervals["f1_macro"]["lower"],
                "macro_f1_ci_upper": intervals["f1_macro"]["upper"],
                "n": len(group),
            })

    if not rows:
        raise RuntimeError("No test predictions found.")

    results = pd.DataFrame(rows)

    output = RESULTS / "shared_evaluation_results.csv"
    results.to_csv(output, index=False)

    print("\nShared Evaluation Results:")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()