
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main():
    naive = pd.read_csv(RESULTS / "naive_benchmark_metrics.csv")
    knn = pd.read_csv(RESULTS / "baseline_model_metrics.csv")

    naive = naive[naive["split"] == "test"]
    knn = knn[knn["split"] == "test"]

    rows = []

    for _, model in knn.iterrows():
        benchmark = naive[naive["language"] == model["language"]]

        if len(benchmark) != 1:
            raise ValueError(f"Missing benchmark for {model['language']}")

        benchmark = benchmark.iloc[0]

        rows.append({
            "language": model["language"],
            "variant": model["variant"],
            "naive_accuracy": benchmark["accuracy"],
            "knn_accuracy": model["accuracy"],
            "naive_precision": benchmark["precision"],
            "knn_precision": model["precision"],
            "naive_recall": benchmark["recall"],
            "knn_recall": model["recall"],
            "naive_f1": benchmark["f1"],
            "knn_f1": model["f1"],
            "f1_improvement": round(
                model["f1"] - benchmark["f1"], 4
            )
        })

    comparison = pd.DataFrame(rows)

    output = RESULTS / "model_comparison.csv"
    comparison.to_csv(output, index=False)

    print("\nModel Comparison (Test Set)\n")
    print(comparison.to_string(index=False))
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
