"""
Naive benchmark 

A majority-class DummyClassifier, trained separately for each language on
<language>_train.csv and evaluated on the validation and test splits.

Run from the top folder of the repository:
    python src/naive_benchmark.py

Outputs (results/):
    naive_benchmark_metrics.csv     one row per language x split (baseline table rows)
    naive_benchmark_predictions.csv per-post predictions (id, language, split, y_true, y_pred)
                                    for the shared evaluation code
    naive_benchmark_run_info.json   seed, library versions, majority class per language

Notes:
- Label 1 = at risk. Precision, Recall and F1 are reported for the at-risk class
  (pos_label=1), plus macro-F1 across both classes.
- The masked files are not evaluated separately: the dummy model ignores the text,
  so its results on the masked data are identical by construction.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


# Let Python find the src folder so we can use Person 1's(Sevara's) load_splits()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_pipeline import load_splits  # noqa: E402

SEED = 42   # same seed as the data pipeline
LANGUAGES = ["arabic", "chinese", "portuguese", "thai"]
OUT_DIR = ROOT / "results"


def set_seeds(seed: int = SEED) -> None:
    # Fix the random seed so results are the same every run
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),  # average of both classes
    }


def main() -> None:
    set_seeds()
    OUT_DIR.mkdir(exist_ok=True)

    metric_rows, pred_frames, majority = [], [], {}

    # Train and test one model per language
    for lang in LANGUAGES:
        train, val, test = load_splits(lang)

        # Features are irrelevant to a majority-class model, so a placeholder matrix is used.
        # Basically a Model that always predicts the most common training label
        # It ignores the text, so we just give it zeros as input
        model = DummyClassifier(strategy="most_frequent", random_state=SEED)
        model.fit(np.zeros((len(train), 1)), train["label"])

        # Save which label it always predicts
        majority[lang] = int(model.classes_[np.argmax(model.class_prior_)])


        # Score on validation and test
        for split_name, df in (("val", val), ("test", test)):
            y_pred = model.predict(np.zeros((len(df), 1)))

            # Add a row to the results table
            metric_rows.append({
                "model": "Naive (majority class)",
                "language": lang.capitalize(),
                "split": split_name,
                "n": len(df),
                **compute_metrics(df["label"], y_pred),
            })

            # Keep each post's prediction for Person 4(I think Firdous)
            pred_frames.append(pd.DataFrame({
                "id": df["id"], "language": lang.capitalize(), "split": split_name,
                "y_true": df["label"], "y_pred": y_pred,
            }))

    # Save the results table and the predictions
    metrics = pd.DataFrame(metric_rows).round(4)
    metrics.to_csv(OUT_DIR / "naive_benchmark_metrics.csv", index=False)
    pd.concat(pred_frames).to_csv(OUT_DIR / "naive_benchmark_predictions.csv", index=False)

    # Save the seed and library versions used
    run_info = {
        "seed": SEED,
        "majority_class_per_language": majority,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    with open(OUT_DIR / "naive_benchmark_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(metrics.to_string(index=False))
    print(f"\nMajority class per language: {majority}")


if __name__ == "__main__":
    main()