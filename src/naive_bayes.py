from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import load_splits
from src.evaluate import (
    evaluate_model,
    bootstrap_confidence_intervals,
)

LANGUAGES = ["arabic", "chinese", "portuguese", "thai"]
ALPHA_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
RESULTS_DIR = ROOT / "results"


def run_naive_bayes(language):
    train, val, test = load_splits(language)

    best_model = None
    best_alpha = None
    best_score = -1

    tuning_rows = []

    for alpha in ALPHA_VALUES:
        model = Pipeline([
            ("vectorizer", CountVectorizer(
                analyzer="char",
                ngram_range=(2, 3)
            )),
            ("classifier", MultinomialNB(alpha=alpha)),
        ])

        model.fit(train["text"], train["label"])
        val_predictions = model.predict(val["text"])
        val_metrics = evaluate_model(
            val["label"], val_predictions
        )

        tuning_rows.append({
            "language": language,
            "alpha": alpha,
            **val_metrics,
        })

        if val_metrics["f1_macro"] > best_score:
            best_score = val_metrics["f1_macro"]
            best_alpha = alpha
            best_model = model

    test_predictions = best_model.predict(test["text"])
    test_metrics = evaluate_model(
        test["label"], test_predictions
    )

    intervals = bootstrap_confidence_intervals(
        test["label"], test_predictions
    )

    print(f"\nLanguage: {language}")
    print(f"Best alpha: {best_alpha}")
    print(f"Validation Macro-F1: {best_score:.4f}")
    print(f"Test Macro-F1: {test_metrics['f1_macro']:.4f}")

    result = {
        "model": "Multinomial Naive Bayes",
        "language": language,
        "best_alpha": best_alpha,
        "validation_macro_f1": best_score,
        **test_metrics,
    }

    interval_rows = [
        {
            "language": language,
            "metric": metric,
            **bounds,
        }
        for metric, bounds in intervals.items()
    ]

    prediction_rows = [
        {
            "model": "Multinomial Naive Bayes",
            "language": language.capitalize(),
            "split": "test",
            "id": post_id,
            "y_true": actual,
            "y_pred": predicted,
        }
        for post_id, actual, predicted in zip(
            test["id"],
            test["label"],
            test_predictions,
        )
    ]

    return result, tuning_rows, interval_rows, prediction_rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    results = []
    tuning = []
    intervals = []
    predictions = []

    for language in LANGUAGES:
        result, tune, ci, preds = run_naive_bayes(language)
        results.append(result)
        tuning.extend(tune)
        intervals.extend(ci)
        predictions.extend(preds)

    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "naive_bayes_test_results.csv",
        index=False,
    )
    pd.DataFrame(tuning).to_csv(
        RESULTS_DIR / "naive_bayes_tuning.csv",
        index=False,
    )
    pd.DataFrame(intervals).to_csv(
        RESULTS_DIR / "naive_bayes_confidence_intervals.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(
        RESULTS_DIR / "naive_bayes_predictions.csv",
        index=False,
    )

    print("\nAll Naive Bayes experiments completed.")


if __name__ == "__main__":
    main()