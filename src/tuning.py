"""
Shared tuning protocol for Logistic Regression and the re-tuned KNN
(Milestone 5, Person 2).

Both models go through exactly the same steps, so the comparison is fair:
  - same splits (data_pipeline.load_splits), same seed (42)
  - same character 2-3 gram features as Naive Bayes and the M4 KNN baseline,
    used either as raw counts or as TF-IDF
  - the vectorizer is fitted on the training split only
  - every configuration is scored on the validation split with Macro-F1
  - the best configuration (already fitted on train) is evaluated ONCE on test
  - test scores come from evaluate.evaluate_model, with
    evaluate.bootstrap_confidence_intervals (2000 stratified resamples, 95%)

Ties on validation Macro-F1 keep the configuration that comes first in the grid.
"""

from itertools import product
from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import load_splits
from src.evaluate import SEED, evaluate_model, bootstrap_confidence_intervals

LANGUAGES = ["arabic", "chinese", "portuguese", "thai"]
RESULTS_DIR = ROOT / "results"
NGRAM_RANGE = (2, 3)

FEATURES = {
    "counts": lambda: CountVectorizer(analyzer="char", ngram_range=NGRAM_RANGE),
    "tfidf": lambda: TfidfVectorizer(analyzer="char", ngram_range=NGRAM_RANGE),
}


def tune_language(language, model_name, make_classifier, grid):
    """Grid-search one model for one language.

    make_classifier(**params) returns an unfitted sklearn classifier.
    grid is a dict of hyperparameter name -> list of values; "features" is
    handled here and must be one of the keys.
    """
    train, val, test = load_splits(language)
    lang = language.capitalize()

    param_names = [k for k in grid if k != "features"]
    tuning_rows = []
    best = {"score": -1.0}

    for feature in grid["features"]:
        vectorizer = FEATURES[feature]()
        X_train = vectorizer.fit_transform(train["text"])
        X_val = vectorizer.transform(val["text"])

        for values in product(*(grid[k] for k in param_names)):
            params = dict(zip(param_names, values))
            clf = make_classifier(**params)

            start = time.perf_counter()
            clf.fit(X_train, train["label"])
            val_pred = clf.predict(X_val)
            seconds = time.perf_counter() - start

            metrics = evaluate_model(val["label"], val_pred)
            tuning_rows.append({
                "model": model_name,
                "language": lang,
                "features": feature,
                **{k: ("none" if v is None else v) for k, v in params.items()},
                "n_features": X_train.shape[1],
                **{f"val_{m}": v for m, v in metrics.items()},
                "fit_predict_seconds": round(seconds, 3),
            })

            if metrics["f1_macro"] > best["score"]:
                best = {
                    "score": metrics["f1_macro"],
                    "val_metrics": metrics,
                    "features": feature,
                    "params": params,
                    "clf": clf,
                    "vectorizer": vectorizer,
                }

    # The best classifier is already fitted on the training split; test is
    # transformed with the same training-only vectorizer and scored once.
    X_test = best["vectorizer"].transform(test["text"])
    test_pred = best["clf"].predict(X_test)

    test_metrics = evaluate_model(test["label"], test_pred)
    intervals = bootstrap_confidence_intervals(test["label"], test_pred)

    shown = ", ".join(f"{k}={v}" for k, v in best["params"].items())
    print(f"{lang:<11} best: features={best['features']}, {shown} | "
          f"val Macro-F1={best['score']:.4f} | test Macro-F1={test_metrics['f1_macro']:.4f} "
          f"[{intervals['f1_macro']['lower']:.3f}, {intervals['f1_macro']['upper']:.3f}]",
          flush=True)

    result = {
        "model": model_name,
        "language": lang,
        "best_features": best["features"],
        **{f"best_{k}": ("none" if v is None else v) for k, v in best["params"].items()},
        "n_configs_tried": len(tuning_rows),
        **{f"val_{m}": v for m, v in best["val_metrics"].items()},
        **{f"test_{m}": v for m, v in test_metrics.items()},
        "test_macro_f1_ci_lower": intervals["f1_macro"]["lower"],
        "test_macro_f1_ci_upper": intervals["f1_macro"]["upper"],
    }

    ci_rows = [
        {"model": model_name, "language": lang, "metric": m, **bounds}
        for m, bounds in intervals.items()
    ]

    prediction_rows = [
        {"model": model_name, "language": lang, "split": "test",
         "id": i, "y_true": t, "y_pred": p}
        for i, t, p in zip(test["id"], test["label"], test_pred)
    ]

    return result, tuning_rows, ci_rows, prediction_rows


def run_all(model_name, make_classifier, grid, file_prefix):
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"\n=== {model_name} (seed {SEED}) ===")

    results, tuning, cis, preds = [], [], [], []
    for language in LANGUAGES:
        r, t, c, p = tune_language(language, model_name, make_classifier, grid)
        results.append(r)
        tuning.extend(t)
        cis.extend(c)
        preds.extend(p)

    pd.DataFrame(results).to_csv(RESULTS_DIR / f"{file_prefix}_test_results.csv", index=False)
    pd.DataFrame(tuning).to_csv(RESULTS_DIR / f"{file_prefix}_tuning.csv", index=False)
    pd.DataFrame(cis).to_csv(RESULTS_DIR / f"{file_prefix}_confidence_intervals.csv", index=False)
    pd.DataFrame(preds).to_csv(RESULTS_DIR / f"{file_prefix}_predictions.csv", index=False)

    print(f"Saved results/{file_prefix}_{{test_results,tuning,confidence_intervals,predictions}}.csv")
    return pd.DataFrame(results), pd.DataFrame(tuning)