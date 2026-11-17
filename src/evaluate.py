import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

SEED = 42


def evaluate_model(y_true, y_pred):
    """Calculate the same evaluation metrics for every model."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def bootstrap_confidence_intervals(
    y_true, y_pred, n_bootstrap=2000, seed=SEED
):
    """Calculate 95% confidence intervals by resampling test predictions."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Prediction arrays must have equal, nonzero lengths.")

    rng = np.random.default_rng(seed)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))

        if len(np.unique(y_true[indices])) < 2:
            continue

        scores = evaluate_model(
            y_true[indices], y_pred[indices]
        )
        bootstrap_scores.append(scores)

    intervals = {}

    for metric in bootstrap_scores[0]:
        values = [scores[metric] for scores in bootstrap_scores]

        intervals[metric] = {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }

    return intervals
