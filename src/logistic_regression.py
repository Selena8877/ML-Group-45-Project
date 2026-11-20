from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import SEED
from src.tuning import run_all

MODEL_NAME = "Logistic Regression"

GRID = {
    "features": ["counts", "tfidf"],
    "class_weight": [None, "balanced"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
}


def make_classifier(class_weight, C):
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="liblinear",      # deterministic, fast on sparse n-gram features
        max_iter=5000,
        random_state=SEED,
    )


if __name__ == "__main__":
    run_all(MODEL_NAME, make_classifier, GRID, "logistic_regression")