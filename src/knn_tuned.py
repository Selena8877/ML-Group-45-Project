from pathlib import Path
import sys

from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tuning import run_all

MODEL_NAME = "KNN (tuned)"

GRID = {
    "features": ["counts", "tfidf"],
    "weights": ["uniform", "distance"],
    "n_neighbors": [1, 3, 5, 9, 15, 25, 41],
}


def make_classifier(weights, n_neighbors):
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="cosine",
        algorithm="brute",
    )


if __name__ == "__main__":
    run_all(MODEL_NAME, make_classifier, GRID, "knn_tuned")