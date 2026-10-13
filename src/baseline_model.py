
from pathlib import Path
import sys

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)

# Locate the project folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import load_splits


# Load the keyword-masked datasets
def load_masked_splits(language):
    folder = ROOT / "data" / "processed"

    train = pd.read_csv(folder / f"{language}_train_masked.csv")
    val = pd.read_csv(folder / f"{language}_val_masked.csv")
    test = pd.read_csv(folder / f"{language}_test_masked.csv")

    return train, val, test


def evaluate_model(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


def run_knn(language, masked=False):
    if masked:
        train, val, test = load_masked_splits(language)
    else:
        train, val, test = load_splits(language)

    variant = "masked" if masked else "original"

    print(f"\nLanguage: {language.capitalize()} ({variant})")
    print("Training posts:", len(train))
    print("Validation posts:", len(val))
    print("Test posts:", len(test))

    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(2, 3)
    )

    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    X_test = vectorizer.transform(test["text"])

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    k_values = [1, 3, 5, 7, 9, 11]

    best_k = None
    best_f1 = -1

    for k in k_values:
        model = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute"
        )

        model.fit(X_train, train["label"])

        predictions = model.predict(X_val)

        macro_f1 = f1_score(
            val["label"],
            predictions,
            average="macro",
            zero_division=0
        )

        print(f"K={k}: Macro-F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_k = k

    print(f"Best K for {language.capitalize()}: {best_k}")

    final_model = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="cosine",
        algorithm="brute"
    )

    final_model.fit(X_train, train["label"])

    val_predictions = final_model.predict(X_val)
    test_predictions = final_model.predict(X_test)

    val_metrics = evaluate_model(
        val["label"],
        val_predictions
    )

    test_metrics = evaluate_model(
        test["label"],
        test_predictions
    )

    print("\nValidation Results:")
    print(val_metrics)

    print("\nTest Results:")
    print(test_metrics)

    return {
        "language": language,
        "variant": variant,
        "best_k": best_k,
        "validation": val_metrics,
        "test": test_metrics,
        "val_ids": val["id"].tolist(),
        "val_true": val["label"].tolist(),
        "val_predictions": val_predictions.tolist(),
        "test_ids": test["id"].tolist(),
        "test_true": test["label"].tolist(),
        "test_predictions": test_predictions.tolist()
    }


def main():
    languages = ["arabic", "chinese", "portuguese", "thai"]

    all_experiments = []

    for language in languages:
        original = run_knn(language)
        all_experiments.append(original)

        masked = run_knn(language, masked=True)
        all_experiments.append(masked)

    # Create evaluation metrics table
    metrics_rows = []

    for experiment in all_experiments:
        for split in ["validation", "test"]:
            metrics_rows.append({
                "model": "KNN (CountVectorizer)",
                "language": experiment["language"].capitalize(),
                "variant": experiment["variant"],
                "split": "val" if split == "validation" else "test",
                "best_k": experiment["best_k"],
                **experiment[split]
            })

    metrics_df = pd.DataFrame(metrics_rows)

    # Create individual predictions table
    prediction_rows = []

    for experiment in all_experiments:
        for split in ["val", "test"]:
            ids = experiment[f"{split}_ids"]
            actual = experiment[f"{split}_true"]
            predicted = experiment[f"{split}_predictions"]

            for post_id, true_label, predicted_label in zip(
                ids, actual, predicted
            ):
                prediction_rows.append({
                    "model": "KNN (CountVectorizer)",
                    "language": experiment["language"].capitalize(),
                    "variant": experiment["variant"],
                    "split": split,
                    "id": post_id,
                    "y_true": true_label,
                    "y_pred": predicted_label
                })

    predictions_df = pd.DataFrame(prediction_rows)

    # Save both files
    results_folder = ROOT / "results"
    results_folder.mkdir(exist_ok=True)

    metrics_file = results_folder / "baseline_model_metrics.csv"
    predictions_file = results_folder / "baseline_model_predictions.csv"

    metrics_df.round(4).to_csv(metrics_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    print("\nAll experiments completed!")
    print("Total experiments:", len(all_experiments))
    print("Metrics rows:", len(metrics_df))
    print("Individual predictions:", len(predictions_df))
    print("Metrics saved to:", metrics_file)
    print("Predictions saved to:", predictions_file)


if __name__ == "__main__":
    main()
