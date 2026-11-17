from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import load_splits
from src.evaluate import evaluate_model

LANGUAGES = ["arabic", "chinese", "portuguese", "thai"]
SEEDS = [42, 43, 44, 45, 46]
ALPHA_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

RESULTS_DIR = ROOT / "results"


def create_split(language, pool, seed):
    """Create an 80/20 split without mixing Chinese users."""

    if language == "chinese":
        users = pool.groupby("user_id")["label"].first()

        train_users, val_users = train_test_split(
            users.index.to_numpy(),
            test_size=0.20,
            random_state=seed,
            stratify=users.to_numpy(),
        )

        train = pool[pool["user_id"].isin(train_users)]
        val = pool[pool["user_id"].isin(val_users)]

        assert set(train["user_id"]).isdisjoint(
            set(val["user_id"])
        )
    else:
        train, val = train_test_split(
            pool,
            test_size=0.20,
            random_state=seed,
            stratify=pool["label"],
        )

    assert set(train["text"]).isdisjoint(set(val["text"]))

    return train, val


def run_experiment(language, seed, pool):
    """Train and tune Naive Bayes for one language and seed."""

    train, val = create_split(language, pool, seed)

    best_alpha = None
    best_score = -1

    for alpha in ALPHA_VALUES:
        model = Pipeline([
            ("vectorizer", CountVectorizer(
                analyzer="char",
                ngram_range=(2, 3),
            )),
            ("classifier", MultinomialNB(alpha=alpha)),
        ])

        model.fit(train["text"], train["label"])

        predictions = model.predict(val["text"])

        metrics = evaluate_model(
            val["label"], predictions
        )

        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_alpha = alpha

    print(
        f"{language.capitalize()} | "
        f"Seed: {seed} | "
        f"Best alpha: {best_alpha} | "
        f"Validation Macro-F1: {best_score:.4f}",
        flush=True,
    )

    return {
        "language": language,
        "seed": seed,
        "train_size": len(train),
        "val_size": len(val),
        "best_alpha": best_alpha,
        "validation_macro_f1": best_score,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    results = []

    for language in LANGUAGES:
        train, val, test = load_splits(language)

        pool = pd.concat(
            [train, val],
            ignore_index=True,
        )

        assert set(pool["text"]).isdisjoint(
            set(test["text"])
        )

        if language == "chinese":
            assert set(pool["user_id"]).isdisjoint(
                set(test["user_id"])
            )

        for seed in SEEDS:
            results.append(
                run_experiment(language, seed, pool)
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "naive_bayes_seed_variance.csv",
        index=False,
    )

    summary = (
        results_df.groupby("language")["validation_macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.to_csv(
        RESULTS_DIR / "naive_bayes_seed_summary.csv",
        index=False,
    )

    print("\nFive-seed summary:")
    print(summary.round(4).to_string(index=False))

    print("\nAll five-seed experiments completed.")


if __name__ == "__main__":
    main()