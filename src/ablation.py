
"""Milestone 5: sensitivity and ablation experiments for Logistic Regression.

Run from the repository root: python -m src.ablation
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.data_pipeline import load_splits
from src.evaluate import SEED, evaluate_model
from src.logistic_regression import make_classifier
from src.tuning import LANGUAGES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

NGRAMS = (2, 3)
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def load_best_settings():
    path = RESULTS / "logistic_regression_test_results.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 'python -m src.logistic_regression' first."
        )

    rows = pd.read_csv(path).set_index("language")
    settings = {}

    for language in LANGUAGES:
        row = rows.loc[language.capitalize()]
        weight = str(row["best_class_weight"]).lower()

        settings[language] = {
            "features": str(row["best_features"]).lower(),
            "C": float(row["best_C"]),
            "class_weight": None if weight in ("none", "nan", "") else weight,
        }

    return settings


def make_vectorizer(features, ngrams):
    if features == "tfidf":
        return TfidfVectorizer(analyzer="char", ngram_range=ngrams)

    if features == "counts":
        return CountVectorizer(analyzer="char", ngram_range=ngrams)

    raise ValueError(f"Unknown feature representation: {features}")


def balance_training(train):
    """Duplicate minority-class examples in training only."""

    counts = train["label"].value_counts()

    if len(counts) < 2 or counts.min() == counts.max():
        return train.copy()

    majority_size = int(counts.max())
    pieces = []

    for label in sorted(counts.index):
        group = train[train["label"] == label]

        if len(group) < majority_size:
            group = group.sample(
                n=majority_size,
                replace=True,
                random_state=SEED
            )

        pieces.append(group)

    balanced = pd.concat(pieces, ignore_index=True)

    return balanced.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


def train_fraction(train, fraction):
    if fraction == 1.0:
        return train.copy()

    subset, _ = train_test_split(
        train,
        train_size=fraction,
        stratify=train["label"],
        random_state=SEED
    )

    return subset.copy()


def run_experiment(
    language,
    experiment,
    variant,
    train,
    val,
    settings,
    features=None,
    ngrams=NGRAMS
):
    features = features or settings["features"]
    start = time.perf_counter()

    vectorizer = make_vectorizer(features, ngrams)

    x_train = vectorizer.fit_transform(
        train["text"].fillna("").astype(str)
    )

    x_val = vectorizer.transform(
        val["text"].fillna("").astype(str)
    )

    model = make_classifier(
        settings["class_weight"],
        settings["C"]
    )

    model.fit(x_train, train["label"])

    prediction = model.predict(x_val)
    scores = evaluate_model(val["label"], prediction)

    return {
        "language": language.capitalize(),
        "experiment": experiment,
        "variant": variant,
        "features": features,
        "ngram_range": f"{ngrams[0]}-{ngrams[1]}",
        "C": settings["C"],
        "class_weight": settings["class_weight"] or "none",
        "seed": SEED,
        "train_rows": len(train),
        "val_rows": len(val),
        "n_features": x_train.shape[1],
        **{f"val_{key}": value for key, value in scores.items()},
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }


def run_masking(language, train, val, settings):
    masked_train, masked_val, _ = load_splits(
        language,
        masked=True
    )

    # Ensure the original and masked splits contain the same examples.
    assert train["id"].tolist() == masked_train["id"].tolist()
    assert val["id"].tolist() == masked_val["id"].tolist()

    assert train["label"].tolist() == masked_train["label"].tolist()
    assert val["label"].tolist() == masked_val["label"].tolist()

    return [
        run_experiment(
            language, "keyword_masking", "off",
            train, val, settings
        ),
        run_experiment(
            language, "keyword_masking", "on",
            masked_train, masked_val, settings
        ),
    ]


def run_oversampling(language, train, val, settings):
    return [
        run_experiment(
            language, "oversampling", "off",
            train, val, settings
        ),
        run_experiment(
            language, "oversampling", "on",
            balance_training(train), val, settings
        ),
    ]


def run_ngrams(language, train, val, settings):
    return [
        run_experiment(
            language, "ngram_range", "2-3",
            train, val, settings,
            ngrams=(2, 3)
        ),
        run_experiment(
            language, "ngram_range", "1-4",
            train, val, settings,
            ngrams=(1, 4)
        ),
    ]


def run_features(language, train, val, settings):
    return [
        run_experiment(
            language, "feature_representation", "counts",
            train, val, settings,
            features="counts"
        ),
        run_experiment(
            language, "feature_representation", "tfidf",
            train, val, settings,
            features="tfidf"
        ),
    ]


def run_learning_curve(language, train, val, settings):
    results = []

    for fraction in FRACTIONS:
        subset = train_fraction(train, fraction)

        result = run_experiment(
            language,
            "learning_curve",
            f"{int(fraction * 100)}%",
            subset,
            val,
            settings
        )

        results.append(result)

    return results


def save_figures(results):
    ablations = results[
        results["experiment"] != "learning_curve"
    ]

    experiments = [
        "keyword_masking",
        "oversampling",
        "ngram_range",
        "feature_representation"
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        constrained_layout=True
    )

    for ax, experiment in zip(axes.flat, experiments):
        subset = ablations[
            ablations["experiment"] == experiment
        ]

        pivot = subset.pivot(
            index="language",
            columns="variant",
            values="val_f1_macro"
        )

        pivot = pivot.reindex(
            [language.capitalize() for language in LANGUAGES]
        )

        pivot.plot(
            kind="bar",
            ax=ax,
            rot=20
        )

        ax.set_title(
            experiment.replace("_", " ").title()
        )
        ax.set_xlabel("")
        ax.set_ylabel("Validation macro-F1")
        ax.set_ylim(0, 1)
        ax.legend(title="Variant", fontsize=8)

    fig.savefig(
        FIGURES / "ablation_comparison.png",
        dpi=160
    )

    plt.close(fig)

    curves = results[
        results["experiment"] == "learning_curve"
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for language, group in curves.groupby("language"):
        ordered = group.assign(
            pct=group["variant"].str.rstrip("%").astype(int)
        ).sort_values("pct")

        ax.plot(
            ordered["pct"],
            ordered["val_f1_macro"],
            marker="o",
            label=language
        )

    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Learning curve by language")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIGURES / "learning_curve.png",
        dpi=160
    )

    plt.close(fig)


def save_results(rows):
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS / "ablation_results.csv",
        index=False
    )

    results[
        results["experiment"] == "learning_curve"
    ].to_csv(
        RESULTS / "learning_curve.csv",
        index=False
    )

    results.to_csv(
        RESULTS / "ablation_log.csv",
        index=False
    )

    save_figures(results)

    print("\nResults and figures saved successfully.")


def main():
    settings_by_language = load_best_settings()
    rows = []

    experiments = (
        run_masking,
        run_oversampling,
        run_ngrams,
        run_features,
        run_learning_curve
    )

    for language in LANGUAGES:
        print(f"\n[{language.upper()}]", flush=True)

        train, val, _ = load_splits(language)
        settings = settings_by_language[language]

        for experiment in experiments:
            experiment_results = experiment(
                language,
                train,
                val,
                settings
            )

            for result in experiment_results:
                rows.append(result)

                print(
                    f"  {result['experiment']:<24} "
                    f"{result['variant']:<7} "
                    f"macro-F1={result['val_f1_macro']:.4f}",
                    flush=True
                )

    save_results(rows)

    print(
        f"Completed {len(rows)} experiment configurations."
    )


if __name__ == "__main__":
    main()
