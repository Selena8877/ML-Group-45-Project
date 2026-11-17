"""
Main results table and figure (Milestone 5, Person 3).

Reads results/shared_evaluation_results.csv, so any model added to that file
(e.g. Logistic Regression) appears automatically.

Run from the repository root:
    python src/main_results.py

Outputs (results/):
    main_results_table.csv   one row per model and language (test set, original text)
    main_results_table.md    the same table, ready to paste into the report
    main_results_figure.png  test Macro-F1 with 95% bootstrap confidence intervals
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LANGUAGES = ["Arabic", "Chinese", "Portuguese", "Thai"]


def load_results():
    df = pd.read_csv(RESULTS / "shared_evaluation_results.csv")
    df = df[df["split"] == "test"].copy()
    # Main comparison uses the original (unmasked) text; masking is an ablation.
    if "variant" in df.columns:
        df = df[df["variant"].isna() | (df["variant"] == "original")]
    df["language"] = df["language"].str.capitalize()
    return df


def model_order(df):
    # Naive first, then the other models from weakest to strongest mean Macro-F1.
    means = df.groupby("model")["f1_macro"].mean().sort_values()
    naive = [m for m in means.index if m.lower().startswith("naive (")]
    return naive + [m for m in means.index if m not in naive]


def make_table(df, models):
    cols = ["model", "language", "accuracy", "precision", "recall", "f1",
            "f1_macro", "macro_f1_ci_lower", "macro_f1_ci_upper"]
    table = df[cols].copy()
    table["model"] = pd.Categorical(table["model"], models, ordered=True)
    table["language"] = pd.Categorical(table["language"], LANGUAGES, ordered=True)
    table = table.sort_values(["language", "model"]).round(3)
    table.to_csv(RESULTS / "main_results_table.csv", index=False)

    # Markdown version for the report.
    lines = ["| Language | Model | Accuracy | Precision (1) | Recall (1) | F1 (1) | Macro-F1 [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        ci = f"{r.f1_macro:.3f} [{r.macro_f1_ci_lower:.3f}–{r.macro_f1_ci_upper:.3f}]"
        lines.append(f"| {r.language} | {r.model} | {r.accuracy:.3f} | {r.precision:.3f} | "
                     f"{r.recall:.3f} | {r.f1:.3f} | {ci} |")
    (RESULTS / "main_results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def make_figure(df, models):
    colors = ["#9e9e9e", "#7a9cc6", "#2b5c8a", "#c8611f", "#5a8f5a", "#8a5a9e"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("language").reindex(LANGUAGES)
        x = [j + (i - (len(models) - 1) / 2) * width for j in range(len(LANGUAGES))]

        err = [sub["f1_macro"] - sub["macro_f1_ci_lower"],
               sub["macro_f1_ci_upper"] - sub["f1_macro"]]
        
        ax.bar(x, sub["f1_macro"], width, label=model, color=colors[i % len(colors)],
               yerr=err, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
        
    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES)
    ax.set_ylabel("Test Macro-F1")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "main_results_figure.png", dpi=300)


def main():
    df = load_results()
    models = model_order(df)

    table = make_table(df, models)
    make_figure(df, models)
    print(table.to_string(index=False))
    print("\nSaved main_results_table.csv, main_results_table.md and main_results_figure.png in results/")


if __name__ == "__main__":
    main()