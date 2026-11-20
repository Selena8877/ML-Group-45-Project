from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LANGUAGES = ["Arabic", "Chinese", "Portuguese", "Thai"]

SEARCH_SPACE = {
    "KNN (CountVectorizer)": ("counts; k ∈ {1,3,5,7,9,11}; uniform; cosine", 6),
    "Multinomial Naive Bayes": ("counts; alpha ∈ {0.01,0.1,0.5,1,2,5}", 6),
    "KNN (tuned)": ("counts|tfidf × weights ∈ {uniform,distance} × k ∈ {1,3,5,9,15,25,41}; cosine", 28),
    "Logistic Regression": ("counts|tfidf × class_weight ∈ {none,balanced} × C ∈ {0.001,…,1000}", 28),
}


def selected_rows():
    rows = []

    knn = pd.read_csv(RESULTS / "baseline_model_metrics.csv")
    knn = knn[knn["variant"] == "original"]
    for lang in LANGUAGES:
        val = knn[(knn["language"] == lang) & (knn["split"] == "val")].iloc[0]
        rows.append({"model": "KNN (CountVectorizer)", "language": lang,
                     "selected": f"counts, k={int(val['best_k'])}, uniform",
                     "val_macro_f1": val["f1_macro"]})

    nb = pd.read_csv(RESULTS / "naive_bayes_test_results.csv")
    for _, r in nb.iterrows():
        rows.append({"model": "Multinomial Naive Bayes", "language": r["language"].capitalize(),
                     "selected": f"counts, alpha={r['best_alpha']:g}",
                     "val_macro_f1": r["validation_macro_f1"]})

    knn_t = pd.read_csv(RESULTS / "knn_tuned_test_results.csv")
    for _, r in knn_t.iterrows():
        rows.append({"model": "KNN (tuned)", "language": r["language"],
                     "selected": f"{r['best_features']}, k={int(r['best_n_neighbors'])}, {r['best_weights']}",
                     "val_macro_f1": r["val_f1_macro"]})

    lr = pd.read_csv(RESULTS / "logistic_regression_test_results.csv")
    for _, r in lr.iterrows():
        rows.append({"model": "Logistic Regression", "language": r["language"],
                     "selected": f"{r['best_features']}, C={r['best_C']:g}, class_weight={r['best_class_weight']}",
                     "val_macro_f1": r["val_f1_macro"]})

    return pd.DataFrame(rows)


def main():
    table = selected_rows()
    table["search_space"] = table["model"].map(lambda m: SEARCH_SPACE[m][0])
    table["n_configs"] = table["model"].map(lambda m: SEARCH_SPACE[m][1])

    shared = pd.read_csv(RESULTS / "shared_evaluation_results.csv")
    shared = shared[(shared["split"] == "test") &
                    (shared["variant"].isna() | (shared["variant"] == "original"))]
    table = table.merge(
        shared[["model", "language", "f1_macro", "macro_f1_ci_lower", "macro_f1_ci_upper"]]
        .rename(columns={"f1_macro": "test_macro_f1"}),
        on=["model", "language"], how="left")

    order = list(SEARCH_SPACE)
    table["model"] = pd.Categorical(table["model"], order, ordered=True)
    table["language"] = pd.Categorical(table["language"], LANGUAGES, ordered=True)
    table = table.sort_values(["language", "model"])
    cols = ["language", "model", "search_space", "n_configs", "selected",
            "val_macro_f1", "test_macro_f1", "macro_f1_ci_lower", "macro_f1_ci_upper"]
    table = table[cols]
    table.round(4).to_csv(RESULTS / "hyperparameter_table.csv", index=False)

    lines = ["| Language | Model | Configs tried | Selected configuration | Val Macro-F1 | Test Macro-F1 [95% CI] |",
             "|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r.language} | {r.model} | {r.n_configs} | {r.selected} | {r.val_macro_f1:.3f} | "
                     f"{r.test_macro_f1:.3f} [{r.macro_f1_ci_lower:.3f}–{r.macro_f1_ci_upper:.3f}] |")
    lines += ["", "Search spaces (all use character 2–3 grams fitted on train only; selection by validation Macro-F1, seed 42):", ""]
    lines += [f"- **{m}**: {s} ({n} configurations)" for m, (s, n) in SEARCH_SPACE.items()]
    (RESULTS / "hyperparameter_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Effect of each tuning choice: best validation Macro-F1 reachable at each level
    effects = []
    for prefix, model, choices in [
        ("logistic_regression", "Logistic Regression", ["features", "class_weight"]),
        ("knn_tuned", "KNN (tuned)", ["features", "weights"]),
    ]:
        tune = pd.read_csv(RESULTS / f"{prefix}_tuning.csv")
        for choice in choices:
            best = tune.groupby(["language", choice])["val_f1_macro"].max()
            for (lang, level), score in best.items():
                effects.append({"model": model, "language": lang, "choice": choice,
                                "level": level, "best_val_macro_f1": round(score, 4)})
    pd.DataFrame(effects).to_csv(RESULTS / "tuning_effects.csv", index=False)

    print((RESULTS / "hyperparameter_table.md").read_text(encoding="utf-8"))
    print(pd.DataFrame(effects).to_string(index=False))


if __name__ == "__main__":
    main()