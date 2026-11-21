# Experiment Log: Person 3 (Main Results + Error Analysis)

Milestone 5. Covers requirement 4 (main results table and figure) and requirement 5 (error analysis).

## How to reproduce

From the repository root, after the model scripts and `src/verify_evaluation.py` have produced `results/shared_evaluation_results.csv`:

```bash
python src/main_results.py          # main table + figure
python src/error_analysis_best.py   # error analysis of the best model
```

Both scripts only read existing results; they train no models and use no randomness.

## Files added or changed

| File | What it is |
|---|---|
| `src/main_results.py` | Builds the main results table and figure from `shared_evaluation_results.csv` (new) |
| `src/error_analysis_best.py` | Error analysis of the best model, compared with KNN (tuned) (new) |
| `results/main_results_table.{csv,md}` | Test metrics for every model and language, Macro-F1 with 95% CI |
| `results/main_results_figure.png` | Test Macro-F1 per model and language with 95% CI error bars |
| `results/best_model_error_summary.csv` | TP/TN/FP/FN per language, plus errors fixed and new versus KNN (tuned) |
| `results/best_model_error_patterns.csv` | Median post length and keyword rate by outcome (TP/TN/FP/FN) |
| `results/best_model_error_examples.csv` | Every misclassified test post with its text, label, prediction and KNN prediction |

## Configuration

- **Data:** test split only (100 posts per language, 50/50), original (unmasked) text. Masked results are left to the ablation section.
- **Metrics:** taken from `shared_evaluation_results.csv` (Person 1's `evaluate.py`): accuracy, at-risk precision/recall/F1, Macro-F1 with 95% stratified bootstrap CI (2,000 resamples).
- **Best model:** highest mean test Macro-F1 across the four languages (selected automatically; can be overridden by passing a model name).
- **Comparison model for errors:** KNN (tuned), so the error comparison is against a fairly tuned baseline, not the untuned Milestone 3 KNN.
- **Keyword check:** اكتئاب (Arabic), 抑郁 (Chinese), "suicid" (Portuguese), ซึมเศร้า (Thai).

## Runs

| Date | Run | Inputs | Result / observation |
|---|---|---|---|
| Nov 17 | Main results + error analysis (draft) | Naive, KNN (CountVectorizer), Naive Bayes | Naive Bayes best; beat KNN in all languages but CIs overlapped except Thai. Fixed a bug where `misclassified_predictions.csv` was read as a full prediction file and doubled KNN error counts. |
| Nov 20 | Main results + error analysis (final) | + Logistic Regression, KNN (tuned) | Logistic Regression best (mean test Macro-F1). Top in Arabic (0.950), Portuguese (0.655), Thai (0.736); Naive Bayes slightly ahead in Chinese (0.859 vs 0.849). Only Thai's gain over KNN is outside the CIs. |
| Nov 20 | Review fixes (from Sevara) | — | Comparison model changed from KNN (CountVectorizer) to KNN (tuned); figure legend moved below the plot so it no longer covers the Thai bars. |

## Key observations

- **Errors by type (LR, test):** Arabic 3 FP / 2 FN; Chinese 12 FP / 3 FN; Portuguese 6 FP / 27 FN; Thai 1 FP / 24 FN.
- **Persistent errors:** all 24 Thai and 24 of 27 Portuguese false negatives were also missed by KNN (tuned); 11 of 12 Chinese false positives were shared.
- **Keyword rarely present in misses:** "suicid-" in 4% of missed Portuguese posts; no keyword in missed Thai posts.
- **Figurative language:** all 6 Portuguese false positives use self-harm phrases as exaggeration ("vou me matar").
- **Chinese false positives** are ordinary personal posts, consistent with user-level labelling.