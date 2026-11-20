# Experiment Log: Person 2 (Logistic Regression + KNN re-tune)

Milestone 5. Covers requirement 2 (tuning procedure) and the Person 2 part of requirement 1 (fair comparison).

## How to reproduce

From the repository root (Python 3.13, versions from `requirements.txt`: scikit-learn 1.9.1, pandas 3.0.5):

```bash
python src/logistic_regression.py     # ~1 min
python src/knn_tuned.py               # ~1 min
python src/verify_evaluation.py       # rebuilds shared_evaluation_results.csv (now includes LR and KNN (tuned))
python src/hyperparameter_table.py    # hyperparameter table + tuning effects
python src/main_results.py            # Person 3: main table + figure
python src/error_analysis_best.py     # Person 3: error analysis of the best model
```

## Files added or changed

| File | What it is |
|---|---|
| `src/tuning.py` | Shared tuning protocol used by both LR and KNN (new) |
| `src/logistic_regression.py` | LR grid search per language (new) |
| `src/knn_tuned.py` | KNN re-tune with TF-IDF and distance weighting (new) |
| `src/hyperparameter_table.py` | Builds the hyperparameter table and tuning-effect summary (new) |
| `src/verify_evaluation.py` | **Changed:** two prediction files added to `PREDICTION_FILES` / `MODEL_NAMES` so LR and KNN (tuned) flow into the shared results, main table, figure and error analysis |
| `results/logistic_regression_{tuning,test_results,confidence_intervals,predictions}.csv` | LR outputs |
| `results/knn_tuned_{tuning,test_results,confidence_intervals,predictions}.csv` | KNN (tuned) outputs |
| `results/hyperparameter_table.{csv,md}` | Search space, selected configuration, val and test Macro-F1 for all tuned models |
| `results/tuning_effects.csv` | Best validation Macro-F1 for each level of each tuning choice |

## Shared protocol (identical for LR and KNN)

- Splits from `data_pipeline.load_splits` (Person 1's leakage-safe splits), seed 42.
- Features: character 2–3 grams (same as NB and the M4 KNN baseline), either raw counts (`CountVectorizer`) or TF-IDF (`TfidfVectorizer`, default l2 norm and smoothed idf). The vectorizer is fitted on **train only**.
- Each configuration is trained on train and scored on validation. **Selection criterion: validation Macro-F1.** Ties keep the configuration that comes first in the grid.
- The selected model is scored **once** on the 100-post balanced test set with `evaluate.evaluate_model`. 95% CIs come from `evaluate.bootstrap_confidence_intervals` (2000 stratified resamples).
- Equal budget: **28 configurations per language for both models**.

---

## Entry LR-1: Logistic Regression grid search

**Config.** `LogisticRegression(solver="liblinear", max_iter=5000, random_state=42)`
Grid: features ∈ {counts, tfidf} × class_weight ∈ {none, balanced} × C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000} = 28 configurations × 4 languages = 112 fits.

**Selected configuration and scores**

| Language | Features | C | class_weight | Val Macro-F1 | Test Acc | Test P(1) | Test R(1) | Test Macro-F1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Arabic | tfidf | 10 | none | 0.943 | 0.95 | 0.941 | 0.96 | **0.950** [0.910–0.990] |
| Chinese | tfidf | 1 | none* | 0.714 | 0.85 | 0.797 | 0.94 | **0.849** [0.777–0.910] |
| Portuguese | tfidf | 10 | balanced | 0.763 | 0.67 | 0.793 | 0.46 | **0.655** [0.563–0.743] |
| Thai | tfidf | 10 | balanced | 0.784 | 0.75 | 0.963 | 0.52 | **0.736** [0.644–0.815] |

\* Chinese: `none` and `balanced` tied exactly (0.7135). `none` was kept because it comes first in the grid; Chinese train is 50/50, so the two settings are almost the same model.

**Observations**
- TF-IDF beat raw counts in all four languages (best val Macro-F1: Arabic 0.943 vs 0.937, Chinese 0.714 vs 0.704, Portuguese 0.763 vs 0.725, Thai 0.784 vs 0.776). The largest gain is in Portuguese.
- `class_weight="balanced"` was selected exactly where the classes are imbalanced: Portuguese and Thai, which are about 12–13% positive in train/val. For the roughly balanced Arabic and Chinese data it made no difference.
- Moderate regularisation won: C = 1–10. At C = 0.001 the model often collapses to one class: val Macro-F1 ≈ 0.24–0.47 for TF-IDF in Portuguese and Thai, and for unweighted counts in Portuguese and Thai. Counts on Arabic and Chinese hold up (0.68–0.91). Large C (100–1000) loses only slightly.
- LR has the highest test Macro-F1 in Arabic, Portuguese and Thai. In Chinese it is second to NB (0.849 vs 0.859). Mean over languages: LR 0.798, NB 0.745, KNN (tuned) 0.665, KNN baseline 0.660.
- **The CIs overlap.** With n = 100 test posts, the 95% CIs are about ±0.07–0.09 wide. LR vs NB overlap in every language. The clearest gaps are LR vs KNN in Thai (0.736 [0.644–0.815] vs 0.454 [0.376–0.524], no overlap) and in Arabic (LR's lower bound 0.910 is above the baseline KNN's upper bound 0.900 and just touches the tuned KNN's 0.910). A paired test on the same test posts would be needed to call the LR vs NB differences significant.
- **Val/test gaps:**
  - Chinese val (0.71) is much lower than test (0.85) for every model, so the test users seem easier than the val users.
  - Portuguese goes the other way (0.76 → 0.66). Train/val are 12% positive but test is 50% positive, so the model's conservative threshold costs recall on test (R = 0.46).
  - Thai shows the same pattern: precision 0.96 but recall 0.52.
  - Possible follow-up (not done): tune the decision threshold on validation.

## Entry KNN-1: KNN re-tune (same tuning effort as LR)

**Why.** The M4 baseline (`baseline_model.py`) tuned only k ∈ {1,3,5,7,9,11} on raw counts (6 configurations). To make the comparison fair, KNN gets the same 28-configuration budget and the same feature options as LR.

**Config.** `KNeighborsClassifier(metric="cosine", algorithm="brute")`
Grid: features ∈ {counts, tfidf} × weights ∈ {uniform, distance} × k ∈ {1, 3, 5, 9, 15, 25, 41} = 28 configurations × 4 languages.

*Grid revision (logged for transparency):* the first run used k ∈ {1,3,5,7,9,11,15}. Arabic and Chinese both picked k = 15, the edge of the grid, and validation scores were still rising. k was therefore re-spaced to {1,3,5,9,15,25,41}, which keeps the same 28-configuration budget. Only the second run is reported. (First run, for the record: Arabic test 0.880, Chinese 0.819; Portuguese and Thai were unchanged.)

**Selected configuration and scores**

| Language | Features | k | weights | Val Macro-F1 (baseline → tuned) | Test Macro-F1 (baseline → tuned) [95% CI] | Test R(1) |
|---|---|---|---|---|---|---|
| Arabic | tfidf | 25 | distance | 0.831 → 0.914 | 0.838 → **0.850** [0.779–0.910] | 0.90 |
| Chinese | tfidf | 41† | uniform | 0.666 → 0.711 | 0.788 → **0.838** [0.767–0.909] | 0.94 |
| Portuguese | tfidf | 5 | uniform | 0.616 → 0.668 | 0.558 → **0.517** [0.422–0.609] | 0.24 |
| Thai | tfidf | 9 | uniform | 0.628 → 0.663 | 0.454 → **0.454** [0.376–0.524] | 0.12 |

† Chinese still picks the largest k. The curve is flat by then (0.702 at k = 25, 0.711 at k = 41), so a larger k is unlikely to change the conclusion.

**Observations**
- TF-IDF beat counts for KNN in every language, by a much larger margin than for LR (Arabic 0.914 vs 0.851 val). Down-weighting very common character n-grams matters a lot for cosine-distance neighbours.
- Distance weighting barely mattered. It was identical to uniform in Portuguese and Thai and within 0.01 in Arabic and Chinese.
- Re-tuning raised **validation** Macro-F1 in every language (+0.04 to +0.08). On **test** it helped Chinese (+0.05) and Arabic (+0.01), left Thai unchanged and lowered Portuguese (−0.04). All of these changes are well inside the CIs.
- **Conclusion for fairness:** with the same tuning budget, KNN stays well below LR and NB in Portuguese and Thai, so its weak M4 result was not just a tuning problem.
- KNN has no `class_weight` equivalent. On the imbalanced languages its positive-class recall stays very low (Portuguese 0.24, Thai 0.12), because nearest neighbours are mostly majority-class posts. This is the main reason it trails LR there. Oversampling is tested in Person 4's ablation.

## Notes and limitations (for the report)
- NB (Person 1) searched 6 configurations on counts only, while LR and KNN searched 28, including TF-IDF. NB's tuning budget is therefore smaller. MNB is normally used with counts, but this should be stated.
- All selection used validation Macro-F1 only; the test set was scored once per model. The KNN grid revision above was based on validation scores, not test scores.
- These are single-seed (42) results. Person 1's 5-seed variance analysis currently covers NB only.
- `liblinear` with a fixed `random_state` is deterministic: re-running gives identical predictions.