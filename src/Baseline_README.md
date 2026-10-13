# KNN Baseline Model
## Milestone 3

## 1. Overview

This project uses K-Nearest Neighbors (KNN) as a baseline for multilingual text classification across four languages: Arabic, Chinese, Brazilian Portuguese, and Thai.

We tested both original and masked versions of each dataset to see how masking affects model performance. KNN was chosen because it provides a relatively simple, interpretable baseline that we can compare against other approaches.

Each language is evaluated independently, as the datasets differ in language, content, and classification objectives.

## 2. Methodology

Since KNN cannot process raw text directly, we used `CountVectorizer` to convert text into numerical features. Character-level 2–3 n-grams were selected to provide a consistent representation across languages without relying on language-specific word tokenization.

The model uses Scikit-learn's `KNeighborsClassifier` with cosine distance and brute-force neighbor search.

| Parameter | Configuration |
|---|---|
| Model | K-Nearest Neighbors |
| Text representation | Character-level 2–3 n-grams |
| Vectorizer | `CountVectorizer` |
| Distance metric | Cosine |
| Neighbor search | Brute force |
| K values tested | 1, 3, 5, 7, 9, 11 |
| Model selection | Highest validation Macro-F1 |

For each language and dataset version, the vectorizer is fitted on the training data only. We then evaluate different K values on the validation set and select the one with the highest Macro-F1 score.

The selected configuration is evaluated on the held-out test set. This keeps the test data separate from feature fitting and hyperparameter selection.

## 3. Evaluation and Results

We evaluated the model using accuracy, precision, recall, F1-score, and Macro-F1.

Macro-F1 is particularly useful here because it gives equal importance to both classes rather than allowing the majority class to dominate the evaluation.

### Test Results

The table below shows the Macro-F1 scores obtained from the original and masked datasets.

| Language | Original Macro-F1 | Masked Macro-F1 |
|---|---:|---:|
| Arabic | 0.8377 | 0.8279 |
| Chinese | 0.7883 | 0.8271 |
| Brazilian Portuguese | 0.5578 | 0.5660 |
| Thai | 0.4544 | 0.4544 |

The results show that masking did not affect every language in the same way.

- **Arabic:** Performance decreased slightly after masking.
- **Chinese:** Masking improved Macro-F1 from 0.7883 to 0.8271.
- **Brazilian Portuguese:** Performance improved slightly.
- **Thai:** Macro-F1 remained unchanged.

Arabic achieved the highest original-data Macro-F1, while Chinese showed the largest improvement after masking.

These differences are worth investigating further, particularly because the datasets do not all represent the same classification task. The positive class relates to depression in the Arabic and Chinese datasets, while the Thai and Portuguese datasets focus on suicidal ideation.

The complete evaluation results, including accuracy, precision, recall, F1-score, and selected K values, are available in:

`results/baseline_model_metrics.csv`

Individual predictions and their corresponding true labels are available in:

`results/baseline_model_predictions.csv`

## 4. Reproducibility

The implementation is available in:

`src/baseline_model.py`

### Dependencies

The baseline uses Python and the following libraries:

- NumPy
- Pandas
- Scikit-learn

The repository also contains a `requirements.txt` file. However, its NumPy version specification needs to be checked before relying on it for a fresh installation.

### Running the Model

From the repository's root directory, install the required packages:

```bash
python -m pip install numpy pandas scikit-learn
```

Run the baseline:

```bash
python src/baseline_model.py
```

The script uses the prepared training, validation, and test datasets, including the masked CSV files in `data/processed/`.

It runs the original and masked experiments for all four languages and saves two output files:

```text
results/
├── baseline_model_metrics.csv
└── baseline_model_predictions.csv
```

A successful run should produce:

- 8 experiments (4 languages × 2 dataset versions)
- 16 evaluation rows (validation and test results)
- 3,858 individual predictions

The script uses the same model-selection procedure for each experiment, allowing the results to be reproduced and compared consistently.

## 5. Model Limitations

KNN provides a straightforward baseline, but its performance depends on the selected K value, distance metric, and text representation.

Character-level n-grams capture patterns in text without directly representing semantic meaning. Additionally, KNN can be computationally expensive because predictions require comparisons with training examples.

These limitations should be considered when comparing the baseline with other models.

## 6. Summary

The KNN baseline was successfully implemented and evaluated across four languages using original and masked datasets.

The model uses character-level n-grams, cosine distance, and validation-based hyperparameter selection. All evaluation metrics and individual predictions have been saved to support further comparison and error analysis.

Overall, the baseline establishes a reproducible reference point for evaluating how different text representations and modeling approaches perform across the selected datasets.
