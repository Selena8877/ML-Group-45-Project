## Run it
```bash
python src/data_pipeline.py            # builds data/processed/*.csv + split_summary.csv
python src/data_pipeline.py --masked   # also builds keyword-masked copies (*_masked.csv)
```
Needs only `pandas` and `numpy`. It doesn't use scikit-learn, so it also runs on laptops where Windows blocks SciPy. The model code will still need `scikit-learn` in `requirements.txt`. It's deterministic: seed 42, and a rerun produces byte-identical files.

## Use it in your code
```python
from src.data_pipeline import load_splits
train, val, test = load_splits("arabic")      # "arabic" | "chinese" | "portuguese" | "thai"
X_train, y_train = train["text"], train["label"]
X_val,   y_val   = val["text"],   val["label"]
X_test,  y_test  = test["text"],  test["label"]   # test also has prefix / middle / suffix
```

| Column | Meaning |
|---|---|
| `id` | post id (test keeps the Milestone 2 ids, e.g. `ar_001`) |
| `language`, `split` | e.g. `Thai`, `train` |
| `label` | 1 = at risk (depression for AR/ZH, suicidal thoughts for TH/PT), 0 = not |
| `text` | normalized post text. **This is the only model input.** |
| `prefix`, `middle`, `suffix` | test only, normalized the same way |
| `user_id` | Chinese only. Use it for grouping, **never as a feature** |

## What the pipeline does
1. **Test set** = the fixed 100-post (50/50) sample from Milestone 2. Those posts are removed from training. For Chinese, every post by those 100 users is also removed.
2. **Normalization**, applied identically to every split:
   - All languages: NFKC.
   - Arabic: remove diacritics and tatweel; أ/إ/آ/ٱ→ا, ى→ي, ة→ه.
   - Thai: 3 or more repeated Thai characters → 1.
   - Portuguese: lowercase, accents kept.
3. **Leakage guards:** drop texts that have conflicting labels, drop duplicates, and drop anything identical to a test post after normalization.
4. **Chinese downsampling:** 3 random posts per user (900 users), giving 2,695 posts.
5. **80/20 train/val**, stratified by label. Chinese is also grouped by user, so no user appears in two splits.
6. **Text only:** Thai emotion columns and Chinese `self_reported` / `suicidal_ideation` are dropped.

`check_no_leakage()` asserts that there is no text overlap between any two splits and no Chinese user overlap.

## For the model code (Persons 2–3)
- Fit TF-IDF **on `train` only**, e.g. inside an sklearn `Pipeline`.
- Use `class_weight="balanced"` for Thai and Portuguese, where training is only about 12–13% positive.
- Use the **training** labels for the majority-class benchmark. For TH/PT that class is 0, so on the 50/50 test set the benchmark gets about 0.50 accuracy and an F1 of 0 for the positive class. This is expected.

## Known issue for the discrepancy analysis (Person 4)
**Chinese train/test mismatch:** in Milestone 2, depressed-user test posts had to contain 抑郁 ("depression"), so 50/50 positive test posts contain it. Training posts were sampled at random from each user, and only 51 of the 1,076 positive training posts contain it. Expect validation scores to be clearly lower than test scores for Chinese. That gap comes from how the test set was built, not from the model.
