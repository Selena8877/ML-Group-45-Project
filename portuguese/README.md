# Portuguese (Brazilian) - cleaned data

- **Raw data:** `../dataset_ideacao (3).csv` ("ideação" = ideation)
- **Label:** 1 = suicidal thoughts (meant literally), 0 = no suicidal thoughts (often suicide words used figuratively, e.g., "suicídio social")
- **Code:** `portuguese_cleaning.ipynb`

- Note: like Thai, this is **suicidal thoughts**, not depression.

## Files

| File | What it's for |
|---|---|
| `portuguese_sample_split.csv` | **Main file.** 50 + 50 tweets split into full / prefix / middle / suffix |
| `portuguese_cleaning_log.csv` | How many tweets were left after each cleaning step |
| `portuguese_clean_before_length_filter.csv` | All 2,940 cleaned tweets (for charts) |
| `portuguese_clean_full.csv` | 1,241 cleaned tweets with 15+ words (for training) |

## What I did
1. Removed links, hashtags, `RT` and invisible characters; usernames → [USER]
2. Removed contradictory and duplicate tweets (3,788 → 2,940; 832 removed)
3. Kept tweets with at least 15 words (1,241)
4. Picked 50 random tweets from each label (seed = 42)
5. Split each tweet into 3 equal parts

## Things to watch out for
- **Same condition as Thai** (suicidal thoughts), different from Arabic and Chinese (depression)
- **Many duplicates** in the raw data (22% removed)
- **Unbalanced raw data:** 802 vs. 2,138 after cleaning; the sample is 50/50
- **Positives are shorter** in the full data (median 8 vs. 14 words); the 15-word filter removes this gap in the sample (both 20.5)
- **Some labels may be noisy**, e.g. exaggerations like "vou me matar" ("I'm going to kill myself") labeled 1
- Contains very sensitive content
