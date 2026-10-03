# Arabic - cleaned data

- **Raw source:** `../Arabic_Depression_10.000_Tweets.xlsx` (main page of this repo)
- **Task:** binary classification – 1 = depression, 0 = not depression
- **Code:** `arabic_cleaning.ipynb` (Colab: upload the raw xlsx, then Runtime - Run all)

## Which file should I use?

| File | Use it for | Rows |
|---|---|---|
| `arabic_sample_split.csv` | **Modeling** - 50+50 sample split into full / prefix / middle / suffix | 100 |
| `arabic_cleaning_log.csv` | **Memo / charts** - rows left after each cleaning step | 7 |
| `arabic_clean_before_length_filter.csv` | **Charts** - length histogram (all cleaned tweets, before the 15-word cut) | 9,962 |
| `arabic_clean_full.csv` | **Training** (only if fine-tuning) – cleaned tweets with ≥ 15 words | 2,730 |

## Columns in `arabic_sample_split.csv`
- `id` – post ID (ar_001 … ar_100)
- `language` - Arabic
- `label` - 1 = depression, 0 = not
- `full` / `prefix` / `middle` / `suffix` - the 4 versions of each tweet
- `n_words` - number of words in the full tweet
- `has_keyword` - True if the tweet contains "اكتئاب" (depression)
- `kw_in_prefix` / `kw_in_middle` / `kw_in_suffix` - which third contains that word

## Cleaning steps

| Step | Rows left | Label 1 | Label 0 |
|---|---|---|---|
| 0. Raw data | 10,000 | 5,000 | 5,000 |
| 1. Remove empty tweets | 10,000 | 5,000 | 5,000 |
| 2. Clean text* | 10,000 | 5,000 | 5,000 |
| 3. Remove contradictory tweets | 10,000 | 5,000 | 5,000 |
| 4. Remove duplicates | 9,962 | 4,992 | 4,970 |
| 5. Keep tweets with ≥ 15 words | 2,730 | 1,425 | 1,305 |
| 6. Balanced random sample (seed = 42) | 100 | 50 | 50 |

\* Removed Excel `_x000D_` codes, HTML codes, invisible text-direction marks, `RT`, and links; replaced usernames with `[USER]`. **Kept** punctuation, emojis and hashtags.

**Why ≥ 15 words?** Each post is split into thirds, so 15 words gives about 5 words per part (enough for a meaningful phrase) while keeping 1,300+ tweets per class.

**Hashtags kept in Arabic** (they were rare); the other languages removed them because of hashtag spam

**How splitting works:** each tweet is cut into 3 parts by word count. If it doesn't divide evenly, the extra words go to the earlier parts (e.g., 16 words → 6/5/5).

## Known issues (for the risks section)
- **Keyword leakage:** "اكتئاب" appears in 1,795 / 5,000 positive tweets but only 2 / 5,000 negatives, so a model could learn the word instead of the meaning.
- **Negatives look like religious/positive tweets** (prayers, blessings), so a model may learn *sentiment* instead of *depression*.
- **Labels are not clinical:** many positives use "depression" casually or as a joke.
- **Short tweets:** median 9 words; ~73% were removed by the 15-word filter.
- **Small sample:** with 50 per class, small changes can shift patterns (e.g., keyword position), so don't over-interpret.
