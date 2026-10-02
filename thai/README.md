# Thai - cleaned data

- **Raw data:** `../SIED-Thai.csv`
- **Label:** 1 = suicidal thoughts, 0 = no suicidal thoughts
- **Code:** `thai_cleaning.ipynb`

- Note: Arabic is about **depression**, but Thai is about **suicidal thoughts**.

## Files

| File | What it's for |
|---|---|
| `thai_sample_split.csv` | **Main file.** 50 + 50 tweets split into full / prefix/middle/suffix |
| `thai_cleaning_log.csv` | How many tweets were left after each cleaning step |
| `thai_clean_before_length_filter.csv` | All 2,376 cleaned tweets (for charts) |
| `thai_clean_full.csv` | 1,285 cleaned tweets with 15+ words (for training) |

## What I did
1. Removed hashtags, links, usernames, and extra symbols
2. Removed duplicate and contradictory tweets (2,400 to 2,376)
3. Kept tweets with at least 15 words ( 1,285)
4. Picked 50 random tweets from each label (seed = 42)
5. Split each tweet into 3 equal parts

Thai has no spaces between words, so I used **PyThaiNLP** to count and split words.

## Things to watch out for
- Different topic from Arabic (suicidal thoughts vs. depression)
- Original data is unbalanced (379 vs. 2,021), but the sample is 50/50
- Hashtags were removed because many posts ended in hashtag spam, and the "depression" hashtag would give away the answer
- Contains very sensitive content
