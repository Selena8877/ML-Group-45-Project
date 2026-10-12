# Data Pipeline (Milestone 3)

This directory includes the finalized datasets to be used for model training and evaluation. The language subdirectories in this repo (`arabic/`, `chinese/`, `portuguese/`, `thai/`) have the cleaned dataset created at Milestone 2, but this dataset cannot be used for modeling yet since it includes the test tweets, the additional columns that may leak the label and the Chinese dataset is much bigger compared to other datasets.

The python file `src/data_pipeline.py` resolves these issues and the cleaned dataset is stored at `data/processed/`. The following README describes the pipeline and its steps and the contents of the generated files.

## How to run it

From the top folder of the repository:

```bash
python src/data_pipeline.py --masked
```

This script requires just `pandas` and `numpy`. It runs in a few seconds, printing how many posts are remaining after each stage, followed by a summary table. When `--masked` is not specified, it skips the keyword-masked versions as explained below.

It is deterministic; all random operations use seed 42, ensuring the exact same files are generated when running the script again.

To load the data into another script:

```python
from src.data_pipeline import load_splits

train, val, test = load_splits("arabic")   # or "chinese", "portuguese", "thai"
```

## What the pipeline does

### 1. Separating the test set

As discussed in Milestone 2, 100 posts were chosen per language (50 at-risk and 50 non-at-risk), and these were divided into a prefix, middle, and suffix. This constitutes our test set of 100 posts.

However, these 100 posts also exist in the complete cleaned datasets. Training on these files will cause the model to encounter posts it has seen before in the test set, thus giving artificially good results. Therefore, the first step is to remove these 100 test posts from the training data. However, the cleaned files do not have an id column matching that of the test samples. As a result, posts were identified based on their text. All 100 posts were identified in all languages.

However, Chinese had an additional step. In this language, labels are assigned to users, not posts, and each of the 100 test posts belongs to a unique user. Simply removing the 100 posts will leave the rest of the posts belonging to the same user in the training data. Therefore, for Chinese language, all posts from the 100 test users were removed.

### 2. Normalizing the text

There are multiple ways to encode the same writing. Some messages use full-width characters, Arabic has an option for vowels, and Thai people like to double their letters. All of these appear to be different words to the model, hence the standardization of the text.

| Language | What we changed |
|---|---|
| All | Unicode NFKC normalization, which converts characters with several forms (for example full-width digits and punctuation) to one standard form |
| Arabic | Removed diacritics (short vowel marks) and tatweel (the character used to stretch words). Converted أ إ آ ٱ to ا, ى to ي, and ة to ه |
| Thai | Shortened any character repeated three or more times to a single character (มากกกกก becomes มาก) |
| Portuguese | Converted to lowercase. Accents were kept because removing them can change the meaning of a word |
| Chinese | NFKC only |

ئ and ؤ were retained as they appear in Arabic because any modification of them would change the keyword اكتئاب that was used in the keyword analysis.

This normalization is performed on all three datasets (training, validation, and test), including the prefix, middle, and suffix columns of the test dataset. Any difference in normalization between two splits would render the testing results invalid.

### 3. Removing duplicates after normalization

Normalization will transform two posts that have some difference into identical posts. After normalization, we:

- removed posts with texts appearing with both labels because we cannot determine the correct label;
- retained only one of any duplicate posts;
- removed any training post that became identical with any test post.

This resulted in 12 posts being deleted in Arabic. In Chinese, there were 781 posts with identical texts by different users. These posts were not deleted in the Milestone 2 cleaning because duplicates were only deleted per user.

### 4. Downsampling Chinese

After the procedures described above, there were 68,000 posts in Chinese, whereas the other languages had 1,100-2,600 posts. It would be unfair to train the model on much more data for one language compared to the other languages.

For Chinese, we sampled up to 3 posts per user, and thus got 2,695 posts from 900 users. Sampling an equal number of posts per user ensures that highly active users do not make up the majority of the dataset. The number of posts per user is controlled by `CHINESE_POSTS_PER_USER` at the beginning of the script.

### 5. Splitting into training and validation

The rest of the posts were allocated into training and validation sets in an 80/20 ratio. It is important to note that the allocation is stratified, which means that both datasets have the same distribution of risky posts as the whole dataset. This is especially important for Thai and Portuguese, because only 13% of posts are at risk there.

In case of Chinese, it is users who are being allocated into the datasets, not posts.

### 6. Keeping only the text

The files generated for the outputs have only the text of the post and the label. We removed the Thai emotion columns and the Chinese `self_reported` and `suicidal_ideation` columns since they are highly correlated with the label and provide a cheat to the model. This was done as part of the "text-only inputs" task defined in Milestone 2.

### Leakage check

Before writing the files, the script performs a leakage check to ensure that no post is in more than one split, that no Chinese user is in more than one split, and that each test set has exactly 50 posts per class.

## Results

Number of posts after each step:

| Step | Arabic | Chinese | Portuguese | Thai |
|---|---|---|---|---|
| Cleaned data from Milestone 2 | 2,730 | 75,869 | 1,241 | 1,285 |
| Remove the 100 test posts | 2,630 | 75,740 | 1,141 | 1,185 |
| Remove other posts by test users | | 68,674 | | |
| Normalize and remove duplicates | 2,618 | 67,846 | 1,141 | 1,185 |
| Downsample (3 posts per user) | | 2,695 | | |

Final splits (the percentage is the share of at-risk posts):

| Language | Train | Validation | Test |
|---|---|---|---|
| Arabic | 2,094 (52.3%) | 524 (52.3%) | 100 (50%) |
| Chinese | 2,155 (49.9%) | 540 (50.0%) | 100 (50%) |
| Portuguese | 913 (12.4%) | 228 (12.3%) | 100 (50%) |
| Thai | 948 (12.9%) | 237 (12.7%) | 100 (50%) |

For Chinese, training covers 720 users, validation 180 users and test 100 users.

## Output files

All files are in `data/processed/`.

| File | Contents |
|---|---|
| `<language>_train.csv` | Training data |
| `<language>_val.csv` | Validation data, used for checking and tuning models |
| `<language>_test.csv` | The 100 test posts from Milestone 2, normalized |
| `*_masked.csv` | The same files with the main keyword replaced by `[KW]` |
| `split_summary.csv` | Post and label counts for every split (the Results table above) |
| `pipeline_log.csv` | Number of posts after each step |

Columns:

| Column | Description |
|---|---|
| `id` | Post id. Test posts keep their Milestone 2 ids (for example `ar_001`) |
| `language` | Arabic, Chinese, Portuguese or Thai |
| `split` | train, val or test |
| `label` | 1 = at risk, 0 = not at risk. "At risk" means depression for Arabic and Chinese, and suicidal thoughts for Thai and Portuguese |
| `text` | The normalized post. This is the only column meant to be used as model input |
| `prefix`, `middle`, `suffix` | Test files only. The three parts of each post, normalized the same way |
| `user_id` | Chinese only. Kept so posts can be grouped by user. It should not be used as a feature |

The masked dataset replaces اكتئاب (Arabic), 抑郁 (Chinese), ซึมเศร้า (Thai) and any words that start with "suicid" (Portuguese) with `[KW]`. Comparing the results from the two datasets will reveal to what extent the model is relying on this word.

## Notes and limitations

- **Imbalance in class varies between training and test.** While the test sets are balanced 50/50, the Thai and Portuguese training data is about 13% at risk. This is normal, as the test samples were purposefully balanced in Milestone 2, but it influences the interpretation of metrics. For instance, a classifier that predicts the majority class scores high on validation, but low on test.
- **The Chinese test set is less difficult than the training data.** In Milestone 2, test posts by depressed users were required to contain 抑郁 ("depression"). All 50 at-risk test posts contain this term. Training posts were selected randomly, and only 51 out of 1,076 at-risk posts contain it. This means Chinese scores would be higher on test than on validation.
- **Chinese labels refer to users rather than posts.** A post by a depressed user does not necessarily contain any trace of depression, which means the Chinese training labels are noisier than the labels of other languages.
- **There are few at-risk examples in Thai and Portuguese.** Excluding 50 at-risk test posts leaves 152 at-risk posts in Thai and 141 in Portuguese.