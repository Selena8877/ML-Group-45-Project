# Chinese - cleaned data

- **Raw data:** SWDD - Sina Weibo Depression Dataset (Cai et al., 2023) – [GitHub](https://github.com/ethan-nicholas-tsai/SWDD); Google Drive link in the main README
- **Label:** 1 = depressed user, 0 = non-depressed (control) user
- **Code:** `chinese_cleaning.ipynb`

- Note: labels are **per user**, not per post. We picked **one post per user**.

## Files

| File | What it's for |
|---|---|
| `chinese_sample_split.csv` | **Main file.** 50 + 50 posts split into full / prefix/middle/suffix |
| `chinese_cleaning_log.csv` | How many posts/users were left after each step |
| `chinese_post_lengths.csv` | Lengths of all 140,709 cleaned posts (for charts, no text) |
| `chinese_clean_full.csv` | 75,869 cleaned posts with 15–100 words (for training) |

## What I did
1. Used the **first 500 users** from each file (the control file is 4.5 GB)
2. Kept only users' **own posts** (removed reposts): 146,590 posts
3. Removed HTML, links, hashtags (#topic#), Weibo filler ("网页链接", "转发微博", "xxx的微博视频", "我在: …"); usernames → [USER]
4. Removed empty and duplicate posts → 140,709 posts
5. Kept posts with 15–100 words, and removed posts Weibo had cut off ("... 全文")
6. Picked **one random post per user**. For depressed users, the post had to mention **抑郁** (depression)
7. Picked 50 users from each class (seed = 42) and split each post into 3 equal parts

Chinese has no spaces between words, so I used **jieba** to count and split words.

## Things to watch out for
- **Labels are per user:** we chose depressed users' posts that mention 抑郁 so the post actually contains evidence
- **Keyword leakage:** because of that, positive posts contain 抑郁, and control posts don't
- **Not a random sample of users:** first 500 users per file
- **Some control users are advertisers** (product ads), and many posts end with a **location check-in** (e.g. 宁波, 韶关)
- Contains sensitive mental-health content
