from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Settings
SEED = 42
VAL_SIZE = 0.20
CHINESE_POSTS_PER_USER = 3      # about 2,700 Chinese posts in total, close to Arabic
LANGUAGES = ["arabic", "chinese", "portuguese", "thai"]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed"

# The text column is called "tweet" in three of the cleaned files and "text" in Chinese
TEXT_COL = {"arabic": "tweet", "portuguese": "tweet", "thai": "tweet", "chinese": "text"}

# Keywords that get replaced with [KW] in the masked version of the data.
# This is for the experiment that checks how much the models rely on one word.
KEYWORDS = {
    "arabic": [r"اكتئاب"],
    "chinese": [r"抑郁"],
    "thai": [r"ซึมเศร้า"],
    "portuguese": [r"\bsuicid\w*"],            # matches suicídio, suicida, suicidar, etc.
}

TEST_EXTRA_COLS = ["prefix", "middle", "suffix"]


# Text normalization
_AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_AR_TATWEEL = re.compile(r"ـ")
_AR_ALEF = re.compile(r"[أإآٱ]")
_TH_REPEAT = re.compile(r"([฀-๿])\1{2,}")   # the same Thai character 3 or more times in a row
_SPACES = re.compile(r"\s+")


def normalize_text(text: str, language: str) -> str:
    """Clean up one post. The same function is used for train, validation and test."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # NFKC turns different versions of the same character into one standard form,
    # for example full-width digits and punctuation into normal ones
    t = unicodedata.normalize("NFKC", text)

    if language == "arabic":
        t = _AR_DIACRITICS.sub("", t)           # short vowel marks
        t = _AR_TATWEEL.sub("", t)              # tatweel, used to stretch words
        t = _AR_ALEF.sub("ا", t)                # all alef forms become a plain alef
        t = t.replace("ى", "ي")
        t = t.replace("ة", "ه")
        # We don't touch ئ or ؤ, otherwise the keyword اكتئاب would change
    elif language == "thai":
        t = _TH_REPEAT.sub(r"\1", t)            
    elif language == "portuguese":
        # Lowercase but keep accents, since removing them can change the meaning.
        # [USER] is put back in capitals so it matches the other languages.
        t = t.lower().replace("[user]", "[USER]")
    # Chinese only needs the NFKC step

    return _SPACES.sub(" ", t).strip()


def mask_keywords(text: str, language: str) -> str:
    for pat in KEYWORDS[language]:
        text = re.sub(pat, "[KW]", text, flags=re.IGNORECASE)
    return text


def _stratified_pick(items: np.ndarray, labels: np.ndarray, frac: float) -> set:
    """Pick the same fraction of items from each label, using the fixed seed."""
    rng = np.random.default_rng(SEED)
    picked = []
    for lab in sorted(np.unique(labels)):
        # Sorting first means the result only depends on the seed, not on row order
        group = np.sort(items[labels == lab])
        rng.shuffle(group)
        picked.extend(group[: int(round(len(group) * frac))])
    return set(picked)


# Loading the cleaned Milestone 2 files
def _load_raw(language: str, data_root: Path = REPO_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(data_root / language / f"{language}_clean_full.csv", encoding="utf-8-sig")
    test = pd.read_csv(data_root / language / f"{language}_sample_split.csv", encoding="utf-8-sig")
    full = full.rename(columns={TEXT_COL[language]: "text"})
    return full, test


# Building the splits for one language
def build_language(language: str, data_root: Path = REPO_ROOT,
                   mask: bool = False) -> dict[str, pd.DataFrame]:
    """Build the train, val and test tables for one language.

    Also returns a log of how many posts were left after each step.
    """
    full, test = _load_raw(language, data_root)
    log = [("clean_full (M2)", len(full))]

    # The test posts are still inside the full dataset, so they have to be taken
    # out before anything else. There is no shared id column, so we match on the text.
    pool = full[~full["text"].isin(set(test["full"]))].copy()
    log.append(("remove 100 test posts", len(pool)))

    # Chinese labels belong to users, not posts. If a test user's other posts were
    # in training, the model could learn that user's writing style, so they all go.
    if language == "chinese":
        pool = pool[~pool["user_id"].isin(set(test["user_id"]))]
        log.append(("remove all posts of test users", len(pool)))

    # Normalize everything, including the prefix/middle/suffix parts of the test set
    pool["text"] = pool["text"].map(lambda s: normalize_text(s, language))
    test = test.copy()
    test["text"] = test["full"].map(lambda s: normalize_text(s, language))
    for c in TEST_EXTRA_COLS:
        test[c] = test[c].map(lambda s: normalize_text(s, language))

    # Normalizing can make two different posts identical, so we clean up again
    pool = pool[pool["text"] != ""]
    # If the same text has both labels we can't trust either one, so drop all copies
    n_labels = pool.groupby("text")["label"].nunique()
    pool = pool[~pool["text"].isin(n_labels[n_labels > 1].index)]
    # Keep one copy of repeated posts (Chinese had the same post under different users)
    pool = pool.drop_duplicates(subset="text", keep="first")
    pool = pool[~pool["text"].isin(set(test["text"]))]
    log.append(("normalize + drop dup/conflicting/test-identical", len(pool)))

    # Chinese has about 68k posts left at this point, far more than the other
    # languages. Taking the same number of posts from each user keeps the size
    # comparable and stops very active users from dominating.
    if language == "chinese":
        pool = (pool.sample(frac=1, random_state=SEED)
                    .groupby("user_id").head(CHINESE_POSTS_PER_USER)
                    .sort_values(["user_id"], kind="stable"))
        log.append((f"downsample to <= {CHINESE_POSTS_PER_USER} posts/user", len(pool)))

    # 80/20 train/validation split with the same label balance in both
    pool = pool.reset_index(drop=True)
    if language == "chinese":
        # Split users instead of posts so nobody ends up in both train and val.
        # Each user has a single label, so picking 20% of users per label keeps it balanced.
        users = pool.groupby("user_id")["label"].first()
        val_users = _stratified_pick(users.index.to_numpy(), users.to_numpy(), VAL_SIZE)
        is_val = pool["user_id"].isin(val_users)
    else:
        val_rows = _stratified_pick(pool.index.to_numpy(), pool["label"].to_numpy(), VAL_SIZE)
        is_val = pool.index.isin(val_rows)
    train, val = pool[~is_val], pool[is_val]

    # Final output: only the columns the models are allowed to use. This drops the
    # Thai emotion scores and the Chinese symptom columns, which are too closely
    # tied to the label.
    prefix = {"arabic": "ar", "chinese": "zh", "portuguese": "pt", "thai": "th"}[language]
    lang_name = test["language"].iloc[0]

    def _finish(df: pd.DataFrame, split: str) -> pd.DataFrame:
        df = df.reset_index(drop=True).copy()
        out = pd.DataFrame({
            "id": [f"{prefix}_{split}_{i+1:05d}" for i in range(len(df))],
            "language": lang_name,
            "split": split,
            "label": df["label"].astype(int).values,
            "text": df["text"].values,
        })
        if language == "chinese":
            out["user_id"] = df["user_id"].values   # only for grouping, not a feature
        return out

    train_out, val_out = _finish(train, "train"), _finish(val, "val")

    # The test set keeps its original Milestone 2 ids and the three post parts
    test_out = test[["id", "language"]].assign(split="test")
    test_out["label"] = test["label"].astype(int)
    test_out["text"] = test["text"]
    for c in TEST_EXTRA_COLS:
        test_out[c] = test[c]
    if language == "chinese":
        test_out["user_id"] = test["user_id"]

    if mask:
        for df in (train_out, val_out, test_out):
            for c in ["text"] + [c for c in TEST_EXTRA_COLS if c in df]:
                df[c] = df[c].map(lambda s: mask_keywords(s, language))

    return {"train": train_out, "val": val_out, "test": test_out, "log": log}


# Checks and summary
def check_no_leakage(language: str, splits: dict[str, pd.DataFrame]) -> None:
    """Stop the script with an error if any post (or Chinese user) is in two splits."""
    tr, va, te = (set(splits[s]["text"]) for s in ("train", "val", "test"))
    assert not (tr & va), f"{language}: train/val text overlap"
    assert not (tr & te), f"{language}: train/test text overlap"
    assert not (va & te), f"{language}: val/test text overlap"
    if language == "chinese":
        u = {s: set(splits[s]["user_id"]) for s in ("train", "val", "test")}
        assert not (u["train"] & u["val"]), "chinese: user in train and val"
        assert not (u["train"] & u["test"]), "chinese: user in train and test"
        assert not (u["val"] & u["test"]), "chinese: user in val and test"
    # The test set should still be exactly 50 posts per class
    assert len(splits["test"]) == 100 and splits["test"]["label"].sum() == 50


def summarize(language: str, splits: dict[str, pd.DataFrame]) -> list[dict]:
    """Count posts and labels in each split. This is what goes in the baseline table."""
    rows = []
    for s in ("train", "val", "test"):
        df = splits[s]
        n, n1 = len(df), int(df["label"].sum())
        row = {"language": df["language"].iloc[0], "split": s, "n": n,
               "label_1": n1, "label_0": n - n1, "pct_label_1": round(100 * n1 / n, 1)}
        if "user_id" in df:
            row["users"] = df["user_id"].nunique()
        rows.append(row)
    return rows


# Main functions
def build_all(data_root: Path = REPO_ROOT, out_dir: Path = OUT_DIR, mask: bool = False,
              verbose: bool = True) -> pd.DataFrame:
    """Build, check and save the splits for all four languages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_masked" if mask else ""
    summary, logs = [], []
    for lang in LANGUAGES:
        splits = build_language(lang, data_root, mask=mask)
        check_no_leakage(lang, splits)
        for s in ("train", "val", "test"):
            splits[s].to_csv(out_dir / f"{lang}_{s}{suffix}.csv", index=False, encoding="utf-8-sig")
        summary += summarize(lang, splits)
        logs += [{"language": lang, "step": st, "rows": n} for st, n in splits["log"]]
        if verbose:
            print(f"[{lang}] " + " -> ".join(f"{st}: {n:,}" for st, n in splits["log"]))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / f"split_summary{suffix}.csv", index=False)
    pd.DataFrame(logs).to_csv(out_dir / f"pipeline_log{suffix}.csv", index=False)
    return summary_df


def load_splits(language: str, masked: bool = False, out_dir: Path = OUT_DIR
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train, val and test tables for one language.

    If the CSV files don't exist yet, they are built first.

    Columns in every table: id, language, split, label, text
    The test table also has: prefix, middle, suffix
    Chinese tables also have user_id, which should not be used as a feature.
    """
    language = language.lower()
    suffix = "_masked" if masked else ""
    paths = [out_dir / f"{language}_{s}{suffix}.csv" for s in ("train", "val", "test")]
    if not all(p.exists() for p in paths):
        build_all(out_dir=out_dir, mask=masked, verbose=False)
    return tuple(pd.read_csv(p, encoding="utf-8-sig") for p in paths)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build leakage-safe train/val/test splits.")
    ap.add_argument("--masked", action="store_true", help="also build keyword-masked copies")
    args = ap.parse_args()

    np.random.seed(SEED)
    summary = build_all()
    print("\nSplit summary:\n" + summary.to_string(index=False))
    if args.masked:
        build_all(mask=True, verbose=False)
        print("\nKeyword-masked copies written (*_masked.csv).")
    print(f"\nFiles written to {OUT_DIR}")