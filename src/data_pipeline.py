from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# --------------------------------------------------------------------------------------
# Configuration - agreed group settings
# --------------------------------------------------------------------------------------
SEED = 42
VAL_SIZE = 0.20                 # 80/20 train/validation
CHINESE_POSTS_PER_USER = 3      # downsampling: 3 posts per user, about 2,400 posts total
LANGUAGES = ["arabic", "chinese", "portuguese", "thai"]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed"

# Column holding the post text in each *_clean_full.csv
TEXT_COL = {"arabic": "tweet", "portuguese": "tweet", "thai": "tweet", "chinese": "text"}

# Keywords for the optional masking experiment (M2, planned step 5)
KEYWORDS = {
    "arabic": [r"اكتئاب"],                     # after normalization إكتئاب -> اكتئاب
    "chinese": [r"抑郁"],
    "thai": [r"ซึมเศร้า"],
    "portuguese": [r"\bsuicid\w*"],            # suicídio, suicida, suicidar, ...
}

TEST_EXTRA_COLS = ["prefix", "middle", "suffix"]


# --------------------------------------------------------------------------------------
# 3. Text normalization
# --------------------------------------------------------------------------------------
_AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_AR_TATWEEL = re.compile(r"ـ")
_AR_ALEF = re.compile(r"[أإآٱ]")
_TH_REPEAT = re.compile(r"([฀-๿])\1{2,}")   # 3+ identical Thai chars -> 1
_SPACES = re.compile(r"\s+")


def normalize_text(text: str, language: str) -> str:
    """Apply the same normalization to every split. Returns a cleaned string."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = unicodedata.normalize("NFKC", text)

    if language == "arabic":
        t = _AR_DIACRITICS.sub("", t)           # remove harakat (short-vowel marks)
        t = _AR_TATWEEL.sub("", t)              # remove tatweel (stretching char)
        t = _AR_ALEF.sub("ا", t)                # أ إ آ ٱ -> ا
        t = t.replace("ى", "ي")                 # alef maqsura -> ya
        t = t.replace("ة", "ه")                 # ta marbuta -> ha
        # ئ / ؤ are kept so the keyword اكتئاب is unchanged
    elif language == "thai":
        t = _TH_REPEAT.sub(r"\1", t)            # มากกกกก -> มาก  (ๆ mark is untouched)
    elif language == "portuguese":
        t = t.lower().replace("[user]", "[USER]")   # lowercase, keep accents
    # chinese: NFKC only (full-width -> half-width punctuation/digits)

    return _SPACES.sub(" ", t).strip()


def mask_keywords(text: str, language: str) -> str:
    for pat in KEYWORDS[language]:
        text = re.sub(pat, "[KW]", text, flags=re.IGNORECASE)
    return text


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _load_raw(language: str, data_root: Path = REPO_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(data_root / language / f"{language}_clean_full.csv", encoding="utf-8-sig")
    test = pd.read_csv(data_root / language / f"{language}_sample_split.csv", encoding="utf-8-sig")
    full = full.rename(columns={TEXT_COL[language]: "text"})
    return full, test


# --------------------------------------------------------------------------------------
# 1 + 2. Leakage-safe split (+ Chinese downsampling)
# --------------------------------------------------------------------------------------
def build_language(language: str, data_root: Path = REPO_ROOT,
                   mask: bool = False) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} DataFrames and a 'log' of the steps."""
    full, test = _load_raw(language, data_root)
    log = [("clean_full (M2)", len(full))]

    # --- 1a. remove TEST posts (matched on the exact cleaned text) --------------------
    pool = full[~full["text"].isin(set(test["full"]))].copy()
    log.append(("remove 100 test posts", len(pool)))

    # --- 1b. Chinese: remove every post by the 100 test users -------------------------
    if language == "chinese":
        pool = pool[~pool["user_id"].isin(set(test["user_id"]))]
        log.append(("remove all posts of test users", len(pool)))

    # --- normalize text (same function for train/val/test) ----------------------------
    pool["text"] = pool["text"].map(lambda s: normalize_text(s, language))
    test = test.copy()
    test["text"] = test["full"].map(lambda s: normalize_text(s, language))
    for c in TEST_EXTRA_COLS:
        test[c] = test[c].map(lambda s: normalize_text(s, language))

    # --- extra leakage guards after normalization -------------------------------------
    pool = pool[pool["text"] != ""]
    # same text with different labels -> label is unreliable, drop all copies
    n_labels = pool.groupby("text")["label"].nunique()
    pool = pool[~pool["text"].isin(n_labels[n_labels > 1].index)]
    # duplicate texts (e.g. same Weibo post from different users) -> keep one copy
    pool = pool.drop_duplicates(subset="text", keep="first")
    # anything that became identical to a test post after normalization
    pool = pool[~pool["text"].isin(set(test["text"]))]
    log.append(("normalize + drop dup/conflicting/test-identical", len(pool)))

    # --- 2. Chinese downsampling: the same number of posts per user -------------------
    if language == "chinese":
        # shuffle once with the seed, then keep the first k posts of each user
        # (works the same on every pandas version)
        pool = (pool.sample(frac=1, random_state=SEED)
                    .groupby("user_id").head(CHINESE_POSTS_PER_USER)
                    .sort_values(["user_id"], kind="stable"))
        log.append((f"downsample to <= {CHINESE_POSTS_PER_USER} posts/user", len(pool)))

    # --- 1c. 80/20 train/val, stratified (Chinese: grouped by user) -------------------
    pool = pool.reset_index(drop=True)
    if language == "chinese":
        sgkf = StratifiedGroupKFold(n_splits=int(round(1 / VAL_SIZE)), shuffle=True,
                                    random_state=SEED)
        tr_idx, va_idx = next(sgkf.split(pool, pool["label"], groups=pool["user_id"]))
        train, val = pool.iloc[tr_idx], pool.iloc[va_idx]
    else:
        train, val = train_test_split(pool, test_size=VAL_SIZE, stratify=pool["label"],
                                      random_state=SEED)

    # --- 6. text-only output schema ---------------------------------------------------
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
            out["user_id"] = df["user_id"].values   # kept ONLY for grouping, not a feature
        return out

    train_out, val_out = _finish(train, "train"), _finish(val, "val")
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


# --------------------------------------------------------------------------------------
# Checks + summary
# --------------------------------------------------------------------------------------
def check_no_leakage(language: str, splits: dict[str, pd.DataFrame]) -> None:
    tr, va, te = (set(splits[s]["text"]) for s in ("train", "val", "test"))
    assert not (tr & va), f"{language}: train/val text overlap"
    assert not (tr & te), f"{language}: train/test text overlap"
    assert not (va & te), f"{language}: val/test text overlap"
    if language == "chinese":
        u = {s: set(splits[s]["user_id"]) for s in ("train", "val", "test")}
        assert not (u["train"] & u["val"]), "chinese: user in train and val"
        assert not (u["train"] & u["test"]), "chinese: user in train and test"
        assert not (u["val"] & u["test"]), "chinese: user in val and test"
    assert len(splits["test"]) == 100 and splits["test"]["label"].sum() == 50


def summarize(language: str, splits: dict[str, pd.DataFrame]) -> list[dict]:
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

