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


