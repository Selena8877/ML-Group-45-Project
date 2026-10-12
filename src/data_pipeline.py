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


