#!/usr/bin/env python3
"""
prepare_chinese_depglm.py
Replace the NOMADD-4 Chinese dataset (SWDD) with DepGLM (DW/train.csv).

Pipeline (Assignment 1, steps 1-8), following the paper's Appendix C/D rules:
  1. Acquire raw CSV  -> data/raw/depglm/train.csv
  2. Relabel          -> 0 = normal / "not depressed", 1-3 = depression / "depressed"
  3. Deduplicate      -> exact duplicate posts removed (first kept), count logged
  4. Structural clean -> strip HTML, normalise whitespace, drop empty posts
  5. Ethics screening -> DeepSeek prompt from Appendix D.3 (or regex heuristics with --dry-run)
  6. Stratified sample-> 2,500 per class, seed=42 (if a class is short: both classes reduced to its size + warning)
  7. Write Phase 1 JSON to data/phase1/
  8. Verify output (counts, balance, duplicates, required fields)

Usage:
  python scripts/prepare_chinese_depglm.py --dry-run          # regex ethics screening, no API
  DEEPSEEK_API_KEY=... python scripts/prepare_chinese_depglm.py  # real DeepSeek screening

Output schema, metadata keys, sampling (random.Random(seed), balanced) and file naming
(chinese_{N}samples_seed{seed}.json) follow scripts/prepare_data.py and the existing
data/phase1/chinese_5000samples_seed42.json. "index" = row position in the raw DepGLM CSV.
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import urllib.request
from datetime import datetime, timezone

import pandas as pd

RAW_URL = "https://raw.githubusercontent.com/cocollapsar/DepGLM/main/DW/train.csv"
SEED = 42
PER_CLASS = 2500

# Label conventions used across the existing Phase 1 files
LABEL_MAP = {"depression": "depressed", "normal": "not depressed"}   # internal label -> ground truth

log = logging.getLogger("prepare_chinese_depglm")


# ------------------------------------------------------------------ 1. acquisition
def acquire(raw_path):
    """Download the DepGLM CSV once and keep it under data/raw/."""
    if os.path.exists(raw_path):
        log.info("Raw file already present: %s", raw_path)
        return
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    log.info("Downloading %s", RAW_URL)
    urllib.request.urlretrieve(RAW_URL, raw_path)


def load_raw(raw_path):
    # pandas handles the 1,086 quoted posts that contain line breaks
    # (the file has ~12k physical lines but only ~8k posts)
    df = pd.read_csv(raw_path, encoding="utf-8")
    df = df.rename(columns={"post_text": "post"})
    df["index"] = range(len(df))                # original row position in DepGLM train.csv (kept as the sample "index")
    df["post"] = df["post"].astype(str)
    return df


# ------------------------------------------------------------------ 2. relabel
def relabel(df):
    """DepGLM severity 0-3 -> binary. Original severity kept for analysis."""
    df = df.copy()
    df["severity"] = df["label"].astype(int)
    df["label"] = df["severity"].apply(lambda x: "normal" if x == 0 else "depression")
    return df


# ------------------------------------------------------------------ 4. structural cleaning
HTML_TAG = re.compile(r"<[^>]+>")


def clean_post(text):
    text = HTML_TAG.sub(" ", text)              # strip HTML (same as SWDD pipeline)
    text = re.sub(r"\s+", " ", text).strip()    # whitespace normalisation (also joins multi-line posts)
    return text


# ------------------------------------------------------------------ 5. ethics screening
SYSTEM_PROMPT = (
    "You are a dataset quality-control assistant for an academic NLP corpus. "
    "You review social-media posts and decide whether they must be REMOVED from the "
    "dataset for ethical or quality reasons. You respond only with the exact format "
    "specified. Do not add explanations."
)

USER_TEMPLATE = """You are reviewing posts for an academic mental-health NLP dataset. Your task is to decide whether the post below must be REMOVED from the dataset.
Remove the post ONLY if it meets at least one of these five criteria:
--- PII --- contains a real personal email address, phone number, or home address
--- EXPLICIT_SEXUAL --- contains pornographic or sexually explicit content
--- GRAPHIC_VIOLENCE --- contains gore, torture, or graphic descriptions of physical harm to a body
--- SPAM --- consists mainly of repeated characters or words, multiple unrelated URLs, or unreadable gibberish
--- UNUSABLE --- fewer than 4 meaningful words, pure encoding artefacts, or completely untranslatable output
Do NOT remove a post solely because it discusses depression, suicide, self-harm, hopelessness, or other distressing mental-health topics --- that content is exactly what this corpus studies.
Post (Chinese Simplified):
\"\"\"{post_text}\"\"\"
Respond on two lines ONLY:
Decision: KEEP or REMOVE
Reason: <one of PII | EXPLICIT_SEXUAL | GRAPHIC_VIOLENCE | SPAM | UNUSABLE | N/A>"""

# Regex heuristics: used with --dry-run, and as the fallback on API errors (paper, Appendix D.3)
RE_PII = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"            # email
    r"|(?<!\d)1[3-9]\d{9}(?!\d)"          # mainland mobile number
    r"|(?<!\d)\d{17}[\dXx](?!\d)"         # national ID number
)
RE_SEXUAL = re.compile(r"色情|黄片|约炮|裸聊|AV女优")      # pornographic terms only (not passing mentions)
RE_VIOLENCE = re.compile(r"碎尸|肢解|血肉模糊|开膛|剁成")
RE_REPEAT = re.compile(r"(.)\1{4,}")                      # runs of 5+ identical characters
RE_URL = re.compile(r"https?://\S+")
RE_MEANINGFUL = re.compile(r"[一-鿿A-Za-z0-9]")


def is_spam(text):
    """SPAM = consists MAINLY of repeated characters, or multiple URLs.
    Emotional repetition inside a real post ("啊啊啊啊", "......") is kept."""
    repeated = sum(len(m.group(0)) for m in RE_REPEAT.finditer(text))
    return repeated > 0.5 * len(text) or len(RE_URL.findall(text)) >= 3


def regex_screen(text):
    """Return (decision, reason). Distressing mental-health content is never a reason."""
    if RE_PII.search(text):
        return "REMOVE", "PII"
    if RE_SEXUAL.search(text):
        return "REMOVE", "EXPLICIT_SEXUAL"
    if RE_VIOLENCE.search(text):
        return "REMOVE", "GRAPHIC_VIOLENCE"
    if is_spam(text):
        return "REMOVE", "SPAM"
    if len(RE_MEANINGFUL.findall(text)) < 4:      # Chinese has no spaces: count meaningful characters
        return "REMOVE", "UNUSABLE"
    return "KEEP", "N/A"


def deepseek_screen(text, api_key):
    """Call deepseek-chat with the paper's prompt; parse the two-line reply."""
    import requests  # only needed for real screening
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(post_text=text)},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"]
    decision = re.search(r"Decision:\s*(KEEP|REMOVE)", reply, re.I)
    reason = re.search(r"Reason:\s*([A-Z_/]+)", reply, re.I)
    if not decision:
        raise ValueError(f"Unparseable reply: {reply!r}")
    return decision.group(1).upper(), (reason.group(1).upper() if reason else "N/A")


def ethics_screen(df, dry_run, api_key):
    decisions, reasons, methods = [], [], []
    for i, text in enumerate(df["post"]):
        method = "regex"
        if dry_run or not api_key:
            d, r = regex_screen(text)
        else:
            try:
                d, r = deepseek_screen(text, api_key)
                method = "deepseek"
            except Exception as e:                # fall back instead of discarding the post
                log.warning("API error on post %d (%s) - using regex fallback", i, e)
                d, r = regex_screen(text)
        decisions.append(d)
        reasons.append(r)
        methods.append(method)
        if (i + 1) % 500 == 0:
            log.info("Screened %d / %d posts", i + 1, len(df))
    df = df.copy()
    df["ethics_decision"], df["ethics_reason"], df["ethics_method"] = decisions, reasons, methods
    return df


# ------------------------------------------------------------------ 6. sampling
def stratified_sample(items, n_each, seed, warnings):
    """Same logic as scripts/prepare_data.py::_stratified_sample (random.Random(seed)).
    If a class is short, BOTH classes are reduced to that size so the set stays balanced."""
    rng = random.Random(seed)
    depressed = [x for x in items if x["label"] == "depression"]
    normal = [x for x in items if x["label"] == "normal"]
    for name, pool in [("depressed", depressed), ("normal", normal)]:
        if len(pool) < n_each:
            msg = f"Only {len(pool)} {name} available (wanted {n_each})"
            log.warning(msg)
            warnings.append(msg)
            n_each = min(n_each, len(pool))
    selected = rng.sample(depressed, n_each) + rng.sample(normal, n_each)
    rng.shuffle(selected)
    return selected


# ------------------------------------------------------------------ 7. output (same schema as chinese_5000samples_seed42.json)
def build_records(df):
    return [
        {
            "index": int(row["index"]),
            "post": row["post"],
            "ground_truth": LABEL_MAP[row["label"]],
            "label": row["label"],
        }
        for _, row in df.iterrows()
    ]


def build_metadata(selected, total_available, seed):
    return {
        "language": "chinese",
        "experiment": "phase1",
        "dataset": "DepGLM (DW/train.csv)",
        "seed": seed,
        "total_available": total_available,
        "total_sampled": len(selected),
        "depressed_count": sum(1 for s in selected if s["label"] == "depression"),
        "normal_count": sum(1 for s in selected if s["label"] == "normal"),
        "post_field": "Original Chinese DepGLM text (model input)",
        "word_count_note": "Character count (CJK text has no word spaces)",
        "created": datetime.now().isoformat(),
    }


# ------------------------------------------------------------------ 8. verification
def verify(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert set(data) == {"metadata", "samples"}, "Top level must be {metadata, samples}"
    records = data["samples"]
    required = {"index": int, "post": str, "ground_truth": str, "label": str}
    for r in records:
        assert set(r) == set(required), f"Fields must be exactly {list(required)}: {r}"
        for key, typ in required.items():
            assert key in r and isinstance(r[key], typ), f"Bad/missing field '{key}' in {r}"
        assert r["post"].strip(), f"Empty post at index {r['index']}"
        assert r["label"] in LABEL_MAP, f"Bad label {r['label']}"
        assert r["ground_truth"] == LABEL_MAP[r["label"]], f"Label mismatch at index {r['index']}"
    posts = [r["post"] for r in records]
    assert len(posts) == len(set(posts)), "Duplicate posts found"
    assert len({r["index"] for r in records}) == len(records), "Duplicate indices"
    meta = data["metadata"]
    assert meta["total_sampled"] == len(records), "metadata total_sampled mismatch"
    counts = pd.Series([r["label"] for r in records]).value_counts().to_dict()
    log.info("VERIFIED %s: %d posts, class counts %s, no duplicates, all fields valid",
             path, len(records), counts)
    return counts


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/depglm/train.csv")
    ap.add_argument("--out-dir", default="data/phase1")
    ap.add_argument("--filter-log", default="data/phase1/chinese_depglm_filter_log.json")
    ap.add_argument("--dry-run", action="store_true", help="Use regex ethics heuristics instead of the DeepSeek API")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    random.seed(args.seed)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not args.dry_run and not api_key:
        log.warning("DEEPSEEK_API_KEY not set - falling back to regex screening (same as --dry-run)")

    flog = {"source": RAW_URL, "seed": args.seed, "dry_run": args.dry_run or not api_key,
            "created_utc": datetime.now(timezone.utc).isoformat(), "steps": [], "warnings": []}

    def step(name, df, **extra):
        counts = df["label"].value_counts().to_dict() if "label" in df else {}
        entry = {"step": name, "rows": len(df), "class_counts": counts, **extra}
        flog["steps"].append(entry)
        log.info("%-22s rows=%-5d %s %s", name, len(df), counts, extra or "")

    # 1. acquire + load
    acquire(args.raw)
    df = load_raw(args.raw)
    flog["raw_severity_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
    flog["raw_multiline_posts"] = int(df["post"].str.contains("\n").sum())

    # 2. relabel
    df = relabel(df)
    step("raw (relabelled)", df)

    # 3. deduplicate (before any cleaning)
    before = len(df)
    df = df.drop_duplicates(subset="post", keep="first")
    step("deduplicate", df, removed=before - len(df))

    # 4. structural cleaning
    df["post"] = df["post"].apply(clean_post)
    before = len(df)
    df = df[df["post"] != ""]
    step("structural clean", df, removed_empty=before - len(df))

    # 5. ethics screening
    df = ethics_screen(df, args.dry_run, api_key)
    removed = df[df["ethics_decision"] == "REMOVE"]
    df = df[df["ethics_decision"] == "KEEP"]
    step("ethics screening", df, removed=len(removed),
         removed_by_reason=removed["ethics_reason"].value_counts().to_dict())

    # 6. stratified sample
    selected = stratified_sample(build_records(df), args.per_class, args.seed, flog["warnings"])
    step("stratified sample", pd.DataFrame(selected))

    # 7. write output + filter log (file name follows prepare_data.py: chinese_{N}samples_seed{seed}.json)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"chinese_{len(selected)}samples_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": build_metadata(selected, len(df), args.seed), "samples": selected},
                  f, ensure_ascii=False, indent=2)
    with open(args.filter_log, "w", encoding="utf-8") as f:
        json.dump(flog, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s and %s", out_path, args.filter_log)

    # 8. verify
    verify(out_path)


if __name__ == "__main__":
    main()
