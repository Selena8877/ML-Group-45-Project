from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# load prompts.py directly from its file: importing src.evaluation.prompts runs
# src/evaluation/__init__.py, which imports sampler.py (not in the repo)
_spec = importlib.util.spec_from_file_location(
    "nomadd_prompts", PROJECT_ROOT / "src" / "evaluation" / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)
PHASE1_PROMPTS = _prompts.PHASE1_PROMPTS
LANGUAGE_FEWSHOT_PROMPT = _prompts.LANGUAGE_FEWSHOT_PROMPT

SEED = 42
random.seed(SEED)

ABLATION_DIR = PROJECT_ROOT / "ablation"
PHASE2_DIR = PROJECT_ROOT / "results" / "phase2"
PHASE2_AUDIT_DIR = PROJECT_ROOT / "results" / "phase2_audit"  # default output folder of create_phase2_audit.py
PHASE1_DIR = PROJECT_ROOT / "data" / "phase1"
LOG_PATH = ABLATION_DIR / "ablation_phase2.log"
CHECKPOINT_DIR = ABLATION_DIR / "checkpoints_phase2"
RESULTS_DIR = ABLATION_DIR / "results_phase2"

LANGUAGES = ["arabic", "bengali", "chinese", "urdu"]
MODELS = ["gemini", "openai", "claude", "llama", "gemma", "qwen"]  # same keys as scripts/run_phase1.py
CONDITIONS = [
    "FULL_FS", "NO_CHECKLIST", "NO_GUIDANCE", "NO_EXAMPLES",
    "CHECKLIST_ONLY", "GUIDANCE_ONLY", "EXAMPLES_ONLY",
]

EXPECTED_POSTS = 250
POST_TOLERANCE = 10  # flag a language with fewer than 240 or more than 260 posts

log = logging.getLogger("ablation_phase2")


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(level)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    screen = logging.StreamHandler(sys.stdout)
    screen.setFormatter(fmt)
    log.addHandler(screen)

    file = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file.setFormatter(fmt)
    log.addHandler(file)

    log.propagate = False
    return log


# ---------------------------------------------------------------------------
# phase 2 data
# ---------------------------------------------------------------------------

def _phase2_candidates(language: str) -> list[Path]:
    # newest audit output first, the older human-evaluation workbook last
    name = f"phase2_audit_{language}_seed{SEED}.xlsx"
    return [
        PHASE2_DIR / name,
        PHASE2_AUDIT_DIR / name,
        PHASE2_DIR / f"{language.capitalize()} HE.xlsx",
    ]


def _norm(col) -> str:
    # "post\noriginal" and "post_original" both become "post_original"
    return re.sub(r"[\s_]+", "_", str(col).strip().lower())


def _read_phase2_sheet(path: Path) -> pd.DataFrame:
    # the workbooks have a banner row above the column names, so search for the header row
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str)
    for row in range(min(5, len(raw))):
        names = [_norm(c) for c in raw.iloc[row]]
        if "post_original" in names and "ground_truth" in names:
            df = raw.iloc[row + 1:].copy()
            df.columns = names
            return df.reset_index(drop=True)
    raise ValueError(f"{path.name}: no header row with 'post original' and 'ground truth'")


def _normalise_label(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"[\s_]+", " ", str(value).strip().lower())
    if text in {"depressed", "depression", "1"}:
        return "depressed"
    if text in {"not depressed", "normal", "non depressed", "non-depressed", "0"}:
        return "not depressed"
    return None


def _load_phase1_index(language: str, phase2_posts: list[str]) -> tuple[dict[str, int], str, str]:
    # maps post text to its phase 1 index. if data/phase1 has several files for one
    # language (e.g. swdd and depglm chinese), the file containing the most phase 2
    # posts is used. urdu samples have no "index" field, so list position is used
    # instead, as in run_phase1.py
    best = ({}, "missing", "none", -1)
    wanted = set(phase2_posts)
    for path in sorted(PHASE1_DIR.glob(f"{language}_*samples_seed{SEED}.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        samples = data["samples"] if isinstance(data, dict) else data
        lookup = {}
        for pos, s in enumerate(samples):
            lookup.setdefault(str(s["post"]).strip(), s.get("index", pos))
        hits = len(wanted & lookup.keys())
        if hits > best[3]:
            meta = data.get("metadata", {}) if isinstance(data, dict) else {}
            best = (lookup, meta.get("dataset", "unknown"), path.name, hits)
    return best[0], best[1], best[2]


# per-language load summary, used by the dry run
DATA_SOURCES: dict[str, dict] = {}


def load_phase2_posts(languages: list[str]) -> dict[str, list[dict]]:
    # returns {language: [{language, index, post, ground_truth}, ...]}
    posts_by_lang: dict[str, list[dict]] = {}

    for lang in languages:
        path = next((p for p in _phase2_candidates(lang) if p.exists()), None)
        if path is None:
            log.error(f"[{lang}] no Phase 2 file found. Looked for: "
                      + ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in _phase2_candidates(lang)))
            posts_by_lang[lang] = []
            continue

        df = _read_phase2_sheet(path)
        n_rows = len(df)
        df["post_original"] = df["post_original"].astype("string").str.strip()
        df["label"] = df["ground_truth"].map(_normalise_label)

        empty = df["post_original"].isna() | (df["post_original"] == "")
        bad_label = ~empty & df["label"].isna()
        if bad_label.any():
            log.warning(f"[{lang}] {bad_label.sum()} rows skipped: unknown ground truth "
                        f"{sorted(df.loc[bad_label, 'ground_truth'].astype(str).unique())[:5]}")
        df = df[~empty & ~bad_label]

        dupes = df.duplicated(["post_original", "label"], keep="first")
        if dupes.any():
            log.warning(f"[{lang}] {dupes.sum()} exact duplicate rows dropped (kept first)")
        df = df[~dupes]

        conflicts = df.groupby("post_original")["label"].nunique().gt(1).sum()
        if conflicts:
            log.warning(f"[{lang}] {conflicts} posts appear with BOTH labels; kept as separate rows")

        lookup, dataset, phase1_file = _load_phase1_index(lang, list(df["post_original"]))
        posts, used, unmatched = [], set(), 0
        for row_no, (text, label) in enumerate(zip(df["post_original"], df["label"]), start=1):
            idx = lookup.get(text)
            if idx is None or idx in used:
                # no phase 1 match: stable row-based id, offset so it cannot clash with a real index
                idx = 900000 + row_no
                unmatched += 1
            used.add(idx)
            posts.append({"language": lang, "index": int(idx), "post": str(text), "ground_truth": label})

        counts = Counter(p["ground_truth"] for p in posts)
        DATA_SOURCES[lang] = {
            "file": str(path.relative_to(PROJECT_ROOT)),
            "rows_in_file": n_rows,
            "posts": len(posts),
            "depressed": counts["depressed"],
            "not_depressed": counts["not depressed"],
            "phase1_dataset": dataset,
            "phase1_file": phase1_file,
            "phase1_matched": len(posts) - unmatched,
        }
        posts_by_lang[lang] = posts

        log.info(f"[{lang}] loaded {len(posts)} posts from {path.name} "
                 f"(depressed={counts['depressed']}, not depressed={counts['not depressed']}; "
                 f"{len(posts) - unmatched}/{len(posts)} matched to Phase 1 '{dataset}')")
        if abs(len(posts) - EXPECTED_POSTS) > POST_TOLERANCE:
            log.warning(f"[{lang}] expected about {EXPECTED_POSTS} posts, got {len(posts)}")

    _check_chinese_is_depglm()
    return posts_by_lang


def _check_chinese_is_depglm() -> None:
    info = DATA_SOURCES.get("chinese")
    if not info:
        return
    is_depglm = "depglm" in info["phase1_dataset"].lower()
    info["is_depglm"] = is_depglm
    if not is_depglm:
        log.warning("[chinese] Phase 2 posts are NOT from DepGLM "
                    f"(matched Phase 1 dataset = '{info['phase1_dataset']}'). Expected the DepGLM "
                    "Phase 1 file in data/phase1/ and phase2_audit_chinese_seed42.xlsx in results/phase2/.")
    elif info["phase1_matched"] < info["posts"]:
        log.warning(f"[chinese] only {info['phase1_matched']}/{info['posts']} Phase 2 posts found in "
                    "the DepGLM Phase 1 file")


# ---------------------------------------------------------------------------
# prompt variants
# ---------------------------------------------------------------------------

# every few-shot prompt in prompts.py has the same headings. a section starts at
# its marker and ends where the next one starts:
#   header     role + "read the full post" instruction
#   checklist  depression indicators + "do not classify as depressed" exclusions
#   guidance   script/language note (not in every language) + context cautions
#   examples   in-context examples
#   footer     output format + {post_text}
SECTION_MARKERS = {
    "checklist": ["Consider these indicators of depression:"],
    "guidance":  ["The post may be in", "Context cautions"],  # whichever comes first
    "examples":  ["Here are examples to guide your classification:"],
    "footer":    ["Now classify the following post."],
}
SECTION_ORDER = ["header", "checklist", "guidance", "examples", "footer"]

CONDITION_SECTIONS = {
    "FULL_FS":        ["header", "checklist", "guidance", "examples", "footer"],
    "NO_CHECKLIST":   ["header", "guidance", "examples", "footer"],
    "NO_GUIDANCE":    ["header", "checklist", "examples", "footer"],
    "NO_EXAMPLES":    ["header", "checklist", "guidance", "footer"],
    "CHECKLIST_ONLY": ["header", "checklist", "footer"],
    "GUIDANCE_ONLY":  ["header", "guidance", "footer"],
    "EXAMPLES_ONLY":  ["header", "examples", "footer"],
}

# text that appears only in its own section, used to check which sections a variant contains
SECTION_SIGNATURES = {
    "header":    "You are a mental health text classifier",
    "checklist": "Consider these indicators of depression:",
    "guidance":  "Context cautions",
    "examples":  "Here are examples to guide your classification:",
    "footer":    "Now classify the following post.",
}


def split_sections(template: str, language: str) -> dict[str, str]:
    starts = {"header": 0}
    for section, markers in SECTION_MARKERS.items():
        found = [template.find(m) for m in markers if template.find(m) != -1]
        assert found, f"[{language}] marker for '{section}' not found: {markers}"
        starts[section] = min(found)

    positions = [starts[s] for s in SECTION_ORDER]
    assert positions == sorted(positions), \
        f"[{language}] sections are out of order: {dict(zip(SECTION_ORDER, positions))}"

    sections = {}
    for i, name in enumerate(SECTION_ORDER):
        end = starts[SECTION_ORDER[i + 1]] if i + 1 < len(SECTION_ORDER) else len(template)
        sections[name] = template[starts[name]:end]

    # the split must be lossless
    assert "".join(sections[s] for s in SECTION_ORDER) == template, \
        f"[{language}] sections do not add back up to the original prompt"
    return sections


def validate_variant(prompt: str, language: str, condition: str) -> None:
    tag = f"[{language}/{condition}]"
    wanted = set(CONDITION_SECTIONS[condition])

    for section, signature in SECTION_SIGNATURES.items():
        present = signature in prompt
        if section in wanted:
            assert present, f"{tag} missing section '{section}'"
        else:
            assert not present, f"{tag} contains section '{section}' that should be removed"

    has_examples = bool(re.search(r"^Example \d+", prompt, flags=re.M))
    assert has_examples == ("examples" in wanted), f"{tag} example blocks present={has_examples}"

    assert prompt.count("{post_text}") == 1, \
        f"{tag} needs exactly one {{post_text}}, found {prompt.count('{post_text}')}"
    # prompts are filled with str.format, so any stray brace would fail here
    rendered = prompt.format(post_text="<<TEST POST>>")
    assert "<<TEST POST>>" in rendered and "{post_text}" not in rendered, f"{tag} {{post_text}} did not fill in"

    assert "Classification: Depressed OR Not Depressed" in prompt, f"{tag} output format line missing"
    assert prompt.rstrip().endswith("Response:"), f"{tag} prompt must end with 'Response:'"


def build_prompts(languages: list[str], conditions: list[str]) -> dict[str, dict[str, str]]:
    # returns prompts[language][condition]
    prompts: dict[str, dict[str, str]] = {}
    for lang in languages:
        template = PHASE1_PROMPTS[LANGUAGE_FEWSHOT_PROMPT[lang]]
        sections = split_sections(template, lang)
        prompts[lang] = {}
        for cond in conditions:
            prompt = "".join(sections[s] for s in CONDITION_SECTIONS[cond])
            validate_variant(prompt, lang, cond)
            prompts[lang][cond] = prompt

        assert prompts[lang].get("FULL_FS", template) == template, \
            f"[{lang}] FULL_FS must equal the original few-shot prompt"
    return prompts


def prompt_summary(prompts: dict[str, dict[str, str]]) -> pd.DataFrame:
    # character count per variant (language x condition)
    rows = [{"language": lang, "condition": cond, "chars": len(p)}
            for lang, by_cond in prompts.items() for cond, p in by_cond.items()]
    return pd.DataFrame(rows).pivot(index="language", columns="condition", values="chars")


# ---------------------------------------------------------------------------
# model execution
# ---------------------------------------------------------------------------

def checkpoint_completion(model: str, language: str, condition: str, posts: list[dict]) -> float | None:
    # % of posts already saved in ablation/checkpoints_phase2/, or none if there is no checkpoint
    return None


def run_ablation(models: list[str], posts_by_lang: dict[str, list[dict]],
                 prompts: dict[str, dict[str, str]], conditions: list[str]) -> None:
    raise NotImplementedError("model execution is not implemented yet")


# ---------------------------------------------------------------------------
# command line + dry run
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NOMADD-4 Phase 2 prompt-component ablation (7 conditions).")
    p.add_argument("--model", nargs="+", choices=MODELS + ["all"], default=["all"],
                   help="models to run (default: all 6)")
    p.add_argument("--language", nargs="+", choices=LANGUAGES + ["all"], default=["all"],
                   help="languages to run (default: all 4)")
    p.add_argument("--condition", nargs="+", choices=CONDITIONS + ["all"], default=["all"],
                   help="prompt conditions to run (default: all 7)")
    p.add_argument("--dry-run", action="store_true",
                   help="check data, prompts and checkpoints and count API calls; no model calls")
    args = p.parse_args(argv)

    args.model = MODELS if "all" in args.model else args.model
    args.language = LANGUAGES if "all" in args.language else args.language
    # keep the fixed condition order so the baseline condition always comes first
    args.condition = CONDITIONS if "all" in args.condition else [c for c in CONDITIONS if c in args.condition]
    return args


def dry_run(args, posts_by_lang, prompts) -> int:
    # returns 0 if everything checks out, 1 if there are issues
    problems = 0
    log.info("=" * 70)
    log.info("DRY RUN: no model calls will be made")
    log.info(f"models={args.model}")
    log.info(f"languages={args.language}")
    log.info(f"conditions={args.condition}")

    log.info("-" * 70)
    log.info(f"1. Prompt variants: {sum(len(v) for v in prompts.values())} built, all checks passed "
             "(right sections, one {post_text}, output format, ends with 'Response:')")
    for line in prompt_summary(prompts).to_string().splitlines():
        log.info("   " + line)

    log.info("-" * 70)
    log.info("2. Phase 2 data")
    for lang in args.language:
        info = DATA_SOURCES.get(lang)
        if not info:
            log.error(f"   {lang:<8} NOT LOADED")
            problems += 1
            continue
        ok = abs(info["posts"] - EXPECTED_POSTS) <= POST_TOLERANCE
        problems += not ok
        log.info(f"   {lang:<8} {info['posts']:>4} posts  dep={info['depressed']:<4} not={info['not_depressed']:<4} "
                 f"{'OK ' if ok else 'CHECK'}  {info['file']}  (Phase 1: {info['phase1_dataset']}, "
                 f"{info['phase1_matched']}/{info['posts']} matched)")
    if "chinese" in DATA_SOURCES and not DATA_SOURCES["chinese"].get("is_depglm"):
        log.warning("   chinese  still the OLD SWDD sample, not DepGLM (see warning above)")
        problems += 1

    log.info("-" * 70)
    log.info("3. Checkpoints (ablation/checkpoints_phase2/)")
    total = sum(len(posts_by_lang.get(l, [])) for l in args.language) * len(args.condition) * len(args.model)
    remaining = 0
    for model in args.model:
        for lang in args.language:
            for cond in args.condition:
                posts = posts_by_lang.get(lang, [])
                pct = checkpoint_completion(model, lang, cond, posts)
                done = 0 if pct is None else round(len(posts) * pct / 100)
                remaining += len(posts) - done
                if pct is not None:
                    log.info(f"   {model:<7} {lang:<8} {cond:<15} {pct:5.1f}% done")
    if remaining == total:
        log.info("   no completed predictions found (fresh run)")

    log.info("-" * 70)
    n_posts = sum(len(posts_by_lang.get(l, [])) for l in args.language)
    log.info(f"4. Projected API calls: {total:,} in total = {n_posts} posts x "
             f"{len(args.condition)} conditions x {len(args.model)} models "
             f"({n_posts * len(args.condition):,} per model); still to do after checkpoints: {remaining:,}")
    for lang in args.language:
        n = len(posts_by_lang.get(lang, []))
        log.info(f"   {lang:<8} {n} x {len(args.condition)} x {len(args.model)} = {n * len(args.condition) * len(args.model):,}")

    log.info("=" * 70)
    log.info("DRY RUN OK" if problems == 0 else f"DRY RUN finished with {problems} issue(s) to check")
    return 0 if problems == 0 else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    log.info(f"Phase 2 ablation | seed={SEED} | log file: {LOG_PATH.relative_to(PROJECT_ROOT)}")

    try:
        prompts = build_prompts(args.language, args.condition)
    except AssertionError as err:
        log.error(f"Prompt validation failed: {err}")
        return 2

    posts_by_lang = load_phase2_posts(args.language)

    if args.dry_run:
        return dry_run(args, posts_by_lang, prompts)

    if not any(posts_by_lang.values()):
        log.error("No Phase 2 posts loaded; nothing to run.")
        return 1
    run_ablation(args.model, posts_by_lang, prompts, args.condition)
    return 0


if __name__ == "__main__":
    sys.exit(main())