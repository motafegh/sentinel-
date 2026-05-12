#!/usr/bin/env python3
"""
verify_splits.py — Three-check dataset split audit.

CHECK 1 — LABEL LEAKAGE (content-hash duplicates across splits)
  The path-based MD5 used as filenames does NOT detect contracts whose text
  is identical but stored under different paths (e.g. same vuln appears in
  two BCCC categories). This check builds a content-MD5 for every .sol file
  found in the source dirs, then looks for the same content hash appearing
  in more than one split.

CHECK 2 — CLASS IMBALANCE SKEW
  For each of the 10 classes, reports positive counts in train / val / test
  and flags any class with 0 positives in val or test (metric is unmeasurable).
  Also reports the class-level split ratio (ideal: each class ≈ 70/15/15).

CHECK 3 — SPLIT RATIO DRIFT
  Reports current row counts and percentage ratios.
  Intentional future drift: Phase 2 augmentation (CEI + DoS) adds to train only.

USAGE
─────
  source ml/.venv/bin/activate
  PYTHONPATH=. python ml/scripts/verify_splits.py
  PYTHONPATH=. python ml/scripts/verify_splits.py --no-content-check   # skip slow dup scan
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from ml.src.utils.hash_utils import get_contract_hash, get_contract_hash_from_content  # noqa: E402

DEFAULT_CSV        = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index.csv"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"

SOURCE_DIRS = [
    PROJECT_ROOT / "BCCC-SCsVul-2024" / "SourceCodes",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI-processed",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-curated",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-wild",
    PROJECT_ROOT / "ml" / "data" / "augmented",
]

CLASS_NAMES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]

IDEAL_TRAIN_PCT = 70.0
IDEAL_VAL_PCT   = 15.0
IDEAL_TEST_PCT  = 15.0
RATIO_WARN_TOL  = 2.0   # warn if a split is > ±2pp from ideal


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(csv_path: Path, splits_dir: Path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"md5_stem": "md5"})

    tr_idx = np.load(splits_dir / "train_indices.npy")
    va_idx = np.load(splits_dir / "val_indices.npy")
    te_idx = np.load(splits_dir / "test_indices.npy")

    return df, tr_idx, va_idx, te_idx


# ── Check 3 — Split ratio ─────────────────────────────────────────────────────

def check_ratio(df: pd.DataFrame, tr_idx, va_idx, te_idx) -> bool:
    total = len(df)
    tr_n, va_n, te_n = len(tr_idx), len(va_idx), len(te_idx)
    covered = tr_n + va_n + te_n

    tr_pct = 100 * tr_n / total
    va_pct = 100 * va_n / total
    te_pct = 100 * te_n / total

    logger.info("─" * 60)
    logger.info("CHECK 3 — Split ratio")
    logger.info(f"  CSV rows   : {total:,}")
    logger.info(f"  Split total: {covered:,}  (delta = {total - covered:+d})")
    logger.info(
        f"  train={tr_n:,} ({tr_pct:.1f}%)  "
        f"val={va_n:,} ({va_pct:.1f}%)  "
        f"test={te_n:,} ({te_pct:.1f}%)"
    )
    logger.info(f"  Ideal: {IDEAL_TRAIN_PCT:.0f} / {IDEAL_VAL_PCT:.0f} / {IDEAL_TEST_PCT:.0f}")

    issues = []
    if abs(tr_pct - IDEAL_TRAIN_PCT) > RATIO_WARN_TOL:
        issues.append(f"train {tr_pct:.1f}% vs ideal {IDEAL_TRAIN_PCT}%")
    if abs(va_pct - IDEAL_VAL_PCT) > RATIO_WARN_TOL:
        issues.append(f"val {va_pct:.1f}% vs ideal {IDEAL_VAL_PCT}%")
    if abs(te_pct - IDEAL_TEST_PCT) > RATIO_WARN_TOL:
        issues.append(f"test {te_pct:.1f}% vs ideal {IDEAL_TEST_PCT}%")

    if total != covered:
        logger.warning(f"  WARNING: {total - covered} rows in CSV are not in any split")
    if issues:
        for iss in issues:
            logger.warning(f"  DRIFT: {iss}")
        return False

    logger.info("  PASS — ratios within ±{RATIO_WARN_TOL}pp of ideal")
    return True


# ── Check 2 — Class imbalance skew ───────────────────────────────────────────

def check_class_balance(df: pd.DataFrame, tr_idx, va_idx, te_idx) -> bool:
    logger.info("─" * 60)
    logger.info("CHECK 2 — Class imbalance skew across splits")

    tr_df = df.iloc[tr_idx]
    va_df = df.iloc[va_idx]
    te_df = df.iloc[te_idx]

    header = f"  {'Class':<28} {'Train':>7} {'Val':>6} {'Test':>6}  {'Val%':>5} {'Test%':>6}  Status"
    logger.info(header)

    failures = []
    warnings = []

    for cls in CLASS_NAMES:
        tr_pos = int(tr_df[cls].sum())
        va_pos = int(va_df[cls].sum())
        te_pos = int(te_df[cls].sum())
        total_pos = tr_pos + va_pos + te_pos

        if total_pos == 0:
            val_pct = test_pct = 0.0
        else:
            val_pct  = 100 * va_pos / total_pos
            test_pct = 100 * te_pos / total_pos

        status = "OK"
        if va_pos == 0 or te_pos == 0:
            status = "FAIL — zero positives in " + (
                "val+test" if va_pos == 0 and te_pos == 0 else
                "val" if va_pos == 0 else "test"
            )
            failures.append(cls)
        elif val_pct < 5.0 or test_pct < 5.0:
            status = f"WARN — low representation (val={val_pct:.1f}% test={test_pct:.1f}%)"
            warnings.append(cls)

        logger.info(
            f"  {cls:<28} {tr_pos:>7,} {va_pos:>6,} {te_pos:>6,}  "
            f"{val_pct:>5.1f}% {test_pct:>6.1f}%  {status}"
        )

    if failures:
        logger.error(f"  FAIL — {len(failures)} class(es) have zero val/test positives: {failures}")
        return False
    if warnings:
        logger.warning(f"  WARN — {len(warnings)} class(es) have <5% representation: {warnings}")
    if not failures and not warnings:
        logger.info("  PASS — all classes represented in val and test")
    return len(failures) == 0


# ── Check 1 — Content-hash leakage ───────────────────────────────────────────

def _content_hash(path: Path) -> str:
    return get_contract_hash_from_content(path.read_text(encoding="utf-8", errors="ignore"))


def _build_path_md5_to_path(target_path_md5s: set[str]) -> dict[str, Path]:
    """Rebuild the path-MD5 → file path map (same as reextract_graphs.py)."""
    mapping: dict[str, Path] = {}
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            continue
        for sol in src_dir.rglob("*.sol"):
            rel = sol.relative_to(PROJECT_ROOT)
            path_md5 = get_contract_hash(rel)
            if path_md5 in target_path_md5s:
                mapping[path_md5] = sol
    return mapping


def check_leakage(df: pd.DataFrame, tr_idx, va_idx, te_idx) -> bool:
    logger.info("─" * 60)
    logger.info("CHECK 1 — Label leakage (content-hash duplicates across splits)")

    all_path_md5s = set(df["md5"].tolist())
    logger.info(f"  Scanning source dirs to map {len(all_path_md5s):,} path-MD5s → files...")

    path_md5_to_file = _build_path_md5_to_path(all_path_md5s)
    logger.info(f"  Mapped {len(path_md5_to_file):,}/{len(all_path_md5s):,} contracts to .sol files")

    unmapped = len(all_path_md5s) - len(path_md5_to_file)
    if unmapped:
        logger.info(f"  {unmapped} contracts not found in source dirs (token-only orphans) — excluded from check")

    # Assign split membership to each path-MD5
    tr_set = set(df.iloc[tr_idx]["md5"].tolist())
    va_set = set(df.iloc[va_idx]["md5"].tolist())
    te_set = set(df.iloc[te_idx]["md5"].tolist())

    # Build content-hash → list of (split, path_md5, file_path) for all mapped contracts
    logger.info(f"  Computing content hashes for {len(path_md5_to_file):,} files...")
    content_to_entries: dict[str, list] = defaultdict(list)

    from tqdm import tqdm
    for path_md5, file_path in tqdm(path_md5_to_file.items(), desc="  hashing", unit="file",
                                     leave=False, dynamic_ncols=True):
        try:
            chash = _content_hash(file_path)
        except OSError:
            continue

        split = "train" if path_md5 in tr_set else "val" if path_md5 in va_set else "test"
        content_to_entries[chash].append((split, path_md5, file_path))

    # Find content hashes that appear in multiple splits
    leaks = {
        chash: entries
        for chash, entries in content_to_entries.items()
        if len({e[0] for e in entries}) > 1   # more than one distinct split
    }

    # Summarise
    intra_dup_count = sum(
        1 for entries in content_to_entries.values() if len(entries) > 1
    )
    cross_split_count = len(leaks)

    logger.info(f"  Total unique content hashes : {len(content_to_entries):,}")
    logger.info(f"  Intra-split exact duplicates: {intra_dup_count:,} content hashes appear > once within same split")
    logger.info(f"  Cross-split duplicates      : {cross_split_count}")

    if leaks:
        logger.error(f"  FAIL — {cross_split_count} content hash(es) appear in multiple splits:")
        for chash, entries in list(leaks.items())[:20]:
            splits_involved = sorted({e[0] for e in entries})
            paths = [str(e[2].relative_to(PROJECT_ROOT)) for e in entries]
            logger.error(f"    {chash[:12]}... splits={splits_involved}")
            for p in paths[:4]:
                logger.error(f"      {p}")
        if cross_split_count > 20:
            logger.error(f"    ... and {cross_split_count - 20} more")
        return False

    logger.info("  PASS — no contract content appears in more than one split")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify dataset splits: leakage, class balance, ratio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--multilabel-csv",  type=Path, default=DEFAULT_CSV)
    p.add_argument("--splits-dir",      type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--no-content-check", action="store_true",
                   help="Skip content-hash leakage scan (faster but less thorough)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("Dataset split verification — three checks")
    logger.info(f"CSV    : {args.multilabel_csv}")
    logger.info(f"Splits : {args.splits_dir}")

    df, tr_idx, va_idx, te_idx = load_data(args.multilabel_csv, args.splits_dir)

    results = {}

    results["ratio"]   = check_ratio(df, tr_idx, va_idx, te_idx)
    results["balance"] = check_class_balance(df, tr_idx, va_idx, te_idx)

    if args.no_content_check:
        logger.info("─" * 60)
        logger.info("CHECK 1 — SKIPPED (--no-content-check)")
        results["leakage"] = None
    else:
        results["leakage"] = check_leakage(df, tr_idx, va_idx, te_idx)

    # ── Final verdict ─────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("SUMMARY")
    pass_str = lambda r: "PASS" if r else ("FAIL" if r is False else "SKIP")
    logger.info(f"  Check 1 leakage  : {pass_str(results['leakage'])}")
    logger.info(f"  Check 2 balance  : {pass_str(results['balance'])}")
    logger.info(f"  Check 3 ratio    : {pass_str(results['ratio'])}")

    any_fail = any(v is False for v in results.values())
    if any_fail:
        logger.error("OVERALL: FAIL — at least one check failed")
        sys.exit(1)
    else:
        logger.info("OVERALL: PASS")


if __name__ == "__main__":
    main()
