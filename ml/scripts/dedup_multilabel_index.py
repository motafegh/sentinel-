#!/usr/bin/env python3
"""
dedup_multilabel_index.py — Content-deduplicate multilabel_index.csv and rebuild splits.

PROBLEM
───────
BCCC-SCsVul-2024 stores the same .sol file in multiple category directories
(one copy per vulnerability class). Path-based MD5 creates separate rows for
each copy. Random splitting scatters these copies across train/val/test, leaking
training data into validation and test (34.9% of rows affected).

FIX
───
1. Hash the content (not the path) of every .sol file.
2. Group all rows that share a content hash.
3. Merge their labels with OR (a contract in both Timestamp/ and IntegerUO/ →
   both flags = 1 — correct multi-label annotation).
4. Keep the alphabetically-first path-MD5 as the canonical row identifier.
5. Write a new deduped CSV: multilabel_index_deduped.csv (44,420 rows vs 68,523).
6. Rebuild train/val/test splits on the deduped index (stratified by label density,
   same 70/15/15 ratio). Write to splits/deduped/.

AFTER THIS SCRIPT
─────────────────
- Update create_cache.py (or pass --label-csv flag) to use the deduped CSV.
- Retrain using the deduped CSV and new splits.
- Old graph .pt files for non-canonical duplicate MD5s remain on disk (harmless;
  they will simply not be loaded since their MD5s are absent from the deduped CSV).

USAGE
─────
  source ml/.venv/bin/activate
  PYTHONPATH=. python ml/scripts/dedup_multilabel_index.py
  PYTHONPATH=. python ml/scripts/dedup_multilabel_index.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DEFAULT_CSV    = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index.csv"
DEFAULT_OUT    = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index_deduped.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "ml" / "data" / "splits" / "deduped"

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

TRAIN_PCT = 0.70
VAL_PCT   = 0.15
TEST_PCT  = 0.15
SEED      = 42


# ── Step 1 — Rebuild path-MD5 → content-hash map ─────────────────────────────

def build_content_map(all_path_md5s: set[str]) -> dict[str, str]:
    """Return path_md5 → content_md5 for all mapped files."""
    path_md5_to_content: dict[str, str] = {}

    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            continue
        for sol in src_dir.rglob("*.sol"):
            rel = sol.relative_to(PROJECT_ROOT)
            pm = hashlib.md5(str(rel).encode("utf-8")).hexdigest()
            if pm not in all_path_md5s:
                continue
            if pm in path_md5_to_content:
                continue   # already mapped (can't appear twice in different source dirs with same path)
            cm = hashlib.md5(sol.read_bytes()).hexdigest()
            path_md5_to_content[pm] = cm

    return path_md5_to_content


# ── Step 2 — Deduplicate and merge labels ─────────────────────────────────────

def deduplicate(df: pd.DataFrame, path_md5_to_content: dict[str, str]) -> pd.DataFrame:
    """
    Group rows by content hash, OR labels, keep alphabetically-first MD5.
    Vectorised: attaches content_hash column, groups by it, aggregates in one pass.
    Rows whose source file was not found (orphans) are kept as-is (1-element group).
    """
    # Attach content hash and canonical MD5 columns
    df = df.copy()
    df["_content_hash"] = df["md5"].map(path_md5_to_content)

    orphan_mask = df["_content_hash"].isna()
    n_orphans   = orphan_mask.sum()
    logger.info(f"  Orphan rows (no .sol found): {n_orphans:,}  — kept as-is")

    # For orphans: use their own md5 as the group key so they survive groupby
    df.loc[orphan_mask, "_content_hash"] = df.loc[orphan_mask, "md5"]

    # Canonical MD5 per content group = alphabetically first md5 in that group
    df["_canonical"] = df.groupby("_content_hash")["md5"].transform("min")

    # Aggregate: OR labels (max of 0/1 = OR), canonical MD5 as group id
    agg = (
        df.groupby("_content_hash", sort=False)
          .agg(md5=("_canonical", "first"), **{c: (c, "max") for c in CLASS_NAMES})
          .reset_index(drop=True)
    )

    logger.info(f"  Content groups : {len(agg):,}")
    return agg[["md5"] + CLASS_NAMES]


# ── Step 3 — Rebuild splits ───────────────────────────────────────────────────

def rebuild_splits(
    deduped_df: pd.DataFrame,
    splits_dir: Path,
    dry_run: bool,
) -> dict[str, np.ndarray]:
    """
    Stratified 70/15/15 split on the deduplicated index.

    Stratification key: number of positive labels per row (0 = safe, 1 = single,
    2+ = multi-label). This preserves the proportion of safe vs vulnerable contracts
    and avoids putting all rare DoS positives in one split.
    """
    rng = np.random.default_rng(SEED)

    n = len(deduped_df)
    all_idx = np.arange(n)

    # Stratify by label count bucket: 0, 1, 2, 3+
    label_counts = deduped_df[CLASS_NAMES].sum(axis=1).clip(upper=3).astype(int).values

    train_idx, val_idx, test_idx = [], [], []

    for bucket in range(4):
        bucket_idx = all_idx[label_counts == bucket]
        rng.shuffle(bucket_idx)
        n_b = len(bucket_idx)
        n_train = int(n_b * TRAIN_PCT)
        n_val   = int(n_b * VAL_PCT)
        train_idx.append(bucket_idx[:n_train])
        val_idx.append(bucket_idx[n_train : n_train + n_val])
        test_idx.append(bucket_idx[n_train + n_val:])

    train_arr = np.concatenate(train_idx)
    val_arr   = np.concatenate(val_idx)
    test_arr  = np.concatenate(test_idx)

    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    rng.shuffle(test_arr)

    logger.info(
        f"  Split sizes — train={len(train_arr):,}  val={len(val_arr):,}  test={len(test_arr):,}"
        f"  total={len(train_arr)+len(val_arr)+len(test_arr):,}"
    )

    # Verify no overlap
    assert len(set(train_arr) & set(val_arr)) == 0
    assert len(set(train_arr) & set(test_arr)) == 0
    assert len(set(val_arr) & set(test_arr)) == 0

    if not dry_run:
        splits_dir.mkdir(parents=True, exist_ok=True)
        np.save(splits_dir / "train_indices.npy", train_arr)
        np.save(splits_dir / "val_indices.npy",   val_arr)
        np.save(splits_dir / "test_indices.npy",  test_arr)
        logger.info(f"  Splits written to {splits_dir}")

    return {"train": train_arr, "val": val_arr, "test": test_arr}


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_class_balance(
    deduped_df: pd.DataFrame,
    splits: dict[str, np.ndarray],
) -> None:
    tr_df = deduped_df.iloc[splits["train"]]
    va_df = deduped_df.iloc[splits["val"]]
    te_df = deduped_df.iloc[splits["test"]]

    logger.info(f"  {'Class':<28} {'Train':>7} {'Val':>6} {'Test':>6}  {'Val%':>5} {'Test%':>6}")
    for cls in CLASS_NAMES:
        tr_pos = int(tr_df[cls].sum())
        va_pos = int(va_df[cls].sum())
        te_pos = int(te_df[cls].sum())
        total  = tr_pos + va_pos + te_pos
        vp     = 100 * va_pos / total if total else 0
        tp     = 100 * te_pos / total if total else 0
        logger.info(
            f"  {cls:<28} {tr_pos:>7,} {va_pos:>6,} {te_pos:>6,}  {vp:>5.1f}% {tp:>6.1f}%"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Content-dedup multilabel_index.csv and rebuild splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--multilabel-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-csv",        type=Path, default=DEFAULT_OUT)
    p.add_argument("--splits-dir",     type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--dry-run",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    if args.dry_run:
        logger.info("DRY RUN — no files written")

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.multilabel_csv} ...")
    df = pd.read_csv(args.multilabel_csv).rename(columns={"md5_stem": "md5"})
    logger.info(f"  {len(df):,} rows loaded")

    # ── Map path-MD5 → content hash ───────────────────────────────────────────
    logger.info("Scanning source dirs to build content-hash map ...")
    path_md5_to_content = build_content_map(set(df["md5"]))
    logger.info(f"  {len(path_md5_to_content):,}/{len(df):,} rows mapped to content hash")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    logger.info("Deduplicating by content hash (OR labels) ...")
    deduped_df = deduplicate(df, path_md5_to_content)
    logger.info(
        f"  {len(df):,} → {len(deduped_df):,} rows  "
        f"(removed {len(df) - len(deduped_df):,} duplicates)"
    )

    # Label comparison
    logger.info("  Label counts before → after dedup:")
    for cls in CLASS_NAMES:
        before = int(df[cls].sum())
        after  = int(deduped_df[cls].sum())
        logger.info(f"    {cls:<28} {before:>7,} → {after:>7,}  ({after-before:+,})")

    # ── Rebuild splits ────────────────────────────────────────────────────────
    logger.info(f"Rebuilding stratified 70/15/15 splits → {args.splits_dir} ...")
    splits = rebuild_splits(deduped_df, args.splits_dir, dry_run=args.dry_run)

    # Per-class balance in new splits
    logger.info("Per-class balance in new splits:")
    report_class_balance(deduped_df, splits)

    # ── Write deduped CSV ─────────────────────────────────────────────────────
    if not args.dry_run:
        out_csv = args.out_csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        deduped_df.rename(columns={"md5": "md5_stem"}).to_csv(out_csv, index=False)
        logger.info(f"Deduped CSV written: {out_csv}")

    logger.info("═" * 60)
    logger.info("DONE")
    logger.info(f"  Original    : {len(df):,} rows, splits in ml/data/splits/")
    logger.info(f"  Deduplicated: {len(deduped_df):,} rows, splits in {args.splits_dir}")
    logger.info("")
    if not args.dry_run:
        logger.info("Next steps:")
        logger.info("  python ml/scripts/create_cache.py \\")
        logger.info(f"    --label-csv {args.out_csv} \\")
        logger.info(f"    --splits-dir {args.splits_dir} \\")
        logger.info("    --output ml/data/cached_dataset_deduped.pkl")
        logger.info("  Then retrain using --label-csv and --splits-dir flags.")


if __name__ == "__main__":
    main()
