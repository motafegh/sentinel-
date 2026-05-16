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

OPTIONAL: --relabel-timestamp
─────────────────────────────
When enabled, performs source-verified Timestamp label relabeling AFTER deduplication.
For every row where Timestamp=1:
  a. Load the graph .pt file and check if uses_block_globals (feat[2]) fires for
     any function node (any value > 0.5 in x[:, 2]).
  b. Also grep the source .sol file for block.timestamp, block.number, now,
     block.difficulty, blockhash(.
  c. If NEITHER source grep NOR feature activation → set Timestamp=0.
  d. If EITHER confirms → keep Timestamp=1.
This is conservative: only removes labels where BOTH checks fail.

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
  PYTHONPATH=. python ml/scripts/dedup_multilabel_index.py --relabel-timestamp
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from ml.src.utils.hash_utils import get_contract_hash  # noqa: E402
DEFAULT_CSV    = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index.csv"
DEFAULT_OUT    = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index_deduped.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "ml" / "data" / "splits" / "deduped"
DEFAULT_GRAPHS = PROJECT_ROOT / "ml" / "data" / "graphs"

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

# ── Block-global source patterns for Timestamp relabeling ─────────────────────

_BLOCK_GLOBAL_PATTERNS = [
    re.compile(r'\bblock\s*\.\s*timestamp\b', re.IGNORECASE),
    re.compile(r'\bblock\s*\.\s*number\b',   re.IGNORECASE),
    re.compile(r'\bblock\s*\.\s*difficulty\b', re.IGNORECASE),
    re.compile(r'\bblock\s*\.\s*prevrandao\b', re.IGNORECASE),
    re.compile(r'\bblockhash\s*\(',          re.IGNORECASE),
    re.compile(r'\bnow\b'),  # Solidity <0.7 alias for block.timestamp
]


# ── Step 1 — Rebuild path-MD5 → content-hash map ─────────────────────────────

def build_content_map(all_path_md5s: set[str], source_dirs: list[Path]) -> dict[str, str]:
    """Return path_md5 → content_md5 for all mapped files."""
    path_md5_to_content: dict[str, str] = {}

    for src_dir in source_dirs:
        if not src_dir.exists():
            continue
        for sol in src_dir.rglob("*.sol"):
            rel = sol.relative_to(PROJECT_ROOT)
            pm = get_contract_hash(rel)
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


# ── Step 3 — Source-verified Timestamp relabeling ────────────────────────────

def _find_source_for_md5(md5: str, source_dirs: list[Path]) -> Path | None:
    """Walk source dirs to find the .sol file whose path-MD5 matches *md5*.

    NOTE: This is O(N × |source files|). Call build_md5_to_sol_map() once and
    use the returned dict directly — don't call this in a tight loop.
    """
    for src_dir in source_dirs:
        if not src_dir.exists():
            continue
        for sol in src_dir.rglob("*.sol"):
            rel = sol.relative_to(PROJECT_ROOT)
            if get_contract_hash(rel) == md5:
                return sol
    return None


def _build_md5_to_sol_map(
    md5_set: set[str], source_dirs: list[Path]
) -> dict[str, Path]:
    """Scan source dirs ONCE and return {md5 → absolute .sol path} for all md5s in md5_set."""
    md5_map: dict[str, Path] = {}
    for src_dir in source_dirs:
        if not src_dir.exists():
            continue
        for sol in src_dir.rglob("*.sol"):
            rel = sol.relative_to(PROJECT_ROOT)
            m = get_contract_hash(rel)
            if m in md5_set and m not in md5_map:
                md5_map[m] = sol
    return md5_map


def _source_has_block_globals(sol_path: Path) -> bool:
    """Return True if any block-global pattern matches the source file."""
    try:
        text = sol_path.read_text(errors="replace")
    except OSError:
        return False
    return any(pat.search(text) for pat in _BLOCK_GLOBAL_PATTERNS)


def _graph_has_block_globals(md5: str, graphs_dir: Path) -> bool:
    """Return True if uses_block_globals (feat index 2) fires in any function node."""
    pt_path = graphs_dir / f"{md5}.pt"
    if not pt_path.exists():
        return False

    import torch

    try:
        data = torch.load(pt_path, weights_only=False)
    except Exception:
        return False

    if not hasattr(data, "x") or data.x is None:
        return False

    # feat[2] = uses_block_globals; any node with value > 0.5 means it fires
    try:
        block_global_feats = data.x[:, 2]
        return bool((block_global_feats > 0.5).any())
    except (IndexError, RuntimeError):
        return False


def relabel_timestamp(
    deduped_df: pd.DataFrame,
    graphs_dir: Path,
    source_dirs: list[Path],
    dry_run: bool,
) -> pd.DataFrame:
    """
    Source-verified Timestamp label relabeling.

    For every row where Timestamp=1:
      - Check graph features (uses_block_globals, feat[2] > 0.5 on any node)
      - Check source patterns (block.timestamp, block.number, etc.)
      - If NEITHER confirms → set Timestamp=0
      - If EITHER confirms → keep Timestamp=1
    """
    ts_mask = deduped_df["Timestamp"] == 1
    ts_indices = deduped_df.index[ts_mask].tolist()
    n_ts = len(ts_indices)
    logger.info(f"  Timestamp=1 rows to verify: {n_ts:,}")

    # Pre-build md5 → .sol path map with a single source-dir scan (not per-row)
    ts_md5_set = set(deduped_df.loc[ts_mask, "md5"])
    logger.info("  Building md5→source map (single scan) ...")
    md5_to_sol = _build_md5_to_sol_map(ts_md5_set, source_dirs)
    logger.info(f"  {len(md5_to_sol):,}/{n_ts:,} Timestamp rows mapped to source file")

    n_removed = 0
    n_confirmed_graph = 0
    n_confirmed_source = 0
    n_confirmed_both = 0
    n_no_graph = 0
    n_no_source = 0

    for idx in ts_indices:
        md5 = deduped_df.at[idx, "md5"]

        # Check graph features
        graph_confirms = _graph_has_block_globals(md5, graphs_dir)

        # Check source patterns (use pre-built map — no per-row directory scan)
        sol_path = md5_to_sol.get(md5)
        source_confirms = False
        if sol_path is not None:
            source_confirms = _source_has_block_globals(sol_path)
        else:
            n_no_source += 1

        if not graph_confirms and not source_confirms:
            if not dry_run:
                deduped_df.at[idx, "Timestamp"] = 0
            n_removed += 1

        # Track confirmation sources
        if graph_confirms and source_confirms:
            n_confirmed_both += 1
        elif graph_confirms:
            n_confirmed_graph += 1
        elif source_confirms:
            n_confirmed_source += 1

        if not graph_confirms and sol_path is None:
            n_no_graph += 1

    logger.info(f"  Timestamp relabeling results:")
    logger.info(f"    Removed (neither source nor graph confirms): {n_removed:,}")
    logger.info(f"    Confirmed by both source + graph: {n_confirmed_both:,}")
    logger.info(f"    Confirmed by graph only: {n_confirmed_graph:,}")
    logger.info(f"    Confirmed by source only: {n_confirmed_source:,}")
    logger.info(f"    No graph .pt found: {n_no_graph:,}")
    logger.info(f"    No source .sol found: {n_no_source:,}")
    logger.info(
        f"    Timestamp=1 remaining: {n_ts - n_removed:,} / {n_ts:,}  "
        f"({100 * (n_ts - n_removed) / n_ts:.1f}% kept)" if n_ts else
        f"    Timestamp=1 remaining: 0 / 0"
    )

    return deduped_df


# ── Step 4 — Rebuild splits ───────────────────────────────────────────────────

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
    p.add_argument("--graphs-dir",     type=Path, default=DEFAULT_GRAPHS,
                   help="Directory containing graph .pt files for feature-based checks")
    p.add_argument("--source-dirs",    type=Path, nargs="*", default=SOURCE_DIRS,
                   help="Directories to scan for .sol source files")
    p.add_argument("--relabel-timestamp", action="store_true",
                   help="After dedup, verify Timestamp=1 labels against source "
                        "patterns and graph features; remove labels neither confirms")
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
    path_md5_to_content = build_content_map(set(df["md5"]), args.source_dirs)
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

    # ── Source-verified Timestamp relabeling ──────────────────────────────────
    if args.relabel_timestamp:
        logger.info("Source-verified Timestamp relabeling ...")
        ts_before = int(deduped_df["Timestamp"].sum())
        deduped_df = relabel_timestamp(
            deduped_df,
            graphs_dir=args.graphs_dir,
            source_dirs=args.source_dirs,
            dry_run=args.dry_run,
        )
        ts_after = int(deduped_df["Timestamp"].sum())
        logger.info(
            f"  Timestamp labels: {ts_before:,} → {ts_after:,}  "
            f"(removed {ts_before - ts_after:,} unverified labels)"
        )

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
    if args.relabel_timestamp:
        logger.info(f"  Timestamp relabeling: {ts_before - ts_after:,} labels removed")
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
