"""
clean_integeruo_labels.py — Sol-2: Solidity ≥0.8.0 IntegerUO label cleaning

PURPOSE
───────
Removes IntegerUO=1 labels from contracts compiled with Solidity ≥0.8.0 that
do NOT use unchecked{} blocks. Solidity ≥0.8.0 has built-in overflow protection
that reverts on overflow. Only unchecked{} blocks can disable this protection.

RELIABILITY: HIGH
─────────────────
The version check is deterministic — overflow is physically impossible in
0.8.0+ without unchecked{}, by the language spec. The only failure modes are:

- `contract_path` not accessible (graph.contract_path points to a file that no
  longer exists). Those contracts are KEPT (conservative). Check the audit JSON
  to see how many fall into this bucket.
- Range pragmas like `>=0.7.0 <0.9.0` — script uses the MINIMUM version (0.7.0
  here) and conservatively keeps the label rather than assuming 0.8 protection.

Run `--dry-run` first to see removal counts; the audit log lists all removed stems.

EXPECTED REMOVAL: ~1,000–3,000 IntegerUO=1 labels

USAGE
─────
    source ml/.venv/bin/activate
    python -m ml.scripts.clean_integeruo_labels \
        --cache ml/data/cached_dataset_v8.pkl \
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \
        --out ml/data/processed/multilabel_index_sol2.csv \
        --dry-run

OUTPUT
──────
    CSV: updated multilabel_index with IntegerUO labels removed for confirmed 0.8+ contracts
    JSON: audit log with removed stems and "no source path" count
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── Pragma / source parsing ────────────────────────────────────────────────────

def _parse_min_solc_version(sol_path: str) -> tuple[int, int, int]:
    """
    Parse minimum Solidity version from pragma. Returns (major, minor, patch).
    Conservative: for range pragmas, uses the MINIMUM stated version.

    Handles: ^0.8.0, >=0.8.0, =0.8.19, 0.8.0, ~0.8, >=0.7.0 <0.9.0
    """
    try:
        content = Path(sol_path).read_text(errors="ignore")
        match = re.search(r"pragma\s+solidity\s+([^;]+);", content)
        if not match:
            return (0, 0, 0)
        pragma_str = match.group(1).strip()

        versions = re.findall(r"(\d+)\.(\d+)(?:\.(\d+))?", pragma_str)
        if not versions:
            return (0, 0, 0)

        min_ver = None
        for v in versions:
            major, minor = int(v[0]), int(v[1])
            patch = int(v[2]) if v[2] else 0
            ver = (major, minor, patch)
            if min_ver is None or ver < min_ver:
                min_ver = ver
        return min_ver or (0, 0, 0)
    except Exception:
        return (0, 0, 0)


def _has_unchecked_block(sol_path: str) -> bool:
    """Returns True if source contains an unchecked{} block."""
    try:
        content = Path(sol_path).read_text(errors="ignore")
        return bool(re.search(r"\bunchecked\s*\{", content))
    except Exception:
        return True  # conservative: keep label if source unreadable


def _is_overflow_impossible(sol_path: str) -> bool:
    """
    Returns True if IntegerUO is structurally impossible:
    - Solidity >= 0.8.0 AND no unchecked{} blocks.
    """
    major, minor, patch = _parse_min_solc_version(sol_path)
    if major == 0 and minor < 8:
        return False  # could be 0.7.x — overflow possible
    if major == 0 and minor >= 8:
        # overflow protected unless unchecked block opts out
        return not _has_unchecked_block(sol_path)
    return False  # unknown version — keep label


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sol-2: Solidity >=0.8.0 IntegerUO cleaning")
    p.add_argument("--cache", default="ml/data/cached_dataset_v8.pkl",
                   help="Path to cache (for contract_path lookup)")
    p.add_argument("--label-csv", default="ml/data/processed/multilabel_index_cleaned.csv")
    p.add_argument("--out", default="ml/data/processed/multilabel_index_sol2.csv")
    p.add_argument("--audit-json", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-contracts", type=int, default=None,
                   help="Limit to first N IntegerUO+ contracts")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    label_csv  = Path(args.label_csv)
    cache_path = Path(args.cache)
    out_path   = Path(args.out)
    audit_path = Path(args.audit_json) if args.audit_json else out_path.with_suffix(".audit.json")

    if not label_csv.exists():
        log.error(f"Label CSV not found: {label_csv}")
        return 1

    # Load cache for contract_path lookup (optional — fall back gracefully)
    cache = {}
    if cache_path.exists():
        log.info(f"Loading cache: {cache_path}...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        log.info(f"Cache: {len(cache):,} entries")
    else:
        log.warning(f"Cache not found ({cache_path}) — will skip contracts without cached path")

    log.info(f"Loading labels: {label_csv}")
    df = pd.read_csv(label_csv)

    uo_positive = df[df["IntegerUO"] == 1]["md5_stem"].tolist()
    log.info(f"IntegerUO=1 contracts: {len(uo_positive):,}")

    if args.max_contracts:
        uo_positive = uo_positive[:args.max_contracts]

    overflow_impossible: list[str] = []
    overflow_possible:   list[str] = []
    no_path:             list[str] = []

    for i, stem in enumerate(uo_positive):
        if i % 500 == 0:
            log.info(f"  {i}/{len(uo_positive)} checked, {len(overflow_impossible)} to remove...")

        # Get contract path from cache
        contract_path = None
        if stem in cache:
            entry = cache[stem]
            if isinstance(entry, tuple) and len(entry) >= 1:
                graph = entry[0]
                contract_path = getattr(graph, "contract_path", None)

        if contract_path is None or not Path(str(contract_path)).exists():
            no_path.append(stem)
            continue

        if _is_overflow_impossible(str(contract_path)):
            overflow_impossible.append(stem)
        else:
            overflow_possible.append(stem)

    log.info(f"\n=== Sol-2 IntegerUO 0.8.0 results ===")
    log.info(f"IntegerUO+ checked: {len(uo_positive):,}")
    log.info(f"  Overflow possible (keep):      {len(overflow_possible):,}")
    log.info(f"  Overflow impossible (remove):  {len(overflow_impossible):,}")
    log.info(f"  No source path (keep):         {len(no_path):,}")

    audit = {
        "n_integer_uo_positive":      len(uo_positive),
        "n_overflow_possible_kept":   len(overflow_possible),
        "n_overflow_impossible_removed": len(overflow_impossible),
        "n_no_path_kept":             len(no_path),
        "removed_stems":              overflow_impossible,
    }

    if args.dry_run:
        log.info("DRY RUN — no files written")
        return 0

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    log.info(f"Audit → {audit_path}")

    df_out = df.copy()
    mask = df_out["md5_stem"].isin(set(overflow_impossible))
    df_out.loc[mask, "IntegerUO"] = 0
    log.info(f"Updated {mask.sum()} rows: IntegerUO 1→0")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    log.info(f"Updated CSV → {out_path}")

    log.info(f"IntegerUO labels: {int(df['IntegerUO'].sum())} → {int(df_out['IntegerUO'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
