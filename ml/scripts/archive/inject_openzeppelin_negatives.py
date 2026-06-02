"""
inject_openzeppelin_negatives.py — IMP-D2: OpenZeppelin clean negatives

PURPOSE
───────
Downloads OpenZeppelin Contracts and injects a curated subset (~100+ contracts)
as all-zero-label (clean) negatives into the training data. OZ contracts are
formally audited and battle-tested — they provide high-confidence clean signal
that helps the model distinguish vulnerable from safe patterns.

WHY
───
The current training set has ~40% zero-label rows, but these are BCCC contracts
that *might* be vulnerable but weren't labelled that way (absent-positive noise).
OZ contracts are genuinely safe by construction, providing clean anchor negatives.

WHAT THIS DOES
──────────────
1. Clone openzeppelin-contracts to a temp dir (if not already present)
2. Enumerate token/*.sol files (ERC20, ERC721, security, access control)
3. For each .sol file: extract graph + tokenize → write to graphs_dir/tokens_dir
4. Append 0-label rows to the multilabel index CSV
5. Print injection summary

USAGE
─────
    source ml/.venv/bin/activate
    python -m ml.scripts.inject_openzeppelin_negatives \
        --oz-dir /tmp/oz-contracts \
        --graphs-dir ml/data/graphs \
        --tokens-dir ml/data/tokens_windowed \
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \
        --out-csv ml/data/processed/multilabel_index_oz.csv \
        --dry-run

    # Clone fresh (requires network):
    python -m ml.scripts.inject_openzeppelin_negatives \
        --clone \
        --oz-dir /tmp/oz-contracts \
        ...

OUTPUT
──────
    Updated label CSV with OZ contracts as all-zeros labels
    Graph .pt files in graphs_dir
    Token .pt files in tokens_dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# OZ repository URL
_OZ_REPO = "https://github.com/OpenZeppelin/openzeppelin-contracts.git"
_OZ_TAG  = "v5.1.0"

# Curated subset: contracts that are guaranteed clean (no external calls that
# could be exploited, no timestamp-gated value transfers, standard patterns only)
_OZ_WHITELIST_DIRS = [
    "contracts/token/ERC20",
    "contracts/token/ERC721",
    "contracts/access",
    "contracts/security",
    "contracts/utils",
]

# CLASS_NAMES for zero-label rows
_CLASS_NAMES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _md5_stem(sol_path: Path) -> str:
    """Compute MD5 of the file content (matches create_cache.py convention)."""
    content = sol_path.read_bytes()
    return hashlib.md5(content).hexdigest()


def _clone_oz(oz_dir: Path, tag: str) -> None:
    """Clone openzeppelin-contracts at given tag to oz_dir."""
    log.info(f"Cloning OZ {tag} → {oz_dir}...")
    oz_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--depth=1", "--branch", tag,
        _OZ_REPO, str(oz_dir)
    ], check=True)
    log.info("Clone complete")


def _enumerate_oz_sols(oz_dir: Path, whitelist_dirs: list[str]) -> list[Path]:
    """Return .sol files from whitelisted subdirectories."""
    sols: list[Path] = []
    for subdir in whitelist_dirs:
        target = oz_dir / subdir
        if not target.exists():
            log.warning(f"OZ subdir not found: {target}")
            continue
        for sol in sorted(target.glob("*.sol")):
            # Skip interface files (no implementation to extract)
            if sol.name.startswith("I") and sol.name[1].isupper():
                continue
            sols.append(sol)
    log.info(f"Found {len(sols)} OZ .sol files in whitelisted dirs")
    return sols


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMP-D2: Inject OpenZeppelin clean negatives")
    p.add_argument("--oz-dir", default="/tmp/oz-contracts",
                   help="Path to OpenZeppelin contracts checkout")
    p.add_argument("--clone", action="store_true",
                   help="Clone OZ repo before processing (requires network)")
    p.add_argument("--oz-tag", default=_OZ_TAG, help=f"OZ git tag (default: {_OZ_TAG})")
    p.add_argument("--graphs-dir", default="ml/data/graphs",
                   help="Directory to write extracted graph .pt files")
    p.add_argument("--tokens-dir", default="ml/data/tokens_windowed",
                   help="Directory to write tokenized .pt files")
    p.add_argument("--label-csv", default="ml/data/processed/multilabel_index_cleaned.csv")
    p.add_argument("--out-csv", default="ml/data/processed/multilabel_index_oz.csv")
    p.add_argument("--solc-version", default="0.8.20",
                   help="Solidity compiler version for OZ v5 (default: 0.8.20)")
    p.add_argument("--max-contracts", type=int, default=None,
                   help="Limit to first N OZ contracts (for testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="List candidates without extracting")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    oz_dir     = Path(args.oz_dir)
    graphs_dir = Path(args.graphs_dir)
    tokens_dir = Path(args.tokens_dir)
    label_csv  = Path(args.label_csv)
    out_csv    = Path(args.out_csv)

    if args.clone:
        if oz_dir.exists():
            log.info(f"OZ dir already exists: {oz_dir} — skipping clone")
        else:
            _clone_oz(oz_dir, args.oz_tag)

    if not oz_dir.exists():
        log.error(
            f"OZ dir not found: {oz_dir}\n"
            "Run with --clone to download, or point --oz-dir to existing checkout."
        )
        return 1

    sols = _enumerate_oz_sols(oz_dir, _OZ_WHITELIST_DIRS)
    if args.max_contracts:
        sols = sols[:args.max_contracts]

    if args.dry_run:
        log.info("DRY RUN — candidates:")
        for s in sols:
            log.info(f"  {s.relative_to(oz_dir)}")
        return 0

    if not label_csv.exists():
        log.error(f"Label CSV not found: {label_csv}")
        return 1

    import pandas as pd
    import torch

    from ml.src.preprocessing.graph_extractor import extract_contract_graph
    from ml.src.preprocessing.tokenizer import tokenize_contract

    graphs_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(label_csv)
    existing_stems = set(df["md5_stem"].tolist())

    new_rows: list[dict] = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for sol_path in sols:
        stem = _md5_stem(sol_path)
        if stem in existing_stems:
            log.debug(f"  Skip (already in CSV): {sol_path.name}")
            n_skip += 1
            continue

        graph_out = graphs_dir / f"{stem}.pt"
        token_out = tokens_dir / f"{stem}.pt"

        try:
            # Extract graph
            graph = extract_contract_graph(
                str(sol_path),
                solc_version=args.solc_version,
            )
            torch.save(graph, graph_out)

            # Tokenize
            tokens = tokenize_contract(str(sol_path))
            torch.save(tokens, token_out)

            # Zero-label row
            row = {
                "md5_stem": stem,
                "contract_path": str(sol_path),
                "source": "openzeppelin",
            }
            for cls in _CLASS_NAMES:
                row[cls] = 0
            new_rows.append(row)
            n_ok += 1
            log.info(f"  OK: {sol_path.name} → {stem[:8]}")

        except Exception as e:
            log.warning(f"  FAIL: {sol_path.name}: {e}")
            n_fail += 1
            # Clean up partial files
            for p in [graph_out, token_out]:
                if p.exists():
                    p.unlink()

    log.info(f"\n=== IMP-D2 OZ injection results ===")
    log.info(f"  Extracted OK:  {n_ok}")
    log.info(f"  Already in CSV (skipped): {n_skip}")
    log.info(f"  Failed:        {n_fail}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Ensure all class columns exist (fill missing with 0)
        for cls in _CLASS_NAMES:
            if cls not in df_new.columns:
                df_new[cls] = 0
        df_out = pd.concat([df, df_new], ignore_index=True)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        log.info(f"Updated CSV: {len(df)} → {len(df_out)} rows → {out_csv}")
    else:
        log.info("No new contracts injected — CSV unchanged")

    return 0


if __name__ == "__main__":
    sys.exit(main())
