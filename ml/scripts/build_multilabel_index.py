"""
build_multilabel_index.py — Build multilabel_index.csv for Track 3 retraining.

Maps every graph .pt file to an 11-dim multi-hot vulnerability label vector,
sourced from the authoritative BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv.

Two hash systems are in play — never mix them:
  BCCC SHA256:  hash of file contents → .sol filename in BCCC SourceCodes/
  Internal MD5: hash of file path     → .pt filename in ml/data/graphs/
Bridge: graph.contract_path inside each .pt → Path(...).stem = SHA256

Label construction:
  The BCCC CSV has 111,897 rows — one per folder-file occurrence, NOT per
  unique SHA256. A contract in 3 vuln folders → 3 rows, each with a different
  Class=1. To get a multi-hot per contract: GROUP BY SHA256, max() the Class
  columns (max of 0/1 == OR).

Output: ml/data/processed/multilabel_index.csv
  Columns: md5_stem, CallToUnknown, DenialOfService, ExternalBug, GasException,
           IntegerUO, MishandledException, Reentrancy, Timestamp,
           TransactionOrderDependence, UnusedReturn, WeakAccessMod
  Rows: 68,555  (one per .pt file in ml/data/graphs/)

Usage:
    cd ~/projects/sentinel
    poetry run python ml/scripts/build_multilabel_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.serialization
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# ── Path setup ────────────────────────────────────────────────────────────────
# __file__ is ml/scripts/build_multilabel_index.py → parents[2] = project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

logger.remove()
logger.add(sys.stderr, level="INFO")

# Allow PyG classes for weights_only=True loads — same allowlist as dataset.py
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

# ── Class name constants (alphabetical, two classes excluded) ─────────────────
# This order MUST match the training vector index 0-9.
# Any change here requires rebuilding the CSV and retraining from scratch.
#
# EXCLUDED — Class12:NonVulnerable: not a vulnerability type; absence of all
#   11 classes already encodes "safe". Including it would add a redundant output.
#
# EXCLUDED — Class07:WeakAccessMod: 1,918 .sol files exist in BCCC SourceCodes
#   but ZERO were extracted into ml/data/graphs/ .pt files during the original
#   graph extraction pass. Including a class with zero training examples would
#   produce undefined gradients and a permanently near-zero output node.
#   Decision (2026-04-17): drop WeakAccessMod from output vector.
#   If WeakAccessMod .pt files are extracted in a future run, add it back at
#   index 9 (appended so existing trained indices 0-8 stay valid), rebuild CSV,
#   retrain. Until then: 10-class output.
CLASS_NAMES = [
    "CallToUnknown",               # index 0
    "DenialOfService",             # index 1
    "ExternalBug",                 # index 2
    "GasException",                # index 3
    "IntegerUO",                   # index 4
    "MishandledException",         # index 5
    "Reentrancy",                  # index 6
    "Timestamp",                   # index 7
    "TransactionOrderDependence",  # index 8
    "UnusedReturn",                # index 9
]

# BCCC CSV column names in the same order as CLASS_NAMES above.
# Class12:NonVulnerable and Class07:WeakAccessMod intentionally excluded — see above.
CLASS_COLS = [
    "Class08:CallToUnknown",
    "Class09:DenialOfService",
    "Class01:ExternalBug",
    "Class02:GasException",
    "Class10:IntegerUO",
    "Class03:MishandledException",
    "Class11:Reentrancy",
    "Class04:Timestamp",
    "Class05:TransactionOrderDependence",
    "Class06:UnusedReturn",
]

# Paths — all relative to project root so this works from any cwd
_BCCC_CSV   = _PROJECT_ROOT / "BCCC-SCsVul-2024" / "BCCC-SCsVul-2024.csv"
_GRAPHS_DIR = _PROJECT_ROOT / "ml" / "data" / "graphs"
_OUTPUT_CSV = _PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index.csv"


def build_sha256_to_multihot() -> dict[str, list[int]]:
    """
    Read BCCC CSV, group by SHA256, OR the 11 class columns.

    Returns: {sha256_hex_string: [0/1, 0/1, ..., 0/1]} (length-11 list)
    """
    logger.info("Loading BCCC CSV: {}", _BCCC_CSV)
    df = pd.read_csv(_BCCC_CSV, low_memory=False)
    logger.info("BCCC CSV loaded — {} rows, {} columns", len(df), len(df.columns))

    sha256_col = df.columns[1]   # second column is always the SHA256 hash
    logger.info("SHA256 column: '{}'", sha256_col)

    # Verify all expected class columns are present
    missing = [c for c in CLASS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing class columns in BCCC CSV: {missing}")

    # GROUP BY SHA256, take max per class column (max(0,1) == OR for binary cols).
    # This collapses 111k rows into ~68k unique SHA256 entries, each with
    # a correct multi-hot label that captures ALL vulnerability types the
    # contract appears under in BCCC.
    multi_hot_df = (
        df.groupby(sha256_col)[CLASS_COLS]
        .max()
        .reset_index()
        .rename(columns={sha256_col: "sha256"})
    )
    logger.info(
        "Grouped to {} unique SHA256 entries (was {} rows)",
        len(multi_hot_df), len(df)
    )

    # Convert to {sha256: [int, ...]} dict for O(1) lookup per .pt file
    lookup: dict[str, list[int]] = {}
    for _, row in multi_hot_df.iterrows():
        sha256 = row["sha256"]
        vector = [int(row[col]) for col in CLASS_COLS]
        lookup[sha256] = vector

    return lookup


def build_index() -> None:
    sha256_lookup = build_sha256_to_multihot()
    logger.info("SHA256 lookup built — {} unique contracts", len(sha256_lookup))

    pt_files = sorted(_GRAPHS_DIR.glob("*.pt"))
    logger.info("Processing {} graph .pt files from {}", len(pt_files), _GRAPHS_DIR)

    rows: list[list] = []
    unknown_count = 0
    load_errors = 0

    for i, pt_file in enumerate(pt_files):
        if i % 5000 == 0:
            logger.info("  Progress: {}/{}", i, len(pt_files))

        md5_stem = pt_file.stem

        try:
            graph = torch.load(pt_file, weights_only=True)
        except Exception as exc:
            logger.error("Failed to load {} — skipping: {}", pt_file.name, exc)
            load_errors += 1
            continue

        # Extract SHA256 from the BCCC .sol filename embedded in contract_path.
        # Example: "BCCC-SCsVul-2024/SourceCodes/Reentrancy/abc123...sol"
        # → stem = "abc123..."  (64-char SHA256)
        contract_path = getattr(graph, "contract_path", "")
        sha256 = Path(contract_path).stem if contract_path else ""

        if sha256 and sha256 in sha256_lookup:
            label_vector = sha256_lookup[sha256]
        else:
            # SHA256 not found in BCCC — very rare (non-BCCC source or extraction
            # artifact). Log a WARNING and use all-zeros (treated as safe for
            # training purposes — these contracts contribute neutral gradient).
            binary_y = int(graph.y.item()) if hasattr(graph, "y") and graph.y is not None else 0
            logger.warning(
                "SHA256 not in BCCC | md5={} | sha256={} | graph.y={}",
                md5_stem, sha256, binary_y
            )
            label_vector = [0] * 10
            unknown_count += 1

        rows.append([md5_stem] + label_vector)

    # Write CSV
    df_out = pd.DataFrame(rows, columns=["md5_stem"] + CLASS_NAMES)
    _OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(_OUTPUT_CSV, index=False)

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"multilabel_index.csv written: {_OUTPUT_CSV}")
    print(f"{'='*60}")
    print(f"Total rows:                  {len(df_out)}")
    print(f"Multi-label rows (sum > 1):  {(df_out[CLASS_NAMES].sum(axis=1) > 1).sum()}")
    print(f"Safe rows (all zeros):        {(df_out[CLASS_NAMES].sum(axis=1) == 0).sum()}")
    print(f"Unknown (not in BCCC):        {unknown_count}")
    print(f"Load errors (skipped):        {load_errors}")
    print(f"\nPer-class positive count (needed for pos_weight in trainer):")
    for name in CLASS_NAMES:
        pos = int(df_out[name].sum())
        neg = len(df_out) - pos
        pw  = neg / max(pos, 1)
        print(f"  {name:<30}  pos={pos:>6}  neg={neg:>6}  pos_weight={pw:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    build_index()
