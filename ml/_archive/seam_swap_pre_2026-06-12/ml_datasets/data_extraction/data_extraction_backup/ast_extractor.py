#!/usr/bin/env python
"""
AST Extractor V4.3 — Offline Batch Pipeline (Thin Wrapper)

WHAT THIS FILE DOES
───────────────────
Drives the offline dataset-build pipeline: reads the contracts metadata
parquet, resolves the correct solc binary for each version group, spawns
11 worker processes, extracts PyG graphs, and writes <md5_hash>.pt files
to ml/data/graphs/.

This file is the orchestration layer only. Graph construction logic has been
extracted to:

  ml/src/preprocessing/graph_schema.py    — NODE_TYPES, VISIBILITY_MAP, EDGE_TYPES,
                                             FEATURE_NAMES, FEATURE_SCHEMA_VERSION
  ml/src/preprocessing/graph_extractor.py — extract_contract_graph(), typed exceptions,
                                             GraphExtractionConfig

WHY THE EXTRACTION LOGIC WAS MOVED
───────────────────────────────────
Before the refactor, ml/src/inference/preprocess.py contained an identical
copy of node_features() and the edge-building loop. Any change to feature
engineering had to be applied in two places; a missed sync caused silent
inference accuracy regression (model receives different features than it was
trained on, with no error message).

Now both pipelines import from graph_extractor.py, making divergence impossible.

OFFLINE-ONLY RESPONSIBILITIES KEPT HERE
────────────────────────────────────────
  • parse_solc_version() / solc_supports_allow_paths() / get_solc_binary()
      — Version-pinned solc binary resolution for each version group.
        Online inference uses the system solc (no version pinning needed).
  • ASTExtractorV4.extract_batch_with_checkpoint()
      — multiprocessing Pool, checkpoint JSON, progress bars.
        These are batch concerns not relevant to single-request inference.
  • contract_to_pyg()
      — Thin wrapper: builds GraphExtractionConfig, calls extract_contract_graph(),
        attaches offline-specific metadata (contract_hash, contract_path, y),
        returns None on any GraphExtractionError (skip-and-log batch policy).

SCHEMA COMPATIBILITY
────────────────────
Existing .pt files in ml/data/graphs/ were written by pre-refactor versions
of this script. They store edge_attr with shape [E, 1]. New .pt files written
by this version store edge_attr with shape [E] (PyG 1-D convention, aligned
with graph_extractor.py). GNNEncoder ignores edge_attr, so both shapes are
safe to load in the current training/inference pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
from functools import partial
import multiprocessing as mp
import warnings
import json
from datetime import datetime
import re

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from tqdm import tqdm

# ── Path bootstrap ─────────────────────────────────────────────────────────
# ast_extractor.py lives in ml/src/data_extraction/; add ml/ to sys.path so that
# the src package (ml/src/) is importable without installing the project.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.hash_utils import get_contract_hash, get_filename_from_hash
from src.preprocessing.graph_extractor import (
    GraphExtractionConfig,
    GraphExtractionError,
    extract_contract_graph,
)
from src.preprocessing.graph_schema import NODE_TYPES, VISIBILITY_MAP, EDGE_TYPES  # re-exported for any external callers

logger = logging.getLogger(__name__)

try:
    from slither import Slither
    from slither.core.declarations import Contract, Function, Modifier, Event
except ImportError as e:
    raise ImportError("Slither not installed. Run: pip install slither-analyzer") from e

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("torch-geometric not installed. Run: pip install torch-geometric")


# ─────────────────────────────────────────────────────────────────────────────
# Solc version utilities (offline-only — not needed for online inference)
# ─────────────────────────────────────────────────────────────────────────────

def parse_solc_version(version: str) -> tuple:
    """
    Parse a solc version string into a (major, minor, patch) tuple.

    Examples:
        '0.4.26' → (0, 4, 26)
        '0.8.19' → (0, 8, 19)

    Returns (0, 0, 0) on parse failure rather than raising, so callers can
    safely compare tuples without guarding against exceptions.
    """
    try:
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
        if match:
            return tuple(int(x) for x in match.groups())
    except Exception:
        pass
    return (0, 0, 0)


def solc_supports_allow_paths(version: str) -> bool:
    """
    Return True if this solc version supports the --allow-paths flag.

    --allow-paths was introduced in solc 0.5.0. Passing it to an older solc
    binary causes a startup error that aborts the entire extraction process.
    """
    major, minor, _ = parse_solc_version(version)
    return (major, minor) >= (0, 5)


def get_project_root() -> Path:
    """Return the repository root (three levels up from this script)."""
    return Path(__file__).resolve().parent.parent.parent


def get_solc_binary(version: str) -> Optional[str]:
    """
    Resolve the pinned solc binary path inside the Poetry virtualenv.

    solc-select installs version-specific binaries under
    .venv/.solc-select/artifacts/solc-{version}/. This ensures each contract
    group is compiled by a solc binary matching its pragma version, producing
    ASTs that match what the pragma author intended.

    Returns None if the binary is not found; the caller must decide whether to
    skip those contracts or fall back to the system solc.
    """
    if not version:
        return None

    # A19: use get_project_root() for deterministic resolution regardless of
    # the working directory at launch time.  Path.cwd() silently produced the
    # wrong path when the script was invoked from any directory other than the
    # repo root.
    venv_path = get_project_root() / ".venv" / ".solc-select" / "artifacts" / f"solc-{version}"
    candidates = [
        venv_path / f"solc-{version}",
        venv_path / "solc",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Module-level multiprocessing worker
# ─────────────────────────────────────────────────────────────────────────────

# The 10 vulnerability class columns in multilabel_index_cleaned.csv, in order.
_LABEL_COLUMNS: List[str] = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]


def _labeled_pool_worker(
    path_label_pair: tuple,
    extractor: "ASTExtractorV4",
    solc_binary: Optional[str],
    solc_version: str,
) -> Optional["Data"]:
    """Picklable multiprocessing worker.

    Accepts a (contract_path, label) tuple from pool.imap and delegates to
    ASTExtractorV4.contract_to_pyg.  Defined at module level (not as a lambda
    or nested function) so Python's pickle can serialise it for subprocess IPC
    on non-fork platforms.
    """
    contract_path, label = path_label_pair
    return extractor.contract_to_pyg(
        contract_path,
        solc_binary=solc_binary,
        solc_version=solc_version,
        label=label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor class
# ─────────────────────────────────────────────────────────────────────────────

class ASTExtractorV4:
    """
    Offline batch graph extractor with checkpoint/resume and multiprocessing.

    Instantiate once per run; reuse for multiple version groups.

    Public API:
      contract_to_pyg(path, solc_binary, solc_version, label) → Data | None
      extract_batch_with_checkpoint(df, ...)                   → list[Data]
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose     = verbose
        self.project_root = get_project_root()

    # ── Single-contract extraction ─────────────────────────────────────────

    def contract_to_pyg(
        self,
        contract_path: str,
        solc_binary:   Optional[str] = None,
        solc_version:  Optional[str] = None,
        label:         Union[List[int], int, None] = None,
    ) -> Optional[Data]:
        """
        Convert one Solidity file to a PyG Data object and return it.

        Wraps extract_contract_graph() with the offline batch error policy:
          • GraphExtractionError → log (if verbose) and return None (skip).
          • RuntimeError (Slither not installed) → re-raise (abort the process).

        Attaches offline-specific metadata that the shared extractor does not set:
          data.contract_hash  — MD5 of the contract file path (matches tokenizer)
          data.contract_path  — original path string (for provenance tracking)
          data.y              — vulnerability label tensor

        Label storage:
          • List/tuple of ints (multi-label, e.g. from label_csv):
              stored as float32 tensor of shape [NUM_CLASSES].
          • Single int (legacy binary mode, 0 or 1):
              stored as long tensor of shape [1].
          • None (no label provided):
              stored as long zeros tensor of shape [1] (backward-compat default).

        Args:
            contract_path: Path to .sol file.
            solc_binary:   Pinned solc binary path, or None for system solc.
            solc_version:  Version string for --allow-paths gating, e.g. "0.8.19".
            label:         Multi-hot label vector [NUM_CLASSES] from label_map,
                           or a single int for legacy binary mode, or None.

        Returns:
            PyG Data on success, None on any graph extraction failure.
        """
        config = GraphExtractionConfig(
            solc_binary=solc_binary,
            solc_version=solc_version,
            # Pass project_root as allow_paths; graph_extractor checks version
            # before injecting --allow-paths into solc_args.
            allow_paths=str(self.project_root) if solc_version else None,
        )

        try:
            graph = extract_contract_graph(Path(contract_path), config)
        except RuntimeError:
            # Slither not installed — fatal; surface immediately.
            raise
        except GraphExtractionError as exc:
            # A21: use logger instead of print() — workers run under QueueHandler/
            # QueueListener so concurrent print() calls interleave on stdout.
            logger.warning("Skipped %s: %s", Path(contract_path).name, exc)
            return None
        except Exception as exc:
            # Unexpected error — log and skip rather than crash the pool worker.
            # A21: logger.warning is concurrency-safe; print() is not.
            logger.warning(
                "Unexpected error for %s: %s",
                Path(contract_path).name, exc, exc_info=True,
            )
            return None

        # Attach offline-specific metadata.
        contract_hash = get_contract_hash(contract_path)
        graph.contract_hash = contract_hash
        graph.contract_path = str(contract_path)

        # A20: store ground-truth label supplied from label_map (not hardcoded 0).
        # Multi-hot list  → float32 [NUM_CLASSES] (used by DualPathDataset multi-label mode).
        # Single int      → long   [1]            (legacy binary mode fallback).
        # None            → long   [1] = 0        (backward-compat; warns if verbose).
        if isinstance(label, (list, tuple)):
            graph.y = torch.tensor(label, dtype=torch.float32)
        elif isinstance(label, int):
            graph.y = torch.tensor([label], dtype=torch.long)
        else:
            # A21: logger is concurrency-safe; print() is not.
            logger.warning("No label for %s — storing zero.", Path(contract_path).name)
            graph.y = torch.zeros(1, dtype=torch.long)

        return graph

    # ── Batch extraction with checkpoint/resume ────────────────────────────

    def extract_batch_with_checkpoint(
        self,
        df:               pd.DataFrame,
        n_workers:        int  = 11,
        chunksize:        int  = 50,
        output_dir:       Path = Path("ml/data/graphs"),
        checkpoint_every: int  = 500,
        label_csv:        Optional[Path] = None,
    ) -> List[Data]:
        """
        Extract graphs for every contract in df, with parallel workers and resume.

        Checkpoint JSON is written every `checkpoint_every` contracts so the
        run can be interrupted (Ctrl-C) and resumed with --resume without
        re-processing already-saved graphs.

        Args:
            df:               DataFrame with columns: contract_path, detected_version.
            n_workers:        Pool size (default 11; set to 1 for debugging).
            chunksize:        imap chunksize (batching reduces IPC overhead).
            output_dir:       Directory where <md5>.pt files are written.
            checkpoint_every: Checkpoint frequency in contracts processed.
            label_csv:        Path to multilabel_index_cleaned.csv.  Each contract's
                              md5 hash is looked up in this CSV and the 10-dim binary
                              label vector is stored in graph.y.  If None, graph.y
                              defaults to zeros (backward-compatible binary mode).

        Returns:
            List of Data objects extracted in this run (not counting resumed ones).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = output_dir / "checkpoint.json"

        # ── A20: Build label_map from ground-truth CSV before any pool work ──
        # Maps md5_stem → [10 binary ints] for every known contract.
        # This must be loaded BEFORE pool.imap so all workers receive correct
        # labels — the old code hardcoded label=0 for every contract.
        label_map: Dict[str, List[int]] = {}
        if label_csv is not None:
            label_csv = Path(label_csv)
            print(f"📋 Loading label map from {label_csv} ...")
            _ldf = pd.read_csv(label_csv)
            for _, _row in _ldf.iterrows():
                _stem = str(_row["md5_stem"])
                label_map[_stem] = [int(_row[col]) for col in _LABEL_COLUMNS]
            print(f"   {len(label_map):,} label entries loaded.")
        else:
            print("⚠️  No label_csv provided — graph.y will default to zeros (binary mode).")

        # ── Load checkpoint ────────────────────────────────────────────────
        processed_hashes: set = set()
        if checkpoint_file.exists():
            print("📂 Loading checkpoint...")
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                processed_hashes = set(checkpoint.get("processed", []))
            print(f"   Found {len(processed_hashes):,} already processed contracts")

        # ── Filter already-processed contracts ────────────────────────────
        if processed_hashes:
            df = df.copy()
            df["_temp_hash"] = df["contract_path"].apply(get_contract_hash)
            df = df[~df["_temp_hash"].isin(processed_hashes)].drop(columns=["_temp_hash"])
            print(f"   Remaining: {len(df):,} contracts")

        # ── Resolve pinned solc binaries ───────────────────────────────────
        if "solc_binary" not in df.columns:
            print("🔧 Resolving solc binaries...")
            df = df.copy()
            df["solc_binary"] = df["detected_version"].apply(get_solc_binary)
            found = df["solc_binary"].notna().sum()
            print(f"   Binary found for {found:,}/{len(df):,} contracts")

        df = df[df["solc_binary"].notna()].copy()

        # ── Process per version group ──────────────────────────────────────
        groups = df.groupby("detected_version")
        all_data: List[Data] = []
        total_processed = len(processed_hashes)

        print(f"\n🚀 Processing {len(groups)} version groups...\n")

        for version, group in tqdm(groups, desc="Version groups"):
            solc_bin = group.iloc[0]["solc_binary"]

            # A20: Build (contract_path, label) pairs for this version group.
            # Look up each contract's md5 hash in label_map; None means unknown.
            contract_paths = group["contract_path"].tolist()
            path_label_pairs = [
                (path, label_map.get(get_contract_hash(path)))
                for path in contract_paths
            ]

            # Gate 0.1: assert every contract in this batch has a label entry.
            # If label_csv was not provided, label_map is empty and all labels
            # are None — this is accepted (backward-compat binary mode).
            if label_map:
                missing = [p for p, lbl in path_label_pairs if lbl is None]
                assert len(missing) == 0, (
                    f"[A20] label_map is missing {len(missing)} contract(s) "
                    f"in version group {version}. "
                    f"First missing: {missing[0] if missing else '—'}. "
                    "Ensure label_csv covers all contracts before extraction."
                )

            worker = partial(
                _labeled_pool_worker,
                extractor=self,
                solc_binary=solc_bin,
                solc_version=version,
            )

            # A22: track torch.save() failures per version group.
            failed_saves: List[str] = []

            with mp.Pool(processes=n_workers) as pool:
                results = []
                for result in tqdm(
                    pool.imap(worker, path_label_pairs, chunksize=chunksize),
                    total=len(group),
                    desc=f"  v{version}",
                    leave=False,
                ):
                    if result is None:
                        continue

                    results.append(result)
                    filename   = get_filename_from_hash(result.contract_hash)
                    graph_file = output_dir / filename

                    # A22: guard torch.save() so a single disk/permission error
                    # doesn't crash the entire batch; failed paths are collected
                    # and re-raised as a group after all contracts are processed.
                    try:
                        torch.save(result, graph_file)
                    except (OSError, IOError) as save_err:
                        logger.error(
                            "torch.save failed for %s: %s", graph_file, save_err
                        )
                        failed_saves.append(str(graph_file))
                        continue  # don't mark as processed — it wasn't written

                    processed_hashes.add(result.contract_hash)
                    total_processed += 1

                    if total_processed % checkpoint_every == 0:
                        with open(checkpoint_file, "w") as f:
                            json.dump(
                                {
                                    "processed":  list(processed_hashes),
                                    "total":      total_processed,
                                    "timestamp":  datetime.now().isoformat(),
                                    "completed":  False,
                                },
                                f,
                                indent=2,
                            )
                        logger.info("Checkpoint: %d contracts processed.", total_processed)

                all_data.extend(results)

            # A22: after the batch, surface all save failures as a single exception.
            if failed_saves:
                raise RuntimeError(
                    f"torch.save() failed for {len(failed_saves)} graph(s) "
                    f"in version group {version}:\n"
                    + "\n".join(f"  {p}" for p in failed_saves)
                )

        # ── Final checkpoint ───────────────────────────────────────────────
        with open(checkpoint_file, "w") as f:
            json.dump(
                {
                    "processed": list(processed_hashes),
                    "total":     total_processed,
                    "timestamp": datetime.now().isoformat(),
                    "completed": True,
                },
                f,
                indent=2,
            )

        print(f"\n✅ Successfully processed {len(all_data):,} NEW graphs")
        print(f"📁 Total graphs now: {total_processed:,}")
        return all_data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AST Extractor V4.3 — Offline Batch Pipeline")
    parser.add_argument("--input",            default="ml/data/processed/contracts_metadata.parquet")
    parser.add_argument("--output",           default="ml/data/graphs")
    parser.add_argument("--workers",   type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--test",      action="store_true", help="Process first 100 contracts only")
    parser.add_argument("--resume",    action="store_true", help="Skip already-processed contracts")
    parser.add_argument("--force",     action="store_true", help="Delete checkpoint and reprocess all contracts (full re-extraction with new schema)")
    parser.add_argument("--verbose",   action="store_true")
    # A20: ground-truth label CSV; required for correct graph.y in re-extraction.
    parser.add_argument(
        "--label-csv",
        default="ml/data/processed/multilabel_index_cleaned.csv",
        help="Path to multilabel_index_cleaned.csv supplying per-contract labels "
             "(default: ml/data/processed/multilabel_index_cleaned.csv). "
             "Pass --label-csv '' to disable (backward-compat binary mode).",
    )
    args = parser.parse_args()

    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive")

    print("=" * 70)
    print("🚀 AST Extractor V4.3 — Offline Batch Pipeline (Thin Wrapper)")
    print("=" * 70)
    print(f"📅 Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Output:    {args.output}")
    print(f"⚙️  Workers:  {args.workers}")
    print(f"📋 Label CSV: {args.label_csv or '(none — binary mode)'}")
    print("=" * 70)
    print()

    if args.force:
        checkpoint_file = Path(args.output) / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("🗑️  FORCE MODE: deleted checkpoint — all contracts will be reprocessed")
        else:
            print("🗑️  FORCE MODE: no checkpoint found — starting fresh")
        print()

    df = pd.read_parquet(args.input)
    df = df[df["success"] == True].copy()
    print(f"✅ Loaded {len(df):,} successful contracts")

    if args.test:
        df = df.head(100)
        print(f"🧪 TEST MODE: {len(df)} contracts")

    if args.resume:
        print("🔄 RESUME MODE: skipping already processed contracts")

    print()

    extractor = ASTExtractorV4(verbose=args.verbose)
    print(f"💾 Checkpoints every {args.checkpoint_every} contracts  (Ctrl-C to stop safely)")
    print()

    _label_csv = Path(args.label_csv) if args.label_csv else None
    graphs = extractor.extract_batch_with_checkpoint(
        df,
        n_workers=args.workers,
        output_dir=Path(args.output),
        checkpoint_every=args.checkpoint_every,
        label_csv=_label_csv,
    )

    print()
    print("=" * 70)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Graphs created: {len(graphs):,}")
    print(f"Output directory: {args.output}")
    print("=" * 70)
