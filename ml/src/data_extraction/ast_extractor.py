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
from pathlib import Path
from typing import Optional, List
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

    venv_path = Path.cwd() / ".venv" / ".solc-select" / "artifacts" / f"solc-{version}"
    candidates = [
        venv_path / f"solc-{version}",
        venv_path / "solc",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def get_project_root() -> Path:
    """Return the repository root (three levels up from this script)."""
    return Path(__file__).resolve().parent.parent.parent


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
        label:         int = 0,
    ) -> Optional[Data]:
        """
        Convert one Solidity file to a PyG Data object and return it.

        Wraps extract_contract_graph() with the offline batch error policy:
          • GraphExtractionError → log (if verbose) and return None (skip).
          • RuntimeError (Slither not installed) → re-raise (abort the process).

        Attaches offline-specific metadata that the shared extractor does not set:
          data.contract_hash  — MD5 of the contract file path (matches tokenizer)
          data.contract_path  — original path string (for provenance tracking)
          data.y              — vulnerability label tensor [label]

        Args:
            contract_path: Path to .sol file.
            solc_binary:   Pinned solc binary path, or None for system solc.
            solc_version:  Version string for --allow-paths gating, e.g. "0.8.19".
            label:         Integer vulnerability label (0 = safe, 1 = vulnerable).

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
            if self.verbose:
                print(f"  Skipped {Path(contract_path).name}: {exc}")
            return None
        except Exception as exc:
            # Unexpected error — log and skip rather than crash the pool worker.
            if self.verbose:
                print(f"  Unexpected error for {Path(contract_path).name}: {exc}")
            return None

        # Attach offline-specific metadata
        contract_hash = get_contract_hash(contract_path)
        graph.contract_hash = contract_hash
        graph.contract_path = str(contract_path)
        graph.y = torch.tensor([label], dtype=torch.long)

        return graph

    # ── Batch extraction with checkpoint/resume ────────────────────────────

    def extract_batch_with_checkpoint(
        self,
        df:               pd.DataFrame,
        n_workers:        int  = 11,
        chunksize:        int  = 50,
        output_dir:       Path = Path("ml/data/graphs"),
        checkpoint_every: int  = 500,
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

        Returns:
            List of Data objects extracted in this run (not counting resumed ones).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = output_dir / "checkpoint.json"

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

            worker = partial(
                self.contract_to_pyg,
                solc_binary=solc_bin,
                solc_version=version,
                label=0,
            )

            with mp.Pool(processes=n_workers) as pool:
                results = []
                for result in tqdm(
                    pool.imap(worker, group["contract_path"].tolist(), chunksize=chunksize),
                    total=len(group),
                    desc=f"  v{version}",
                    leave=False,
                ):
                    if result is None:
                        continue

                    results.append(result)
                    filename   = get_filename_from_hash(result.contract_hash)
                    graph_file = output_dir / filename
                    torch.save(result, graph_file)

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
                        if self.verbose:
                            print(f"💾 Checkpoint: {total_processed:,} contracts")

                all_data.extend(results)

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
    args = parser.parse_args()

    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive")

    print("=" * 70)
    print("🚀 AST Extractor V4.3 — Offline Batch Pipeline (Thin Wrapper)")
    print("=" * 70)
    print(f"📅 Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Output:  {args.output}")
    print(f"⚙️  Workers: {args.workers}")
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

    graphs = extractor.extract_batch_with_checkpoint(
        df,
        n_workers=args.workers,
        output_dir=Path(args.output),
        checkpoint_every=args.checkpoint_every,
    )

    print()
    print("=" * 70)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Graphs created: {len(graphs):,}")
    print(f"Output directory: {args.output}")
    print("=" * 70)
