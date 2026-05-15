"""
create_cache.py — Build the DualPathDataset RAM cache.

Loads every paired (graph, tokens) sample into a single pickle file so
DualPathDataset.__getitem__ reads from RAM instead of individual .pt files.
On the RTX-3070 workstation this cuts per-epoch I/O time by ~30%.

Usage
─────
    cd /home/motafeq/projects/sentinel
    source ml/.venv/bin/activate
    PYTHONPATH=/home/motafeq/projects/sentinel python ml/scripts/create_cache.py

    Optional flags:
      --graphs-dir   ml/data/graphs              (default)
      --tokens-dir   ml/data/tokens              (default)
      --label-csv    ml/data/processed/multilabel_index_deduped.csv  (default)
      --output       ml/data/cached_dataset_deduped.pkl  (default)
      --workers      8   parallel loaders        (default: 8)

Output
──────
ml/data/cached_dataset_deduped.pkl   dict  md5_stem → (graph, tokens)
    Only stems present in multilabel_index_deduped.csv AND both graph/token dirs are cached.
    Orphan .pt files (no matching pair, not in label index) are skipped.

Re-run after any re-extraction (ast_extractor.py --force).
The trainer detects a stale cache automatically via a spot-check at dataset init.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.serialization
from tqdm import tqdm

# ── safe globals — same list as DualPathDataset ─────────────────────────────
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION

torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _load_pair(stem: str, graphs_dir: Path, tokens_dir: Path):
    """Load one (graph, tokens) pair. Returns (stem, graph, tokens) or raises."""
    graph  = torch.load(graphs_dir / f"{stem}.pt", weights_only=True)
    tokens = torch.load(tokens_dir / f"{stem}.pt", weights_only=True)
    return stem, graph, tokens


def build_cache(
    graphs_dir: Path,
    tokens_dir:  Path,
    label_csv:   Path,
    output:      Path,
    workers:     int,
) -> None:
    # ── 1. Build the valid stem set from multilabel_index.csv ────────────────
    log.info(f"Loading label index from {label_csv}")
    label_stems: set[str] = set()
    with open(label_csv, newline="") as f:
        for row in csv.DictReader(f):
            label_stems.add(row["md5_stem"])
    log.info(f"  {len(label_stems):,} stems in label index")

    # ── 2. Find paired stems (graph ∩ tokens ∩ label_index) ─────────────────
    graph_stems = {p.stem for p in graphs_dir.glob("*.pt")}
    token_stems = {p.stem for p in tokens_dir.glob("*.pt")}
    paired      = sorted(label_stems & graph_stems & token_stems)

    skipped_no_token = len(label_stems & graph_stems) - len(paired)
    skipped_no_graph = len(label_stems & token_stems) - len(paired)
    log.info(f"  Paired stems: {len(paired):,}")
    if skipped_no_token:
        log.warning(f"  Skipped {skipped_no_token} stems: graph present, token missing")
    if skipped_no_graph:
        log.warning(f"  Skipped {skipped_no_graph} stems: token present, graph missing")

    # ── 3. Estimate memory ───────────────────────────────────────────────────
    #  Rough: ~1.4 GB for 68K pairs (graph tensors 0.14 GB + token tensors 0.56 GB
    #  + pickle overhead). Check we have headroom.
    import shutil
    free_gb = shutil.disk_usage(output.parent).free / 1e9
    log.info(f"  Disk free at {output.parent}: {free_gb:.1f} GB (need ~2 GB)")

    # ── 4. Load in parallel ──────────────────────────────────────────────────
    log.info(f"Loading {len(paired):,} pairs with {workers} workers …")
    cached: dict = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_load_pair, stem, graphs_dir, tokens_dir): stem
            for stem in paired
        }
        with tqdm(total=len(paired), unit="pair", desc="Caching") as bar:
            for fut in as_completed(futures):
                stem = futures[fut]
                try:
                    _, graph, tokens = fut.result()
                    cached[stem] = (graph, tokens)
                except Exception as exc:
                    log.warning(f"  Failed to load {stem}: {exc}")
                    errors += 1
                bar.update(1)

    log.info(f"Loaded {len(cached):,} pairs  ({errors} errors skipped)")

    # Fix D2 (H15): embed schema version so DualPathDataset can detect stale caches.
    # The sentinel key "__schema_version__" is checked on load; any mismatch raises
    # RuntimeError rather than silently feeding the model mismatched features.
    cached["__schema_version__"] = FEATURE_SCHEMA_VERSION
    log.info(f"  Schema version embedded: FEATURE_SCHEMA_VERSION={FEATURE_SCHEMA_VERSION!r}")

    # ── 5. Write pickle ──────────────────────────────────────────────────────
    log.info(f"Writing cache to {output} …")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = output.stat().st_size / 1e9
    log.info(f"✅ Cache written: {len(cached):,} pairs, {size_gb:.2f} GB")
    log.info(f"   Pass --cache-path {output} to train.py (already the default)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build DualPathDataset RAM cache")
    p.add_argument("--graphs-dir", default="ml/data/graphs")
    p.add_argument("--tokens-dir", default="ml/data/tokens")
    p.add_argument("--label-csv",  default="ml/data/processed/multilabel_index_deduped.csv")
    p.add_argument("--output",     default="ml/data/cached_dataset_deduped.pkl")
    p.add_argument("--workers",    type=int, default=8)
    args = p.parse_args()

    build_cache(
        graphs_dir = Path(args.graphs_dir),
        tokens_dir  = Path(args.tokens_dir),
        label_csv   = Path(args.label_csv),
        output      = Path(args.output),
        workers     = args.workers,
    )


if __name__ == "__main__":
    main()
