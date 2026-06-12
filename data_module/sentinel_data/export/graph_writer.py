"""Stage 7A — graph_writer: writes sharded PyG graph .pt files.

Per format_schema/v1.yaml: each shard is a torch_geometric.data.Batch
of `shard_size` contracts, written as `graphs-{shard:05d}.pt`.

The contract order is driven by the split JSONL (train → val → test,
in JSONL line order). Contracts with no .pt representation are skipped
with a warning and are NOT included in the shard — the parquet tables
have all 22,356 rows, but the shards only cover contracts that have reps.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.serialization
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

logger = logging.getLogger(__name__)

# Must register PyG types before weights_only=True can deserialize them.
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

# Attributes that Batch.from_data_list() can't merge (str / non-tensor metadata).
_EXCLUDE = ("sha256", "source", "pragma", "schema_version")


def _load_split_jsonl(splits_dir: Path) -> list[tuple[str, str, str]]:
    """Return ordered [(sha256, source, split)] from all 3 JSONL files."""
    rows: list[tuple[str, str, str]] = []
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Split JSONL not found: {path}")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            rows.append((c["sha256"], c.get("source", "unknown"), split_name))
    return rows


def write_graphs_shards(
    rep_root: Path,
    splits_dir: Path,
    output_dir: Path,
    shard_size: int = 5000,
) -> tuple[list[Path], dict[str, int]]:
    """Write sharded PyG graph .pt files.

    Returns:
        (shard_paths, shard_index) — shard_index maps sha256 → shard number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = _load_split_jsonl(splits_dir)

    shard_index: dict[str, int] = {}
    shard_paths: list[Path] = []

    current_graphs: list[Data] = []
    current_ids: list[str] = []
    shard_num = 0
    skipped = 0

    def _flush() -> None:
        nonlocal shard_num
        if not current_graphs:
            return
        batch = Batch.from_data_list(current_graphs, exclude_keys=list(_EXCLUDE))
        path = output_dir / f"graphs-{shard_num:05d}.pt"
        torch.save(batch, path)
        shard_paths.append(path)
        for sha in current_ids:
            shard_index[sha] = shard_num
        shard_num += 1
        current_graphs.clear()
        current_ids.clear()

    for sha, source, _ in contracts:
        pt_path = rep_root / source / f"{sha}.pt"
        if not pt_path.exists():
            skipped += 1
            continue
        graph: Data = torch.load(pt_path, weights_only=True)
        current_graphs.append(graph)
        current_ids.append(sha)
        if len(current_graphs) >= shard_size:
            _flush()

    _flush()  # final partial shard

    if skipped:
        logger.warning("graph_writer: skipped %d contracts with no .pt file", skipped)

    return shard_paths, shard_index


__all__ = ["write_graphs_shards"]
