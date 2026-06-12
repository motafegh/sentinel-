"""Stage 7B — SentinelDataset: PyTorch Dataset backed by a v2 export artifact.

Returns 5-tuples per item:
    (graph, tokens, y, contract_id, confidence_tier)

    graph           : PyG Data — x[n_nodes,12], edge_index[2,E], edge_attr[E]
    tokens          : dict — "input_ids"[4,512] int64, "attention_mask"[4,512] int64
    y               : float32 Tensor[10] — multi-label targets
    contract_id     : str (sha256 of the contract)
    confidence_tier : str | None  ("T0", "T1", "T2", or None for NonVulnerable)

Three gates are checked at construction time (hard ValueError on failure):
    1. format schema version must be "v1"
    2. graph schema version must match FEATURE_SCHEMA_VERSION
    3. artifact hash must be intact (data-integrity check)

Shard loading is LRU-cached (default 4 shards; set SENTINEL_SHARD_CACHE_SIZE env var).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.serialization
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from sentinel_data.export.export import SentinelDatasetExport
from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION

# Register PyG safe globals for weights_only deserialization.
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

_SHARD_CACHE_SIZE = int(os.environ.get("SENTINEL_SHARD_CACHE_SIZE", "4"))
_PAD_TOKEN_ID = 1  # graphcodebert-base pad_token_id (RoBERTa vocab)

_EXPECTED_FORMAT_SCHEMA = "v1"


@lru_cache(maxsize=_SHARD_CACHE_SIZE)
def _load_graph_shard(path: Path) -> Batch:
    return torch.load(path, weights_only=False)


@lru_cache(maxsize=_SHARD_CACHE_SIZE)
def _load_token_shard(path: Path) -> torch.Tensor:
    return torch.load(path, weights_only=True)


class SentinelDataset(Dataset):
    """PyTorch Dataset that reads from a `chunk_export()` output directory.

    Args:
        split: "train", "val", or "test"
        export_dir: Path to the export directory (must contain manifest.json).
        model_graph_schema_version: Override the expected graph schema version
            (default: FEATURE_SCHEMA_VERSION from sentinel_data.representation).
            Pass a different value to test schema-mismatch handling.
    """

    def __init__(
        self,
        split: str,
        export_dir: Path,
        model_graph_schema_version: Optional[str] = None,
    ) -> None:
        self.split = split
        self.export = SentinelDatasetExport(Path(export_dir))

        expected_schema = model_graph_schema_version or FEATURE_SCHEMA_VERSION

        # Gate 1 — format schema version
        if self.export.manifest.schema_version != _EXPECTED_FORMAT_SCHEMA:
            raise ValueError(
                f"Export format schema version mismatch: "
                f"expected '{_EXPECTED_FORMAT_SCHEMA}', "
                f"got '{self.export.manifest.schema_version}'. "
                f"Re-run `sentinel-data export` with a compatible version."
            )

        # Gate 2 — graph schema version
        if self.export.manifest.graph_schema_version != expected_schema:
            raise ValueError(
                f"Graph schema version mismatch: "
                f"model expects '{expected_schema}', "
                f"export has '{self.export.manifest.graph_schema_version}'. "
                f"Re-export with matching FEATURE_SCHEMA_VERSION."
            )

        # Gate 3 — artifact hash integrity
        if not self.export.verify_artifact_hash():
            raise ValueError(
                f"Export artifact hash mismatch — data files may be corrupt or tampered. "
                f"Re-run `sentinel-data export` to regenerate a clean artifact."
            )

        # Build label lookup: {contract_id: (y_tensor, confidence_tier)}
        labels_df = pd.read_parquet(self.export.labels_path).set_index("contract_id")
        class_cols = [f"class_{i}" for i in range(10)]
        self._label_lookup: dict[str, tuple[torch.Tensor, str | None]] = {}
        for cid, row in labels_df.iterrows():
            y = torch.tensor([row[c] for c in class_cols], dtype=torch.float32)
            tier = row["confidence_tier"]
            self._label_lookup[cid] = (y, None if pd.isna(tier) else str(tier))

        # Contract list for this split, filtered to only those with representations.
        shard_index = self.export.shard_index
        all_ids = self.export.get_split_contract_ids(split)
        self._contract_ids: list[str] = [sha for sha in all_ids if sha in shard_index]

        # Precompute num_nodes per contract. Used by _build_weighted_sampler's
        # "timestamp-size" mode (trainer.py). All shards are touched once at init
        # — LRU-cached so subsequent __getitem__ calls hit the cache.
        self._num_nodes_map: dict[str, int] = {}
        for contract_id, entry in shard_index.items():
            if contract_id not in self._contract_ids:
                continue  # skip other splits
            shard_num = entry["shard"]
            pos = entry["pos_in_shard"]
            graph_shard = _load_graph_shard(
                self.export.graphs_dir / f"graphs-{shard_num:05d}.pt"
            )
            self._num_nodes_map[contract_id] = int(graph_shard.get_example(pos).num_nodes)

    @property
    def num_nodes_map(self) -> dict[str, int]:
        """contract_id → num_nodes for every contract in this split.

        Read-only view of the precomputed map. Used by the trainer's
        timestamp-size weighted sampler mode.
        """
        return self._num_nodes_map

    def __len__(self) -> int:
        return len(self._contract_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[Data, dict[str, torch.Tensor], torch.Tensor, str, str | None]:
        contract_id = self._contract_ids[idx]
        entry = self.export.shard_index[contract_id]
        shard_num: int = entry["shard"]
        pos: int = entry["pos_in_shard"]

        # Load graph
        graph_shard = _load_graph_shard(self.export.graphs_dir / f"graphs-{shard_num:05d}.pt")
        graph: Data = graph_shard.get_example(pos)

        # Load tokens: [4, 512] int64 input_ids
        token_shard = _load_token_shard(self.export.tokens_dir / f"tokens-{shard_num:05d}.pt")
        input_ids: torch.Tensor = token_shard[pos]  # [4, 512]
        attention_mask = (input_ids != _PAD_TOKEN_ID).long()  # [4, 512]
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Labels
        y, confidence_tier = self._label_lookup[contract_id]

        return graph, tokens, y, contract_id, confidence_tier
