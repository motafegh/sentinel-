"""Stage 7B — sentinel_collate_fn: collate function for SentinelDataset.

Handles the 5-tuple returned by SentinelDataset.__getitem__:
    (graph, tokens, y, contract_id, confidence_tier)

Returns a 5-tuple batch:
    (graphs_batch, tokens_batch, y_batch, contract_ids, confidence_tiers)

    graphs_batch    : PyG Batch — merged graph batch
    tokens_batch    : dict — "input_ids"[B,4,512], "attention_mask"[B,4,512]
    y_batch         : float32 Tensor[B,10]
    contract_ids    : list[str]            — NOT a tensor
    confidence_tiers: list[str | None]     — NOT a tensor (contains None values)
"""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Batch, Data

# Attributes that Batch.from_data_list() cannot merge (str/non-tensor metadata).
# Mirrors the exclusion list from dual_path_dataset.py for backward compatibility.
_EXCLUDE_KEYS = [
    "contract_hash", "contract_path", "contract_name",
    "node_metadata", "num_edges", "num_nodes", "y",
]


def sentinel_collate_fn(
    batch: list[tuple[Data, dict, torch.Tensor, str, Optional[str]]],
) -> tuple[Batch, dict[str, torch.Tensor], torch.Tensor, list[str], list[Optional[str]]]:
    """Collate a list of SentinelDataset items into a training batch.

    Args:
        batch: list of (graph, tokens, y, contract_id, confidence_tier) tuples.

    Returns:
        (graphs_batch, tokens_batch, y_batch, contract_ids, confidence_tiers)
    """
    graphs, tokens_list, ys, contract_ids, confidence_tiers = zip(*batch)

    graphs_batch = Batch.from_data_list(list(graphs), exclude_keys=_EXCLUDE_KEYS)

    input_ids = torch.stack([t["input_ids"] for t in tokens_list], dim=0)      # [B,4,512]
    attention_mask = torch.stack([t["attention_mask"] for t in tokens_list], dim=0)  # [B,4,512]
    tokens_batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    y_batch = torch.stack(list(ys), dim=0)  # [B,10]

    return graphs_batch, tokens_batch, y_batch, list(contract_ids), list(confidence_tiers)
