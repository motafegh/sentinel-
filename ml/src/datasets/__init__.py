"""Dataset package exports.

Imports are lazy so the DATA-vNext training path does not import the historical
v1 export/label stack merely because Python initializes this package.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "SentinelDataset",
    "sentinel_collate_fn",
    "CANONICAL_G7_BINDING_DIGEST",
    "MODEL_SELECTION_ROLES",
    "TRAIN_ROLES",
    "VNextTrainingDataset",
    "vnext_collate_fn",
]


def __getattr__(name: str) -> Any:
    if name == "SentinelDataset":
        from ml.src.datasets.sentinel_dataset import SentinelDataset
        return SentinelDataset
    if name == "sentinel_collate_fn":
        from ml.src.datasets.collate import sentinel_collate_fn
        return sentinel_collate_fn
    if name in {
        "CANONICAL_G7_BINDING_DIGEST",
        "MODEL_SELECTION_ROLES",
        "TRAIN_ROLES",
        "VNextTrainingDataset",
        "vnext_collate_fn",
    }:
        from ml.src.datasets import vnext_dataset as module
        return getattr(module, name)
    raise AttributeError(name)
