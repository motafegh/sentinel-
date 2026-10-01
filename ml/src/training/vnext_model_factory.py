"""Frozen model construction for R4 Phase 8."""
from __future__ import annotations

import torch

from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.vnext_phase8_config import FROZEN_ARCHITECTURE
from sentinel_data.preprocessing.r4_versions import V10_GRAPH_SCHEMA_VERSION


def build_phase8_model(device: torch.device) -> SentinelModel:
    """Historical Phase-8/G7 factory; preserves the original v9 default."""
    return SentinelModel(**FROZEN_ARCHITECTURE).to(device)


def build_phase8_v10_model(device: torch.device) -> SentinelModel:
    """Build the frozen architecture explicitly against accepted graph schema V10."""
    return SentinelModel(
        **FROZEN_ARCHITECTURE,
        graph_schema_version=V10_GRAPH_SCHEMA_VERSION,
    ).to(device)


__all__ = ["build_phase8_model", "build_phase8_v10_model"]
