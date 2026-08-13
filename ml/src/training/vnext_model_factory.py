"""Frozen model construction for R4 Phase 8."""
from __future__ import annotations

import torch

from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.vnext_phase8_config import FROZEN_ARCHITECTURE


def build_phase8_model(device: torch.device) -> SentinelModel:
    return SentinelModel(**FROZEN_ARCHITECTURE).to(device)


__all__ = ["build_phase8_model"]
