"""Mask-aware Phase-8 loss and model-selection metrics."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ml.src.training.losses import AsymmetricLoss

NONE = 0
WEAK = 1
STRONG = 2


def validate_positive_targets(targets: torch.Tensor, mask: torch.Tensor) -> None:
    if targets.shape != mask.shape:
        raise ValueError("target/mask shape mismatch")
    if mask.dtype is not torch.bool:
        raise TypeError("mask must be bool")
    if not mask.any():
        raise ValueError("zero authorized cells")
    selected = targets[mask]
    if not torch.isfinite(selected).all():
        raise ValueError("authorized cell has null target")
    if not torch.all(selected == 1.0):
        raise ValueError("Phase-8 authorized supervision must be positive-only")


def strength_weights(
    strength_codes: torch.Tensor,
    loss_mask: torch.Tensor,
    weak_positive_weight: float,
) -> torch.Tensor:
    if not 0.0 < weak_positive_weight <= 1.0:
        raise ValueError("weak_positive_weight must be in (0,1]")
    if strength_codes.shape != loss_mask.shape:
        raise ValueError("strength/mask shape mismatch")
    if torch.any(loss_mask & (strength_codes == NONE)):
        raise ValueError("loss cell has NONE strength")
    weights = torch.zeros_like(strength_codes, dtype=torch.float32)
    weights = torch.where(strength_codes == STRONG, torch.ones_like(weights), weights)
    weights = torch.where(
        strength_codes == WEAK,
        torch.full_like(weights, float(weak_positive_weight)),
        weights,
    )
    return weights * loss_mask.float()


def masked_mean(
    per_cell: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if per_cell.shape != mask.shape or per_cell.shape != weights.shape:
        raise ValueError("loss/mask/weight shape mismatch")
    effective = weights * mask.float()
    denom = effective.sum()
    if denom.item() <= 0.0:
        raise ValueError("zero effective optimizer weight")
    return (per_cell * effective).sum() / denom


def masked_asl_positive_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    strength_codes: torch.Tensor,
    *,
    weak_positive_weight: float,
    gamma_neg: float = 2.0,
    gamma_pos: float = 1.0,
    clip: float = 0.01,
) -> torch.Tensor:
    validate_positive_targets(targets, loss_mask)
    safe = torch.where(loss_mask, targets, torch.zeros_like(targets))
    criterion = AsymmetricLoss(
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=clip,
        reduction="none",
    ).to(logits.device)
    cells = criterion(logits, safe)
    weights = strength_weights(strength_codes, loss_mask, weak_positive_weight).to(logits.device)
    return masked_mean(cells, loss_mask, weights)


def masked_bce_positive_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    strength_codes: torch.Tensor,
    *,
    weak_positive_weight: float,
    class_multiplier: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_positive_targets(targets, loss_mask)
    safe = torch.where(loss_mask, targets, torch.zeros_like(targets))
    cells = F.binary_cross_entropy_with_logits(logits.float(), safe.float(), reduction="none")
    weights = strength_weights(strength_codes, loss_mask, weak_positive_weight).to(logits.device)
    if class_multiplier is not None:
        if class_multiplier.ndim != 1 or class_multiplier.numel() != logits.shape[1]:
            raise ValueError("class_multiplier must be [C]")
        weights = weights * class_multiplier.to(logits.device).view(1, -1)
    return masked_mean(cells, loss_mask, weights)


def positive_selection_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    metric_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    validate_positive_targets(targets, metric_mask)
    selected_logits = logits.float()[metric_mask]
    probs = torch.sigmoid(selected_logits)
    result: dict[str, Any] = {
        "positive_nll": float(F.softplus(-selected_logits).mean().item()),
        "mean_positive_probability": float(probs.mean().item()),
        "positive_recall_at_fixed_threshold": float((probs >= threshold).float().mean().item()),
        "metric_cells": int(metric_mask.sum().item()),
        "fixed_threshold": float(threshold),
        "per_class": {},
    }
    for c in range(logits.shape[1]):
        cmask = metric_mask[:, c]
        n = int(cmask.sum().item())
        if n == 0:
            continue
        c_logits = logits[:, c].float()[cmask]
        c_probs = torch.sigmoid(c_logits)
        result["per_class"][str(c)] = {
            "cells": n,
            "positive_nll": float(F.softplus(-c_logits).mean().item()),
            "mean_positive_probability": float(c_probs.mean().item()),
            "positive_recall_at_fixed_threshold": float((c_probs >= threshold).float().mean().item()),
        }
    return result


__all__ = [
    "masked_asl_positive_loss",
    "masked_bce_positive_loss",
    "masked_mean",
    "positive_selection_metrics",
    "strength_weights",
    "validate_positive_targets",
]
