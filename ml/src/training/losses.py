"""
losses.py — Loss functions for SENTINEL multi-label classification.

AsymmetricLoss (ASL):
    Addresses positive-negative imbalance by applying different gamma exponents
    to positives and negatives independently, then hard-thresholds small negative
    probabilities to zero-out easy negatives before computing their contribution.

    Reference: Ridnik et al. "Asymmetric Loss For Multi-Label Classification"
    (ICCV 2021).  https://arxiv.org/abs/2009.14119

    Why for SENTINEL:
        Standard BCE: equally weights all (class, sample) pairs — 44K×10 = 440K
        cells, >85% negative (label=0).  The optimizer spends most gradient budget
        suppressing easy negatives (DoS=0 on most contracts) rather than lifting
        rare positives (DoS=1 on only 377 train contracts).

        ASL with gamma_neg=4, gamma_pos=1:
        - Easy negatives (p≈0, y=0) get (1-p)^0 = near-zero weight after clip
        - Hard negatives (p≈0.5, y=0) get moderate weight
        - Positives (y=1) get full upweighted gradient (gamma_pos=1 is mild focus)

        clip (probability margin): shifts p_neg by -clip before focus, so any
        negative with p < clip is treated as zero probability and contributes zero
        gradient.  clip=0.05 removes very-confident negatives from the gradient.

    Per-class gamma/clip (BUG-M3):
        gamma_neg, gamma_pos, and clip can each be a float (scalar, broadcast to
        all classes) or a 1-D Tensor of shape [C] (per-class values).  Tensor
        inputs are registered as buffers so they move with .to(device).

        Example — apply softer negative focus to rare classes:
            gamma_neg = torch.tensor([4.0, 2.0, 4.0, ...])  # shape [C]

AMP Safety:
    Both classes guard against log(0) with .clamp(min=1e-8) on probabilities.
    Operations stay in the float32 regime via explicit .float() casts so they
    are safe inside torch.cuda.amp.autocast() (BF16/FP16 context).
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021).

    Args:
        gamma_neg: Focus parameter for negatives (y=0). Float scalar OR 1-D
                   Tensor of shape [C] for per-class values. Default 4.
        gamma_pos: Focus parameter for positives (y=1). Float scalar OR 1-D
                   Tensor of shape [C]. Default 1.
        clip:      Probability margin shift for negatives. Float scalar OR 1-D
                   Tensor of shape [C]. Default 0.05.
        reduction: "mean" (default), "sum", or "none".
    """

    def __init__(
        self,
        gamma_neg:  Union[float, torch.Tensor] = 4.0,
        gamma_pos:  Union[float, torch.Tensor] = 1.0,
        clip:       Union[float, torch.Tensor] = 0.05,
        pos_weight: "torch.Tensor | None"      = None,
        reduction:  str                        = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

        # Register as buffers so .to(device) / .cuda() propagates automatically.
        # torch.as_tensor converts float → scalar tensor; 1-D tensors pass through.
        self.register_buffer("gamma_neg", torch.as_tensor(gamma_neg, dtype=torch.float32))
        self.register_buffer("gamma_pos", torch.as_tensor(gamma_pos, dtype=torch.float32))
        self.register_buffer("clip",      torch.as_tensor(clip,      dtype=torch.float32))
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None)

    def forward(
        self,
        logits: torch.Tensor,   # [B, C] raw logits (NO sigmoid)
        labels: torch.Tensor,   # [B, C] float {0.0, 1.0} or soft targets
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.

        Args:
            logits: Raw model output before sigmoid, shape [B, C].
            labels: Binary (or soft) targets, shape [B, C], values in [0, 1].

        Returns:
            Scalar loss (or per-element if reduction="none").
        """
        # Work in float32 regardless of AMP context to avoid BF16 precision loss
        logits = logits.float()
        labels = labels.float()

        # Probabilities from sigmoid
        prob = torch.sigmoid(logits)                              # [B, C]

        # Shifted negative probability for the clip margin.
        # self.clip is scalar or [C]; broadcasting handles [B, C] - [C] correctly.
        prob_neg = (prob - self.clip).clamp(min=0.0)              # [B, C]

        # Cross-entropy terms: log(p) for positives, log(1-p) for negatives
        log_pos  = torch.log(prob.clamp(min=1e-8))                # [B, C]
        log_neg  = torch.log((1.0 - prob_neg).clamp(min=1e-8))    # [B, C]

        # Focal weights — scalar or [C] gamma broadcasts correctly over [B, C]
        focal_pos = (1.0 - prob) ** self.gamma_pos                # [B, C]
        focal_neg = prob_neg     ** self.gamma_neg                 # [B, C]

        # Asymmetric loss per cell
        loss_pos = -labels         * focal_pos * log_pos          # [B, C]
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight  # [C] broadcasts over [B, C]
        loss_neg = -(1.0 - labels) * focal_neg * log_neg          # [B, C]
        loss     = loss_pos + loss_neg                            # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
