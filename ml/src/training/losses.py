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

AMP Safety:
    Both classes guard against log(0) with .clamp(min=1e-8) on probabilities.
    Operations stay in the float32 regime via explicit .float() casts so they
    are safe inside torch.cuda.amp.autocast() (BF16/FP16 context).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021).

    Args:
        gamma_neg: Focus parameter for negatives (y=0). Higher → more focus
                   on hard negatives (misclassified negatives). Default 4.
        gamma_pos: Focus parameter for positives (y=1). Lower than gamma_neg
                   gives positives more weight relative to easy negatives.
                   Default 1 (mild focal on positives).
        clip:      Probability margin shift applied to negative probabilities
                   before computing focus weight.  Negatives with p < clip
                   are clipped to zero and contribute zero gradient.
                   Default 0.05.
        reduction: "mean" (default), "sum", or "none".
    """

    def __init__(
        self,
        gamma_neg:  float = 4.0,
        gamma_pos:  float = 1.0,
        clip:       float = 0.05,
        reduction:  str   = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.reduction = reduction

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
        prob = torch.sigmoid(logits)                     # [B, C], range (0, 1)

        # Shifted negative probability for the clip margin
        prob_neg = (prob - self.clip).clamp(min=0.0)     # [B, C]

        # Cross-entropy terms: log(p) for positives, log(1-p) for negatives
        log_pos  = torch.log(prob.clamp(min=1e-8))           # [B, C]
        log_neg  = torch.log((1.0 - prob_neg).clamp(min=1e-8))  # [B, C]

        # Focal weights
        #   pos_weight = (1 - p)^gamma_pos  — focus on uncertain positives
        #   neg_weight = p_shifted^gamma_neg — focus on hard negatives
        focal_pos = (1.0 - prob) ** self.gamma_pos
        focal_neg = prob_neg     ** self.gamma_neg

        # Asymmetric loss per cell
        loss_pos = -labels       * focal_pos * log_pos
        loss_neg = -(1.0 - labels) * focal_neg * log_neg
        loss     = loss_pos + loss_neg                   # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
