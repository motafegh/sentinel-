from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary or multi-label classification on imbalanced data.

    Multiplies standard BCE by (1 - pt)^gamma to crush the loss on easy
    examples, forcing the model to focus on hard (rare) ones.

    SENTINEL uses gamma=2.0, alpha=0.25 (original paper defaults).

    Works element-wise, so it handles both:
        - Binary:     predictions shape (B,),    targets shape (B,)
        - Multi-label: predictions shape (B, C), targets shape (B, C)

    IMPORTANT — this class expects POST-SIGMOID probabilities in [0, 1].
    Do NOT pass raw logits. The _FocalFromLogits wrapper in trainer.py
    handles the sigmoid call before forwarding here.

    Audit fix #6 (2026-05-01):
        Explicit .float() cast on predictions and targets at the top of
        forward() guards against BF16 underflow when called inside an
        AMP autocast block. Without this, probabilities near 0 become
        exactly 0.0 in BF16, log(p) = -inf, loss = nan.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss for a batch.

        Args:
            predictions: Model outputs AFTER sigmoid.
                         Shape (B,) for binary or (B, C) for multi-label.
                         Values must be in [0, 1].
            targets:     Ground truth labels.
                         Same shape as predictions. dtype float, values 0 or 1.

        Returns:
            Scalar loss value averaged over all elements in the batch.
        """
        # ── Audit fix #6: BF16 underflow guard ────────────────────────────
        # Under torch.amp.autocast("cuda"), eligible ops run in BF16.
        # BF16 has ~3 decimal digits of precision: a probability of 0.001
        # silently becomes 0.0 in BF16, making log(p) = -inf.
        # Casting to float32 here prevents nan losses regardless of whether
        # this is called inside an autocast block or not.
        predictions = predictions.float()
        targets = targets.float()
        # ──────────────────────────────────────────────────────────────────

        bce = F.binary_cross_entropy(predictions, targets, reduction="none")

        # pt = model confidence on the correct class
        #   if target==1: pt = p  (we want high p, easy if p is already high)
        #   if target==0: pt = 1-p (we want low p, easy if p is already low)
        pt = torch.where(targets == 1, predictions, 1 - predictions)

        # alpha balances positive vs negative class contribution
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal weight: (1-pt)^gamma down-weights easy examples (high pt)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()


class MultiLabelFocalLoss(nn.Module):
    """
    Per-class Focal Loss for multi-label classification on raw logits.

    Differences from FocalLoss:
      - Accepts RAW LOGITS (applies sigmoid internally). No external wrapper needed.
      - alpha is a per-class List[float] (not a scalar). Enables per-class
        imbalance correction, e.g. alpha[c] = 1 - n_pos[c] / N so rare classes
        receive a higher weight than common ones.
      - Designed for [B, C] multi-label output only.

    Formula (element-wise, averaged over B×C elements):
        p    = sigmoid(logit)
        pt   = p  if target == 1 else (1 - p)
        alpha_t = alpha_c  if target == 1 else (1 - alpha_c)   ← class-balancing
        loss = -alpha_t * (1 - pt)^gamma * log(pt + ε)

    Args:
        alpha:  List of C per-class balance weights in [0, 1].
                Typically set to (1 - class_freq_c) so rare classes get
                alpha closer to 1.0 and common classes closer to 0.0.
        gamma:  Focusing exponent (default 2.0). Higher values down-weight
                easy examples more aggressively.

    Audit fix (2026-05-12):
        alpha_t was previously self.alpha.unsqueeze(0) — the same weight
        applied to both positive AND negative examples per class. This
        inverts class-balancing for rare classes: the loss on negatives
        (which dominate) is inflated by alpha instead of (1-alpha), causing
        the model to over-penalise correct negatives and under-penalise
        missed positives. Fixed by conditioning on targets == 1 to match
        the original focal loss formula.
    """

    def __init__(self, alpha: List[float], gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        # Register as a buffer so it moves to the correct device with .to(device)
        # and is included in state_dict (for checkpoint reproducibility).
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw model logits [B, C]. NOT sigmoid-applied.
            targets: Ground truth labels [B, C]. Values 0 or 1, any dtype.

        Returns:
            Scalar loss averaged over all B×C elements.
        """
        # BF16 guard: sigmoid and log are numerically sensitive under autocast.
        logits  = logits.float()
        targets = targets.float()

        p   = torch.sigmoid(logits)              # [B, C]
        pt  = torch.where(targets == 1, p, 1 - p)  # confidence on correct class

        # alpha_t: α for positives, (1-α) for negatives — per original focal loss.
        # alpha: [C] → broadcast to [1, C] → [B, C] via where
        alpha = self.alpha.unsqueeze(0)          # [1, C] → broadcasts to [B, C]
        alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)  # [B, C]

        # Numerically stable BCE from logits avoids log(0) issues.
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
