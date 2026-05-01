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
