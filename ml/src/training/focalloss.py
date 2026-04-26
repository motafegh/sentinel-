import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification on imbalanced data.

    Multiplies standard BCE by (1 - pt)^gamma to crush the loss
    on easy examples, forcing the model to focus on hard ones.
    SENTINEL uses gamma=2.0, alpha=0.25 (original paper defaults).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss for a batch.

        Args:
            predictions: Model outputs AFTER sigmoid. Shape (B,), values in [0,1].
            targets: Ground truth labels. Shape (B,), dtype float, values 0 or 1.
        Returns:
            Scalar loss value averaged over the batch.
        """
        bce = F.binary_cross_entropy(predictions, targets, reduction="none")
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
