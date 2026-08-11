"""SENTINEL ZKML student model.

The proxy is a frozen 128→64→32→10 network used by the legacy V2 EZKL
circuit. One detail is critical for every consumer:

**The current student is trained by MSE directly against
``sigmoid(teacher_logits)``.** Therefore ``ProxyModel.forward()`` returns the
student's *probability-regression scores*. Consumers must not apply another
sigmoid. The final linear layer is intentionally left unbounded; no clipping is
performed silently because the existing trained artifact must remain
observable exactly as trained.

Changing the architecture or changing the output transform changes the ZK
statement and requires a new circuit version, a newly evaluated student, fresh
ONNX/EZKL artifacts, and a new verifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

# Architecture/proof protocol identity.
CIRCUIT_VERSION = "v2.0"
OUTPUT_SEMANTICS = "teacher_probability_regression_v1"

# The executable parameter count is 10,666. This guard is an engineering
# ceiling, not a claim about a universal EZKL parameter limit.
EZKL_PARAM_LIMIT = 12_000


class ProxyModel(nn.Module):
    """Frozen student network for the legacy V2 proof circuit.

    Architecture::

        fusion[128] → Linear(128,64) → ReLU
                    → Linear(64,32)  → ReLU
                    → Linear(32,10)  → student_scores[10]

    ``student_scores`` approximate teacher probabilities because the current
    distillation target is ``sigmoid(teacher_logits)``. There is no sigmoid in
    this module and callers must not add one merely because the last layer is
    linear.
    """

    FROZEN_INPUT_DIM = 128
    FROZEN_HIDDEN1 = 64
    FROZEN_HIDDEN2 = 32
    FROZEN_NUM_CLASSES = 10

    def __init__(
        self,
        input_dim: int = 128,
        hidden1: int = 64,
        hidden2: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if input_dim != self.FROZEN_INPUT_DIM:
            raise RuntimeError(
                f"input_dim must be {self.FROZEN_INPUT_DIM}; got {input_dim}. "
                "Changing it requires a new circuit version and artifact bundle."
            )
        if hidden1 != self.FROZEN_HIDDEN1:
            raise RuntimeError(
                f"hidden1 must be {self.FROZEN_HIDDEN1}; got {hidden1}. "
                "Changing it invalidates the current circuit artifacts."
            )
        if hidden2 != self.FROZEN_HIDDEN2:
            raise RuntimeError(
                f"hidden2 must be {self.FROZEN_HIDDEN2}; got {hidden2}. "
                "Changing it invalidates the current circuit artifacts."
            )
        if num_classes != self.FROZEN_NUM_CLASSES:
            raise RuntimeError(
                f"num_classes must be {self.FROZEN_NUM_CLASSES}; got {num_classes}. "
                "Changing it requires a new circuit/versioned class protocol."
            )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

        total_params = self.parameter_count()
        if total_params > EZKL_PARAM_LIMIT:
            raise RuntimeError(
                f"ProxyModel has {total_params:,} parameters, exceeding the "
                f"configured engineering ceiling {EZKL_PARAM_LIMIT:,}."
            )

        logger.info(
            "ProxyModel initialised — params={} circuit={} output_semantics={}",
            total_params,
            CIRCUIT_VERSION,
            OUTPUT_SEMANTICS,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return student probability-regression scores with shape ``[B,10]``.

        The values are the direct output of the trained linear student and are
        deliberately not clipped or transformed. A future bounded/calibrated
        score transform must be introduced as a new circuit protocol rather
        than silently changing V2 semantics.
        """
        return self.network(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def circuit_version(self) -> str:
        return CIRCUIT_VERSION

    def output_semantics(self) -> str:
        return OUTPUT_SEMANTICS


__all__ = [
    "CIRCUIT_VERSION",
    "OUTPUT_SEMANTICS",
    "EZKL_PARAM_LIMIT",
    "ProxyModel",
]


if __name__ == "__main__":
    proxy = ProxyModel().eval()
    sample = torch.randn(4, proxy.FROZEN_INPUT_DIM)
    with torch.no_grad():
        scores = proxy(sample)
    print(f"Input shape:      {sample.shape}")
    print(f"Output shape:     {scores.shape}")
    print(f"Total params:     {proxy.parameter_count():,}")
    print(f"Circuit version:  {proxy.circuit_version()}")
    print(f"Output semantics: {proxy.output_semantics()}")
