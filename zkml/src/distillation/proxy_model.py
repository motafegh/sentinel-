"""
proxy_model.py — SENTINEL ZKML Proxy Model (Student)

A tiny neural network trained to mimic the full SentinelModel (teacher)
via knowledge distillation. Small enough to fit inside an EZKL ZK circuit.

RECALL — why this exists:
    Full SentinelModel (~125M params) → too large for ZK circuits (~10K limit)
    Proxy (~2,625 params) → ZK-compatible, agrees with teacher ≥95% of the time

RECALL — what the proxy receives as input:
    NOT raw Solidity. NOT the teacher's final score.
    The 64-dim FusionLayer output — the teacher's richest representation.
    By the time data reaches the proxy, the teacher has already done all
    the hard work: AST extraction, GNN message passing, CodeBERT semantics,
    fusion. The proxy just maps that compressed understanding to a scalar.

RECALL — why the architecture is frozen (ADR-007):
    The ZK circuit is derived from the computation graph structure.
    Changing layer sizes changes the circuit.
    Changing the circuit invalidates the proving and verification keys.
    Only weights retrain — architecture never changes.

CIRCUIT_VERSION tracks the architecture definition.
If you ever need to change the architecture:
  1. Bump CIRCUIT_VERSION
  2. Retrain proxy
  3. Re-export ONNX
  4. Rerun full EZKL pipeline (Steps 1-4)
  5. Redeploy ZKMLVerifier.sol with new verification key
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

# ------------------------------------------------------------------
# Circuit version — tracks architecture definition
# ------------------------------------------------------------------
# This constant identifies which architecture was used to generate
# the EZKL proving and verification keys.
#
# RECALL — why this matters:
#   proving_key.pk and verification_key.vk are derived from the
#   circuit structure (layer sizes + operations).
#   If the architecture changes, the keys become invalid.
#   Mismatched keys produce cryptic EZKL errors deep in the pipeline.
#   CIRCUIT_VERSION makes mismatches detectable at load time.
#
# Bump this whenever you change input_dim, hidden1, or hidden2.
# Then rerun the full EZKL setup pipeline.
CIRCUIT_VERSION = "v1.0"  # Linear(64→32→16→1) + Sigmoid

# ------------------------------------------------------------------
# Hard limits — enforced by assertion, not convention
# ------------------------------------------------------------------
# RECALL — why 10K is the limit:
#   EZKL converts each model operation into arithmetic constraints.
#   The number of constraints scales with parameter count.
#   Beyond ~10K params, circuit compilation becomes impractical
#   and proving time grows to hours. Keep the proxy well below this.
EZKL_PARAM_LIMIT = 10_000


class ProxyModel(nn.Module):
    """
    Tiny student model for knowledge distillation + ZK proof generation.

    Architecture: Linear(64→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)
    Parameters:   2,625 (well within EZKL's ~10K circuit limit)
    Circuit:      CIRCUIT_VERSION = "v1.0"

    RECALL — the architecture is intentionally frozen:
        Changing layer sizes = changing the ZK circuit structure =
        invalidating proving_key.pk and verification_key.vk.
        The constructor accepts dimension args for testing flexibility,
        but asserts they match the frozen architecture before proceeding.
        Do not pass non-default values in production.

    Args:
        input_dim:  Must be 64 — matches FusionLayer output_dim
        hidden1:    Must be 32 — frozen by ADR-007
        hidden2:    Must be 16 — frozen by ADR-007

    Example:
        proxy = ProxyModel()
        features = torch.randn(4, 64)  # batch of 4 contracts
        scores = proxy(features)       # [4] risk scores in [0, 1]
    """

    # Frozen architecture constants — single source of truth
    # Change here = change CIRCUIT_VERSION + rerun full EZKL pipeline
    FROZEN_INPUT_DIM = 64
    FROZEN_HIDDEN1   = 32
    FROZEN_HIDDEN2   = 16

    def __init__(
        self,
        input_dim: int = 64,
        hidden1:   int = 32,
        hidden2:   int = 16,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Architecture freeze guard
        # ------------------------------------------------------------------
        # RECALL — why we assert rather than silently accept:
        #   Passing hidden1=128 would create a different circuit.
        #   That circuit would need new EZKL keys.
        #   Training would succeed but EZKL setup would produce keys
        #   incompatible with any previously generated proofs.
        #   The assertion catches this at instantiation, not hours later.
        assert input_dim == self.FROZEN_INPUT_DIM, (
            f"input_dim must be {self.FROZEN_INPUT_DIM} (FusionLayer output). "
            f"Got {input_dim}. Changing this requires bumping CIRCUIT_VERSION "
            f"and rerunning the full EZKL pipeline."
        )
        assert hidden1 == self.FROZEN_HIDDEN1, (
            f"hidden1 must be {self.FROZEN_HIDDEN1} (frozen by ADR-007). "
            f"Got {hidden1}. Architecture changes invalidate EZKL keys."
        )
        assert hidden2 == self.FROZEN_HIDDEN2, (
            f"hidden2 must be {self.FROZEN_HIDDEN2} (frozen by ADR-007). "
            f"Got {hidden2}. Architecture changes invalidate EZKL keys."
        )

        # ------------------------------------------------------------------
        # Network — three linear layers, deliberately minimal
        # ------------------------------------------------------------------
        # RECALL — this is not a simplification, size is a hard ZK constraint.
        # The teacher does all the intelligence work upstream.
        # The proxy just maps the teacher's 64-dim summary to a scalar.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),  # 64 → 32
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),    # 32 → 16
            nn.ReLU(),
            nn.Linear(hidden2, 1),          # 16 → 1
            nn.Sigmoid(),                   # squash to [0, 1]
        )

        # ------------------------------------------------------------------
        # Parameter count enforcement
        # ------------------------------------------------------------------
        # RECALL — why we assert not just log:
        #   Logging warns but doesn't stop execution.
        #   An oversized proxy reaches EZKL compile_model and fails
        #   with a cryptic circuit-too-large error after minutes of work.
        #   Asserting here catches it immediately at instantiation.
        total_params = self.parameter_count()
        assert total_params <= EZKL_PARAM_LIMIT, (
            f"ProxyModel has {total_params:,} parameters — "
            f"exceeds EZKL limit of {EZKL_PARAM_LIMIT:,}. "
            f"Reduce architecture before proceeding."
        )

        logger.info(
            f"ProxyModel initialised — "
            f"{total_params:,} parameters | "
            f"EZKL limit: {EZKL_PARAM_LIMIT:,} | "
            f"Circuit: {CIRCUIT_VERSION}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map 64-dim teacher features to a risk score.

        RECALL — what x actually contains:
            The FusionLayer output from the full SENTINEL model.
            By this point the teacher has already processed: AST graph
            structure (GNN path) + code semantics (CodeBERT path) +
            fusion. This 64-dim vector encodes the teacher's complete
            understanding of the contract. The proxy's job is simple
            by comparison — map that rich summary to a scalar.

        Args:
            x: FusionLayer output from teacher, shape [B, 64]

        Returns:
            Risk scores, shape [B] — values in [0, 1]
        """
        # [B, 64] → network → [B, 1] → squeeze → [B]
        return self.network(x).squeeze(1)

    def parameter_count(self) -> int:
        """Return total parameter count — used to verify ZK compatibility."""
        return sum(p.numel() for p in self.parameters())

    def circuit_version(self) -> str:
        """Return circuit version string — use to detect key mismatches."""
        return CIRCUIT_VERSION


if __name__ == "__main__":
    # Sanity check — run directly to verify shapes and constraints
    proxy = ProxyModel()
    proxy.eval()

    fake_features = torch.randn(4, 64)
    scores = proxy(fake_features)

    print(f"Input shape:     {fake_features.shape}")  # [4, 64]
    print(f"Output shape:    {scores.shape}")          # [4]
    print(f"Score range:     [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Total params:    {proxy.parameter_count():,}")  # 2,625
    print(f"Circuit version: {proxy.circuit_version()}")    # v1.0

    # Test freeze guard — should raise AssertionError
    try:
        bad_proxy = ProxyModel(hidden1=128)
        print("ERROR: freeze guard failed — should have raised AssertionError")
    except AssertionError as e:
        print(f"Freeze guard working: {e}")