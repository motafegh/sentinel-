"""
proxy_model.py — SENTINEL ZKML Proxy Model (Student)

A tiny neural network trained to mimic the full SentinelModel (teacher)
via knowledge distillation. Small enough to fit inside an EZKL ZK circuit.

RECALL — why this exists:
    Full SentinelModel (~125M params) → too large for ZK circuits (~10K limit)
    Proxy (~8K params) → ZK-compatible, agrees with teacher ≥95% of the time

RECALL — what the proxy receives as input:
    NOT raw Solidity. NOT the teacher's final score.
    The 128-dim CrossAttentionFusion output — the teacher's richest
    representation AFTER bidirectional cross-attention fusion.
    CrossAttentionFusion output_dim=128 is LOCKED by SENTINEL-SPEC §6 (ADR-025).
    The proxy maps that 128-dim vector to 10 class logits.

RECALL — why input_dim=128, NOT 64 (CRITICAL — spec change from old binary model):
    OLD architecture (binary, pre-Track 3):
        FusionLayer (concat MLP) → output_dim=64 → proxy input_dim=64
    CURRENT architecture (multi-label, CrossAttentionFusion):
        CrossAttentionFusion → output_dim=128 (ADR-025) → proxy input_dim=128
    Using input_dim=64 on the current checkpoint → silent shape mismatch →
    Linear(64→32) vs expected Linear(128→64) → garbage outputs or runtime error.
    SENTINEL-SPEC §8.2: "proxy inputdim 128 matches CrossAttentionFusion output"
    SENTINEL-SPEC §6: "ZKML proxy input dim depends on this [fusion output_dim]"

RECALL — why the architecture is frozen (ADR-007):
    The ZK circuit is derived from the computation graph structure.
    Changing layer sizes changes the circuit.
    Changing the circuit invalidates the proving and verification keys.
    Only weights retrain — architecture never changes.

CIRCUIT_VERSION tracks the architecture definition.
If you ever need to change the architecture:
  1. Bump CIRCUIT_VERSION
  2. Retrain proxy (train_proxy.py)
  3. Re-export ONNX (export_onnx.py)
  4. Rerun full EZKL pipeline (Steps 1-5 in setup_circuit.py)
  5. Generate new ZKMLVerifier.sol from new verification_key.vk
  6. Redeploy ZKMLVerifier.sol on Sepolia
  7. Update ZKMLVerifier address in AuditRegistry via upgradeToAndCall
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

# ------------------------------------------------------------------
# Circuit version — tracks architecture definition
# ------------------------------------------------------------------
# CHANGELOG:
#   v1.0 — input_dim=64,  hidden=[32,16],   output=1  (binary, old FusionLayer)
#           WRONG: matched pre-Track-3 binary model. CrossAttentionFusion
#           outputs 128-dim, not 64. All v1.0 keys are invalid.
#   v2.0 — input_dim=128, hidden=[64,32],   output=10 (multi-label, current)
#           Matches CrossAttentionFusion output_dim=128 (ADR-025, SENTINEL-SPEC §8.2).
#           Matches NUMCLASSES=10 (SENTINEL-SPEC §8.1).
#           Matches ZKML proxy spec: Linear(128→64→32→10).
#
# RECALL — bumping CIRCUIT_VERSION means:
#   - Old proving_key.pk and verification_key.vk are INVALID
#   - Must rerun: export_onnx.py → setup_circuit.py (Steps 1-5)
#   - Must regenerate ZKMLVerifier.sol and redeploy on Sepolia
#   - Must update ZKMLVerifier address in AuditRegistry
CIRCUIT_VERSION = "v2.0"  # Linear(128→64→32→10) raw logits

# ------------------------------------------------------------------
# Hard limits — enforced at instantiation, not convention
# ------------------------------------------------------------------
EZKL_PARAM_LIMIT = 10_000


class ProxyModel(nn.Module):
    """
    Tiny student model for knowledge distillation + ZK proof generation.

    Architecture: Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→10)
    Parameters:   ~8,330 (within EZKL's circuit limit)
    Circuit:      CIRCUIT_VERSION = "v2.0"

    RECALL — the architecture is intentionally frozen (ADR-007):
        Changing layer sizes = changing the ZK circuit structure =
        invalidating proving_key.pk and verification_key.vk.
        Constructor raises RuntimeError (not assert — ADR-019) if dims
        don't match the frozen architecture. Do not pass non-default
        values in production.

    Args:
        input_dim:   Must be 128 — matches CrossAttentionFusion output_dim (ADR-025)
        hidden1:     Must be 64  — frozen by ADR-007
        hidden2:     Must be 32  — frozen by ADR-007
        num_classes: Must be 10  — matches NUMCLASSES in trainer.py (SENTINEL-SPEC §8.1)

    Example:
        proxy = ProxyModel()
        features = torch.randn(4, 128)  # batch of 4 contracts
        logits = proxy(features)        # [4, 10] raw class logits
        scores = torch.sigmoid(logits)  # [4, 10] probabilities
    """

    # Frozen architecture constants — single source of truth
    FROZEN_INPUT_DIM   = 128   # CrossAttentionFusion output_dim (ADR-025, SPEC §8.2)
    FROZEN_HIDDEN1     = 64
    FROZEN_HIDDEN2     = 32
    FROZEN_NUM_CLASSES = 10    # NUMCLASSES in trainer.py (SPEC §8.1)

    def __init__(
        self,
        input_dim:   int = 128,
        hidden1:     int = 64,
        hidden2:     int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Architecture freeze guard — RuntimeError not assert (ADR-019)
        # ------------------------------------------------------------------
        # RECALL — using RuntimeError not assert:
        #   `python -O` (optimised mode) silently strips `assert` statements.
        #   EZKL pipeline takes minutes; a silent pass-through here means
        #   hours of compute before the shape mismatch surfaces. Fail fast.
        if input_dim != self.FROZEN_INPUT_DIM:
            raise RuntimeError(
                f"input_dim must be {self.FROZEN_INPUT_DIM} — matches "
                f"CrossAttentionFusion output_dim (SENTINEL-SPEC §8.2, ADR-025). "
                f"Got {input_dim}. Changing this requires bumping CIRCUIT_VERSION "
                f"and rerunning the full EZKL pipeline."
            )
        if hidden1 != self.FROZEN_HIDDEN1:
            raise RuntimeError(
                f"hidden1 must be {self.FROZEN_HIDDEN1} (frozen by ADR-007). "
                f"Got {hidden1}. Architecture changes invalidate EZKL keys."
            )
        if hidden2 != self.FROZEN_HIDDEN2:
            raise RuntimeError(
                f"hidden2 must be {self.FROZEN_HIDDEN2} (frozen by ADR-007). "
                f"Got {hidden2}. Architecture changes invalidate EZKL keys."
            )
        if num_classes != self.FROZEN_NUM_CLASSES:
            raise RuntimeError(
                f"num_classes must be {self.FROZEN_NUM_CLASSES} — matches "
                f"NUMCLASSES in trainer.py (SENTINEL-SPEC §8.1). "
                f"Got {num_classes}. Architecture changes invalidate EZKL keys."
            )

        # ------------------------------------------------------------------
        # Network — three linear layers, deliberately minimal
        # ------------------------------------------------------------------
        # Raw logits out — no Sigmoid in network.
        # Sigmoid applied externally for probabilities / loss.
        # EZKL circuit is derived from raw logit computation.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),   # 128 → 64
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),     # 64  → 32
            nn.ReLU(),
            nn.Linear(hidden2, num_classes), # 32  → 10 (raw logits)
        )

        # ------------------------------------------------------------------
        # Parameter count enforcement — RuntimeError not assert (ADR-019)
        # ------------------------------------------------------------------
        total_params = self.parameter_count()
        if total_params > EZKL_PARAM_LIMIT:
            raise RuntimeError(
                f"ProxyModel has {total_params:,} parameters — "
                f"exceeds EZKL limit of {EZKL_PARAM_LIMIT:,}. "
                f"Reduce architecture before proceeding. "
                f"Current: Linear({input_dim}→{hidden1}→{hidden2}→{num_classes})"
            )

        logger.info(
            f"ProxyModel initialised — "
            f"{total_params:,} parameters | "
            f"EZKL limit: {EZKL_PARAM_LIMIT:,} | "
            f"Circuit: {CIRCUIT_VERSION} | "
            f"Architecture: Linear({input_dim}→{hidden1}→{hidden2}→{num_classes})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map 128-dim teacher features to 10 class risk logits.

        Args:
            x: CrossAttentionFusion output from teacher, shape [B, 128]

        Returns:
            Raw logits, shape [B, 10] — one per vulnerability class.
            Apply torch.sigmoid() externally for probabilities.
        """
        return self.network(x)

    def parameter_count(self) -> int:
        """Return total parameter count — used to verify ZK compatibility."""
        return sum(p.numel() for p in self.parameters())

    def circuit_version(self) -> str:
        """Return circuit version string — use to detect key mismatches."""
        return CIRCUIT_VERSION


if __name__ == "__main__":
    proxy = ProxyModel()
    proxy.eval()

    fake_features = torch.randn(4, 128)
    logits = proxy(fake_features)
    scores = torch.sigmoid(logits)

    print(f"Input shape:     {fake_features.shape}")   # [4, 128]
    print(f"Output shape:    {logits.shape}")           # [4, 10]
    print(f"Score range:     [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Total params:    {proxy.parameter_count():,}")
    print(f"Circuit version: {proxy.circuit_version()}")  # v2.0

    # Test freeze guard — must raise RuntimeError
    try:
        bad_proxy = ProxyModel(input_dim=64)
        print("ERROR: freeze guard failed")
    except RuntimeError as e:
        print(f"Freeze guard working: {e}")
