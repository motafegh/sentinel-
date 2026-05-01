"""
export_onnx.py — Export SENTINEL proxy model to ONNX format

RECALL — why this file exists:
    The trained proxy lives in proxy_best.pt — PyTorch format.
    EZKL cannot read .pt files. It speaks ONNX only.
    This file bridges the two worlds:
        proxy_best.pt → export_onnx.py → proxy.onnx → EZKL pipeline

RECALL — what ONNX contains:
    1. The computation graph (structure — what operations happen in order)
    2. The trained weights (values — become the private witness in ZK)
    Once exported, any ONNX-compatible tool can run the model —
    EZKL, C++, mobile, cloud — without needing PyTorch installed.

RECALL — why eval() mode before export:
    Export traces every operation that executes during the forward pass.
    train() mode: Dropout randomly zeros activations → random operations
    ZK circuits must be deterministic — same input = same output always.
    eval() removes Dropout from the traced graph entirely → deterministic ✓

RECALL — why opset_version=11 is non-negotiable:
    EZKL's circuit compiler understands specific ONNX operation variants.
    Those variants are stable in opset 11.
    Higher opsets introduce new variants EZKL cannot convert to constraints.
    Symptom if wrong: compile_model fails with "unsupported op" error.

RECALL — why dummy_input shape is (1, 128):
    PyTorch traces by running the model once on a sample input.
    128 = CrossAttentionFusion output dim — fixed by ADR-025, CIRCUIT_VERSION v2.0.
    1   = batch size for the trace (dynamic_axes handles variable B later).
    Wrong shape here = wrong input expectation in ONNX = EZKL fails.

RECALL — why do_constant_folding=True:
    Pre-computes static operations that don't depend on input.
    Fewer operations → fewer ZK constraints → faster proving → less gas.

Output:
    zkml/models/proxy.onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from zkml.src.distillation.proxy_model import CIRCUIT_VERSION, ProxyModel

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROXY_CHECKPOINT = "zkml/models/proxy_best.pt"
ONNX_OUTPUT      = "zkml/models/proxy.onnx"


def export(
    checkpoint: str = PROXY_CHECKPOINT,
    output:     str = ONNX_OUTPUT,
) -> None:
    """
    Load trained proxy and export to ONNX format for EZKL pipeline.

    Args:
        checkpoint: Path to trained proxy .pt checkpoint
        output:     Path to write .onnx file

    Raises:
        FileNotFoundError: if checkpoint does not exist
        AssertionError:    if ONNX output does not match PyTorch output
    """
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Proxy checkpoint not found: {checkpoint}. "
            f"Run train_proxy.py first."
        )

    logger.info(f"Exporting proxy to ONNX — circuit: {CIRCUIT_VERSION}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Output:     {output}")

    # ------------------------------------------------------------------
    # Step 1 — Load trained proxy
    # ------------------------------------------------------------------
    proxy = ProxyModel()
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",  # export on CPU — ONNX export doesn't need GPU
        weights_only=True,
    )
    proxy.load_state_dict(state_dict)

    # RECALL — eval() removes Dropout from the computation graph.
    # ZK circuits cannot contain randomness — the proof must guarantee
    # "I ran exactly this deterministic computation on these inputs."
    # Dropout breaks that guarantee. eval() fixes it.
    proxy.eval()
    logger.info("Proxy in eval() mode — Dropout removed from graph")

    # ------------------------------------------------------------------
    # Step 2 — Create dummy input for tracing
    # ------------------------------------------------------------------
    # RECALL — PyTorch traces by running the model once on this input.
    # Shape (1, 128):
    #   1   = single contract for the trace
    #   128 = CrossAttentionFusion output dim (ADR-025, CIRCUIT_VERSION v2.0)
    # This shape is what EZKL will expect at gen_settings time.
    dummy_input = torch.randn(1, 128)
    logger.info(f"Dummy input shape: {dummy_input.shape}")

    # ------------------------------------------------------------------
    # Step 3 — Export to ONNX
    # ------------------------------------------------------------------
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        proxy,
        dummy_input,
        output,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        do_constant_folding=True,

        # RECALL — why dynamo=False:
        # New PyTorch defaults to dynamo=True export path which
        # no longer supports opset downconversion below 18.
        # EZKL 23.0.5 requires opset 11 — legacy export path
        # correctly produces it. If EZKL upgrades to support
        # opset 18 in future, revisit this decision.
        dynamo=False,
    )

    file_size_kb = Path(output).stat().st_size / 1024
    logger.info(f"ONNX export complete — {file_size_kb:.1f} KB: {output}")

    # ------------------------------------------------------------------
    # Step 4 — Verify export correctness
    # ------------------------------------------------------------------
    # Run the same input through both the PyTorch model and the ONNX file.
    # If outputs match within tolerance — export is valid.
    # RECALL — why verify:
    #   Silent export bugs (wrong opset, wrong shape) produce an ONNX file
    #   that loads fine but computes wrong results. Catching it here costs
    #   seconds. Catching it at EZKL compile_model costs minutes and produces
    #   cryptic errors far from the actual cause.
    logger.info("Verifying export — comparing PyTorch vs ONNX outputs...")

    import onnxruntime as ort

    session = ort.InferenceSession(output)

    # Use a fixed verification input — reproducible check
    verify_input = torch.randn(4, 128)  # batch of 4 for a stronger check

    # PyTorch output
    with torch.no_grad():
        pt_output = proxy(verify_input).numpy()

    # ONNX output
    onnx_output = session.run(
        ["output"],
        {"input": verify_input.numpy()},
    )[0]

    max_diff = np.abs(onnx_output - pt_output).max()
    logger.info(f"Max output difference PyTorch vs ONNX: {max_diff:.8f}")

    # Tolerance: 1e-5 is strict enough to catch real errors,
    # loose enough to allow for float32 precision differences
    assert max_diff < 1e-5, (
        f"ONNX verification failed — max diff {max_diff:.8f} exceeds 1e-5. "
        f"Check opset_version (must be 11) and eval() mode."
    )

    logger.info("Verification passed — ONNX output matches PyTorch ✓")
    logger.info(f"Circuit version: {CIRCUIT_VERSION}")
    logger.info("Ready for EZKL pipeline — next: gen_settings")


if __name__ == "__main__":
    export()