"""Export a versioned SENTINEL proxy checkpoint to ONNX.

The exporter preserves and validates checkpoint lineage when available. The
legacy V2 graph remains 128→64→32→10 and exports the direct student-score
output (no sigmoid). Opset 11 is retained for compatibility with the currently
tracked EZKL 23.0.5 V2 bundle; a future circuit version may revisit that choice
only with fresh setup/proof/verifier validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from zkml.src.distillation.proxy_model import (
    CIRCUIT_VERSION,
    OUTPUT_SEMANTICS,
    ProxyModel,
)

PROXY_CHECKPOINT = Path("zkml/models/proxy_best.pt")
ONNX_OUTPUT = Path("zkml/models/proxy.onnx")
ONNX_MANIFEST = Path("zkml/models/proxy.onnx.manifest.json")
OPSET_VERSION = 11


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_proxy(checkpoint: Path) -> tuple[ProxyModel, dict[str, Any]]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"proxy checkpoint not found: {checkpoint}")
    payload: Any = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
        raw_metadata = payload.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
    else:
        # Historical V2 checkpoint: raw state_dict, retained for compatibility.
        state_dict = payload
        metadata = {
            "schema": "legacy_raw_state_dict",
            "circuit_version": CIRCUIT_VERSION,
            "output_semantics": OUTPUT_SEMANTICS,
            "lineage_complete": False,
        }

    if metadata.get("circuit_version", CIRCUIT_VERSION) != CIRCUIT_VERSION:
        raise RuntimeError(
            f"checkpoint circuit_version={metadata.get('circuit_version')!r} "
            f"does not match exporter {CIRCUIT_VERSION!r}"
        )
    if metadata.get("output_semantics", OUTPUT_SEMANTICS) != OUTPUT_SEMANTICS:
        raise RuntimeError(
            f"checkpoint output_semantics={metadata.get('output_semantics')!r} "
            f"does not match {OUTPUT_SEMANTICS!r}"
        )

    proxy = ProxyModel().eval()
    proxy.load_state_dict(state_dict)
    return proxy, metadata


def export(
    checkpoint: Path = PROXY_CHECKPOINT,
    output: Path = ONNX_OUTPUT,
    manifest_output: Path = ONNX_MANIFEST,
) -> dict[str, Any]:
    proxy, checkpoint_metadata = _load_proxy(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Fixed trace input: deterministic export diagnostics.
    generator = torch.Generator(device="cpu").manual_seed(42)
    dummy_input = torch.randn(1, 128, generator=generator)

    torch.onnx.export(
        proxy,
        dummy_input,
        str(output),
        opset_version=OPSET_VERSION,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        do_constant_folding=True,
        dynamo=False,
    )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"ONNX export did not produce a usable file: {output}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to verify PyTorch↔ONNX parity; refusing an unverified export"
        ) from exc

    session = ort.InferenceSession(str(output))
    verify_input = torch.randn(4, 128, generator=generator)
    with torch.no_grad():
        pt_output = proxy(verify_input).numpy()
    onnx_output = session.run(["output"], {"input": verify_input.numpy()})[0]
    if tuple(onnx_output.shape) != (4, 10):
        raise RuntimeError(f"ONNX output shape must be [4,10], got {onnx_output.shape}")
    max_diff = float(np.abs(onnx_output - pt_output).max())
    if not np.isfinite(max_diff) or max_diff >= 1e-5:
        raise RuntimeError(
            f"PyTorch↔ONNX parity failed: max_diff={max_diff!r}, tolerance=1e-5"
        )

    external_data = Path(str(output) + ".data")
    manifest = {
        "schema": "sentinel-zkml-onnx-export-v1",
        "circuit_version": CIRCUIT_VERSION,
        "output_semantics": OUTPUT_SEMANTICS,
        "input_dim": 128,
        "num_classes": 10,
        "opset_version": OPSET_VERSION,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": _sha256(checkpoint),
            "metadata": checkpoint_metadata,
        },
        "onnx": {
            "path": output.as_posix(),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "onnx_external_data": None,
        "verification": {
            "batch_shape": [4, 128],
            "output_shape": [4, 10],
            "max_abs_diff": max_diff,
            "tolerance": 1e-5,
            "passed": True,
        },
    }
    if external_data.exists():
        manifest["onnx_external_data"] = {
            "path": external_data.as_posix(),
            "sha256": _sha256(external_data),
            "size_bytes": external_data.stat().st_size,
        }

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("ONNX export verified and bound: {}", manifest_output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROXY_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ONNX_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=ONNX_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export(args.checkpoint, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
