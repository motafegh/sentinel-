#!/usr/bin/env python3
"""Read-only inventory of the historical Run12 checkpoint for R4 Phase 8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

DEFAULT_CHECKPOINT = Path("ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt")

ARCH_KEYS = (
    "architecture", "num_classes", "fusion_output_dim", "fusion_dropout",
    "gnn_hidden_dim", "gnn_layers", "gnn_heads", "gnn_dropout",
    "use_edge_attr", "gnn_edge_emb_dim", "gnn_use_jk", "gnn_jk_mode",
    "gnn_phase2_edge_types", "lora_r", "lora_alpha", "lora_dropout",
    "lora_target_modules", "gnn_prefix_k", "gnn_prefix_warmup_epochs",
    "fusion_max_nodes", "drop_complexity_feature", "appnp_alpha",
)

TRAINING_CONTEXT_KEYS = (
    "loss_fn", "asl_gamma_neg", "asl_gamma_pos", "asl_clip", "batch_size",
    "gradient_accumulation_steps", "lr", "weight_decay", "gnn_lr_multiplier",
    "lora_lr_multiplier", "fusion_lr_multiplier", "warmup_pct", "threshold",
    "eval_threshold", "use_weighted_sampler", "label_smoothing",
    "class_label_smoothing",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def state_dict_summary(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"present": False}
    tensors = [(str(k), v) for k, v in state.items() if torch.is_tensor(v)]
    prefixes: dict[str, int] = {}
    for key, _ in tensors:
        prefix = key.split(".", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    return {
        "present": True,
        "tensor_count": len(tensors),
        "total_tensor_elements": sum(int(v.numel()) for _, v in tensors),
        "top_level_prefix_tensor_counts": dict(sorted(prefixes.items())),
        "first_tensor_shapes": {key: list(value.shape) for key, value in tensors[:20]},
    }


def inventory(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint root must be dict, got {type(checkpoint).__name__}")

    config_raw = checkpoint.get("config") or {}
    if hasattr(config_raw, "__dict__") and not isinstance(config_raw, dict):
        config_raw = vars(config_raw)
    if not isinstance(config_raw, dict):
        config_raw = {"_raw": repr(config_raw)}

    return {
        "schema": "sentinel-r4-phase8-run12-inventory-v1",
        "checkpoint": {
            "logical_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "top_level_keys": sorted(str(k) for k in checkpoint),
        "model_version": jsonable(checkpoint.get("model_version")),
        "epoch": jsonable(checkpoint.get("epoch")),
        "best_f1": jsonable(checkpoint.get("best_f1")),
        "architecture_config": {key: jsonable(config_raw.get(key)) for key in ARCH_KEYS},
        "historical_training_context": {
            key: jsonable(config_raw.get(key))
            for key in TRAINING_CONTEXT_KEYS if key in config_raw
        },
        "full_config": jsonable(config_raw),
        "model_state": state_dict_summary(checkpoint.get("model")),
        "optimizer_state_present": isinstance(checkpoint.get("optimizer"), dict),
        "scheduler_state_present": isinstance(checkpoint.get("scheduler"), dict),
        "phase8_reuse_policy": {
            "reuse_architecture_config": True,
            "reuse_model_weights": False,
            "reuse_optimizer_state": False,
            "reuse_scheduler_state": False,
            "reuse_historical_thresholds": False,
            "reuse_historical_label_smoothing": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inventory(args.checkpoint)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
