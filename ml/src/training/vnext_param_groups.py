"""Parameter groups retained for the R4 Phase-8 baseline."""
from __future__ import annotations

from typing import Any

import torch

from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.vnext_phase8_config import Phase8Settings


def build_parameter_groups(model: SentinelModel, settings: Phase8Settings):
    buckets: dict[str, list[torch.nn.Parameter]] = {
        "gnn": [], "lora": [], "fusion": [], "prefix": [], "other": []
    }
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if name.startswith("gnn.") or name.startswith("gnn_eye_proj.") or name.startswith("cfg_eye_proj."):
            buckets["gnn"].append(param)
        elif "lora_" in name:
            buckets["lora"].append(param)
        elif name.startswith("fusion.") or name.startswith("transformer_eye_proj.") or name.startswith("classifier.") or name.startswith("aux_"):
            buckets["fusion"].append(param)
        elif name.startswith("gnn_to_bert_proj.") or name.startswith("prefix_type_embedding."):
            buckets["prefix"].append(param)
        else:
            buckets["other"].append(param)

    specs = [
        ("gnn", settings.lr * settings.gnn_lr_multiplier, None),
        ("lora", settings.lr * settings.lora_lr_multiplier, 0.0),
        ("fusion", settings.lr * settings.fusion_lr_multiplier, None),
        ("prefix", settings.lr * settings.prefix_proj_lr_multiplier, None),
        ("other", settings.lr, None),
    ]
    groups: list[dict[str, Any]] = []
    max_lrs: list[float] = []
    for bucket_name, lr, weight_decay in specs:
        if not buckets[bucket_name]:
            continue
        group: dict[str, Any] = {"params": buckets[bucket_name], "lr": lr, "name": bucket_name}
        if weight_decay is not None:
            group["weight_decay"] = weight_decay
        groups.append(group)
        max_lrs.append(lr)
    return groups, max_lrs


__all__ = ["build_parameter_groups"]
