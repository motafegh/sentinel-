"""Frozen R4 Phase-8 architecture and baseline training settings."""
from __future__ import annotations

from dataclasses import dataclass

ARCHITECTURE = "four_eye_v8"
MODEL_VERSION = "v8.1"

FROZEN_ARCHITECTURE = {
    "num_classes": 10,
    "fusion_output_dim": 128,
    "dropout": 0.3,
    "gnn_hidden_dim": 256,
    "gnn_num_layers": 8,
    "gnn_heads": 8,
    "gnn_dropout": 0.2,
    "use_edge_attr": True,
    "gnn_edge_emb_dim": 64,
    "gnn_use_jk": True,
    "gnn_jk_mode": "attention",
    "gnn_phase2_edge_types": None,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "value"],
    "gnn_prefix_k": 48,
    "gnn_prefix_warmup_epochs": 15,
    "fusion_max_nodes": 2048,
    "drop_complexity_feature": True,
    "appnp_alpha": 0.2,
}


@dataclass(frozen=True)
class Phase8Settings:
    seed: int = 20260813
    weak_positive_weight: float = 0.25
    epochs: int = 100
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-2
    warmup_pct: float = 0.10
    gnn_lr_multiplier: float = 2.5
    lora_lr_multiplier: float = 0.3
    fusion_lr_multiplier: float = 0.5
    prefix_proj_lr_multiplier: float = 5.0
    aux_loss_weight: float = 0.3
    aux_loss_warmup_epochs: int = 8
    aux_phase2_loss_weight: float = 0.2
    jk_entropy_reg_lambda: float = 0.005
    grad_clip: float = 1.0
    fixed_diagnostic_threshold: float = 0.5


__all__ = ["ARCHITECTURE", "FROZEN_ARCHITECTURE", "MODEL_VERSION", "Phase8Settings"]
