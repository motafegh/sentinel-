"""
generate_calibration.py — Extract calibration data for EZKL Step 2

Extracts 128-dim CrossAttentionFusion features from the teacher model
on the val split of the v2 export (SentinelDataset, replaces deleted DualPathDataset).

Output: zkml/ezkl/calibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.datasets.sentinel_dataset import SentinelDataset
from ml.src.datasets.collate import sentinel_collate_fn
from ml.src.models.sentinel_model import SentinelModel
from torch_geometric.loader import DataLoader

TEACHER_CHECKPOINT  = "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
EXPORT_DIR          = "data_module/data/exports/sentinel-v2-baseline-2026-06-12"
CALIBRATION_OUTPUT  = "zkml/ezkl/calibration.json"
N_CALIBRATION_SAMPLES = 200


@torch.no_grad()
def generate(
    n_samples: int = N_CALIBRATION_SAMPLES,
    output:    str = CALIBRATION_OUTPUT,
    device:    str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    logger.info(f"Generating {n_samples} calibration samples on: {device}")

    checkpoint = torch.load(TEACHER_CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config     = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config     = {}
    teacher = SentinelModel(
        num_classes=config.get("num_classes", 10),
        fusion_output_dim=config.get("fusion_output_dim", 128),
        gnn_prefix_k=config.get("gnn_prefix_k", 0),
        gnn_prefix_warmup_epochs=config.get("gnn_prefix_warmup_epochs", 15),
        use_edge_attr=config.get("use_edge_attr", True),
        gnn_hidden_dim=config.get("gnn_hidden_dim", 256),
        gnn_num_layers=config.get("gnn_layers", 8),
        gnn_heads=config.get("gnn_heads", 8),
        gnn_dropout=config.get("gnn_dropout", 0.2),
        gnn_use_jk=config.get("gnn_use_jk", True),
        gnn_jk_mode=config.get("gnn_jk_mode", "attention"),
        fusion_max_nodes=config.get("fusion_max_nodes", 1024),
    ).to(device)
    teacher.load_state_dict(state_dict)
    teacher.float()
    teacher.eval()
    logger.info(f"Teacher loaded — num_classes={config.get('num_classes',10)}")

    val_dataset = SentinelDataset(split="val", export_dir=EXPORT_DIR)
    n_available = min(n_samples, len(val_dataset))
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                         collate_fn=sentinel_collate_fn)
    logger.info(f"Val set — {len(val_dataset)} contracts, using {n_available}")

    all_features = []
    for graphs, tokens, y, cids, tiers in loader:
        graphs = graphs.to(device)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        _, aux = teacher(graphs, input_ids, attention_mask, return_aux=True)
        fusion = aux["fusion_embedding"].cpu()  # [B, 128]
        all_features.append(fusion)
        if sum(f.shape[0] for f in all_features) >= n_samples:
            break

    features_tensor = torch.cat(all_features)[:n_samples]
    logger.info(f"Features — shape: {features_tensor.shape}")

    calibration_data = {"input_data": features_tensor.numpy().tolist()}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(calibration_data, f)

    flat = features_tensor.numpy().flatten()
    logger.info(f"Calibration saved: {n_samples} x 128 | "
                f"range [{flat.min():.4f}, {flat.max():.4f}] | "
                f"mean {flat.mean():.4f}")


if __name__ == "__main__":
    generate()
