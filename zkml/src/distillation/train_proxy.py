"""
train_proxy.py — Knowledge Distillation Training Loop

Trains the ProxyModel (student) to mimic the full SentinelModel (teacher).
Uses SentinelDataset (replaces deleted DualPathDataset) to load the v2 export.

Usage:
    cd ~/projects/sentinel
    source ml/.venv/bin/activate
    python zkml/src/distillation/train_proxy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.datasets.sentinel_dataset import SentinelDataset
from ml.src.datasets.collate import sentinel_collate_fn
from ml.src.models.sentinel_model import SentinelModel
from zkml.src.distillation.proxy_model import CIRCUIT_VERSION, ProxyModel

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

TEACHER_CHECKPOINT = "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
PROXY_CHECKPOINT   = "zkml/models/proxy_best.pt"
EXPORT_DIR         = "data_module/data/exports/sentinel-v2-baseline-2026-06-12"

BATCH_SIZE       = 64
EPOCHS           = 50
LR               = 1e-3
AGREEMENT_TARGET = 0.95
THRESHOLD        = 0.50
RANDOM_SEED      = 42


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    teacher: SentinelModel,
    graphs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 128-dim fusion embedding and per-class teacher scores via model.forward."""
    teacher.eval()
    graphs         = graphs.to(device)
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits, aux = teacher(graphs, input_ids, attention_mask, return_aux=True)
    features = aux["fusion_embedding"]  # [B, 128]
    scores = torch.sigmoid(logits.float())  # [B, 10]
    return features.cpu(), scores.cpu()


# ------------------------------------------------------------------
# Agreement rate
# ------------------------------------------------------------------

def compute_agreement(
    proxy_scores: torch.Tensor,    # [B, 10]
    teacher_scores: torch.Tensor,  # [B, 10]
    threshold: float = THRESHOLD,
) -> float:
    proxy_labels   = (proxy_scores   >= threshold).long()
    teacher_labels = (teacher_scores >= threshold).long()
    matches = (proxy_labels == teacher_labels).float()
    return matches.mean().item()


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    logger.info(f"Starting proxy distillation on: {device}")
    logger.info(f"Teacher: {TEACHER_CHECKPOINT} | Circuit: {CIRCUIT_VERSION}")

    # ── Load teacher ──────────────────────────────────────────────────
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
    teacher.float()  # Normalize BF16 AMP checkpoints to float32 (matches Predictor)
    teacher.eval()
    logger.info(f"Teacher loaded — num_classes={config.get('num_classes',10)}")

    # ── Load data (SentinelDataset, replaces deleted DualPathDataset) ──
    train_dataset = SentinelDataset(split="train", export_dir=EXPORT_DIR)
    val_dataset   = SentinelDataset(split="val",   export_dir=EXPORT_DIR)
    teacher_train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, collate_fn=sentinel_collate_fn)
    teacher_val_loader   = PyGDataLoader(val_dataset,   batch_size=BATCH_SIZE,
                                          shuffle=False, collate_fn=sentinel_collate_fn)
    logger.info(f"Data — train={len(train_dataset)} val={len(val_dataset)}")

    # ── Extract teacher features once ─────────────────────────────────
    logger.info("Extracting teacher features (one-time)...")
    train_features_list, train_scores_list = [], []
    val_features_list,   val_scores_list   = [], []

    for graphs, tokens, y, cids, tiers in teacher_train_loader:
        feats, scores = extract_features(teacher, graphs,
                                         tokens["input_ids"], tokens["attention_mask"], device)
        train_features_list.append(feats)
        train_scores_list.append(scores)

    for graphs, tokens, y, cids, tiers in teacher_val_loader:
        feats, scores = extract_features(teacher, graphs,
                                         tokens["input_ids"], tokens["attention_mask"], device)
        val_features_list.append(feats)
        val_scores_list.append(scores)

    train_features       = torch.cat(train_features_list)    # [N_train, 128]
    train_teacher_scores = torch.cat(train_scores_list)      # [N_train, 10]
    val_features         = torch.cat(val_features_list)      # [N_val, 128]
    val_teacher_scores   = torch.cat(val_scores_list)        # [N_val, 10]
    logger.info(f"Features — train: {train_features.shape}, val: {val_features.shape}")

    # ── Proxy DataLoaders from cached features ────────────────────────
    proxy_train_dataset = TensorDataset(train_features, train_teacher_scores)
    proxy_val_dataset   = TensorDataset(val_features,   val_teacher_scores)
    proxy_train_loader = DataLoader(proxy_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    proxy_val_loader   = DataLoader(proxy_val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Train proxy ───────────────────────────────────────────────────
    proxy     = ProxyModel().to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(proxy.parameters(), lr=LR)
    best_agreement = 0.0
    Path("zkml/models").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        proxy.train()
        train_loss = 0.0
        for features_batch, teacher_scores_batch in proxy_train_loader:
            features_batch = features_batch.to(device)
            teacher_scores_batch = teacher_scores_batch.to(device)
            optimiser.zero_grad()
            loss = criterion(proxy(features_batch), teacher_scores_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(proxy_train_loader)

        proxy.eval()
        all_proxy_scores   = []
        all_teacher_scores = []
        with torch.no_grad():
            for features_batch, teacher_scores_batch in proxy_val_loader:
                all_proxy_scores.append(proxy(features_batch.to(device)).cpu())
                all_teacher_scores.append(teacher_scores_batch)
        all_proxy_scores   = torch.cat(all_proxy_scores)
        all_teacher_scores = torch.cat(all_teacher_scores)
        agreement = compute_agreement(all_proxy_scores, all_teacher_scores)

        logger.info(f"Epoch {epoch:>3}/{EPOCHS} | Loss: {train_loss:.6f} | "
                    f"Agreement: {agreement:.4f} "
                    f"({'TARGET MET' if agreement >= AGREEMENT_TARGET else f'target: {AGREEMENT_TARGET:.0%}'})")
        if agreement > best_agreement:
            best_agreement = agreement
            torch.save(proxy.state_dict(), PROXY_CHECKPOINT)
        if agreement >= AGREEMENT_TARGET:
            logger.info(f"Target reached at epoch {epoch}.")
            break

    logger.info(f"Distillation complete — best: {best_agreement:.4f} | circuit: {CIRCUIT_VERSION}")


if __name__ == "__main__":
    train()
