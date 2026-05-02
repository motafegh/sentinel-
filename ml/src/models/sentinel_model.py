"""
sentinel_model.py — SENTINEL Dual-Path Model (Cross-Attention Upgrade)

WHAT CHANGED FROM ORIGINAL:
    1. GNNEncoder forward() returns (node_embs [N,64], batch [N]) — unpacked here
    2. TransformerEncoder forward() returns [B, 512, 768] — all tokens
    3. CrossAttentionFusion replaces FusionLayer — receives node_embs, batch,
       token_embs, AND attention_mask (now required so fusion can mask PAD tokens)
    4. Fusion output: [B, 128] instead of [B, 64]
    5. Classifier: Linear(128, num_classes)
    6. fusion_output_dim default: 128

REVIEW FIXES APPLIED:
    #3  num_classes default changed from 1 → 10.
        The model is a Track 3 multi-label detector; defaulting to 1 silently
        built a binary head when num_classes was forgotten. Callers that need
        binary mode must now pass num_classes=1 explicitly.

WHAT DID NOT CHANGE:
    - NO Sigmoid inside model — applied externally (predictor.py for inference;
      BCEWithLogitsLoss applies it internally during training)
    - Binary mode: squeeze(1) on [B,1] → [B]
    - Multi-label mode: [B, num_classes] kept as-is
    - parameter_summary() helper
    - Checkpoint format: {"model", "optimizer", "epoch", "best_f1", "config"}
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Batch

from ml.src.models.gnn_encoder import GNNEncoder
from ml.src.models.transformer_encoder import TransformerEncoder
from ml.src.models.fusion_layer import CrossAttentionFusion


class SentinelModel(nn.Module):
    """
    Dual-path smart contract vulnerability detection model.

    Two parallel paths:
        GNN path:         contract graph  → node embeddings [N, gnn_hidden_dim]
        Transformer path: token sequence  → all token embeddings [B, 512, 768]

    CrossAttentionFusion:
        Nodes attend to real tokens (PAD positions masked).
        Tokens attend to real nodes (padding positions masked).
        Both pools use masked mean. Pool AFTER enrichment.
        Output: [B, fusion_output_dim]

    Classifier:
        Linear(fusion_output_dim, num_classes) → raw logits (no Sigmoid).
        Sigmoid applied externally.

    Args:
        num_classes:          10 for Track 3 multi-label (default).
                              Pass num_classes=1 explicitly for binary mode.
        fusion_output_dim:    Width of the fused representation (default: 128).
        dropout:              Shared dropout for fusion + classifier (default: 0.3).

        --- GNN hyperparameters ---
        gnn_hidden_dim:       GNN node embedding width (default: 64).
        gnn_heads:            GAT attention heads (default: 8).
        gnn_dropout:          GNN attention + node dropout (default: 0.2).
        use_edge_attr:        Feed edge type embeddings into GATConv (default: True).
        gnn_edge_emb_dim:     Dimension of learned edge-type embedding (default: 16).

        --- LoRA hyperparameters (forwarded to TransformerEncoder) ---
        lora_r:               LoRA rank (default: 8).
        lora_alpha:           LoRA scale alpha (default: 16; effective scale = alpha/r).
        lora_dropout:         LoRA path dropout (default: 0.1).
        lora_target_modules:  Which attention projections to adapt (default: ["query","value"]).
    """

    def __init__(
        self,
        fusion_output_dim:    int            = 128,
        dropout:              float          = 0.3,
        # Fix #3: default changed from 1 → 10.
        num_classes:          int            = 10,
        # GNN architecture
        gnn_hidden_dim:       int            = 64,
        gnn_heads:            int            = 8,
        gnn_dropout:          float          = 0.2,
        use_edge_attr:        bool           = True,
        gnn_edge_emb_dim:     int            = 16,
        # LoRA architecture
        lora_r:               int            = 8,
        lora_alpha:           int            = 16,
        lora_dropout:         float          = 0.1,
        lora_target_modules:  Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.num_classes  = num_classes
        self.use_edge_attr = use_edge_attr

        self.gnn = GNNEncoder(
            hidden_dim=gnn_hidden_dim,
            heads=gnn_heads,
            dropout=gnn_dropout,
            use_edge_attr=use_edge_attr,
            edge_emb_dim=gnn_edge_emb_dim,
        )
        self.transformer = TransformerEncoder(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        self.fusion = CrossAttentionFusion(
            node_dim=gnn_hidden_dim,   # must match GNNEncoder output width
            token_dim=768,
            attn_dim=256,
            num_heads=8,
            output_dim=fusion_output_dim,
            dropout=dropout,
        )

        # Classifier head — raw logits only, NO Sigmoid.
        self.classifier = nn.Linear(fusion_output_dim, num_classes)

        logger.info(
            f"SentinelModel initialised — cross-attention architecture | "
            f"num_classes={num_classes} | fusion_output={fusion_output_dim} | "
            f"gnn_hidden={gnn_hidden_dim} heads={gnn_heads} edge_attr={use_edge_attr} | "
            f"lora_r={lora_r} lora_alpha={lora_alpha}"
        )

    def forward(
        self,
        graphs:         Batch,          # PyG Batch — batched contract graphs
        input_ids:      torch.Tensor,   # [B, 512] — CodeBERT token IDs
        attention_mask: torch.Tensor,   # [B, 512] — 1=real token, 0=PAD
    ) -> torch.Tensor:
        """
        Full forward pass: contract graph + tokens → vulnerability logits.

        attention_mask is forwarded all the way into CrossAttentionFusion so
        the PAD token positions are masked out of both cross-attention directions
        and the token masked-mean pool (review fixes #2 and #6).

        Returns:
            Binary mode   (num_classes=1):  [B]     raw logits
            Multi-label   (num_classes>1):  [B, 10] raw logits
        """
        # ── GNN path ──────────────────────────────────────────────────────
        # Returns un-pooled node embeddings so each node can query tokens
        # BEFORE pooling loses node-level resolution.
        edge_attr = getattr(graphs, "edge_attr", None) if self.use_edge_attr else None
        node_embs, batch = self.gnn(graphs.x, graphs.edge_index, graphs.batch, edge_attr)
        # node_embs: [N, gnn_hidden_dim]  — N = total nodes across B contracts
        # batch:     [N]                  — maps each node to its contract index 0…B-1

        # ── Transformer path ───────────────────────────────────────────────
        # Returns all 512 token positions, not just the CLS summary.
        token_embs = self.transformer(input_ids, attention_mask)
        # token_embs: [B, 512, 768]

        # ── Cross-attention fusion ─────────────────────────────────────────
        # attention_mask is threaded in so fusion can mask PAD token keys
        # (fix #2) and compute masked token pooling (fix #6).
        # Pooling happens INSIDE fusion, AFTER cross-attention enrichment.
        fused = self.fusion(node_embs, batch, token_embs, attention_mask)
        # fused: [B, 128]

        # ── Classifier ────────────────────────────────────────────────────
        logits = self.classifier(fused)  # [B, num_classes]

        # Binary mode: squeeze [B, 1] → [B] for BCEWithLogitsLoss compatibility
        if self.num_classes == 1:
            logits = logits.squeeze(1)

        logger.debug(
            f"SentinelModel forward — "
            f"nodes: {node_embs.shape} | tokens: {token_embs.shape} | "
            f"fused: {fused.shape} | logits: {logits.shape}"
        )

        return logits

    def parameter_summary(self) -> None:
        """Log trainable vs frozen parameter counts per sub-module."""
        components = {
            "GNNEncoder":           self.gnn,
            "TransformerEncoder":   self.transformer,
            "CrossAttentionFusion": self.fusion,
            "Classifier":           self.classifier,
        }
        total_trainable = total_frozen = 0

        for name, module in components.items():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen    = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            total_trainable += trainable
            total_frozen    += frozen
            logger.info(f"{name}: {trainable:,} trainable | {frozen:,} frozen")

        logger.info(
            f"Total: {total_trainable:,} trainable | "
            f"{total_frozen:,} frozen | "
            f"{total_trainable + total_frozen:,} total"
        )
