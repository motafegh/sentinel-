"""
sentinel_model.py — SENTINEL Three-Eye Model (v5 architecture)

V5 CHANGES FROM V4
───────────────────
Three-eye classifier architecture: instead of routing everything through a
single 128-dim fused bottleneck, the classifier receives three independent
128-dim vectors — one from each modality — concatenated to [B, 384].

  GNN eye         (structural opinion):
    global_max_pool(node_embs, batch)  → [B, 128]
    global_mean_pool(node_embs, batch) → [B, 128]
    cat                                → [B, 256]
    gnn_eye_proj  Linear(256,128)+ReLU+Dropout → [B, 128]

  Transformer eye (semantic opinion):
    token_embs[:, 0, :]   → CLS token   [B, 768]
    transformer_eye_proj  Linear(768,128)+ReLU+Dropout → [B, 128]

  Fused eye       (joint structural+semantic opinion):
    CrossAttentionFusion(node_embs, token_embs) → [B, 128]

  Classifier:
    cat([gnn_eye, transformer_eye, fused_eye])  → [B, 384]
    Linear(384, num_classes)                    → raw logits [B, num_classes]

Auxiliary heads (training only — prevents eye dominance):
  aux_gnn         = Linear(128, num_classes)(gnn_eye)         → [B, num_classes]
  aux_transformer = Linear(128, num_classes)(transformer_eye) → [B, num_classes]
  aux_fused       = Linear(128, num_classes)(fused_eye)       → [B, num_classes]

  forward(..., return_aux=True)  returns (logits, {"gnn": ..., "transformer": ..., "fused": ...})
  forward(..., return_aux=False) returns logits only [DEFAULT — zero inference overhead]

  Trainer loss: main_loss + λ * (loss_gnn + loss_transformer + loss_fused)
  λ=0.1 keeps each eye's gradient signal alive even if the main classifier
  learns to weight one eye heavily.  Auxiliary heads add ~3.9K parameters.

WHAT DID NOT CHANGE
───────────────────
- CrossAttentionFusion implementation (only its output is now one of three eyes)
- TransformerEncoder (CodeBERT frozen, LoRA adapters on Q+V)
- No Sigmoid inside model — applied externally in predictor and BCEWithLogitsLoss
- Checkpoint format: {"model", "optimizer", "epoch", "best_f1", "config"}
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool, global_mean_pool

from ml.src.models.fusion_layer import CrossAttentionFusion
from ml.src.models.gnn_encoder import GNNEncoder
from ml.src.models.transformer_encoder import TransformerEncoder


class SentinelModel(nn.Module):
    """
    Three-eye smart contract vulnerability detection model (v5).

    See module docstring for the full architecture description.

    Args:
        num_classes:          10 for Track 3 multi-label (default).
        fusion_output_dim:    Width of each eye output = fused output_dim (default: 128).
        dropout:              Dropout for eye projections and classifier (default: 0.3).

        --- GNN hyperparameters ---
        gnn_hidden_dim:       GNN node embedding width (default: 128).
        gnn_heads:            GAT attention heads (default: 8).
        gnn_dropout:          GNN attention + node dropout (default: 0.2).
        use_edge_attr:        Feed edge type embeddings into GATConv (default: True).
        gnn_edge_emb_dim:     Dimension of learned edge-type embedding (default: 32).

        --- LoRA hyperparameters ---
        lora_r:               LoRA rank (default: 16 in v5; was 8 in v4).
        lora_alpha:           LoRA scale (default: 32; effective scale = alpha/r = 2.0).
        lora_dropout:         LoRA path dropout (default: 0.1).
        lora_target_modules:  Attention projections to adapt (default: ["query","value"]).
    """

    def __init__(
        self,
        fusion_output_dim:    int                 = 128,
        dropout:              float               = 0.3,
        num_classes:          int                 = 10,
        # GNN architecture
        gnn_hidden_dim:       int                 = 128,
        gnn_heads:            int                 = 8,
        gnn_dropout:          float               = 0.2,
        use_edge_attr:        bool                = True,
        gnn_edge_emb_dim:     int                 = 32,
        # LoRA architecture
        lora_r:               int                 = 16,
        lora_alpha:           int                 = 32,
        lora_dropout:         float               = 0.1,
        lora_target_modules:  Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.num_classes   = num_classes
        self.use_edge_attr = use_edge_attr
        eye_dim = fusion_output_dim  # all three eyes output this width

        # ── Sub-modules ────────────────────────────────────────────────────
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
            node_dim=gnn_hidden_dim,   # must match GNNEncoder.hidden_dim
            token_dim=768,
            attn_dim=256,
            num_heads=8,
            output_dim=fusion_output_dim,
            dropout=dropout,
        )

        # ── Eye projections ────────────────────────────────────────────────
        # GNN eye: max+mean pool → [B, 2*gnn_hidden_dim] → [B, eye_dim]
        self.gnn_eye_proj = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim, eye_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Transformer eye: CLS token → [B, 768] → [B, eye_dim]
        self.transformer_eye_proj = nn.Sequential(
            nn.Linear(768, eye_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Main classifier ────────────────────────────────────────────────
        # All three eyes concatenated: [B, 3*eye_dim] → [B, num_classes]
        # No Sigmoid — applied externally.
        self.classifier = nn.Linear(3 * eye_dim, num_classes)

        # ── Auxiliary heads (training only — eye-independence enforcement) ──
        # Each head produces independent logits for its eye so the loss keeps
        # that eye's gradient alive even if the main classifier learns to
        # downweight it.  Discarded at inference via return_aux=False.
        self.aux_gnn         = nn.Linear(eye_dim, num_classes)
        self.aux_transformer = nn.Linear(eye_dim, num_classes)
        self.aux_fused       = nn.Linear(eye_dim, num_classes)

        logger.info(
            f"SentinelModel v5 (three-eye) initialised | "
            f"num_classes={num_classes} | eye_dim={eye_dim} | "
            f"classifier_in={3 * eye_dim} | "
            f"gnn_hidden={gnn_hidden_dim} heads={gnn_heads} | "
            f"lora_r={lora_r} lora_alpha={lora_alpha}"
        )

    def forward(
        self,
        graphs:         Batch,                   # PyG Batch — batched contract graphs
        input_ids:      torch.Tensor,            # [B, 512]
        attention_mask: torch.Tensor,            # [B, 512] — 1=real token, 0=PAD
        return_aux:     bool = False,            # True during training only
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Three-eye forward pass.

        Args:
            graphs:         Batched PyG graph (from DataLoader via Batch.from_data_list).
            input_ids:      CodeBERT token IDs     [B, 512].
            attention_mask: CodeBERT attention mask [B, 512].
            return_aux:     When True, also return per-eye auxiliary logits for
                            the auxiliary loss computation in trainer.py.
                            Always False at inference — zero overhead.

        Returns:
            return_aux=False (default / inference):
                logits  [B, num_classes]  — raw logits, NO Sigmoid

            return_aux=True (training):
                (logits [B, num_classes],
                 {"gnn": [B, C], "transformer": [B, C], "fused": [B, C]})
        """
        # ── GNN path: node embeddings ─────────────────────────────────────
        edge_attr = getattr(graphs, "edge_attr", None) if self.use_edge_attr else None
        node_embs, batch = self.gnn(graphs.x, graphs.edge_index, graphs.batch, edge_attr)
        # node_embs: [N, gnn_hidden_dim]  batch: [N]

        # ── GNN eye: max+mean pool → project ─────────────────────────────
        # Max pool captures "is there at least one node with feature X?"
        # (existential bias — a contract is vulnerable if ANY function is).
        # Mean pool captures the "typical node" character of the contract.
        # Concatenating both lets the classifier choose its own weighting.
        gnn_max  = global_max_pool(node_embs, batch)   # [B, gnn_hidden_dim]
        gnn_mean = global_mean_pool(node_embs, batch)  # [B, gnn_hidden_dim]
        gnn_eye  = self.gnn_eye_proj(
            torch.cat([gnn_max, gnn_mean], dim=1)      # [B, 2*gnn_hidden_dim]
        )                                               # [B, eye_dim]

        # ── Transformer path ──────────────────────────────────────────────
        token_embs = self.transformer(input_ids, attention_mask)
        # token_embs: [B, 512, 768]

        # ── Transformer eye: CLS token → project ─────────────────────────
        # CLS token at position 0 is CodeBERT's hierarchical sequence summary
        # (12-layer bidirectional self-attention over all 512 positions).
        # It is order-aware and distinct from the masked-mean pool inside fusion.
        transformer_eye = self.transformer_eye_proj(
            token_embs[:, 0, :]   # [B, 768]
        )                          # [B, eye_dim]

        # ── Fused eye: cross-attention fusion ────────────────────────────
        # Encodes joint evidence that neither modality holds alone:
        # "which structural patterns co-occur with which token sequences?"
        fused_eye = self.fusion(node_embs, batch, token_embs, attention_mask)
        # fused_eye: [B, eye_dim]

        # ── Main classifier ───────────────────────────────────────────────
        combined = torch.cat([gnn_eye, transformer_eye, fused_eye], dim=1)  # [B, 3*eye_dim]
        logits   = self.classifier(combined)  # [B, num_classes]

        if self.num_classes == 1:
            logits = logits.squeeze(1)  # [B,1] → [B] for binary BCEWithLogitsLoss

        logger.debug(
            f"SentinelModel forward — nodes: {node_embs.shape} | "
            f"tokens: {token_embs.shape} | "
            f"gnn_eye: {gnn_eye.shape} | tf_eye: {transformer_eye.shape} | "
            f"fused_eye: {fused_eye.shape} | logits: {logits.shape}"
        )

        if not return_aux:
            return logits

        # ── Auxiliary heads (training only) ───────────────────────────────
        aux = {
            "gnn":         self.aux_gnn(gnn_eye),          # [B, num_classes]
            "transformer": self.aux_transformer(transformer_eye),
            "fused":       self.aux_fused(fused_eye),
        }
        return logits, aux

    def parameter_summary(self) -> None:
        """Log trainable vs frozen parameter counts per sub-module."""
        components = {
            "GNNEncoder":            self.gnn,
            "TransformerEncoder":    self.transformer,
            "CrossAttentionFusion":  self.fusion,
            "gnn_eye_proj":          self.gnn_eye_proj,
            "transformer_eye_proj":  self.transformer_eye_proj,
            "Classifier (3×eye→C)":  self.classifier,
            "aux_gnn":               self.aux_gnn,
            "aux_transformer":       self.aux_transformer,
            "aux_fused":             self.aux_fused,
        }
        total_trainable = total_frozen = 0

        for name, module in components.items():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen    = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            total_trainable += trainable
            total_frozen    += frozen
            logger.info(f"  {name}: {trainable:,} trainable | {frozen:,} frozen")

        logger.info(
            f"Total: {total_trainable:,} trainable | "
            f"{total_frozen:,} frozen | "
            f"{total_trainable + total_frozen:,} total"
        )
