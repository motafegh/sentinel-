"""
sentinel_model.py — SENTINEL Three-Eye Model (v5 architecture)

V5 CHANGES FROM V4
───────────────────
Three-eye classifier architecture: instead of routing everything through a
single 128-dim fused bottleneck, the classifier receives three independent
128-dim vectors — one from each modality — concatenated to [B, 384].
GNN is now a three-phase, four-layer GAT (see gnn_encoder.py) that encodes
execution order via CFG CONTROL_FLOW edges and propagates it back up to
function nodes via reversed CONTAINS edges (Phase 3).

  GNN eye         (structural opinion):
    Pool over FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes only.
    After Phase 3 (reverse-CONTAINS), these nodes carry aggregated CFG signal.
    Pooling over all nodes was dominated by CFG_RETURN (77% of CFG node mass),
    drowning the CFG_CALL/WRITE/COND signal that encodes execution order.
    Falls back to all-node pool if no function-level nodes exist (ghost graphs).
    global_max_pool(func_embs, func_batch)  → [B, 128]
    global_mean_pool(func_embs, func_batch) → [B, 128]
    cat                                     → [B, 256]
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
  λ=0.3 keeps each eye's gradient signal alive even if the main classifier
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
from ml.src.preprocessing.graph_schema import NODE_TYPES

# ── Schema-derived constant (single source of truth) ──────────────────────────
# Derived from graph_schema.NODE_TYPES so that adding a new node type here
# automatically propagates to the model's denormalisation without manual edits.
# node feature[0] is stored as float(type_id) / _MAX_TYPE_ID in the extractor;
# multiply back by _MAX_TYPE_ID and round to recover the integer type_id.
_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 12.0 for v2 schema (ids 0–12)

# Pooling target: function-declaration node types only.
# After Phase 3 reverse-CONTAINS, these nodes carry aggregated CFG signal.
_FUNC_TYPE_IDS: frozenset[int] = frozenset({  # FUNCTION MODIFIER FALLBACK RECEIVE CONSTRUCTOR
    NODE_TYPES["FUNCTION"],
    NODE_TYPES["MODIFIER"],
    NODE_TYPES["FALLBACK"],
    NODE_TYPES["RECEIVE"],
    NODE_TYPES["CONSTRUCTOR"],
})


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
        gnn_num_layers:       Number of GAT layers (default: 4; validated in TrainConfig).
        gnn_heads:            GAT attention heads for Phase 1 (default: 8).
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
        gnn_num_layers:       int                 = 4,
        gnn_heads:            int                 = 8,
        gnn_dropout:          float               = 0.2,
        use_edge_attr:        bool                = True,
        gnn_edge_emb_dim:     int                 = 32,
        # JK connections (Phase 1-A1, 2026-05-14)
        gnn_use_jk:           bool                = True,
        gnn_jk_mode:          str                 = 'attention',
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
            num_layers=gnn_num_layers,
            use_jk=gnn_use_jk,
            jk_mode=gnn_jk_mode,
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
            f"gnn_hidden={gnn_hidden_dim} heads={gnn_heads} layers={gnn_num_layers} "
            f"use_jk={gnn_use_jk} jk_mode={gnn_jk_mode} | "
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

        # ── GNN eye: function-level pool → project ───────────────────────
        # Pool only over function-level nodes (FUNCTION/MODIFIER/FALLBACK/
        # RECEIVE/CONSTRUCTOR).  After Phase 3 reverse-CONTAINS aggregation,
        # these nodes carry CFG ordering signal.  Pooling over all nodes was
        # dominated by CFG_RETURN (77% of CFG node mass, median 93%), which
        # caused the GNN eye gradient share to collapse to ~7% by epoch 43.
        #
        # node feature[0] is stored normalised as type_id / _MAX_TYPE_ID;
        # multiply back by _MAX_TYPE_ID (from graph_schema) and round to
        # recover the integer type_id.  _MAX_TYPE_ID is the module-level
        # constant derived from NODE_TYPES so it tracks schema changes.
        # Use .float() before * to guard against AMP/BF16 round-trip precision loss.
        node_type_ids = (graphs.x[:, 0].float() * _MAX_TYPE_ID).round().long()
        func_mask = torch.zeros(node_embs.size(0), dtype=torch.bool,
                                device=node_embs.device)
        for tid in _FUNC_TYPE_IDS:
            func_mask |= (node_type_ids == tid)

        # Per-graph fallback: a graph with NO function-level nodes (ghost graph
        # or interface-only contract) would produce zero rows for its batch index,
        # causing global_max/mean_pool to silently drop it and return B-k outputs
        # instead of B. Fix: for such graphs, include ALL their nodes in the pool.
        if batch.numel() == 0:
            # Empty batch guard: return zero logits (and zero aux dict when requested)
            # rather than crashing in batch.max(). Must return the correct type so the
            # training loop can unpack (logits, aux) without a ValueError.
            B   = input_ids.size(0)
            dev = node_embs.device
            zeros = torch.zeros(B, self.num_classes, device=dev)
            if not return_aux:
                return zeros
            aux_zeros = {
                "gnn":         torch.zeros(B, self.num_classes, device=dev),
                "transformer": torch.zeros(B, self.num_classes, device=dev),
                "fused":       torch.zeros(B, self.num_classes, device=dev),
            }
            return zeros, aux_zeros
        num_graphs = int(batch.max().item()) + 1
        graph_has_func = torch.zeros(num_graphs, dtype=torch.bool, device=node_embs.device)
        if func_mask.any():
            graph_has_func[batch[func_mask]] = True
        # Nodes belonging to graphs that have no function nodes → use as fallback
        fallback_mask = ~graph_has_func[batch]
        pool_mask  = func_mask | fallback_mask
        pool_embs  = node_embs[pool_mask]
        pool_batch = batch[pool_mask]

        gnn_max  = global_max_pool(pool_embs, pool_batch, size=num_graphs)   # [B, gnn_hidden_dim]
        gnn_mean = global_mean_pool(pool_embs, pool_batch, size=num_graphs)  # [B, gnn_hidden_dim]
        gnn_eye  = self.gnn_eye_proj(
            torch.cat([gnn_max, gnn_mean], dim=1)           # [B, 2*gnn_hidden_dim]
        )                                                    # [B, eye_dim]

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

        if not return_aux:
            return logits

        # ── Auxiliary heads (training only) ───────────────────────────────
        aux_gnn   = self.aux_gnn(gnn_eye)
        aux_tf    = self.aux_transformer(transformer_eye)
        aux_fused = self.aux_fused(fused_eye)
        if self.num_classes == 1:
            aux_gnn   = aux_gnn.squeeze(1)
            aux_tf    = aux_tf.squeeze(1)
            aux_fused = aux_fused.squeeze(1)
        aux = {
            "gnn":         aux_gnn,    # [B, num_classes] or [B] for binary
            "transformer": aux_tf,
            "fused":       aux_fused,
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
