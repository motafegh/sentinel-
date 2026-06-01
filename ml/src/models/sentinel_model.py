"""
sentinel_model.py — SENTINEL Three-Eye Model (v8 architecture)

v8 ARCHITECTURE
───────────────
Three-eye classifier: three independent 128-dim vectors concatenated to [B, 384].
GNN is a three-phase, 7-layer GAT (2+3+2) that encodes execution order via
CFG CONTROL_FLOW edges (3 hops: CEI + ENTRY pattern) plus CALL_ENTRY(8)/RETURN_TO(9)
ICFG-Lite edges, and propagates back up via reversed CONTAINS edges (Phase 3).
NODE_FEATURE_DIM=11, NUM_EDGE_TYPES=11.

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
    WindowAttentionPooler → learned-attention over W window-CLS tokens → [B, 768]
    transformer_eye_proj  Linear(768,128)+ReLU+Dropout → [B, 128]
    (single-window fallback: returns CLS at position 0 with zero overhead)

  Fused eye       (joint structural+semantic opinion):
    CrossAttentionFusion(node_embs, token_embs) → [B, 128]

  Classifier:
    cat([gnn_eye, transformer_eye, fused_eye])  → [B, 384]
    Linear(384, 192) → ReLU → Dropout → Linear(192, num_classes) → raw logits [B, num_classes]

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
from ml.src.models.transformer_encoder import TransformerEncoder, WindowAttentionPooler
from ml.src.preprocessing.graph_schema import NODE_TYPES

# ── Schema-derived constant (single source of truth) ──────────────────────────
# Derived from graph_schema.NODE_TYPES so that adding a new node type here
# automatically propagates to the model's denormalisation without manual edits.
# node feature[0] is stored as float(type_id) / _MAX_TYPE_ID in the extractor;
# multiply back by _MAX_TYPE_ID and round to recover the integer type_id.
_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 12.0 for v2 schema (ids 0–12)
# A32: mirror of A3 assert in graph_extractor.py — both files must agree on the
# normalization divisor.  Fires at import so any NODE_TYPES addition is caught
# before it silently misaligns the model's type-id recovery with the stored graphs.
assert _MAX_TYPE_ID == 12.0, (
    f"_MAX_TYPE_ID is {_MAX_TYPE_ID} but expected 12.0. "
    "A new node type was added — update normalization in graph_extractor.py "
    "and sentinel_model.py (this file) before re-extracting graphs."
)

# Pooling target: function-declaration node types only.
# After Phase 3 reverse-CONTAINS, these nodes carry aggregated CFG signal.
_FUNC_TYPE_IDS: frozenset[int] = frozenset({  # FUNCTION MODIFIER FALLBACK RECEIVE CONSTRUCTOR
    NODE_TYPES["FUNCTION"],
    NODE_TYPES["MODIFIER"],
    NODE_TYPES["FALLBACK"],
    NODE_TYPES["RECEIVE"],
    NODE_TYPES["CONSTRUCTOR"],
})
# Pre-built CPU tensor of func type IDs — moved to module level to avoid
# allocating a new tensor on every forward pass (BUG-L1 perf fix).
# .to(device) in forward() is a no-op if already on the right device.
_FUNC_IDS_CPU: torch.Tensor = torch.tensor(sorted(_FUNC_TYPE_IDS), dtype=torch.long)
assert _FUNC_IDS_CPU.numel() == len(_FUNC_TYPE_IDS) and _FUNC_IDS_CPU.min() >= 0, (
    f"NC-2: _FUNC_IDS_CPU has unexpected shape or values: {_FUNC_IDS_CPU.tolist()}"
)

# ── GNN prefix injection constants (Phase 1) ──────────────────────────────────
# Selection priority for K-capped truncation: lower number = selected first.
# Entry-point nodes carry the most vulnerability-relevant signal after Phase 3.
_PREFIX_NODE_PRIORITY: dict[int, int] = {
    NODE_TYPES["CONSTRUCTOR"]: 0,  # unique entry point — always include first
    NODE_TYPES["FALLBACK"]:    1,  # plain-transfer entry — reentrancy-critical
    NODE_TYPES["RECEIVE"]:     2,  # receive hook — reentrancy-critical
    NODE_TYPES["MODIFIER"]:    3,  # access control — important for auth checks
    NODE_TYPES["FUNCTION"]:    4,  # general — selected last if K forces truncation
}
# Embedding index (0-4) for prefix_type_embedding.  Stable mapping independent
# of the raw NODE_TYPES integer values, which differ from these indices.
_PREFIX_TYPE_IDX: dict[int, int] = {
    NODE_TYPES["FUNCTION"]:    0,
    NODE_TYPES["MODIFIER"]:    1,
    NODE_TYPES["FALLBACK"]:    2,
    NODE_TYPES["RECEIVE"]:     3,
    NODE_TYPES["CONSTRUCTOR"]: 4,
}
_NUM_PREFIX_TYPES: int = 5  # FUNCTION MODIFIER FALLBACK RECEIVE CONSTRUCTOR


class SentinelModel(nn.Module):
    """
    Three-eye smart contract vulnerability detection model (v8).

    See module docstring for the full architecture description.

    Args:
        num_classes:          10 for Track 3 multi-label (default).
        fusion_output_dim:    Width of each eye output = fused output_dim (default: 128).
        dropout:              Dropout for eye projections and classifier (default: 0.3).

        --- GNN hyperparameters ---
        gnn_hidden_dim:       GNN node embedding width (default: 128).
        gnn_num_layers:       Number of GAT layers (default: 7; 2+3+2 phases, v7).
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
        gnn_hidden_dim:       int                 = 256,
        gnn_num_layers:       int                 = 8,   # 2+3+3 phases (IMP-G3 adds conv4c)
        gnn_heads:            int                 = 8,
        gnn_dropout:          float               = 0.2,
        use_edge_attr:        bool                = True,
        gnn_edge_emb_dim:     int                 = 64,
        # JK connections (Phase 1-A1, 2026-05-14)
        gnn_use_jk:           bool                = True,
        gnn_jk_mode:          str                 = 'attention',
        # Phase 2 ablation control
        gnn_phase2_edge_types: list[int]|None     = None,
        # LoRA architecture
        lora_r:               int                 = 16,
        lora_alpha:           int                 = 32,
        lora_dropout:         float               = 0.1,
        lora_target_modules:  Optional[List[str]] = None,
        # GNN prefix injection (Phase 1)
        gnn_prefix_k:               int   = 0,    # 0 = disabled; 48 for Phase 1
        gnn_prefix_warmup_epochs:   int   = 15,   # epochs prefix is suppressed
        # CrossAttentionFusion node padding limit (IMP-D1: increase to 2048)
        fusion_max_nodes:           int   = 1024,
    ) -> None:
        super().__init__()

        self.num_classes             = num_classes
        self.use_edge_attr           = use_edge_attr
        self.gnn_prefix_k            = gnn_prefix_k
        self.gnn_prefix_warmup_epochs = gnn_prefix_warmup_epochs
        self._current_epoch: int     = 0
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
            phase2_edge_types=gnn_phase2_edge_types,
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
            max_nodes=fusion_max_nodes,
        )

        # ── GNN prefix injection modules (Phase 1; only when gnn_prefix_k > 0) ─
        if gnn_prefix_k > 0:
            # Projects GNN node embeddings [K, gnn_hidden_dim] → [K, 768]
            # so they can be prepended to GraphCodeBERT's input_embeds.
            self.gnn_to_bert_proj = nn.Linear(gnn_hidden_dim, 768)
            # Type-specific bias per declaration node type (5 types).
            # Added to proj output so the transformer can distinguish node roles.
            self.prefix_type_embedding = nn.Embedding(_NUM_PREFIX_TYPES, 768)

        # ── Eye projections ────────────────────────────────────────────────
        # GNN eye: max+mean pool → [B, 2*gnn_hidden_dim] → [B, eye_dim]
        self.gnn_eye_proj = nn.Sequential(
            nn.Linear(2 * gnn_hidden_dim, eye_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Transformer eye: window-attention pooled CLS → [B, 768] → [B, eye_dim]
        # prefix_k shifts CLS position from 0 → prefix_k within each window.
        self.window_pooler = WindowAttentionPooler(hidden_dim=768, window_size=512, prefix_k=gnn_prefix_k)
        self.transformer_eye_proj = nn.Sequential(
            nn.Linear(768, eye_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Main classifier ────────────────────────────────────────────────
        # All three eyes concatenated: [B, 3*eye_dim] → hidden → [B, num_classes]
        # Hidden layer at 192 adds capacity without overfitting on 44K contracts.
        # No Sigmoid — applied externally.
        _cls_hidden = 192
        self.classifier = nn.Sequential(
            nn.Linear(3 * eye_dim, _cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(_cls_hidden, num_classes),
        )

        # ── Auxiliary heads (training only — eye-independence enforcement) ──
        # Each head produces independent logits for its eye so the loss keeps
        # that eye's gradient alive even if the main classifier learns to
        # downweight it.  Discarded at inference via return_aux=False.
        self.aux_gnn         = nn.Linear(eye_dim, num_classes)
        self.aux_transformer = nn.Linear(eye_dim, num_classes)
        self.aux_fused       = nn.Linear(eye_dim, num_classes)

        # Phase 2 CEI auxiliary head (Interp-2): supervises Phase 2 node embeddings
        # directly so the JK entropy regulariser alone does not starve Phase 2 of signal.
        # Input: Phase 2 output pooled over function nodes [B, gnn_hidden_dim].
        self.aux_phase2 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),  # 256 → 128
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gnn_hidden_dim // 2, num_classes),     # 128 → 10
        )

        logger.info(
            f"SentinelModel v8 (three-eye) initialised | "
            f"num_classes={num_classes} | eye_dim={eye_dim} | "
            f"classifier [{3 * eye_dim}→192→{num_classes}] | "
            f"gnn_hidden={gnn_hidden_dim} heads={gnn_heads} layers={gnn_num_layers} "
            f"use_jk={gnn_use_jk} jk_mode={gnn_jk_mode} | "
            f"lora_r={lora_r} lora_alpha={lora_alpha} | "
            f"gnn_prefix_k={gnn_prefix_k} warmup={gnn_prefix_warmup_epochs}"
        )

    def select_prefix_nodes(
        self,
        node_embs:    torch.Tensor,  # [N, gnn_hidden_dim]
        batch:        torch.Tensor,  # [N] — graph index per node
        node_type_ids: torch.Tensor, # [N] — integer type IDs
        num_graphs:   int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select up to K=gnn_prefix_k declaration-level nodes per graph, project
        to [B, K, 768] for injection as GraphCodeBERT prefix tokens.

        Selection priority (lower = selected first if K forces truncation):
            CONSTRUCTOR(6) > FALLBACK(4) > RECEIVE(5) > MODIFIER(2) > FUNCTION(1)
        Within FUNCTION nodes, secondary sort by feature[10] (external_call_count)
        descending — nodes with more external calls are selected first (IMP-M1).

        Graphs with fewer than K eligible nodes are zero-padded on the right.
        IMP-M3: the actual node count per graph is returned so TransformerEncoder
        can mask padded positions (attention=0) instead of attending to zero vectors.

        Returns:
            prefix:      [B, K, 768] — projected prefix embeddings.
            node_counts: [B]         — number of real (non-padded) nodes per graph.
        """
        K = self.gnn_prefix_k
        device = node_embs.device
        prefix      = torch.zeros(num_graphs, K, 768, device=device, dtype=node_embs.dtype)
        node_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)  # IMP-M3

        _EXT_CALL_DIM = 10  # feature[10] = external_call_count (log1p-normalised)
        _FUNCTION_ID  = NODE_TYPES["FUNCTION"]

        for g in range(num_graphs):
            g_mask    = batch == g
            g_types   = node_type_ids[g_mask]
            g_embs    = node_embs[g_mask]

            # Indices of eligible nodes within this graph's node list
            eligible_local = [
                i for i, t in enumerate(g_types.tolist())
                if t in _PREFIX_NODE_PRIORITY
            ]
            if not eligible_local:
                continue  # no declaration nodes; prefix stays zero

            # IMP-M1: two-key sort — primary=type priority, secondary=−ext_call_count
            # (FUNCTION nodes only; others use 0.0). Within each priority group,
            # nodes with more external calls are selected first when K forces truncation.
            sort_keys = []
            g_types_list = g_types.tolist()
            for local_idx in eligible_local:
                t    = g_types_list[local_idx]
                prio = _PREFIX_NODE_PRIORITY[t]
                sec  = -g_embs[local_idx, _EXT_CALL_DIM].item() if t == _FUNCTION_ID else 0.0
                sort_keys.append((prio, sec, local_idx))
            sort_keys.sort()   # Python stable tuple sort
            selected_local = [sk[2] for sk in sort_keys[:K]]
            selected = torch.tensor(selected_local, device=device, dtype=torch.long)

            proj = self.gnn_to_bert_proj(g_embs[selected])  # [n_sel, 768]

            # Add type-specific embedding bias so transformer knows node roles
            type_indices = torch.tensor(
                [_PREFIX_TYPE_IDX[g_types[i].item()] for i in selected.tolist()],
                device=device, dtype=torch.long,
            )
            proj = proj + self.prefix_type_embedding(type_indices)  # [n_sel, 768]

            n_sel = proj.shape[0]
            prefix[g, :n_sel]  = proj
            node_counts[g]     = n_sel  # IMP-M3: track real node count

        return prefix, node_counts  # [B, K, 768], [B]

    def forward(
        self,
        graphs:         Batch,                   # PyG Batch — batched contract graphs
        input_ids:      torch.Tensor,            # [B, L] or [B, W, L]
        attention_mask: torch.Tensor,            # [B, L] or [B, W, L]
        return_aux:     bool = False,            # True during training only
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Three-eye forward pass.

        Args:
            graphs:         Batched PyG graph (from DataLoader via Batch.from_data_list).
            input_ids:      CodeBERT token IDs     [B, L] or [B, W, L].
            attention_mask: CodeBERT attention mask [B, L] or [B, W, L].
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
        # ── Windowed token reshape (before GNN — builds flat_mask for fusion) ─
        # Multi-window input [B, W, L]: flatten mask to [B, W*L] for CrossAttentionFusion.
        # Single-window [B, L]: pass through unchanged.
        if input_ids.dim() == 3:
            B_tok, W, L = input_ids.shape
            flat_mask = attention_mask.view(B_tok, W * L)   # [B, W*L] for fusion
        else:
            flat_mask = attention_mask                       # [B, L]

        # ── GNN path: node embeddings ─────────────────────────────────────
        edge_attr = getattr(graphs, "edge_attr", None) if self.use_edge_attr else None
        _gnn_out = self.gnn(
            graphs.x, graphs.edge_index, graphs.batch, edge_attr,
            return_phase2_embs=return_aux,  # gradient-carrying phase2 for CEI aux loss
        )
        if return_aux:
            node_embs, batch, _jk_entropy, _phase2_x = _gnn_out
        else:
            node_embs, batch, _jk_entropy = _gnn_out
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
        func_mask = torch.isin(node_type_ids, _FUNC_IDS_CPU.to(node_embs.device))

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

        # Ghost graph fix (BUG-H2): graphs with no FUNCTION/MODIFIER/FALLBACK/
        # RECEIVE/CONSTRUCTOR nodes (interface-only contracts, Slither failures)
        # now produce a zero GNN embedding instead of pooling over STATE_VAR nodes.
        # global_max/mean_pool returns a zero row for any graph index with no
        # contributing nodes — that is the correct degenerate behavior.
        # The old fallback_mask approach pooled STATE_VARs, injecting misleading
        # variable-type signal for 9% of training samples.
        pool_mask  = func_mask
        pool_embs  = node_embs[pool_mask]
        pool_batch = batch[pool_mask]

        gnn_max  = global_max_pool(pool_embs, pool_batch, size=num_graphs)   # [B, gnn_hidden_dim]
        gnn_mean = global_mean_pool(pool_embs, pool_batch, size=num_graphs)  # [B, gnn_hidden_dim]
        gnn_eye  = self.gnn_eye_proj(
            torch.cat([gnn_max, gnn_mean], dim=1)           # [B, 2*gnn_hidden_dim]
        )                                                    # [B, eye_dim]

        # ── GNN prefix selection (suppressed during warmup and when disabled) ──
        # During warmup the prefix is None; gnn_to_bert_proj is untrained but the
        # GNN itself trains normally.  At epoch warmup+1 the projection starts from
        # random init with a well-trained GNN — it rapidly aligns node embeddings
        # into transformer space over the remaining epochs.
        gnn_prefix:       Optional[torch.Tensor] = None
        gnn_prefix_counts: Optional[torch.Tensor] = None
        if self.gnn_prefix_k > 0 and self._current_epoch >= self.gnn_prefix_warmup_epochs:
            gnn_prefix, gnn_prefix_counts = self.select_prefix_nodes(
                node_embs, batch, node_type_ids, num_graphs
            )
            # gnn_prefix: [B, K, 768]  gnn_prefix_counts: [B]

        # ── Transformer path ──────────────────────────────────────────────
        token_embs = self.transformer(
            input_ids, attention_mask,
            gnn_prefix_nodes=gnn_prefix,
            gnn_prefix_counts=gnn_prefix_counts,
        )
        # token_embs: [B, L, 768] or [B, W*L, 768]

        # ── Transformer eye: window-pooled CLS → project ─────────────────
        # WindowAttentionPooler extracts the CLS of each window and combines them
        # via learned attention weights. Single-window fallback returns CLS at [0].
        transformer_eye = self.transformer_eye_proj(
            self.window_pooler(token_embs)   # [B, 768]
        )                                     # [B, eye_dim]

        # ── Fused eye: cross-attention fusion ────────────────────────────
        # flat_mask is [B, W*L] in multi-window mode so CrossAttentionFusion's
        # key_padding_mask correctly marks padding positions across all windows.
        fused_eye = self.fusion(node_embs, batch, token_embs, flat_mask)
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

        # Phase 2 CEI aux head: pool phase2 embeddings over function nodes
        phase2_pooled    = global_mean_pool(_phase2_x[pool_mask], pool_batch, size=num_graphs)
        aux_phase2_logits = self.aux_phase2(phase2_pooled)   # [B, num_classes]

        if self.num_classes == 1:
            aux_gnn           = aux_gnn.squeeze(1)
            aux_tf            = aux_tf.squeeze(1)
            aux_fused         = aux_fused.squeeze(1)
            aux_phase2_logits = aux_phase2_logits.squeeze(1)
        aux = {
            "gnn":         aux_gnn,           # [B, num_classes] or [B] for binary
            "transformer": aux_tf,
            "fused":       aux_fused,
            "phase2":      aux_phase2_logits, # [B, num_classes] — CEI aux loss target
            "jk_entropy":  _jk_entropy,
        }
        return logits, aux

    @torch.no_grad()
    def compute_prefix_attention_mean(
        self,
        graphs:         "Batch",
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[float]:
        """
        IMP-M2 Tier 2: run one diagnostic forward with output_attentions=True.

        Returns mean attention weight from code token positions → GNN prefix positions,
        averaged over all BERT layers, heads, and sequences.  Returns None when prefix
        is disabled or still in warmup (no prefix injected — nothing to measure).

        Call once per validation epoch (not per training step) with a single batch.
        Add ~15% overhead vs normal eval forward — restrict to one batch per epoch.
        """
        if self.gnn_prefix_k == 0 or self._current_epoch < self.gnn_prefix_warmup_epochs:
            return None

        edge_attr = getattr(graphs, "edge_attr", None) if self.use_edge_attr else None
        node_embs, batch, _ = self.gnn(graphs.x, graphs.edge_index, graphs.batch, edge_attr)
        node_type_ids = (graphs.x[:, 0].float() * _MAX_TYPE_ID).round().long()

        if batch.numel() == 0:
            return None
        num_graphs = int(batch.max().item()) + 1
        gnn_prefix = self.select_prefix_nodes(node_embs, batch, node_type_ids, num_graphs)
        # After IMP-M3: select_prefix_nodes returns (prefix, counts); unpack accordingly.
        if isinstance(gnn_prefix, tuple):
            gnn_prefix, _ = gnn_prefix

        result = self.transformer(
            input_ids, attention_mask,
            gnn_prefix_nodes=gnn_prefix,
            output_attentions=True,
        )
        if isinstance(result, tuple):
            _, prefix_attn_mean = result
            return prefix_attn_mean
        return None

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
        if self.gnn_prefix_k > 0:
            components["gnn_to_bert_proj"]     = self.gnn_to_bert_proj
            components["prefix_type_embedding"] = self.prefix_type_embedding
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
