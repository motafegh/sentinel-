"""
gnn_encoder.py — GNN Encoder for SENTINEL (v5 — three-phase architecture)

THREE-PHASE DESIGN
──────────────────
Phase 1 (Layers 1+2): Structural aggregation
  Edges: types 0–5 (CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS)
  add_self_loops=True
  Layer 1: NODE_FEATURE_DIM→hidden_dim (concat 8 heads)
  Layer 2: hidden_dim→hidden_dim (concat 8 heads) + residual from Layer 1
  Purpose: propagate function-level properties DOWN into CFG_NODE children
  via CONTAINS edges, and aggregate inter-function structural context.

Phase 2 (Layer 3): CFG-directed aggregation
  Edges: type 6 (CONTROL_FLOW only)
  add_self_loops=False  ← CRITICAL — self-loops cancel directional signal
  heads=1, concat=False → output stays hidden_dim (128)
  Purpose: enrich CFG_NODE embeddings with execution-order information.
  One message-passing hop: sufficient for diameter-2 CFGs (require→call→write).
  Known limitation: diameter-4+ CFGs (complex branching) may need 2 hops.
  v5.1 target: gnn_layers=5 for 2 CONTROL_FLOW hops.

Phase 3 (Layer 4): Reverse-CONTAINS aggregation
  Edges: type 7 REVERSE_CONTAINS (runtime-only, CFG_NODE → FUNCTION direction)
  add_self_loops=False
  heads=1, concat=False → output stays hidden_dim (128)
  Purpose: aggregate Phase-2-enriched CFG embeddings UP into FUNCTION nodes.
  This is the path by which execution-order information reaches the function
  nodes that the classifier operates on. Without Phase 3, function-node
  embeddings are order-blind regardless of CFG machinery below them.

  Phase 1-A3 (2026-05-14): Phase 3 now uses REVERSE_CONTAINS (type 7) instead
  of reusing the forward CONTAINS (type 5) embedding. This gives the GNN a
  distinct learned representation for the reverse direction, fixing v5.0
  limitation L2. Type-7 edges are generated at runtime by flipping CONTAINS(5)
  edges in the forward pass — no graph re-extraction needed.

  Zero-message behaviour (correct — do not "fix"):
  FUNCTION nodes with no CFG children receive no Phase 3 messages.
  conv4 returns zero for them; residual x = x + dropout(0) is a no-op.
  They retain their Phase 2 embedding. Adding add_self_loops=True to
  "fix" this would dilute the Phase 2 order signal for functions that DO
  have CFG children.

JK Connections (Phase 1-A1, 2026-05-14)
─────────────────────────────────────────
When use_jk=True, a learned attention over all three phase outputs is used
instead of returning only the final Phase 3 embedding. This prevents Phase 1
structural signal from being over-smoothed by phases 2 and 3, fixing the
gradient collapse observed in v5.1-fix28 where the GNN share dropped to ~10%
by epoch 8.

Implementation: custom _JKAttention module (NOT PyG's built-in JumpingKnowledge
with mode='lstm'). Reasons: (1) simpler gradient flow analysis, (2) output
dimension is exactly hidden_dim with no LSTM overhead, (3) easier to inspect
per-phase attention weights for monitoring.

CRITICAL — live intermediates:
JK aggregation consumes tensors from _live = [] which are collected WITHOUT
.detach(). If .detach() were used (as in the existing _intermediates diagnostic
dict), JK attention weights would receive zero gradients for the entire training
run. The _intermediates diagnostic dict still uses .detach().clone() for
backwards compatibility with test_cfg_embedding_separation.py.

Per-Phase LayerNorm (Phase 1-A2, 2026-05-14)
──────────────────────────────────────────────
self.phase_norm = ModuleList([LayerNorm(hidden_dim) for _ in range(3)])
Applied after each phase's residual connection, before collecting for JK.
Prevents Phase 1 magnitude from dominating the JK attention softmax — without
LayerNorm, Phase 1 (two conv layers + residual) tends to produce higher norms
than Phase 2 and 3 (one layer each), causing JK to ignore later phases.

return_intermediates
────────────────────
forward(return_intermediates=False) — default; returns (x, batch)
forward(return_intermediates=True)  — returns (x, batch, intermediates_dict)
  intermediates_dict keys: "after_phase1", "after_phase2", "after_phase3"
  Each is a detached tensor of shape [N, hidden_dim].
  Used by the pre-flight embedding-separation test (test_cfg_embedding_separation.py).

num_layers validation
─────────────────────
Stored as an attribute for serialisation only. Validation is in
TrainConfig.__post_init__() — fires at startup before data loading or
GPU allocation, not deep inside model construction.

NODE FEATURE SCALING NOTE
─────────────────────────
x[:, 0] = type_id is normalised to [0, 1] in graph_extractor.py (/ 12.0).
This is required: raw type_id 0–12 dominates the dot product and makes
adjacent CFG subtypes (CALL=8/12, WRITE=9/12) indistinguishable.
All other features are already in [0, 1] or small normalised ranges.

PARAMETERS (v5 defaults)
─────────────────────────
  in_channels   = NODE_FEATURE_DIM (12) — never hardcode 8 or 13 here
  hidden_dim    = 128
  heads         = 8 (Phase 1 only; Phases 2+3 use heads=1)
  dropout       = 0.2
  use_edge_attr = True
  edge_emb_dim  = 32  (nn.Embedding(NUM_EDGE_TYPES=8, 32))
  num_layers    = 4
  use_jk        = True   (Phase 1-A1; attention over 3 phase outputs)
  jk_mode       = 'attention'

Total trainable parameters (v5 defaults): ~91K (JK adds ~128 params)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES


# ---------------------------------------------------------------------------
# JK attention aggregator
# ---------------------------------------------------------------------------

class _JKAttention(nn.Module):
    """
    Learned attention aggregation over a list of same-shape embeddings.

    For each node, computes a scalar score per embedding, softmax-normalises,
    and returns the weighted sum.  Output shape equals input shape.

    This is the "attention" JK mode used by GNNEncoder.  We implement it
    here rather than using PyG's JumpingKnowledge(mode='lstm') for two reasons:
      1. Explicit gradient flow — easy to verify all parameters are trained.
      2. No LSTM state overhead — just a single Linear(channels, 1).

    Phase 1-A1 (2026-05-14): non-negotiable gate test_jk_gradient_flow checks
    that self.attn receives non-zero gradients after a backward pass.
    """

    def __init__(self, channels: int, num_phases: int = 3) -> None:
        super().__init__()
        self.attn = nn.Linear(channels, 1, bias=False)
        # Register as buffer so it survives .to(device), save/load, and DDP.
        self.register_buffer("last_weights", torch.zeros(num_phases))

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            xs: list of K tensors each [N, channels]
        Returns:
            [N, channels]  — weighted sum, same shape as each input

        Side effect: stores mean per-phase attention weights in self.last_weights
        as a detached [K] tensor so the trainer can log them without an extra
        forward pass. Phase 2-C1 (2026-05-14).
        """
        stacked = torch.stack(xs, dim=1)        # [N, K, channels]
        scores  = self.attn(stacked)             # [N, K, 1]
        weights = torch.softmax(scores, dim=1)   # [N, K, 1]  (sum-to-1 over K)
        # Update registered buffer in-place (preserves device + save/load).
        self.last_weights.copy_(weights.squeeze(-1).mean(0).detach())  # [K]
        return (weights * stacked).sum(dim=1)    # [N, channels]


# ---------------------------------------------------------------------------
# GNNEncoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """
    Three-phase, four-layer GAT encoder for smart contract graphs.

    Returns node-level embeddings (NOT pooled). SentinelModel pools them
    separately for the GNN eye and feeds them to CrossAttentionFusion.

    Input:
        x:          node features       [N, NODE_FEATURE_DIM]
        edge_index: graph connectivity  [2, E]
        batch:      node→graph mapping  [N]
        edge_attr:  edge type IDs       [E]  int64 in [0, NUM_EDGE_TYPES)
                    Optional; None → edge embedding layer is skipped.
        return_intermediates: when True, also return the intermediates dict.

    Output (return_intermediates=False, default):
        node_embeddings: [N, hidden_dim]  — NOT pooled
        batch:           [N]              — passed through unchanged

    Output (return_intermediates=True):
        node_embeddings: [N, hidden_dim]
        batch:           [N]
        intermediates:   dict with "after_phase1", "after_phase2", "after_phase3"
                         (each [N, hidden_dim], detached)
    """

    def __init__(
        self,
        hidden_dim:    int   = 128,
        heads:         int   = 8,
        dropout:       float = 0.2,
        use_edge_attr: bool  = True,
        edge_emb_dim:  int   = 32,
        num_layers:    int   = 4,
        use_jk:        bool  = True,
        jk_mode:       str   = 'attention',
    ) -> None:
        """
        Args:
            hidden_dim:    Node embedding dimension throughout all phases.
            heads:         Multi-head count for Phase 1 (Phases 2+3 always use 1).
            dropout:       Dropout probability applied after each conv layer.
            use_edge_attr: If True, embed edge types and feed to GATConv.
            edge_emb_dim:  Edge type embedding dimension.
            num_layers:    Stored for serialisation; validation in TrainConfig.
            use_jk:        If True, use JK attention aggregation over all three
                           phase outputs instead of returning only Phase 3.
                           Phase 1-A1 (2026-05-14).
            jk_mode:       JK aggregation mode. 'attention' is the only supported
                           mode (preserved-dim attention over phase outputs).
        """
        super().__init__()

        # Stored for serialisation; validation fires in TrainConfig.__post_init__().
        self.num_layers    = num_layers
        self.hidden_dim    = hidden_dim
        self.use_edge_attr = use_edge_attr
        self.dropout_p     = dropout
        self.use_jk        = use_jk
        self.jk_mode       = jk_mode

        _head_dim = hidden_dim // heads  # 16 per head when hidden=128, heads=8
        if _head_dim * heads != hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}). "
                f"Each head needs hidden_dim/heads = {hidden_dim/heads:.1f} dims."
            )

        # Edge type embedding: [E] int64 → [E, edge_emb_dim]
        # Now covers all NUM_EDGE_TYPES (8) including REVERSE_CONTAINS(7).
        # Type 7 is runtime-only (generated in forward pass from flipped CONTAINS
        # edges), so graph files on disk never contain id=7 — but the embedding
        # table must have 8 rows so index-7 lookups don't crash.
        _edge_dim: int | None = None
        if use_edge_attr:
            self.edge_embedding = nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)
            _edge_dim = edge_emb_dim
        else:
            self.edge_embedding = None

        # ── Phase 1 — structural + CONTAINS forward ─────────────────────────
        # add_self_loops=True: harmless for non-directional structural aggregation.
        # out_channels is PER HEAD in PyG GATConv. With heads=8 and concat=True:
        #   total output = 8 × _head_dim = 8 × 16 = 128 = hidden_dim.
        # WARNING: passing out_channels=128 with heads=8, concat=True gives 1024-dim.
        self.conv1 = GATConv(
            in_channels=NODE_FEATURE_DIM,  # 12
            out_channels=_head_dim,         # 16 per head
            heads=heads,                    # 8
            concat=True,                    # total out = 8*16 = 128
            add_self_loops=True,
            edge_dim=_edge_dim,
        )
        # Layer 2: 128 → 128 with residual (dimensions match for skip connection).
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=_head_dim,
            heads=heads,
            concat=True,
            add_self_loops=True,
            edge_dim=_edge_dim,
        )

        # ── Phase 2 — CONTROL_FLOW directed ─────────────────────────────────
        # add_self_loops=False — CRITICAL.
        # Self-loops add each node as its own predecessor in the attention sum.
        # During CONTROL_FLOW attention, each CFG_NODE would then attend to both
        # its genuine predecessor (execution order signal) AND itself (no order
        # information). The self-loop term partially cancels the directional signal.
        # With add_self_loops=False, only genuine directed CONTROL_FLOW edges participate.
        #
        # heads=1: CONTROL_FLOW is a single relationship type (execution order);
        # one head with full hidden_dim capacity is preferable here.
        self.conv3 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,  # 128 (heads=1, concat=False → no head expansion)
            heads=1,
            concat=False,
            add_self_loops=False,     # CRITICAL — preserve directional signal
            edge_dim=_edge_dim,
        )

        # ── Phase 3 — reverse-CONTAINS ───────────────────────────────────────
        # CFG_NODE nodes (enriched by Phase 2) send messages TO FUNCTION nodes.
        # Uses REVERSE_CONTAINS (type 7) edges — CONTAINS edges with src↔dst flipped
        # AND a distinct type-7 embedding (Phase 1-A3 fix: was reusing type-5).
        # add_self_loops=False — we only want CFG → function aggregation.
        # See module docstring for zero-message behaviour explanation.
        self.conv4 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,
            edge_dim=_edge_dim,
        )

        # ── Per-phase LayerNorm (Phase 1-A2) ────────────────────────────────
        # Applied after each phase's residual connection, before collecting
        # intermediates for JK.  Prevents Phase 1's higher norm (two conv layers)
        # from dominating the JK attention softmax.
        # Always present (regardless of use_jk) for training stability.
        self.phase_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim),  # after Phase 1
            nn.LayerNorm(hidden_dim),  # after Phase 2
            nn.LayerNorm(hidden_dim),  # after Phase 3
        ])

        # ── JK attention aggregator (Phase 1-A1) ────────────────────────────
        # Collects all three (LayerNorm-ed) phase outputs and produces a
        # learned weighted sum.  Output dimension == hidden_dim (preserved).
        # Only instantiated when use_jk=True to keep non-JK checkpoints smaller.
        if use_jk:
            if jk_mode != 'attention':
                raise ValueError(
                    f"jk_mode='{jk_mode}' is not supported. "
                    "Only jk_mode='attention' is implemented."
                )
            self.jk = _JKAttention(hidden_dim)
        else:
            self.jk = None

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:                    torch.Tensor,
        edge_index:           torch.Tensor,
        batch:                torch.Tensor,
        edge_attr:            torch.Tensor | None = None,
        return_intermediates: bool               = False,
    ):
        """
        Three-phase forward pass.

        Args:
            x:                    [N, NODE_FEATURE_DIM]
            edge_index:           [2, E]
            batch:                [N]
            edge_attr:            [E] int64 edge type IDs in [0, NUM_EDGE_TYPES).
                                  Required when use_edge_attr=True.
            return_intermediates: When True, also return per-phase embeddings dict.

        Returns (return_intermediates=False):
            node_embeddings [N, hidden_dim], batch [N]
        Returns (return_intermediates=True):
            node_embeddings [N, hidden_dim], batch [N],
            {"after_phase1": ..., "after_phase2": ..., "after_phase3": ...}
            (diagnostic only — detached tensors, not used for gradients)
        """
        # ── Guards ───────────────────────────────────────────────────────────
        # Bug #1: use_edge_attr=True with edge_attr=None silently disables
        # Phase 2 CONTROL_FLOW masking (cfg_mask becomes all-zeros). Raise
        # early so the caller knows the graph is missing edge_attr.
        if self.use_edge_attr and edge_attr is None:
            raise ValueError(
                "GNNEncoder was built with use_edge_attr=True but edge_attr=None "
                "was passed. All v5 graphs must have graph.edge_attr. "
                "Check GraphExtractionConfig.include_edge_attr=True."
            )
        # Bug #3: out-of-bounds node indices in edge_index produce silent wrong
        # GATConv attention or CUDA illegal-memory-access with no useful traceback.
        if edge_index.numel() > 0 and edge_index.max() >= x.shape[0]:
            raise ValueError(
                f"edge_index contains node index {edge_index.max().item()} "
                f"but x has only {x.shape[0]} nodes. Graph .pt file is corrupted."
            )

        # ── Edge embeddings ──────────────────────────────────────────────────
        e = None
        if self.edge_embedding is not None and edge_attr is not None:
            # Fix C1 (H9): clamp OOB edge_attr before nn.Embedding lookup.
            # Corrupted .pt files with type IDs outside [0, NUM_EDGE_TYPES-1] cause
            # nn.Embedding to throw an unhelpful index-out-of-range CUDA error with
            # no indication of which contract caused it. Clamping lets the forward
            # pass continue on a bad sample with a logged warning instead of crashing
            # the entire training run on an unrecoverable CUDA illegal-memory error.
            if edge_attr.numel() > 0:
                _oob_mask = (edge_attr < 0) | (edge_attr >= NUM_EDGE_TYPES)
                if _oob_mask.any():
                    logger.warning(
                        f"GNNEncoder: {_oob_mask.sum().item()} OOB edge_attr value(s) "
                        f"clamped to [0, {NUM_EDGE_TYPES - 1}] "
                        f"(observed min={edge_attr.min().item()}, max={edge_attr.max().item()}). "
                        "Source .pt file may be corrupted — rerun ast_extractor.py for this contract."
                    )
                    edge_attr = edge_attr.clamp(0, NUM_EDGE_TYPES - 1)
            e = self.edge_embedding(edge_attr)   # [E, edge_emb_dim]

        # ── Edge masks — one per phase ───────────────────────────────────────
        # struct_mask:   types 0–CONTAINS (all structural + CONTAINS forward)
        # cfg_mask:      CONTROL_FLOW only
        # contains_mask: CONTAINS only; used to build Phase 3 reverse edges
        _CONTAINS         = EDGE_TYPES["CONTAINS"]          # 5
        _CONTROL_FLOW     = EDGE_TYPES["CONTROL_FLOW"]       # 6
        _REVERSE_CONTAINS = EDGE_TYPES["REVERSE_CONTAINS"]   # 7 (runtime-only)
        if edge_attr is not None:
            struct_mask   = edge_attr <= _CONTAINS
            cfg_mask      = edge_attr == _CONTROL_FLOW
            contains_mask = edge_attr == _CONTAINS
        else:
            # Without edge_attr: Phase 2 (CFG) and Phase 3 (REVERSE_CONTAINS) are
            # disabled — no edges match cfg_mask or contains_mask. This is degraded
            # mode and almost certainly a data-pipeline error when use_edge_attr=True.
            if self.use_edge_attr:
                logger.warning(
                    "GNNEncoder: use_edge_attr=True but edge_attr is None. "
                    "Phase 2 (CONTROL_FLOW) and Phase 3 (REVERSE_CONTAINS) are disabled. "
                    "Check that your graphs were extracted with graph_extractor.py v5+."
                )
            n_edges = edge_index.shape[1]
            struct_mask   = torch.ones(n_edges, dtype=torch.bool, device=edge_index.device)
            cfg_mask      = torch.zeros(n_edges, dtype=torch.bool, device=edge_index.device)
            contains_mask = torch.zeros(n_edges, dtype=torch.bool, device=edge_index.device)

        struct_ei = edge_index[:, struct_mask]
        struct_ea = e[struct_mask]   if e is not None else None

        cfg_ei    = edge_index[:, cfg_mask]
        cfg_ea    = e[cfg_mask]      if e is not None else None

        # Phase 3: flip CONTAINS edges → CFG_NODE sends to FUNCTION (child → parent).
        # Phase 1-A3 (2026-05-14): assign type-7 (REVERSE_CONTAINS) embeddings instead
        # of reusing the type-5 (CONTAINS) embeddings. The GNN can now learn directional
        # asymmetry — "function-to-CFG" vs "CFG-to-function" are distinct relations.
        rev_contains_ei = edge_index[:, contains_mask].flip(0)   # [2, E_contains]
        if self.edge_embedding is not None:
            n_rev = rev_contains_ei.shape[1]
            if n_rev > 0:
                rev_type_ids = torch.full(
                    (n_rev,), _REVERSE_CONTAINS,
                    dtype=torch.long, device=edge_index.device,
                )
                rev_contains_ea = self.edge_embedding(rev_type_ids)  # [E_contains, edge_emb_dim]
            else:
                rev_contains_ea = None
        else:
            rev_contains_ea = None

        # _live collects LIVE (non-detached) phase outputs for JK aggregation.
        # CRITICAL: do NOT use .detach() here — JK attention parameters must
        # receive gradients through these tensors during backward.
        # The _intermediates dict keeps .detach().clone() for diagnostics only.
        _live: list[torch.Tensor] = []
        _intermediates: dict      = {}

        # ── Phase 1: structural aggregation (Layers 1+2) ────────────────────
        # Layer 1: NODE_FEATURE_DIM→hidden_dim. No residual — dims differ (12≠128).
        x  = self.conv1(x, struct_ei, struct_ea)    # [N, 12] → [N, 128]
        x  = self.relu(x)
        x  = self.dropout(x)
        # Layer 2: hidden_dim→hidden_dim with residual from Layer 1.
        x2 = self.conv2(x, struct_ei, struct_ea)    # [N, 128] → [N, 128]
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                   # residual: identity preserved, only branch dropped
        # Phase 1-A2: LayerNorm before collecting for JK.
        x  = self.phase_norm[0](x)

        _live.append(x)
        _intermediates["after_phase1"] = x.detach().clone()

        # ── Phase 2: CONTROL_FLOW directed (Layer 3) ────────────────────────
        # Non-CFG_NODE nodes have no CONTROL_FLOW edges — GATConv returns zero
        # for nodes with no incoming edges. They carry their Phase 1 embeddings forward.
        x2 = self.conv3(x, cfg_ei, cfg_ea)          # [N, 128] → [N, 128]
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                   # residual
        # Phase 1-A2: LayerNorm before collecting for JK.
        x  = self.phase_norm[1](x)

        _live.append(x)
        _intermediates["after_phase2"] = x.detach().clone()

        # ── Phase 3: reverse-CONTAINS (Layer 4) ─────────────────────────────
        # Phase-2-enriched CFG_NODE embeddings flow UP to FUNCTION nodes.
        # FUNCTION nodes with no CFG children receive zero messages — this is
        # correct behaviour (no-op residual). Do NOT add add_self_loops=True.
        x2 = self.conv4(x, rev_contains_ei, rev_contains_ea)  # [N, 128]
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                   # residual
        # Phase 1-A2: LayerNorm before collecting for JK.
        x  = self.phase_norm[2](x)

        _live.append(x)
        _intermediates["after_phase3"] = x.detach().clone()

        # ── JK aggregation (Phase 1-A1) ──────────────────────────────────────
        # Weighted sum over all three phase outputs using learned per-node
        # attention scores.  When use_jk=False, only Phase 3 output is returned
        # (same behaviour as v5.0, preserved for checkpoint compatibility).
        if self.use_jk and self.jk is not None:
            x = self.jk(_live)   # [N, hidden_dim]
        # else: x is already the Phase 3 output

        if return_intermediates:
            return x, batch, _intermediates
        return x, batch
