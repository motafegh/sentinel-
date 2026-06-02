"""
gnn_encoder.py — GNN Encoder for SENTINEL (v8 — three-phase, 8-layer architecture)

THREE-PHASE DESIGN (v8+IMP: 2+3+3 layers = 8 total)
──────────────────────────────────────────────────────
Phase 1 (Layers 1+2): Structural aggregation + input skip (IMP-G2)
  Edges: types 0–5 (CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS)
  add_self_loops=True
  Layer 1: NODE_FEATURE_DIM→hidden_dim (concat 8 heads) + input_proj skip (IMP-G2)
  Layer 2: hidden_dim→hidden_dim (concat 8 heads) + residual
  Purpose: propagate function-level properties DOWN into CFG_NODE children
  via CONTAINS edges, and aggregate inter-function structural context.
  IMP-G2: input_proj skip = Linear(11, 256, bias=False) added before relu in Layer 1.
  Prevents raw feature loss when GAT attention weights start near-uniform.

Phase 2 (Layers 3+4+5): Layer-specific CFG + ICFG (IMP-G1)
  add_self_loops=False  ← CRITICAL — self-loops cancel directional signal
  heads=1, concat=False → output stays hidden_dim
  IMP-G1: each layer processes a DISTINCT edge subset (vs same cfg_mask before).
  Layer 3 (conv3):  CONTROL_FLOW(6) only — intra-function execution ordering
  Layer 4 (conv3b): CALL_ENTRY(8) + RETURN_TO(9) only — cross-function call structure
  Layer 5 (conv3c): CF(6)+CALL_ENTRY(8)+RETURN_TO(9) joint — integration layer

Phase 3 (Layers 6+7+8): Bidirectional CONTAINS (IMP-G3)
  Layer 6 (conv4):  REVERSE_CONTAINS up — CFG→FUNCTION (Phase 2 signal rises)
  Layer 7 (conv4b): REVERSE_CONTAINS up — second hop (multi-function patterns)
  Layer 8 (conv4c): CONTAINS down (IMP-G3) — FUNCTION→CFG, distributes enriched
    FUNCTION context back to CFG children. All nodes carry Phase 3 depth after this.
  Phase 1-A3 (2026-05-14): type-7 embeddings for upward direction.
  IMP-G3: type-5 (CONTAINS) embeddings for downward direction (conv4c).

  Zero-message behaviour (correct — do not "fix"):
  FUNCTION nodes with no CFG children receive no upward Phase 3 messages.
  conv returns zero; residual x = x + dropout(0) is a no-op.

JK Connections (Phase 1-A1, 2026-05-14)
─────────────────────────────────────────
Learned attention aggregation over all three phase outputs. Prevents Phase 1
structural signal from being over-smoothed by phases 2 and 3.

Per-Phase LayerNorm (Phase 1-A2, 2026-05-14)
──────────────────────────────────────────────
Applied once after each complete phase (after both layers of that phase),
before collecting for JK. Prevents Phase 1's higher norm (two conv layers)
from dominating the JK attention softmax.

PARAMETERS (v8 defaults)
─────────────────────────
  in_channels   = NODE_FEATURE_DIM (11)
  hidden_dim    = 256    (was 128 — doubles capacity for complex vulnerability patterns)
  heads         = 8 (Phase 1 only; Phases 2+3 use heads=1)
  dropout       = 0.2
  use_edge_attr = True
  edge_emb_dim  = 64     (was 32 — 64/8 = 8 dims per edge type vs 4 previously)
  num_layers    = 8      (2+3+3 phases; IMP-G3 added downward CONTAINS pass as conv4c)
  use_jk        = True
  jk_mode       = 'attention'

Total trainable parameters (v8 defaults): ~2.4M GNN (was ~91K at hidden=128)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.nn import GATConv

from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES

# [A27] Architecture is fixed at 8 layers (2+3+3 phases). Enforced in GNNEncoder.__init__.
SENTINEL_GNN_NUM_LAYERS: int = 8

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
        # Per-phase std across all nodes in the batch — same shape [K] as last_weights.
        # Tells the trainer whether the attention is behaving as a global constant
        # (std ≈ 0) or genuinely routing different node types to different phases (std > 0.10).
        self.register_buffer("last_weight_stds", torch.zeros(num_phases))
        # Per-node weights stored in eval mode only (N varies per batch — not a buffer).
        # Shape [N, K] after each forward pass in eval mode; None during training.
        # Read by jk_weight_hist.py diagnostic; zero cost in training.
        self.last_node_weights: "torch.Tensor | None" = None

    def forward(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xs: list of K tensors each [N, channels]
        Returns: tuple([N, channels], scalar entropy)

        Side effects:
          - always:    stores mean per-phase weights in self.last_weights [K]
          - eval mode: stores full per-node weights in self.last_node_weights [N, K]
                       for use by jk_weight_hist.py diagnostic script
        """
        stacked = torch.stack(xs, dim=1)        # [N, K, channels]
        scores  = self.attn(stacked)             # [N, K, 1]
        weights = torch.softmax(scores, dim=1)   # [N, K, 1]  (sum-to-1 over K)
        w_nk    = weights.squeeze(-1)            # [N, K]
        # Update registered buffers in-place (preserves device + save/load).
        self.last_weights.copy_(w_nk.mean(0).detach())      # [K]
        self.last_weight_stds.copy_(w_nk.std(0, unbiased=False).nan_to_num(0.0).detach())  # [A23] unbiased=False avoids NaN when N=1
        # Full per-node weights: only kept in eval mode (N varies, can't be a buffer).
        if not self.training:
            self.last_node_weights = w_nk.detach().cpu()
        output = (weights * stacked).sum(dim=1)
        # C-3: Mean entropy over nodes. Gradient-attached for JK entropy regularizer.
        # H in [0, log(K)]; low H = collapsed attention = one phase dominates.
        jk_entropy = -(w_nk * (w_nk + 1e-8).log()).sum(dim=1).mean()
        return output, jk_entropy


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
        hidden_dim:               int            = 256,
        heads:                    int            = 8,
        dropout:                  float          = 0.2,
        use_edge_attr:            bool           = True,
        edge_emb_dim:             int            = 64,
        num_layers:               int            = 8,
        use_jk:                   bool           = True,
        jk_mode:                  str            = 'attention',
        phase2_edge_types:        list[int]|None = None,
        validate_graph_integrity: bool           = False,  # [A25] off by default; O(E) scan gated here
    ) -> None:
        """
        Args:
            hidden_dim:         Node embedding width (default 256).
            heads:              Multi-head count for Phase 1 (Phases 2+3 always use 1).
            dropout:            Dropout probability applied after each conv layer.
            use_edge_attr:      If True, embed edge types and feed to GATConv.
            edge_emb_dim:       Edge type embedding dimension (default 64).
            num_layers:         Stored for serialisation; validation in TrainConfig.
            use_jk:             If True, use JK attention aggregation over all three
                                phase outputs instead of returning only Phase 3.
            jk_mode:            'attention' only.
            phase2_edge_types:  Edge type IDs to include in Phase 2 cfg_mask.
                                None (default) = all four v8 types:
                                [CF(6), CALL_ENTRY(8), RETURN_TO(9), DEF_USE(10)].
                                Ablation examples:
                                  ICFG-only: [6, 8, 9]
                                  DFG-only:  [6, 10]
        """
        super().__init__()

        # [A27] Architecture is fixed at SENTINEL_GNN_NUM_LAYERS (8) layers.
        if num_layers != SENTINEL_GNN_NUM_LAYERS:
            raise ValueError(
                f"GNNEncoder: num_layers={num_layers} but the three-phase architecture "
                f"is fixed at {SENTINEL_GNN_NUM_LAYERS} layers (2+3+3). "
                "Passing a different value produces a structurally incorrect model. "
                f"Pass num_layers={SENTINEL_GNN_NUM_LAYERS} or omit it."
            )

        self.num_layers               = num_layers
        self.hidden_dim               = hidden_dim
        self.use_edge_attr            = use_edge_attr
        self.dropout_p                = dropout
        self.use_jk                   = use_jk
        self.jk_mode                  = jk_mode
        self.phase2_edge_types        = phase2_edge_types
        self.validate_graph_integrity = validate_graph_integrity  # [A25]

        _head_dim = hidden_dim // heads  # 32 per head when hidden=256, heads=8
        if _head_dim * heads != hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}). "
                f"Each head needs hidden_dim/heads = {hidden_dim/heads:.1f} dims."
            )

        # Edge type embedding: covers all 11 edge types including REVERSE_CONTAINS(7)
        # (runtime-only), CALL_ENTRY(8) + RETURN_TO(9) (v8 ICFG-Lite, on disk), DEF_USE(10).
        _edge_dim: int | None = None
        if use_edge_attr:
            self.edge_embedding = nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)
            _edge_dim = edge_emb_dim
        else:
            self.edge_embedding = None

        # IMP-G2: learned skip connection from raw features to Phase 1 Layer 1 output.
        # Prevents raw 11-dim feature information from being lost when GAT attention
        # weights are poorly initialised. bias=False avoids double-counting the conv bias.
        # 11 × 256 = 2,816 parameters — negligible.
        self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim, bias=False)

        # ── Phase 1 — structural + CONTAINS (Layers 1+2) ────────────────────
        # out_channels is PER HEAD in PyG GATConv. With heads=8 and concat=True:
        #   total output = 8 × _head_dim = 8 × 32 = 256 = hidden_dim.
        self.conv1 = GATConv(
            in_channels=NODE_FEATURE_DIM,  # 11
            out_channels=_head_dim,         # 32 per head
            heads=heads,                    # 8 → total 256
            concat=True,
            add_self_loops=True,
            edge_dim=_edge_dim,
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=_head_dim,
            heads=heads,
            concat=True,
            add_self_loops=True,
            edge_dim=_edge_dim,
        )

        # ── Phase 2 — CFG + ICFG directed (Layers 3+4+5) ───────────────────
        # Edge types: CONTROL_FLOW(6) + CALL_ENTRY(8) + RETURN_TO(9).
        # add_self_loops=False — CRITICAL: self-loops cancel directional CF signal.
        # heads=1: one head with full hidden_dim capacity for execution-order encoding.
        # Layer 3: first hop  (intra-function successors + cross-function CALL_ENTRY).
        # Layer 4: second hop (CEI/CEA pattern + callee body via ICFG).
        # Layer 5 (conv3c): third hop — ENTRY→CALL→TMP→WRITE full CEI sequence.
        self.conv3 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,     # CRITICAL
            edge_dim=_edge_dim,
        )
        self.conv3b = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,     # CRITICAL
            edge_dim=_edge_dim,
        )
        # BUG-H1: Layer 3c — 3rd CF hop.
        # 2 hops (conv3+conv3b): covers CALL→TMP→WRITE (classic 2-step CEI).
        # 3 hops (conv3c):       covers ENTRY→CALL→TMP→WRITE — start-of-function
        # CALL that is followed two steps later by the state WRITE. Captures the
        # full CEI sequence from entry to write in one aggregation pass.
        self.conv3c = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,     # CRITICAL
            edge_dim=_edge_dim,
        )

        # ── Phase 3 — bidirectional CONTAINS (Layers 6+7+8) ─────────────────
        # Layer 6 (conv4):  reverse-CONTAINS up — CFG→FUNCTION (Phase 2 signal rises)
        # Layer 7 (conv4b): reverse-CONTAINS up — second hop (multi-function patterns)
        # Layer 8 (conv4c): forward-CONTAINS down — FUNCTION→CFG (IMP-G3: distribute
        #   enriched FUNCTION context back to CFG children so CrossAttentionFusion
        #   sees equally-enriched embeddings from both FUNCTION and CFG node types)
        self.conv4 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,
            edge_dim=_edge_dim,
        )
        self.conv4b = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,
            edge_dim=_edge_dim,
        )
        # IMP-G3: downward pass — FUNCTION nodes distribute Phase 3 context to CFG children.
        self.conv4c = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,
            edge_dim=_edge_dim,
        )

        # ── Per-phase LayerNorm ──────────────────────────────────────────────
        # Applied once after each complete phase (after both layers), before
        # collecting for JK. Prevents Phase 1's higher norm from dominating JK.
        self.phase_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim),  # after Phase 1
            nn.LayerNorm(hidden_dim),  # after Phase 2
            nn.LayerNorm(hidden_dim),  # after Phase 3
        ])

        # ── JK attention aggregator ──────────────────────────────────────────
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

        # [A26] Cache parameter dtype — avoids next(self.parameters()) on every forward pass.
        # Call refresh_dtype_cache() after any runtime dtype cast (.float(), .half(), etc.).
        self._param_dtype: torch.dtype = next(self.parameters()).dtype

    def refresh_dtype_cache(self) -> None:
        """[A26] Update cached dtype after a runtime cast (e.g., model.float(), model.bfloat16())."""
        self._param_dtype = next(self.parameters()).dtype

    def forward(
        self,
        x:                    torch.Tensor,
        edge_index:           torch.Tensor,
        batch:                torch.Tensor,
        edge_attr:            torch.Tensor | None = None,
        return_intermediates: bool               = False,
        return_phase2_embs:   bool               = False,
    ):
        """
        Three-phase forward pass.

        Args:
            x:                    [N, NODE_FEATURE_DIM]
            edge_index:           [2, E]
            batch:                [N]
            edge_attr:            [E] int64 edge type IDs in [0, NUM_EDGE_TYPES).
                                  Required when use_edge_attr=True.
            return_intermediates: When True, also return per-phase embeddings dict
                                  (diagnostic only — detached tensors, no gradients).
            return_phase2_embs:   When True, return Phase 2 output tensor WITH
                                  gradients attached (for CEI auxiliary loss).
                                  Cannot be combined with return_intermediates.

        Returns (return_intermediates=False, return_phase2_embs=False):
            node_embeddings [N, hidden_dim], batch [N], jk_entropy (scalar)
        Returns (return_intermediates=True):
            node_embeddings [N, hidden_dim], batch [N], jk_entropy (scalar),
            {"after_phase1": ..., "after_phase2": ..., "after_phase3": ...}
            (intermediates dict is diagnostic only — detached tensors, not used for gradients)
        Returns (return_phase2_embs=True):
            node_embeddings [N, hidden_dim], batch [N], jk_entropy (scalar),
            phase2_x [N, hidden_dim] — Phase 2 output WITH gradient (for aux loss)
        """
        # ── Guards ───────────────────────────────────────────────────────────
        if x.shape[1] != NODE_FEATURE_DIM:
            raise ValueError(
                f"GNNEncoder expects {NODE_FEATURE_DIM}-dim node features (schema v8) "
                f"but got {x.shape[1]}. Likely a stale v6 .pt file — re-run reextract_graphs.py."
            )
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
        # [A25] Gated by validate_graph_integrity (default False) — edge_index.max() is O(E)
        # and runs on every forward pass. Enable for debugging/testing only; move the check
        # to DualPathDataset.__getitem__ or the collation function for production validation.
        if self.validate_graph_integrity and edge_index.numel() > 0 and edge_index.max() >= x.shape[0]:
            raise ValueError(
                f"edge_index contains node index {edge_index.max().item()} "
                f"but x has only {x.shape[0]} nodes. Graph .pt file is corrupted."
            )

        # Normalise input dtype to match model parameters (float32).
        # BERT loads in BF16 and can set the global default dtype, causing test
        # tensors created with torch.randn to arrive as BF16.
        # [A26] Use cached _param_dtype — call refresh_dtype_cache() after any dtype cast.
        if x.dtype != self._param_dtype:
            x = x.to(self._param_dtype)

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
                _max_valid = self.edge_embedding.num_embeddings - 1
                _oob_mask = (edge_attr < 0) | (edge_attr > _max_valid)
                if _oob_mask.any():
                    logger.warning(
                        f"GNNEncoder: {_oob_mask.sum().item()} OOB edge_attr value(s) "
                        f"clamped to [0, {_max_valid}] "
                        f"(observed min={edge_attr.min().item()}, max={edge_attr.max().item()}). "
                        "Source .pt file may be corrupted or uses a newer schema than this model."
                    )
                    edge_attr = edge_attr.clamp(0, _max_valid)
            e = self.edge_embedding(edge_attr)   # [E, edge_emb_dim]

        # ── Edge masks — one per phase ───────────────────────────────────────
        # struct_mask:   types 0–CONTAINS (all structural + CONTAINS forward)
        # cfg_mask:      CONTROL_FLOW(6) + CALL_ENTRY(8) + RETURN_TO(9) — Phase 2
        # contains_mask: CONTAINS only; used to build Phase 3 reverse edges
        _CONTAINS         = EDGE_TYPES["CONTAINS"]          # 5
        _CONTROL_FLOW     = EDGE_TYPES["CONTROL_FLOW"]       # 6
        _REVERSE_CONTAINS = EDGE_TYPES["REVERSE_CONTAINS"]   # 7 (runtime-only)
        _CALL_ENTRY       = EDGE_TYPES["CALL_ENTRY"]         # 8 (v8 ICFG-Lite)
        _RETURN_TO        = EDGE_TYPES["RETURN_TO"]          # 9 (v8 ICFG-Lite)
        _DEF_USE          = EDGE_TYPES["DEF_USE"]            # 10 (v8 data-flow)
        if edge_attr is not None:
            struct_mask   = edge_attr <= _CONTAINS
            if self.phase2_edge_types is not None:
                cfg_mask = torch.zeros(edge_attr.shape[0], dtype=torch.bool, device=edge_index.device)
                for _t in self.phase2_edge_types:
                    cfg_mask |= (edge_attr == _t)
            else:
                cfg_mask  = (
                    (edge_attr == _CONTROL_FLOW) |
                    (edge_attr == _CALL_ENTRY)   |
                    (edge_attr == _RETURN_TO)    |
                    (edge_attr == _DEF_USE)
                )
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

        phase2_ei    = edge_index[:, cfg_mask]
        phase2_ea    = e[cfg_mask]      if e is not None else None

        # IMP-G1: layer-specific Phase 2 edge subsets so each layer builds distinct context.
        # Layer 3 (conv3):  CONTROL_FLOW only — intra-function execution order
        # Layer 4 (conv3b): CALL_ENTRY + RETURN_TO only — cross-function call structure
        # Layer 5 (conv3c): joint CF+ICFG — integration layer
        # When phase2_edge_types ablation excludes a type, the subset becomes empty;
        # GATConv on empty edges returns zero which residual handles cleanly.
        #
        # NF-6 (DEFERRED to Run 6): cf_only_ei and icfg_only_ei are computed from raw
        # edge_attr, not from the already-masked phase2_ei. When phase2_edge_types ablation
        # excludes CF(6) or ICFG edges, layers 3/4 still process the full unablated set.
        # Fix by re-applying sub-masks to phase2_ei instead of edge_index (zero training
        # impact for normal runs where phase2_edge_types=None).
        if edge_attr is not None:
            _cf_mask   = (edge_attr == _CONTROL_FLOW)
            _icfg_mask = (edge_attr == _CALL_ENTRY) | (edge_attr == _RETURN_TO)
            cf_only_ei   = edge_index[:, _cf_mask]
            cf_only_ea   = e[_cf_mask]   if e is not None else None
            icfg_only_ei = edge_index[:, _icfg_mask]
            icfg_only_ea = e[_icfg_mask] if e is not None else None
        else:
            cf_only_ei = icfg_only_ei = phase2_ei
            cf_only_ea = icfg_only_ea = phase2_ea

        # Phase 3: CONTAINS edges used in both directions (IMP-G3).
        # fwd_contains_ei: FUNCTION→CFG (original CONTAINS direction)
        # rev_contains_ei: CFG→FUNCTION (flipped — Phase 2 signal rises to FUNCTION)
        # Phase 1-A3 (2026-05-14): type-7 (REVERSE_CONTAINS) embeddings for the upward
        # direction. Type-5 (CONTAINS) embeddings used for the new downward pass (conv4c).
        fwd_contains_ei = edge_index[:, contains_mask]           # [2, E_contains]
        fwd_contains_ea = e[contains_mask] if e is not None else None
        rev_contains_ei = fwd_contains_ei.flip(0)                # [2, E_contains]
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
        # IMP-G2: save raw features before any conv for the skip connection.
        x_init = x  # [N, NODE_FEATURE_DIM]

        # Layer 1: NODE_FEATURE_DIM→hidden_dim.
        # IMP-G2: add learned skip (input_proj) so raw features bypass the first
        # GAT layer — prevents feature loss when attention weights start near-uniform.
        x_skip = self.input_proj(x_init.to(self._param_dtype)).to(x.dtype)  # dtype-safe (IMP-G2) [A26]
        x  = self.conv1(x_init, struct_ei, struct_ea)   # [N, NODE_FEATURE_DIM] → [N, hidden_dim]
        x  = self.relu(x + x_skip)                      # skip added before relu (IMP-G2)
        x  = self.dropout(x)
        # Layer 2: hidden_dim→hidden_dim with residual from Layer 1.
        x2 = self.conv2(x, struct_ei, struct_ea)    # [N, 128] → [N, 128]
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                   # residual: identity preserved, only branch dropped
        # Phase 1-A2: LayerNorm before collecting for JK.
        x  = self.phase_norm[0](x)

        _live.append(x)
        _intermediates["after_phase1"] = x.detach().clone()

        # ── Phase 2: CONTROL_FLOW + ICFG directed (Layers 3+4+5) ────────────
        # IMP-G1: each layer processes a distinct edge subset so the JK mechanism
        # has genuinely different representations to aggregate from.
        # Layer 3 (conv3):  CF only — intra-function execution ordering (CEI within fn)
        # Layer 4 (conv3b): CALL_ENTRY + RETURN_TO — cross-function call structure
        # Layer 5 (conv3c): CF + CALL_ENTRY + RETURN_TO joint — integration layer that
        #                   learns from nodes already enriched by layers 3 and 4.
        x2 = self.conv3(x, cf_only_ei, cf_only_ea)          # Layer 3: CF only
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)
        x2 = self.conv3b(x, icfg_only_ei, icfg_only_ea)     # Layer 4: ICFG only
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)
        x2 = self.conv3c(x, phase2_ei, phase2_ea)            # Layer 5: Phase 2 joint (CF+ICFG)
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)
        x  = self.phase_norm[1](x)                          # LayerNorm after complete Phase 2

        _live.append(x)
        _intermediates["after_phase2"] = x.detach().clone()
        _phase2_x = x  # keep gradient-attached reference for aux loss (return_phase2_embs)

        # ── Phase 3: bidirectional CONTAINS (Layers 6+7+8) ──────────────────
        # Layer 6 (conv4):  CFG→FUNCTION up — Phase-2-enriched CFG signal rises
        # Layer 7 (conv4b): second up hop — multi-function vulnerability propagation
        # Layer 8 (conv4c): FUNCTION→CFG down (IMP-G3) — FUNCTION nodes distribute
        #   their enriched context back to CFG children. After this pass, ALL nodes
        #   carry Phase 3 context so CrossAttentionFusion sees a uniform depth.
        x2 = self.conv4(x, rev_contains_ei, rev_contains_ea)
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)
        x2 = self.conv4b(x, rev_contains_ei, rev_contains_ea)
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)
        # Downward pass: FUNCTION → CFG children (original CONTAINS direction, type-5 embs)
        x2 = self.conv4c(x, fwd_contains_ei, fwd_contains_ea)
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)
        x  = self.phase_norm[2](x)                  # LayerNorm covers full bidirectional Phase 3

        _live.append(x)
        _intermediates["after_phase3"] = x.detach().clone()

        # ── JK aggregation (Phase 1-A1) ──────────────────────────────────────
        # Weighted sum over all three phase outputs using learned per-node
        # attention scores.  When use_jk=False, only Phase 3 output is returned
        # (same behaviour as v5.0, preserved for checkpoint compatibility).
        if self.use_jk and self.jk is not None:
            x, _jk_entropy = self.jk(_live)
        else:
            _jk_entropy = x.new_zeros(1).squeeze()

        if return_intermediates:
            return x, batch, _jk_entropy, _intermediates
        if return_phase2_embs:
            return x, batch, _jk_entropy, _phase2_x
        return x, batch, _jk_entropy
