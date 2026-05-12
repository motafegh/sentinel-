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
  Edges: type 5 CONTAINS, REVERSED (CFG_NODE → FUNCTION direction)
  add_self_loops=False
  heads=1, concat=False → output stays hidden_dim (128)
  Purpose: aggregate Phase-2-enriched CFG embeddings UP into FUNCTION nodes.
  This is the path by which execution-order information reaches the function
  nodes that the classifier operates on. Without Phase 3, function-node
  embeddings are order-blind regardless of CFG machinery below them.

  Known limitation (v5.0): reversed CONTAINS edges use the same type-5
  embedding as forward CONTAINS edges — the GNN cannot encode directional
  asymmetry from the edge attribute alone. GATConv positional asymmetry
  provides partial compensation. v5.1 target: REVERSE_CONTAINS = 7.

  Zero-message behaviour (correct — do not "fix"):
  FUNCTION nodes with no CFG children receive no Phase 3 messages.
  conv4 returns zero for them; residual x = x + dropout(0) is a no-op.
  They retain their Phase 2 embedding. Adding add_self_loops=True to
  "fix" this would dilute the Phase 2 order signal for functions that DO
  have CFG children.

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
  in_channels  = NODE_FEATURE_DIM (12) — never hardcode 8 or 13 here
  hidden_dim   = 128
  heads        = 8 (Phase 1 only; Phases 2+3 use heads=1)
  dropout      = 0.2
  use_edge_attr = True
  edge_emb_dim  = 32  (nn.Embedding(NUM_EDGE_TYPES=7, 32))
  num_layers    = 4

Total trainable parameters (v5 defaults): ~90K
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES


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
    ) -> None:
        super().__init__()

        # Stored for serialisation; validation fires in TrainConfig.__post_init__().
        self.num_layers    = num_layers
        self.hidden_dim    = hidden_dim
        self.use_edge_attr = use_edge_attr
        self.dropout_p     = dropout

        _head_dim = hidden_dim // heads  # 16 per head when hidden=128, heads=8
        if _head_dim * heads != hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}). "
                f"Each head needs hidden_dim/heads = {hidden_dim/heads:.1f} dims."
            )

        # Edge type embedding: [E] int64 → [E, edge_emb_dim]
        # Covers all NUM_EDGE_TYPES (7) edge relation types.
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
        # Uses CONTAINS edges with src↔dst flipped.
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
            edge_attr:            [E] int64 edge type IDs — required for phase masks.
                                  None → all phases use no edge features.
            return_intermediates: When True, also return per-phase embeddings dict.

        Returns (return_intermediates=False):
            node_embeddings [N, hidden_dim], batch [N]
        Returns (return_intermediates=True):
            node_embeddings [N, hidden_dim], batch [N],
            {"after_phase1": ..., "after_phase2": ..., "after_phase3": ...}
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
            e = self.edge_embedding(edge_attr)   # [E, edge_emb_dim]

        # ── Edge masks — one per phase ───────────────────────────────────────
        # struct_mask:   types 0–CONTAINS (all structural + CONTAINS forward)
        # cfg_mask:      CONTROL_FLOW only
        # contains_mask: CONTAINS only; used for Phase 3 reversal
        _CONTAINS     = EDGE_TYPES["CONTAINS"]       # 5
        _CONTROL_FLOW = EDGE_TYPES["CONTROL_FLOW"]   # 6
        if edge_attr is not None:
            struct_mask   = edge_attr <= _CONTAINS
            cfg_mask      = edge_attr == _CONTROL_FLOW
            contains_mask = edge_attr == _CONTAINS
        else:
            # Without edge_attr: all edges participate in all phases (degraded mode)
            n_edges = edge_index.shape[1]
            struct_mask   = torch.ones(n_edges, dtype=torch.bool, device=edge_index.device)
            cfg_mask      = torch.zeros(n_edges, dtype=torch.bool, device=edge_index.device)
            contains_mask = torch.zeros(n_edges, dtype=torch.bool, device=edge_index.device)

        struct_ei = edge_index[:, struct_mask]
        struct_ea = e[struct_mask]   if e is not None else None

        cfg_ei    = edge_index[:, cfg_mask]
        cfg_ea    = e[cfg_mask]      if e is not None else None

        # Phase 3: flip CONTAINS edges so CFG_NODE → FUNCTION (child → parent).
        # Both forward and reversed CONTAINS use the same type-5 embedding — this
        # is the v5.0 known limitation (see module docstring).
        rev_contains_ei = edge_index[:, contains_mask].flip(0)   # [2, E_contains]
        rev_contains_ea = e[contains_mask] if e is not None else None

        _intermediates: dict = {}

        # ── Phase 1: structural aggregation (Layers 1+2) ────────────────────
        # Layer 1: NODE_FEATURE_DIM→hidden_dim. No residual — dims differ (12≠128).
        x  = self.conv1(x, struct_ei, struct_ea)    # [N, 12] → [N, 128]
        x  = self.relu(x)
        x  = self.dropout(x)
        # Layer 2: hidden_dim→hidden_dim with residual from Layer 1.
        x2 = self.conv2(x, struct_ei, struct_ea)    # [N, 128] → [N, 128]
        x2 = self.relu(x2)
        x  = self.dropout(x2 + x)                   # residual ✓ (same dim)

        _intermediates["after_phase1"] = x.detach().clone()

        # ── Phase 2: CONTROL_FLOW directed (Layer 3) ────────────────────────
        # Non-CFG_NODE nodes have no CONTROL_FLOW edges — GATConv returns zero
        # for nodes with no incoming edges. They carry their Phase 1 embeddings forward.
        x2 = self.conv3(x, cfg_ei, cfg_ea)          # [N, 128] → [N, 128]
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                   # residual

        _intermediates["after_phase2"] = x.detach().clone()

        # ── Phase 3: reverse-CONTAINS (Layer 4) ─────────────────────────────
        # Phase-2-enriched CFG_NODE embeddings flow UP to FUNCTION nodes.
        # FUNCTION nodes with no CFG children receive zero messages — this is
        # correct behaviour (no-op residual). Do NOT add add_self_loops=True.
        x2 = self.conv4(x, rev_contains_ei, rev_contains_ea)  # [N, 128]
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                   # residual

        _intermediates["after_phase3"] = x.detach().clone()

        if return_intermediates:
            return x, batch, _intermediates
        return x, batch
