"""
gnn_encoder.py — GNN Encoder for SENTINEL (v5 architecture)

V5 CHANGES FROM V4
───────────────────
- in_channels = NODE_FEATURE_DIM (imported — never hardcoded here)
- hidden_dim  = 128 default (was 64)
- 4-layer GAT (was 3)
- Residual connections on layers 2, 3, and 4 (prevent vanishing gradients)
- num_layers constructor argument (default 4; wired from TrainConfig.gnn_layers)
- Edge embedding: nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim) covers 7 types (was 5)

ARCHITECTURE DETAILS (v5 defaults)
───────────────────────────────────
Layer 1 (GATConv):
  Input:  x [N, NODE_FEATURE_DIM], edge_index [2, E], edge_emb [E, edge_emb_dim]
  Config: heads=8, concat=True → each head outputs hidden_dim/heads = 16 dims
  Output: [N, hidden_dim=128]
  After:  ReLU + Dropout(0.2)

Layer 2 (GATConv):
  Input:  [N, 128]
  Output: [N, 128]
  After:  ReLU + Dropout(0.2) + residual from Layer 1

Layer 3 (GATConv):
  Input:  [N, 128]
  Output: [N, 128]
  After:  ReLU + Dropout(0.2) + residual from Layer 2

Layer 4 (GATConv):
  Input:  [N, 128]
  Config: heads=1, concat=False → output = hidden_dim exactly
  Output: [N, 128]
  After:  no activation — passed directly to SentinelModel for pooling

Edge embeddings:
  nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim) — covers all 7 v2 edge types.
  When edge_attr=None (old .pt files), a zero tensor of shape [E, edge_emb_dim]
  is used — degrades gracefully to type-agnostic attention.

WHY RESIDUALS
─────────────
Without residuals, a 4-layer GAT can suffer vanishing gradients because each
GATConv applies dropout and a learned projection. Adding `x2 = f(x1) + x1`
between layers keeps gradient magnitude stable and allows the model to learn
"what to add" rather than "what to represent from scratch" at each hop.
Shapes always match (all intermediate dims = hidden_dim) so no projection is
needed for the skip connection.

WHY NOT global_mean_pool INSIDE THIS MODULE
────────────────────────────────────────────
Pooling is deferred to SentinelModel so CrossAttentionFusion can enrich each
node with relevant CodeBERT token context before pooling.  SentinelModel also
computes the GNN eye via global_max_pool + global_mean_pool → cat → project.

Total trainable parameters (v5 defaults, 4 layers, edge_emb_dim=32): ~88K
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES


class GNNEncoder(nn.Module):
    """
    4-layer GAT encoder for smart contract graphs with edge-type embeddings.

    Returns node-level embeddings (NOT pooled) so SentinelModel can pool them
    separately for the GNN eye and feed them to CrossAttentionFusion.

    Input:
        x:          node features      [N, NODE_FEATURE_DIM]  — N nodes across batch
        edge_index: graph connectivity [2, E]
        batch:      node→graph mapping [N]
        edge_attr:  edge type IDs      [E]  — int64 in [0, NUM_EDGE_TYPES)
                    Optional. None → zeros used (graceful degradation).

    Output:
        node_embeddings: [N, hidden_dim]  — one vector per node, NOT pooled
        batch:           [N]              — passed through unchanged
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

        if num_layers != 4:
            # Guard against unsupported configs until the loop-based generalisation
            # is implemented. Changing num_layers without updating the forward pass
            # would silently produce wrong shapes.
            raise NotImplementedError(
                f"GNNEncoder currently only supports num_layers=4 (got {num_layers}). "
                "Update the forward() method before using a different depth."
            )

        self.hidden_dim    = hidden_dim
        self.use_edge_attr = use_edge_attr
        self.edge_emb_dim  = edge_emb_dim
        self.num_layers    = num_layers

        head_dim = hidden_dim // heads
        if head_dim * heads != hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}). "
                f"Each head needs hidden_dim/heads = {hidden_dim/heads:.1f} dims."
            )

        # Edge type embedding: int ID [E] → dense vector [E, edge_emb_dim].
        # NUM_EDGE_TYPES from graph_schema — covers all EDGE_TYPES entries.
        _edge_dim: int | None = None
        if use_edge_attr:
            self.edge_emb = nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)
            _edge_dim = edge_emb_dim
        else:
            self.edge_emb = None

        # Layer 1: NODE_FEATURE_DIM input → hidden_dim (concat=True, 8 heads)
        # Uses NODE_FEATURE_DIM from graph_schema — never hardcode 8 here.
        self.conv1 = GATConv(
            in_channels=NODE_FEATURE_DIM, out_channels=head_dim,
            heads=heads, concat=True, dropout=dropout,
            edge_dim=_edge_dim,
        )

        # Layers 2–3: hidden_dim → hidden_dim (same shape; enables residuals)
        self.conv2 = GATConv(
            in_channels=hidden_dim, out_channels=head_dim,
            heads=heads, concat=True, dropout=dropout,
            edge_dim=_edge_dim,
        )
        self.conv3 = GATConv(
            in_channels=hidden_dim, out_channels=head_dim,
            heads=heads, concat=True, dropout=dropout,
            edge_dim=_edge_dim,
        )

        # Layer 4: final aggregation — heads=1, concat=False → output = hidden_dim.
        # No activation after this layer; SentinelModel applies pooling + projection.
        self.conv4 = GATConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            heads=1, concat=False, dropout=dropout,
            edge_dim=_edge_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(
        self,
        x:          torch.Tensor,                    # [N, NODE_FEATURE_DIM]
        edge_index: torch.Tensor,                    # [2, E]
        batch:      torch.Tensor,                    # [N]
        edge_attr:  Optional[torch.Tensor] = None,   # [E] int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_embeddings: [N, hidden_dim] — raw node embeddings, NOT pooled
            batch:           [N]             — unchanged, passed to SentinelModel
        """
        # Build edge embeddings
        if self.edge_emb is not None:
            if edge_attr is not None:
                e = self.edge_emb(edge_attr)   # [E] → [E, edge_emb_dim]
            else:
                # Graceful degradation: old .pt files may lack edge_attr.
                e = torch.zeros(
                    edge_index.shape[1], self.edge_emb_dim,
                    dtype=torch.float32, device=x.device,
                )
        else:
            e = None

        # Layer 1: 1-hop structural context per node
        x1 = self.conv1(x, edge_index, edge_attr=e)   # [N, hidden_dim]
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Layer 2: 2-hop context + residual from layer 1
        x2 = self.conv2(x1, edge_index, edge_attr=e)  # [N, hidden_dim]
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = x2 + x1  # residual: preserves layer-1 signal in deeper layers

        # Layer 3: 3-hop context + residual from layer 2
        x3 = self.conv3(x2, edge_index, edge_attr=e)  # [N, hidden_dim]
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = x3 + x2  # residual

        # Layer 4: 4-hop context, final node embeddings
        # No activation — SentinelModel applies pooling and its own projections
        x4 = self.conv4(x3, edge_index, edge_attr=e)  # [N, hidden_dim]
        x4 = x4 + x3  # residual into final layer keeps gradients healthy

        return x4, batch
