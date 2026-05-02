"""
gnn_encoder.py — GNN Encoder for SENTINEL (Cross-Attention Upgrade)

WHAT CHANGED FROM ORIGINAL:
    - global_mean_pool REMOVED from forward()
    - Returns (node_embeddings [N, hidden_dim], batch [N]) instead of [B, hidden_dim]
    - Pooling is now deferred to CrossAttentionFusion so each node can
      query CodeBERT token embeddings BEFORE averaging destroys node-level detail

P0-B/C CHANGES (2026-05-02):
    - edge_attr support added: GATConv now receives learned edge-type embeddings
      (CALLS, READS, WRITES, EMITS, INHERITS → 5-class → R^edge_emb_dim vector)
      so relation type is visible to attention — previously this information was
      fully discarded even though graph_extractor.py computed it.
    - Architecture is now configurable: hidden_dim, heads, dropout, edge_emb_dim
      are constructor params so TrainConfig can drive hyperparameter search.
    - Defaults are backward-compatible with the original hardcoded architecture
      (hidden_dim=64, heads=8, dropout=0.2, use_edge_attr=True, edge_emb_dim=16).

WHY edge_attr matters:
    A function node CALLS another vs READS a state variable are very different
    structural patterns. Reentrancy requires a CALLS edge back to the caller.
    Without edge_attr, the GATConv attention score is purely based on node
    features — it cannot distinguish "this node calls something" from
    "this node reads something".

WHAT DID NOT CHANGE:
    - 3-layer GAT architecture (conv1 → conv2 → conv3)
    - Node feature input dimension: in_channels=8 (locked by graph_schema.py)
    - ReLU + Dropout pattern between layers
    - Output: node-level embeddings (NOT pooled)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from ml.src.preprocessing.graph_schema import NUM_EDGE_TYPES


class GNNEncoder(nn.Module):
    """
    3-layer GAT encoder for smart contract graphs with optional edge-type embeddings.

    Returns node-level embeddings (NOT pooled) so CrossAttentionFusion
    can enrich each node with relevant CodeBERT token context before pooling.

    Input:
        x:          node features      [N, 8]   — N total nodes across batch
        edge_index: graph connectivity [2, E]
        batch:      node→graph mapping [N]
        edge_attr:  edge type IDs      [E]      — int64, values in [0, NUM_EDGE_TYPES)
                    Optional. If None and use_edge_attr=True, edge information
                    is replaced with zeros (graceful degradation for old .pt files).

    Output:
        node_embeddings: [N, hidden_dim]  — one vector per node (NOT pooled)
        batch:           [N]              — passed through to CrossAttentionFusion
    """

    def __init__(
        self,
        hidden_dim:    int   = 64,
        heads:         int   = 8,
        dropout:       float = 0.2,
        use_edge_attr: bool  = True,
        edge_emb_dim:  int   = 16,
    ) -> None:
        super().__init__()

        self.hidden_dim    = hidden_dim
        self.use_edge_attr = use_edge_attr
        self.edge_emb_dim  = edge_emb_dim

        # Each head gets hidden_dim // heads dimensions; concat=True multiplies back.
        head_dim = hidden_dim // heads
        if head_dim * heads != hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}). "
                f"Each head needs hidden_dim/heads = {hidden_dim/heads} dims."
            )

        # Edge type embedding: int ID [E] → dense vector [E, edge_emb_dim]
        # NUM_EDGE_TYPES=5 from graph_schema (CALLS, READS, WRITES, EMITS, INHERITS)
        _edge_dim = None
        if use_edge_attr:
            self.edge_emb = nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)
            _edge_dim = edge_emb_dim
        else:
            self.edge_emb = None

        # Layer 1: 8-dim input → heads × head_dim = hidden_dim (concat=True)
        self.conv1 = GATConv(
            in_channels=8, out_channels=head_dim,
            heads=heads, concat=True, dropout=dropout,
            edge_dim=_edge_dim,
        )

        # Layer 2: hidden_dim → hidden_dim, node now has 2-hop context
        self.conv2 = GATConv(
            in_channels=hidden_dim, out_channels=head_dim,
            heads=heads, concat=True, dropout=dropout,
            edge_dim=_edge_dim,
        )

        # Layer 3: collapse to clean hidden_dim node embedding
        # concat=False, heads=1 → output stays hidden_dim (not heads×hidden_dim)
        self.conv3 = GATConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            heads=1, concat=False, dropout=dropout,
            edge_dim=_edge_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(
        self,
        x:          torch.Tensor,           # [N, 8]  node features
        edge_index: torch.Tensor,           # [2, E]  edges
        batch:      torch.Tensor,           # [N]     node→graph mapping
        edge_attr:  Optional[torch.Tensor] = None,  # [E]  edge type IDs (int64)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_embeddings: [N, hidden_dim] — raw node embeddings, NOT pooled
            batch:           [N]             — unchanged, passed to CrossAttentionFusion
        """
        # Build edge embeddings if edge_attr is available
        if self.edge_emb is not None:
            if edge_attr is not None:
                e = self.edge_emb(edge_attr)   # [E] → [E, edge_emb_dim]
            else:
                # Graceful degradation: old .pt files may lack edge_attr.
                # Use zeros so the model still runs; attention will treat all
                # edge types as equivalent (same as pre-P0-B behaviour).
                e = torch.zeros(
                    edge_index.shape[1], self.edge_emb_dim,
                    dtype=torch.float32, device=x.device,
                )
        else:
            e = None

        # Layer 1: 1-hop structural context per node
        x = self.conv1(x, edge_index, edge_attr=e)   # [N, 8] → [N, hidden_dim]
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2: 2-hop context — each node now sees neighbours' neighbours
        x = self.conv2(x, edge_index, edge_attr=e)   # [N, hidden_dim] → [N, hidden_dim]
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3: 3-hop context, final node embeddings
        # No relu/dropout here — CrossAttentionFusion applies its own projections
        x = self.conv3(x, edge_index, edge_attr=e)   # [N, hidden_dim] → [N, hidden_dim]

        return x, batch
