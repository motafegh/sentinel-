"""
gnn_encoder.py — GNN Encoder for SENTINEL (Cross-Attention Upgrade)

WHAT CHANGED FROM ORIGINAL:
    - global_mean_pool REMOVED from forward()
    - Returns (node_embeddings [N, 64], batch [N]) instead of [B, 64]
    - Pooling is now deferred to CrossAttentionFusion so each node can
      query CodeBERT token embeddings BEFORE averaging destroys node-level detail

WHY:
    Original architecture pooled here → FusionLayer received two blurry
    contract-level summaries. Cross-attention needs raw node embeddings so
    withdraw() can directly attend to "call.value" and "transfer" tokens.
    Pooling on semantically-enriched nodes produces a far better [B, 64].

WHAT DID NOT CHANGE:
    - 3-layer GAT architecture (locked — retrain required for any change)
    - Node feature dimensions: in_channels=8 (locked)
    - Attention heads, dropout, ReLU — identical
    - Output node dimension: 64 (still feeds into cross-attention at 64-dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GNNEncoder(nn.Module):
    """
    3-layer GAT encoder for smart contract graphs.

    Returns node-level embeddings (NOT pooled) so CrossAttentionFusion
    can enrich each node with relevant CodeBERT token context before pooling.

    Input:
        x:          node features      [N, 8]   — N total nodes across batch
        edge_index: graph connectivity [2, E]
        batch:      node→graph mapping [N]      — e.g. [0,0,0,1,1,2,2,2]

    Output:
        node_embeddings: [N, 64]  — one 64-dim vector per node (NOT pooled)
        batch:           [N]      — passed through so fusion layer knows boundaries
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()

        # Layer 1: 8-dim input → 8 heads × 8 out = 64-dim (concat=True)
        # dropout here = attention coefficient dropout (inside GATConv)
        self.conv1 = GATConv(
            in_channels=8, out_channels=8,
            heads=8, concat=True, dropout=dropout,
        )

        # Layer 2: 64 → 64, node now has 2-hop context
        self.conv2 = GATConv(
            in_channels=64, out_channels=8,
            heads=8, concat=True, dropout=dropout,
        )

        # Layer 3: collapse 8 heads → clean 64-dim node embedding
        # concat=False, heads=1 → output stays 64-dim (not 8×64=512)
        self.conv3 = GATConv(
            in_channels=64, out_channels=64,
            heads=1, concat=False, dropout=dropout,
        )

        # Node feature dropout — applied after each conv layer EXCEPT conv3
        # Not after conv3: node embeddings feed directly into cross-attention,
        # corrupting them here creates train/inference mismatch (same principle
        # as the dropout-before-pooling bug we discussed)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,           # [N, 8]  node features
        edge_index: torch.Tensor,  # [2, E]  edges
        batch: torch.Tensor,       # [N]     node→graph mapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_embeddings: [N, 64] — raw node embeddings, NOT pooled
            batch:           [N]     — unchanged, passed to CrossAttentionFusion
        """
        # Layer 1: 1-hop structural context per node
        x = self.conv1(x, edge_index)   # [N, 8] → [N, 64]
        x = self.relu(x)
        x = self.dropout(x)             # node feature dropout

        # Layer 2: 2-hop context — each node now sees neighbours' neighbours
        x = self.conv2(x, edge_index)   # [N, 64] → [N, 64]
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3: 3-hop context, final node embeddings
        # No relu/dropout here — CrossAttentionFusion applies its own projections
        x = self.conv3(x, edge_index)   # [N, 64] → [N, 64]

        # Return node-level embeddings AND batch tensor
        # Pooling happens in CrossAttentionFusion AFTER cross-attention enrichment
        return x, batch
