"""
fusion_layer.py — Cross-Attention Fusion for SENTINEL (Replaces concat+MLP)

WHAT CHANGED FROM ORIGINAL:
    Complete replacement of FusionLayer with CrossAttentionFusion.

    BEFORE: concat([B,64], [B,768]) → MLP → [B,64]
            Two pooled summaries fused — node/token detail already gone.

    AFTER:  Node embeddings [N,256] attend to token embeddings [B,512,768]
            Token embeddings [B,512,768] attend to node embeddings [N,256]
            Pool AFTER enrichment → [B,128]
            Structural patterns find their semantic counterparts before averaging.

WHY:
    withdraw() as a node can now directly attend to "call.value" and "transfer"
    tokens before pooling dilutes its signal across 30 other functions.
    "call.value" as a token can attend to the withdraw() node that emits it.
    Both directions reinforce the reentrancy signal at fine granularity.

OUTPUT DIM CHANGE:
    BEFORE: [B, 64]   — narrow, one modality dominated
    AFTER:  [B, 128]  — wider, both modalities enriched before compression
    SentinelModel classifier must use: Linear(128, num_classes)

ARCHITECTURE:
    1. Project GNN nodes [N,256] → [N,256]        (node_proj is identity-sized; hidden_dim=256)
    2. Project tokens [B,512,768] → [B,512,256]   (common attention dim)
    3. Pad nodes → [B,max_nodes,256] via to_dense_batch(); build node padding mask
    4. Node→Token cross-attention (key_padding_mask=token PAD positions)
         → enriched nodes [B,max_n,256]
    4b. Zero-out padded node positions in enriched_nodes (Fix #8)
    5. Token→Node cross-attention (key_padding_mask=padded node positions)
         → enriched tokens [B,512,256]
    6. Pool enriched nodes  — masked mean over real nodes  → [B,256]
    7. Pool enriched tokens — masked mean over real tokens → [B,256]
    8. Concatenate → [B,512], project → [B,128]

REVIEW FIXES APPLIED (see inline comments):
    #2  Token PAD mask threaded to node→token key_padding_mask
    #4  Device assertion at start of forward()
    #5  Removed dead global_mean_pool import
    #6  Token pooling now masked mean (was plain mean — included PAD positions)
    #7  Python padding loop replaced with to_dense_batch() (vectorised)
    #8  Zero-out padded positions in enriched_nodes after node→token attention
        Padding nodes went through attention and received nonzero values from
        token content. Pooling already excluded them via node_real_mask, but
        those nonzero values were still present in the tensor passed implicitly
        to any future refactor. Explicit zeroing makes the invariant structural,
        not just documented.
    #26 need_weights=False on both MHA calls — attention weight matrices
        ([B, max_nodes, 512] + [B, 512, max_nodes] ≈ 12.6 MB per forward)
        were computed and allocated but never used anywhere in the codebase.
        need_weights=False lets PyTorch skip weight materialisation entirely
        and use the fused efficient-attention CUDA kernel, saving ~12.6 MB
        VRAM per forward pass and reducing allocator fragmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

_c4_truncation_warned: bool = False  # log once total, not once per unique size


def _scatter_to_dense(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    max_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    torch.compile-compatible replacement for to_dense_batch.

    Uses static max_nodes (a config constant) instead of a data-dependent
    repeat(size), which caused a GuardOnDataDependentSymNode graph break that
    forced the entire CrossAttentionFusion forward to run in eager mode.

    Contracts with more than max_nodes nodes have excess nodes silently
    truncated (affects <1% of the corpus at max_nodes=1024).
    """
    N, D = x.shape
    ones = torch.ones(N, dtype=torch.long, device=x.device)
    counts = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
    counts.scatter_add_(0, batch, ones)

    # Per-graph start offsets: [0, count0, count0+count1, ...]
    offsets = torch.cat([
        x.new_zeros(1, dtype=torch.long),
        counts[:-1].cumsum(0),
    ])                                           # [num_graphs]

    local_idx = torch.arange(N, device=x.device) - offsets[batch]
    _max_n = int(counts.max().item())
    if _max_n > max_nodes:
        global _c4_truncation_warned
        if not _c4_truncation_warned:
            _c4_truncation_warned = True
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"C-4: _scatter_to_dense: graphs exceeding max_nodes={max_nodes} detected "
                f"(first seen: {_max_n} nodes). Excess nodes silently truncated. "
                f"This warning fires once — further occurrences suppressed."
            )
    # C2 fix: compute valid mask BEFORE clamping so excess nodes (local_idx >= max_nodes)
    # are truly dropped. Without this, all excess nodes clamp to position max_nodes-1
    # and overwrite each other (last-write-wins = random embedding at that slot).
    valid     = local_idx < max_nodes
    local_idx = local_idx.clamp(max=max_nodes - 1)

    out  = x.new_zeros(num_graphs, max_nodes, D)
    mask = torch.zeros(num_graphs, max_nodes, dtype=torch.bool, device=x.device)
    out[batch[valid], local_idx[valid]]  = x[valid]
    mask[batch[valid], local_idx[valid]] = True
    return out, mask


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion of GNN node embeddings and
    CodeBERT token embeddings.

    Args:
        node_dim:    GNNEncoder output dimension (default: 256)
        token_dim:   TransformerEncoder output dimension per token (default: 768)
        attn_dim:    Common projection dimension for attention (default: 256).
                     Must be divisible by num_heads.
        num_heads:   Parallel attention heads (default: 8, attn_dim/num_heads=32)
        output_dim:  Final fused embedding dimension (default: 128)
        dropout:     Applied inside MultiheadAttention (default: 0.1)
        max_nodes:   Max GNN nodes for compile-safe dense padding (default: 1024)
    """

    def __init__(
        self,
        node_dim:   int   = 256,
        token_dim:  int   = 768,
        attn_dim:   int   = 256,
        num_heads:  int   = 8,
        output_dim: int   = 128,
        dropout:    float = 0.1,
        max_nodes:  int   = 1024,
    ) -> None:
        super().__init__()

        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads}). "
                f"Each head needs attn_dim/num_heads = {attn_dim / num_heads} dims."
            )

        self.attn_dim   = attn_dim
        self.output_dim = output_dim
        self.max_nodes  = max_nodes

        self.node_proj  = nn.Linear(node_dim,  attn_dim)
        self.token_proj = nn.Linear(token_dim, attn_dim)
        # Normalize token embeddings before projection (BUG-C2).
        # CodeBERT hidden states have L2 norm ~10-15; GNN output after its own
        # LayerNorm has norm ~1. Without this, token keys dominate cross-attention
        # dot products by 10-15×, making node→token attention attend to
        # highest-norm tokens rather than semantically relevant ones.
        self.token_norm = nn.LayerNorm(token_dim)

        # Direction 1: every node queries the 512 tokens.
        # Q=nodes [B,n,256]  K=V=tokens [B,512,256]
        self.node_to_token = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Direction 2: every token queries the graph nodes.
        # Q=tokens [B,512,256]  K=V=nodes [B,n,256]
        self.token_to_node = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(attn_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        logger.info(
            f"CrossAttentionFusion init — "
            f"node_dim={node_dim} token_dim={token_dim} "
            f"attn_dim={attn_dim} heads={num_heads} output={output_dim} "
            f"max_nodes={max_nodes} (static; compile-safe)"
        )

    def forward(
        self,
        node_embs:      torch.Tensor,  # [N, gnn_hidden_dim]  all nodes across the batch (hidden_dim=256)
        batch:          torch.Tensor,  # [N]             node→graph index mapping
        token_embs:     torch.Tensor,  # [B, 512, 768]  all token embeddings
        attention_mask: torch.Tensor,  # [B, 512]        1=real token, 0=PAD
    ) -> torch.Tensor:
        """
        Bidirectional cross-attention fusion.

        Returns:
            [B, 128] — fused, structurally and semantically enriched contract embeddings
        """
        # Fix #4: catch device mismatches early.
        if node_embs.device != token_embs.device:
            raise RuntimeError(
                f"Device mismatch: node_embs on {node_embs.device} "
                f"but token_embs on {token_embs.device}."
            )

        # ── Step 1: Project both modalities to common attention space ──────
        nodes_proj  = self.node_proj(node_embs)                    # [N, hidden_dim] → [N, 256]
        tokens_proj = self.token_proj(self.token_norm(token_embs)) # [B, 512, 768] → [B, 512, 256]

        # ── Step 2: Pad nodes to uniform length across the batch ───────────
        # padded_nodes:   [B, max_nodes, 256]  — zero-padded at trailing positions
        # node_real_mask: [B, max_nodes]        — True=real node, False=padding
        # _scatter_to_dense uses a static self.max_nodes instead of a
        # data-dependent repeat, enabling torch.compile to trace the full
        # fusion forward without a graph break.
        B = token_embs.shape[0]
        padded_nodes, node_real_mask = _scatter_to_dense(
            nodes_proj, batch, num_graphs=B, max_nodes=self.max_nodes
        )
        # MHA key_padding_mask convention: True = IGNORE. Invert node_real_mask.
        node_padding_mask  = ~node_real_mask           # [B, max_nodes] True=pad
        # Fix #2: token PAD mask for node→token attention
        token_padding_mask = (attention_mask == 0)     # [B, 512]       True=PAD

        # ── Step 3: Node → Token cross-attention ──────────────────────────
        # Q=padded_nodes [B,max_nodes,256]  K=V=tokens [B,512,256]
        # Fix #26: need_weights=False — weight matrix [B,max_nodes,512] was
        # allocated (~6.3 MB) and computed every forward but never read.
        # With need_weights=False PyTorch uses the fused efficient-attn kernel.
        enriched_nodes, _ = self.node_to_token(
            query=padded_nodes,
            key=tokens_proj,
            value=tokens_proj,
            key_padding_mask=token_padding_mask,
            need_weights=False,                        # Fix #26
        )
        # enriched_nodes: [B, max_nodes, 256]

        # Fix #8: Zero-out padded node positions in enriched_nodes.
        enriched_nodes = enriched_nodes * node_real_mask.float().unsqueeze(-1)

        # ── Step 4: Token → Node cross-attention ──────────────────────────
        # Q=tokens [B,512,256]  K=V=padded_nodes [B,max_nodes,256]
        # Fix #26: need_weights=False — same rationale as Step 3.
        enriched_tokens, _ = self.token_to_node(
            query=tokens_proj,
            key=padded_nodes,
            value=padded_nodes,
            key_padding_mask=node_padding_mask,
            need_weights=False,                        # Fix #26
        )
        # enriched_tokens: [B, 512, 256]

        # ── Step 5: Masked mean pooling ────────────────────────────────────
        node_weight   = node_real_mask.float().unsqueeze(-1)         # [B, max_nodes, 1]
        node_sum      = (enriched_nodes * node_weight).sum(dim=1)    # [B, 256]
        node_count    = node_weight.sum(dim=1).clamp(min=1.0)        # [B, 1]
        pooled_nodes  = node_sum / node_count                        # [B, 256]

        # Fix #6: masked mean over real tokens only
        token_weight  = attention_mask.float().unsqueeze(-1)         # [B, 512, 1]
        token_sum     = (enriched_tokens * token_weight).sum(dim=1)  # [B, 256]
        token_count   = token_weight.sum(dim=1).clamp(min=1.0)       # [B, 1]
        pooled_tokens = token_sum / token_count                       # [B, 256]

        # ── Step 6: Concatenate and project ───────────────────────────────
        fused  = torch.cat([pooled_nodes, pooled_tokens], dim=1)  # [B, 512]
        output = self.output_proj(fused)                           # [B, 128]

        return output
