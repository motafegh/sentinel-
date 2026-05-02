"""
test_fusion_layer.py — Unit tests for CrossAttentionFusion.

Covers: output shape, device mismatch detection, masked pooling excludes PAD,
attn_dim divisibility validation. No checkpoint or real data needed.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from ml.src.models.fusion_layer import CrossAttentionFusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int = 2,
    n_nodes_per_graph: int = 4,
    token_len: int = 512,
    node_dim: int = 64,
    token_dim: int = 768,
) -> tuple:
    """
    Return (node_embs, batch_idx, token_embs, attention_mask) for a synthetic batch.
    """
    N = batch_size * n_nodes_per_graph
    node_embs = torch.randn(N, node_dim)
    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size), n_nodes_per_graph
    )
    token_embs    = torch.randn(batch_size, token_len, token_dim)
    attention_mask = torch.ones(batch_size, token_len, dtype=torch.long)
    # Simulate PAD: last 50 tokens are PAD for every sample
    attention_mask[:, -50:] = 0

    return node_embs, batch_idx, token_embs, attention_mask


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    """CrossAttentionFusion returns [B, output_dim]."""
    fusion = CrossAttentionFusion(
        node_dim=64, token_dim=768, attn_dim=256, num_heads=8, output_dim=128
    )
    fusion.eval()

    node_embs, batch_idx, token_embs, attn_mask = _make_batch(batch_size=3)
    with torch.no_grad():
        out = fusion(node_embs, batch_idx, token_embs, attn_mask)

    assert out.shape == (3, 128), f"Expected [3, 128], got {out.shape}"


def test_forward_single_sample():
    """Batch size 1 should work without shape errors."""
    fusion = CrossAttentionFusion(node_dim=64, token_dim=768, output_dim=128)
    fusion.eval()

    node_embs, batch_idx, token_embs, attn_mask = _make_batch(batch_size=1)
    with torch.no_grad():
        out = fusion(node_embs, batch_idx, token_embs, attn_mask)

    assert out.shape == (1, 128)


# ---------------------------------------------------------------------------
# attn_dim divisibility
# ---------------------------------------------------------------------------

def test_attn_dim_not_divisible_raises():
    """attn_dim not divisible by num_heads raises ValueError at init."""
    with pytest.raises(ValueError, match="divisible"):
        CrossAttentionFusion(attn_dim=100, num_heads=8)  # 100 / 8 = 12.5


def test_valid_attn_dim():
    """attn_dim=256, num_heads=8 → head_dim=32 — constructs fine."""
    fusion = CrossAttentionFusion(attn_dim=256, num_heads=8)
    assert fusion.attn_dim == 256


# ---------------------------------------------------------------------------
# Device mismatch
# ---------------------------------------------------------------------------

def test_device_mismatch_raises():
    """
    node_embs and token_embs on different devices must raise RuntimeError
    with a clear message — not a cryptic CUDA kernel error.
    Only runs when CUDA is available; skipped otherwise.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — device mismatch test skipped")

    fusion = CrossAttentionFusion().cuda()
    node_embs = torch.randn(4, 64).cpu()       # CPU
    batch_idx = torch.zeros(4, dtype=torch.long).cpu()
    token_embs    = torch.randn(1, 512, 768).cuda()   # GPU
    attention_mask = torch.ones(1, 512).cuda()

    with pytest.raises(RuntimeError, match="Device mismatch"):
        fusion(node_embs, batch_idx, token_embs, attention_mask)


# ---------------------------------------------------------------------------
# Masked pooling — PAD tokens excluded
# ---------------------------------------------------------------------------

def test_masked_pooling_with_all_real_vs_half_pad():
    """
    An all-real-token sequence and a half-PAD sequence with identical real
    content should produce different outputs — confirms PAD positions are
    excluded from the pooled token representation.
    """
    fusion = CrossAttentionFusion(
        node_dim=64, token_dim=768, attn_dim=256, num_heads=8, output_dim=128
    )
    fusion.eval()
    torch.manual_seed(7)

    node_embs  = torch.randn(4, 64)
    batch_idx  = torch.zeros(4, dtype=torch.long)
    token_embs = torch.randn(1, 512, 768)

    # Mask 1: all real tokens
    mask_all = torch.ones(1, 512, dtype=torch.long)
    # Mask 2: first 256 real, last 256 PAD
    mask_half = torch.cat([torch.ones(1, 256), torch.zeros(1, 256)], dim=1).long()

    with torch.no_grad():
        out_all  = fusion(node_embs, batch_idx, token_embs, mask_all)
        out_half = fusion(node_embs, batch_idx, token_embs, mask_half)

    assert not torch.allclose(out_all, out_half, atol=1e-4), \
        "Masked and unmasked token pooling should produce different outputs"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_eval_mode_deterministic():
    """In eval mode, two identical forward passes produce identical outputs."""
    fusion = CrossAttentionFusion(
        node_dim=64, token_dim=768, attn_dim=64, num_heads=8, output_dim=128,
        dropout=0.0,
    )
    fusion.eval()

    node_embs, batch_idx, token_embs, attn_mask = _make_batch(batch_size=2)
    with torch.no_grad():
        out1 = fusion(node_embs, batch_idx, token_embs, attn_mask)
        out2 = fusion(node_embs, batch_idx, token_embs, attn_mask)

    assert torch.allclose(out1, out2), "Eval mode must be deterministic (dropout=0)"
