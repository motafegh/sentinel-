"""
test_gnn_encoder.py — Unit tests for GNNEncoder.

Covers: edge_attr embedding, graceful degradation to zeros, output shape,
head-divisibility validation, and no-edge graphs.
No checkpoint or real data required.
"""
from __future__ import annotations

import pytest
import torch

from ml.src.models.gnn_encoder import GNNEncoder
from ml.src.preprocessing.graph_schema import NUM_EDGE_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_nodes: int = 5, n_edges: int = 4) -> tuple:
    """Return (x, edge_index, batch, edge_attr) for a synthetic graph."""
    x          = torch.randn(n_nodes, 8)
    src        = torch.arange(n_edges)
    dst        = torch.arange(1, n_edges + 1)
    edge_index = torch.stack([src, dst])
    batch      = torch.zeros(n_nodes, dtype=torch.long)
    edge_attr  = torch.randint(0, NUM_EDGE_TYPES, (n_edges,))
    return x, edge_index, batch, edge_attr


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape_with_edge_attr():
    """GNNEncoder returns [N, hidden_dim] node embeddings when edge_attr is supplied."""
    enc = GNNEncoder(hidden_dim=64, heads=8, use_edge_attr=True)
    x, edge_index, batch, edge_attr = _make_graph()

    node_embs, batch_out = enc(x, edge_index, batch, edge_attr=edge_attr)

    assert node_embs.shape == (x.shape[0], 64), \
        f"Expected [{x.shape[0]}, 64], got {node_embs.shape}"
    assert batch_out.shape == batch.shape


def test_output_shape_without_edge_attr():
    """use_edge_attr=False (no embedding layer) still produces correct output."""
    enc = GNNEncoder(hidden_dim=64, heads=8, use_edge_attr=False)
    x, edge_index, batch, _ = _make_graph()

    node_embs, _ = enc(x, edge_index, batch, edge_attr=None)
    assert node_embs.shape == (x.shape[0], 64)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_graceful_degradation_none_edge_attr():
    """
    use_edge_attr=True but edge_attr=None (old .pt files) must not crash.
    The model falls back to zero-vectors so the forward pass completes.
    """
    enc = GNNEncoder(hidden_dim=64, heads=8, use_edge_attr=True)
    x, edge_index, batch, _ = _make_graph()

    # Must not raise
    node_embs, _ = enc(x, edge_index, batch, edge_attr=None)
    assert node_embs.shape == (x.shape[0], 64)


def test_zero_fallback_differs_from_real_edge_attr():
    """
    Outputs with real edge_attr and zero-fallback differ — confirms the
    embedding is actually used (not silently ignored).
    """
    torch.manual_seed(0)
    enc = GNNEncoder(hidden_dim=64, heads=8, use_edge_attr=True)
    enc.eval()

    x, edge_index, batch, edge_attr = _make_graph()
    with torch.no_grad():
        out_real,  _ = enc(x, edge_index, batch, edge_attr=edge_attr)
        out_zeros, _ = enc(x, edge_index, batch, edge_attr=None)

    assert not torch.allclose(out_real, out_zeros, atol=1e-5), \
        "Outputs should differ when edge_attr is supplied vs zero-fallback"


# ---------------------------------------------------------------------------
# Head divisibility validation
# ---------------------------------------------------------------------------

def test_head_divisibility_error():
    """hidden_dim not divisible by heads raises ValueError at construction time."""
    with pytest.raises(ValueError, match="divisible"):
        GNNEncoder(hidden_dim=33, heads=8)   # 33 / 8 = 4.125 — not divisible


def test_valid_head_configuration():
    """hidden_dim=128, heads=8 → head_dim=16 — should construct without error."""
    enc = GNNEncoder(hidden_dim=128, heads=8)
    assert enc.hidden_dim == 128


# ---------------------------------------------------------------------------
# No-edge graph (minimum viable input)
# ---------------------------------------------------------------------------

def test_no_edge_graph():
    """A graph with zero edges must not crash (single isolated node)."""
    enc = GNNEncoder(hidden_dim=64, heads=8, use_edge_attr=True)
    x          = torch.randn(3, 8)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    batch      = torch.zeros(3, dtype=torch.long)
    edge_attr  = torch.zeros(0, dtype=torch.long)

    node_embs, _ = enc(x, edge_index, batch, edge_attr=edge_attr)
    assert node_embs.shape == (3, 64)
