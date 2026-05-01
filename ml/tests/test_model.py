"""
test_model.py — Unit tests for SentinelModel forward pass shapes.

TransformerEncoder loads CodeBERT (~500 MB) which makes it impractical for
unit tests. We replace model.transformer with a lightweight stub that returns
the correct shape [B, 512, 768] so every other component (GNN, fusion,
classifier) is exercised with real weights on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from torch_geometric.data import Batch, Data

from ml.src.models.sentinel_model import SentinelModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubTransformer(nn.Module):
    """Returns zeros with correct CodeBERT shape — no HuggingFace required."""
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B = input_ids.shape[0]
        return torch.zeros(B, 512, 768)


def _make_model(num_classes: int = 10) -> SentinelModel:
    model = SentinelModel(num_classes=num_classes, fusion_output_dim=128)
    model.transformer = _StubTransformer()
    model.eval()
    return model


def _make_batch(batch_size: int, nodes_per_graph: int = 5) -> tuple:
    """
    Build a synthetic PyG Batch + token tensors.

    Returns:
        graphs: PyG Batch with x [N, 8], edge_index [2, E], batch [N]
        input_ids:      [B, 512]
        attention_mask: [B, 512]
    """
    graphs_list = []
    for _ in range(batch_size):
        n = nodes_per_graph
        x          = torch.randn(n, 8)
        # Chain graph: 0→1→2→...→n-1
        src        = torch.arange(n - 1)
        dst        = torch.arange(1, n)
        edge_index = torch.stack([src, dst], dim=0)
        graphs_list.append(Data(x=x, edge_index=edge_index))

    graphs         = Batch.from_data_list(graphs_list)
    input_ids      = torch.ones(batch_size, 512, dtype=torch.long)
    attention_mask = torch.ones(batch_size, 512, dtype=torch.long)
    return graphs, input_ids, attention_mask


# ---------------------------------------------------------------------------
# Forward pass — multi-label (default)
# ---------------------------------------------------------------------------

def test_forward_multilabel_shape():
    model = _make_model(num_classes=10)
    graphs, input_ids, attention_mask = _make_batch(batch_size=4)
    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)
    assert out.shape == (4, 10), f"expected (4, 10), got {out.shape}"


def test_forward_batch_size_1():
    model = _make_model(num_classes=10)
    graphs, input_ids, attention_mask = _make_batch(batch_size=1)
    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)
    assert out.shape == (1, 10)


def test_forward_large_batch():
    model = _make_model(num_classes=10)
    graphs, input_ids, attention_mask = _make_batch(batch_size=8)
    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)
    assert out.shape == (8, 10)


# ---------------------------------------------------------------------------
# Forward pass — binary mode
# ---------------------------------------------------------------------------

def test_forward_binary_shape():
    model = _make_model(num_classes=1)
    graphs, input_ids, attention_mask = _make_batch(batch_size=4)
    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)
    # Binary mode: squeeze(1) applied inside model → [B]
    assert out.shape == (4,), f"expected (4,), got {out.shape}"


# ---------------------------------------------------------------------------
# Forward pass — graph with many nodes
# ---------------------------------------------------------------------------

def test_forward_many_nodes():
    model = _make_model(num_classes=10)
    graphs, input_ids, attention_mask = _make_batch(batch_size=2, nodes_per_graph=50)
    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)
    assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# Output dtype — logits, not probabilities
# ---------------------------------------------------------------------------

def test_forward_returns_logits_not_probabilities():
    """Model must not apply Sigmoid internally — logits can exceed [0, 1]."""
    model = _make_model(num_classes=10)
    # Use non-zero input to get non-trivial logits
    graphs_list = []
    for _ in range(4):
        x          = torch.randn(5, 8) * 5  # large values → large logits expected
        edge_index = torch.stack([torch.arange(4), torch.arange(1, 5)])
        graphs_list.append(Data(x=x, edge_index=edge_index))
    graphs         = Batch.from_data_list(graphs_list)
    input_ids      = torch.ones(4, 512, dtype=torch.long)
    attention_mask = torch.ones(4, 512, dtype=torch.long)

    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)

    # At least some values should lie outside [0, 1] — raw logits
    assert (out.abs() > 1.0).any(), "expected raw logits (some > 1), not sigmoid output"


# ---------------------------------------------------------------------------
# Attention mask — PAD tokens should not affect output
# ---------------------------------------------------------------------------

def test_all_pad_attention_mask_does_not_crash():
    """All-PAD mask is an edge case; model must not crash or produce NaN."""
    model = _make_model(num_classes=10)
    graphs, input_ids, _ = _make_batch(batch_size=2)
    # All attention mask = 0 (all PAD)
    all_pad_mask = torch.zeros(2, 512, dtype=torch.long)
    with torch.no_grad():
        out = model(graphs, input_ids, all_pad_mask)
    assert not torch.isnan(out).any(), "NaN in output with all-PAD attention mask"
    assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# parameter_summary — smoke test (no assertion, just must not raise)
# ---------------------------------------------------------------------------

def test_parameter_summary_runs():
    model = _make_model()
    model.parameter_summary()  # should log without raising
