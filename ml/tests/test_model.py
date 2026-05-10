"""
test_model.py — Unit tests for SentinelModel and GNNEncoder (v5 schema).

TransformerEncoder loads CodeBERT (~500 MB) which makes it impractical for
unit tests. We replace model.transformer with a lightweight stub that returns
the correct shape [B, 512, 768] so every other component (GNN, fusion,
classifier, aux heads) is exercised with real weights on CPU.

V5 CHANGES FROM V4 TESTS
─────────────────────────
- Node features: 8 → 12 dims (NODE_FEATURE_DIM=12).
- edge_attr [E] int64 required on every Data object (GNNEncoder phase masking).
- return_aux=True: forward() now returns (logits, aux_dict) when requested.
- Three-eye classifier: in_features = 3 * fusion_output_dim = 384.
- GNNEncoder return_intermediates=True: returns per-phase embedding dicts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from torch_geometric.data import Batch, Data

from ml.src.models.sentinel_model import SentinelModel
from ml.src.models.gnn_encoder import GNNEncoder
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, EDGE_TYPES


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
    Build a synthetic PyG Batch + token tensors (v2 schema).

    Node features are 12-dim (NODE_FEATURE_DIM=12).
    Edges use type 0 (CALLS) so they pass through Phase 1 structural masking.
    edge_attr is required by GNNEncoder for phase mask computation.

    Returns:
        graphs: PyG Batch with x [N, 12], edge_index [2, E], edge_attr [E], batch [N]
        input_ids:      [B, 512]
        attention_mask: [B, 512]
    """
    graphs_list = []
    for _ in range(batch_size):
        n          = nodes_per_graph
        x          = torch.randn(n, NODE_FEATURE_DIM)
        # Chain graph: 0→1→2→...→n-1
        src        = torch.arange(n - 1)
        dst        = torch.arange(1, n)
        edge_index = torch.stack([src, dst], dim=0)
        # All edges are CALLS (type 0) — pass through Phase 1 struct_mask only
        edge_attr  = torch.zeros(n - 1, dtype=torch.long)
        graphs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    graphs         = Batch.from_data_list(graphs_list)
    input_ids      = torch.ones(batch_size, 512, dtype=torch.long)
    attention_mask = torch.ones(batch_size, 512, dtype=torch.long)
    return graphs, input_ids, attention_mask


def _make_cfg_batch() -> tuple:
    """
    Build a batch with CONTAINS and CONTROL_FLOW edges to exercise GNN Phases 2+3.

    Graph structure (one contract, 3 nodes):
        node 0: FUNCTION (type 1)
        node 1: CFG_NODE_CALL (type 8)   — external call statement
        node 2: CFG_NODE_WRITE (type 9)  — state write statement

    Edges:
        0→1  CONTAINS (type 5)  — function contains call node
        0→2  CONTAINS (type 5)  — function contains write node
        1→2  CONTROL_FLOW (type 6) — call precedes write (reentrancy pattern)
    """
    x = torch.randn(3, NODE_FEATURE_DIM)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    edge_attr  = torch.tensor([
        EDGE_TYPES["CONTAINS"],
        EDGE_TYPES["CONTAINS"],
        EDGE_TYPES["CONTROL_FLOW"],
    ], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([graph])
    input_ids      = torch.ones(1, 512, dtype=torch.long)
    attention_mask = torch.ones(1, 512, dtype=torch.long)
    return batch, input_ids, attention_mask


# ---------------------------------------------------------------------------
# Forward pass — multi-label (default, return_aux=False)
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
# return_aux=True — three-eye auxiliary head shapes (v5)
# ---------------------------------------------------------------------------

def test_forward_return_aux_shapes():
    """return_aux=True must return (logits [B, C], aux dict with three [B, C] tensors)."""
    model = _make_model(num_classes=10)
    model.train()  # aux heads active in training mode
    graphs, input_ids, attention_mask = _make_batch(batch_size=4)
    with torch.no_grad():
        result = model(graphs, input_ids, attention_mask, return_aux=True)

    assert isinstance(result, tuple) and len(result) == 2, (
        f"return_aux=True must return a 2-tuple (logits, aux), got {type(result)}"
    )
    logits, aux = result
    assert logits.shape == (4, 10), f"logits shape: {logits.shape}"
    assert set(aux.keys()) == {"gnn", "transformer", "fused"}, (
        f"aux keys must be {{'gnn', 'transformer', 'fused'}}, got {set(aux.keys())}"
    )
    for key, tensor in aux.items():
        assert tensor.shape == (4, 10), (
            f"aux['{key}'] shape: {tensor.shape}, expected (4, 10)"
        )


def test_forward_return_aux_false_returns_tensor():
    """Default return_aux=False must return a plain tensor, not a tuple."""
    model = _make_model(num_classes=10)
    graphs, input_ids, attention_mask = _make_batch(batch_size=2)
    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask, return_aux=False)
    assert isinstance(out, torch.Tensor), (
        f"return_aux=False must return Tensor, got {type(out)}"
    )
    assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# Three-eye classifier architecture (v5)
# ---------------------------------------------------------------------------

def test_classifier_input_dim_is_384():
    """Classifier input must be 3 × eye_dim = 3 × 128 = 384."""
    model = _make_model(num_classes=10)
    assert model.classifier.in_features == 384, (
        f"classifier.in_features = {model.classifier.in_features}, expected 384 "
        "(3 × 128: gnn_eye + transformer_eye + fused_eye)."
    )


def test_aux_heads_exist_and_output_dim():
    """All three auxiliary heads must exist with correct [eye_dim → num_classes] shape."""
    model = _make_model(num_classes=10)
    assert model.aux_gnn.out_features == 10
    assert model.aux_transformer.out_features == 10
    assert model.aux_fused.out_features == 10
    assert model.aux_gnn.in_features == 128
    assert model.aux_transformer.in_features == 128
    assert model.aux_fused.in_features == 128


# ---------------------------------------------------------------------------
# GNNEncoder — return_intermediates (v5)
# ---------------------------------------------------------------------------

def test_gnn_return_intermediates_keys():
    """GNNEncoder(return_intermediates=True) must return (x, batch, dict)."""
    gnn = GNNEncoder()
    gnn.eval()
    x          = torch.randn(3, NODE_FEATURE_DIM)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    edge_attr  = torch.tensor([
        EDGE_TYPES["CONTAINS"],
        EDGE_TYPES["CONTAINS"],
        EDGE_TYPES["CONTROL_FLOW"],
    ], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)

    with torch.no_grad():
        result = gnn(x, edge_index, batch, edge_attr, return_intermediates=True)

    assert len(result) == 3, f"expected 3-tuple, got {len(result)}"
    node_embs, batch_out, intermediates = result
    assert node_embs.shape == (3, 128), f"node_embs shape: {node_embs.shape}"
    assert set(intermediates.keys()) == {"after_phase1", "after_phase2", "after_phase3"}
    for key, tensor in intermediates.items():
        assert tensor.shape == (3, 128), f"intermediates['{key}'] shape: {tensor.shape}"


def test_gnn_phase3_changes_function_node():
    """
    Phase 3 (reverse-CONTAINS) must change the function node's embedding
    when it has CFG children with CONTROL_FLOW signal.

    If after_phase3[0] == after_phase2[0] for the function node in a graph
    that has CONTAINS edges, Phase 3 is silently broken.
    """
    gnn = GNNEncoder()
    gnn.eval()

    x          = torch.randn(3, NODE_FEATURE_DIM)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    edge_attr  = torch.tensor([
        EDGE_TYPES["CONTAINS"],
        EDGE_TYPES["CONTAINS"],
        EDGE_TYPES["CONTROL_FLOW"],
    ], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)

    with torch.no_grad():
        _, _, intermediates = gnn(
            x, edge_index, batch, edge_attr, return_intermediates=True
        )

    # Function node (index 0) receives Phase 3 messages from CFG children (1, 2)
    # via reversed CONTAINS edges. It must differ from its Phase 2 value.
    emb_p2 = intermediates["after_phase2"][0]
    emb_p3 = intermediates["after_phase3"][0]
    assert (emb_p2 != emb_p3).any(), (
        "Phase 3 did not change node 0 (function node) embedding. "
        "Check that rev_contains_ei is built correctly (flip(0)) and "
        "conv4 uses add_self_loops=False."
    )


def test_gnn_return_intermediates_false_is_2_tuple():
    """Default return_intermediates=False must return (x, batch) only."""
    gnn = GNNEncoder()
    gnn.eval()
    x          = torch.randn(5, NODE_FEATURE_DIM)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr  = torch.zeros(4, dtype=torch.long)
    batch      = torch.zeros(5, dtype=torch.long)

    with torch.no_grad():
        result = gnn(x, edge_index, batch, edge_attr)

    assert isinstance(result, tuple) and len(result) == 2, (
        f"return_intermediates=False must return 2-tuple, got len={len(result)}"
    )
    node_embs, batch_out = result
    assert node_embs.shape == (5, 128)


# ---------------------------------------------------------------------------
# Output dtype — logits, not probabilities
# ---------------------------------------------------------------------------

def test_forward_returns_logits_not_probabilities():
    """Model must not apply Sigmoid internally — raw logits include negative values.

    sigmoid(x) ∈ (0, 1) always. If the output has any negative value, sigmoid
    was not applied. With random-init weights and ReLU eye projections, the
    classifier Linear produces a mix of positive and negative values reliably.
    """
    model = _make_model(num_classes=10)
    graphs_list = []
    for _ in range(4):
        x          = torch.randn(5, NODE_FEATURE_DIM) * 5
        edge_index = torch.stack([torch.arange(4), torch.arange(1, 5)])
        edge_attr  = torch.zeros(4, dtype=torch.long)
        graphs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    graphs         = Batch.from_data_list(graphs_list)
    input_ids      = torch.ones(4, 512, dtype=torch.long)
    attention_mask = torch.ones(4, 512, dtype=torch.long)

    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)

    # sigmoid output is always in (0, 1); negative values prove no sigmoid applied
    assert (out < 0).any(), (
        "expected raw logits (some negative with random init), "
        "but all values ≥ 0 — Sigmoid may have been applied."
    )


# ---------------------------------------------------------------------------
# Attention mask — PAD tokens should not affect output
# ---------------------------------------------------------------------------

def test_mostly_pad_attention_mask_does_not_crash():
    """Sparse attention mask (only first token real) must not crash or produce NaN.

    All-PAD (mask = all zeros) triggers softmax(all=-inf) = NaN inside
    CrossAttentionFusion and cannot occur for valid contract inputs (every
    contract has ≥ 1 real token). This test uses 1 real + 511 PAD tokens —
    the realistic worst case.
    """
    model = _make_model(num_classes=10)
    graphs, input_ids, _ = _make_batch(batch_size=2)
    sparse_mask = torch.zeros(2, 512, dtype=torch.long)
    sparse_mask[:, 0] = 1  # only first token is real
    with torch.no_grad():
        out = model(graphs, input_ids, sparse_mask)
    assert not torch.isnan(out).any(), "NaN in output with mostly-PAD attention mask"
    assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# parameter_summary — smoke test (no assertion, just must not raise)
# ---------------------------------------------------------------------------

def test_parameter_summary_runs():
    model = _make_model()
    model.parameter_summary()


# ---------------------------------------------------------------------------
# CONTAINS + CONTROL_FLOW graph — end-to-end shape check
# ---------------------------------------------------------------------------

def test_forward_with_cfg_edges():
    """Model must handle graphs with CONTAINS and CONTROL_FLOW edges without crashing."""
    model = _make_model(num_classes=10)
    graphs, input_ids, attention_mask = _make_cfg_batch()
    with torch.no_grad():
        out = model(graphs, input_ids, attention_mask)
    assert out.shape == (1, 10)
    assert not torch.isnan(out).any(), "NaN in output with CFG edge types"
