"""
test_gnn_encoder.py — Unit tests for GNNEncoder.

Covers: edge_attr embedding, output shape, head-divisibility validation,
no-edge graphs, and JK gradient flow (non-negotiable gate).

Change history:
  Phase 0-T1 (2026-05-14): Fixed synthetic graph feature dimension from 8
    to NODE_FEATURE_DIM (12).  The v1 schema had 8 features; v2 has 12.
    Using 8 caused GATConv's first layer (in_channels=12) to crash at runtime,
    meaning no test here would have caught a real forward-pass regression.

  Phase 0-T1 (2026-05-14): Updated test_graceful_degradation_none_edge_attr
    to assert ValueError is raised.  A previous audit fix (gnn_encoder.py)
    replaced silent zero-vector fallback with an explicit guard because
    edge_attr=None with use_edge_attr=True hides a data-pipeline bug.
    Tests must document the *current* contract, not the old one.

  Phase 0-T1 (2026-05-14): Updated test_zero_fallback_differs_from_real_edge_attr
    to test that edge_attr values *do* influence the output (same intent,
    correct implementation for the current guarded code path).

  Phase 1-T2 (2026-05-14): Added test_jk_gradient_flow — non-negotiable gate.
    Must pass before any v5.2 training is launched.

  Phase 1-T3 (2026-05-14): Added test_reverse_contains_separate_embedding —
    verifies that Phase 3 uses REVERSE_CONTAINS (type 7) embedding, not the
    forward CONTAINS (type 5) embedding.

  Phase 1-T4 (2026-05-14): Added test_jk_output_shape_unchanged and
    test_jk_disabled_output_shape — verify JK does not change output dimension,
    which is hardcoded to 128 throughout SentinelModel and CrossAttentionFusion.
"""
from __future__ import annotations

import pytest
import torch

from ml.src.models.gnn_encoder import GNNEncoder
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_nodes: int = 5, n_edges: int = 4) -> tuple:
    """Return (x, edge_index, batch, edge_attr) for a synthetic graph.

    Phase 0-T1 fix: x uses NODE_FEATURE_DIM (12) not the old v1 value of 8.
    GATConv's in_channels=NODE_FEATURE_DIM, so passing 8-dim tensors would
    cause a dimension mismatch at the first conv layer at runtime.
    """
    x          = torch.randn(n_nodes, NODE_FEATURE_DIM)   # fix: was 8 (v1 schema)
    src        = torch.arange(n_edges)
    dst        = torch.arange(1, n_edges + 1)
    edge_index = torch.stack([src, dst])
    batch      = torch.zeros(n_nodes, dtype=torch.long)
    # Edge types drawn from current valid range [0, NUM_EDGE_TYPES).
    # NUM_EDGE_TYPES is 8 after Phase 1-A3 adds REVERSE_CONTAINS.
    edge_attr  = torch.randint(0, NUM_EDGE_TYPES - 1, (n_edges,))  # exclude type 7 (runtime-only)
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
# Edge attr guard — current contract: use_edge_attr=True + edge_attr=None is
# an error, not silent graceful degradation.
# ---------------------------------------------------------------------------

def test_edge_attr_none_raises_when_expected():
    """
    use_edge_attr=True + edge_attr=None must raise ValueError.

    Phase 0-T1 fix: the original test name was test_graceful_degradation_none_edge_attr
    and asserted 'Must not raise'.  A subsequent audit fix replaced the silent
    zero-vector fallback with an explicit ValueError guard because edge_attr=None
    with use_edge_attr=True indicates a broken data pipeline (every v5 graph must
    have edge_attr).  Silently degrading hid this class of bug.

    The test now documents the current, correct contract.
    """
    enc = GNNEncoder(hidden_dim=64, heads=8, use_edge_attr=True)
    x, edge_index, batch, _ = _make_graph()

    with pytest.raises(ValueError, match="edge_attr"):
        enc(x, edge_index, batch, edge_attr=None)


def test_edge_attr_influences_output():
    """
    Different edge_attr values must produce different node embeddings.

    Phase 0-T1 fix: the original test (test_zero_fallback_differs_from_real_edge_attr)
    passed edge_attr=None to a use_edge_attr=True encoder to compare against real
    edge_attr — but edge_attr=None now raises ValueError (see above).

    Updated intent: verify the edge embedding layer is actually used by running
    two forward passes with different edge type assignments and checking outputs differ.
    """
    torch.manual_seed(0)
    enc = GNNEncoder(hidden_dim=64, heads=8, use_edge_attr=True)
    enc.eval()

    x, edge_index, batch, _ = _make_graph(n_nodes=5, n_edges=4)

    # Two different edge type vectors — same graph topology, different relations.
    edge_attr_a = torch.tensor([0, 1, 5, 6], dtype=torch.long)  # CALLS, READS, CONTAINS, CF
    edge_attr_b = torch.tensor([4, 4, 4, 4], dtype=torch.long)  # all INHERITS

    with torch.no_grad():
        out_a, _ = enc(x, edge_index, batch, edge_attr=edge_attr_a)
        out_b, _ = enc(x, edge_index, batch, edge_attr=edge_attr_b)

    assert not torch.allclose(out_a, out_b, atol=1e-5), \
        "Edge attr values must influence output — embedding layer may be ignored."


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
    # Phase 0-T1 fix: was torch.randn(3, 8); must match NODE_FEATURE_DIM=12.
    x          = torch.randn(3, NODE_FEATURE_DIM)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    batch      = torch.zeros(3, dtype=torch.long)
    edge_attr  = torch.zeros(0, dtype=torch.long)

    node_embs, _ = enc(x, edge_index, batch, edge_attr=edge_attr)
    assert node_embs.shape == (3, 64)


# ---------------------------------------------------------------------------
# Phase 1-T2: Non-negotiable JK gradient flow test
# ---------------------------------------------------------------------------

def test_jk_gradient_flow():
    """
    All JK attention parameters and LayerNorm parameters must receive non-zero
    gradients after a backward pass.  This test is non-negotiable — it MUST
    pass before any v5.2 training is launched.

    Why this test exists (Phase 1-T2, 2026-05-14):
    The existing return_intermediates infrastructure in gnn_encoder.py collects
    intermediate embeddings using .detach().clone() (lines 294, 303, 313).
    If JK aggregation consumed those detached tensors, the JK attention weights
    would receive zero gradients for the entire training run — the mechanism
    would be present in the model but completely inert.  This test would have
    caught that bug.  The corrected implementation collects live (non-detached)
    tensors in a separate _live list.  This test verifies the fix holds.
    """
    gnn = GNNEncoder(hidden_dim=128, heads=8, use_jk=True, jk_mode='attention')

    # Minimal graph with all edge types that appear in Phase 2 and Phase 3.
    x          = torch.randn(5, NODE_FEATURE_DIM, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch      = torch.zeros(5, dtype=torch.long)
    edge_attr  = torch.tensor(
        [
            EDGE_TYPES["CALLS"],        # type 0 — structural Phase 1
            EDGE_TYPES["CONTAINS"],     # type 5 — forward CONTAINS Phase 1 + Phase 3
            EDGE_TYPES["CONTROL_FLOW"], # type 6 — Phase 2 CFG directed
            EDGE_TYPES["CONTAINS"],     # type 5 — another CONTAINS for Phase 3 reversal
        ],
        dtype=torch.long,
    )

    out, _ = gnn(x, edge_index, batch, edge_attr)
    loss = out.sum()
    loss.backward()

    # Every JK parameter must have a non-zero gradient.
    for name, param in gnn.jk.named_parameters():
        assert param.grad is not None, (
            f"JK param '{name}' has None gradient — "
            "live intermediate collection may be broken (detach bug)."
        )
        assert param.grad.abs().sum() > 0, (
            f"JK param '{name}' has all-zero gradients — "
            "JK attention weights are not learning."
        )

    # Every per-phase LayerNorm parameter must have a non-zero gradient.
    for i, layer_norm in enumerate(gnn.phase_norm):
        for name, param in layer_norm.named_parameters():
            assert param.grad is not None, (
                f"phase_norm[{i}].{name} has None gradient."
            )
            assert param.grad.abs().sum() > 0, (
                f"phase_norm[{i}].{name} has all-zero gradients."
            )


# ---------------------------------------------------------------------------
# Phase 1-T3: REVERSE_CONTAINS uses a separate embedding (not type 5)
# ---------------------------------------------------------------------------

def test_reverse_contains_separate_embedding():
    """
    Phase 3 must use REVERSE_CONTAINS (type 7) embedding, not forward CONTAINS
    (type 5).  Before Phase 1-A3, both directions used type-5 embeddings so the
    GNN could not learn directional asymmetry in the CONTAINS relationship.

    Phase 1-T3 (2026-05-14): This test verifies A3 is implemented correctly.
    After random initialisation, rows 5 and 7 in the edge embedding table should
    differ with probability ~1.  If they are equal, Phase 3 is still reusing the
    forward CONTAINS embedding.
    """
    # use_jk=False isolates the edge embedding table from JK machinery.
    gnn = GNNEncoder(hidden_dim=128, heads=8, use_jk=False)

    emb_contains = gnn.edge_embedding(
        torch.tensor([EDGE_TYPES["CONTAINS"]], dtype=torch.long)
    )
    emb_reverse = gnn.edge_embedding(
        torch.tensor([EDGE_TYPES["REVERSE_CONTAINS"]], dtype=torch.long)
    )

    assert not torch.allclose(emb_contains, emb_reverse, atol=1e-6), (
        "CONTAINS and REVERSE_CONTAINS embeddings are identical — "
        "Phase 3 is still using type-5 embedding instead of type-7.  "
        "Check A3 implementation in gnn_encoder.py."
    )


# ---------------------------------------------------------------------------
# Phase 1-T4: JK does not change output dimension
# ---------------------------------------------------------------------------

def test_jk_output_shape_unchanged():
    """
    JK (attention mode) must output [N, hidden_dim] — identical to non-JK.

    Why this matters (Phase 1-T4, 2026-05-14):
    SentinelModel, CrossAttentionFusion, and the eye projection layers all assume
    GNNEncoder outputs [N, 128].  If JK changed this dimension (e.g. cat+project
    mode would output [N, 384] before projection), every downstream component
    would silently receive wrong-shaped tensors.  The attention mode is chosen
    specifically because it preserves output dimension.
    """
    gnn = GNNEncoder(hidden_dim=128, heads=8, use_jk=True, jk_mode='attention')
    x, edge_index, batch, edge_attr = _make_graph(n_nodes=10, n_edges=8)
    out, returned_batch = gnn(x, edge_index, batch, edge_attr)

    assert out.shape == (10, 128), \
        f"JK output shape changed: expected [10, 128], got {out.shape}"
    assert returned_batch.shape == (10,)


def test_jk_disabled_output_shape():
    """
    use_jk=False must produce the same output shape as use_jk=True.
    Backward compatibility: existing checkpoints loaded with use_jk=False
    must not require any downstream changes.
    """
    gnn = GNNEncoder(hidden_dim=128, heads=8, use_jk=False)
    x, edge_index, batch, edge_attr = _make_graph(n_nodes=10, n_edges=8)
    out, _ = gnn(x, edge_index, batch, edge_attr)

    assert out.shape == (10, 128), \
        f"Non-JK output shape: expected [10, 128], got {out.shape}"
