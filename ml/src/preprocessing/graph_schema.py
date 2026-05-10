"""
graph_schema.py — SENTINEL graph feature schema (single source of truth)

WHY THIS FILE EXISTS
────────────────────
ml/src/data_extraction/ast_extractor.py (offline batch pipeline, ~68K training
contracts) and ml/src/inference/preprocess.py (online inference, one contract
per API request) previously duplicated every constant below verbatim.

Any divergence between the two files would silently corrupt inference: the
model would receive feature vectors encoded differently from the ones it was
trained on — producing wrong predictions with no error signal.

Centralising here ensures atomic propagation: one edit to this file updates
both pipelines simultaneously, making it structurally impossible for them to
drift apart.

CHANGE POLICY
─────────────
Any modification to NODE_TYPES, VISIBILITY_MAP, EDGE_TYPES, or the feature
ordering in FEATURE_NAMES requires ALL of the following steps:

  1. Rebuild all ~68K .pt graph files:
       python ml/src/data_extraction/ast_extractor.py --force
  2. Rebuild all token .pt files:
       python ml/scripts/tokenizer_v1_production.py --force
  3. Retrain the model from scratch:
       python ml/scripts/train.py
       (GNNEncoder has in_channels=NODE_FEATURE_DIM hardcoded in the checkpoint)
  4. Increment FEATURE_SCHEMA_VERSION to invalidate all inference caches:
       FEATURE_SCHEMA_VERSION = "v2"

Skipping any of these steps will cause silent accuracy regression.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Schema version
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SCHEMA_VERSION: str = "v1"
"""
Suffix appended to inference cache keys: "{content_md5}_{FEATURE_SCHEMA_VERSION}".

Bumping this string invalidates all cached (graph.pt, tokens.pt) pairs built
under the previous schema, preventing stale features from being returned.
Increment whenever NODE_TYPES, VISIBILITY_MAP, EDGE_TYPES, or FEATURE_NAMES
change — or whenever _build_node_features() logic changes in graph_extractor.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Structural constants
# ─────────────────────────────────────────────────────────────────────────────

NODE_FEATURE_DIM: int = 8
"""
Number of scalar features per graph node.

GNNEncoder is constructed with in_channels=NODE_FEATURE_DIM and the value is
hardcoded in the saved checkpoint. Changing this number requires a full retrain.
"""

NUM_EDGE_TYPES: int = 5
"""
Number of distinct edge-relation types (width of the EDGE_TYPES vocabulary).

Stored in graph.edge_attr as integer IDs in [0, NUM_EDGE_TYPES).
GNNEncoder embeds these via nn.Embedding(NUM_EDGE_TYPES, gnn_edge_emb_dim)
and adds the embedding to GATConv edge features (P0-B).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Node type vocabulary
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPES: dict[str, int] = {
    "STATE_VAR":   0,
    "FUNCTION":    1,
    "MODIFIER":    2,
    "EVENT":       3,
    "FALLBACK":    4,
    "RECEIVE":     5,
    "CONSTRUCTOR": 6,
    "CONTRACT":    7,
}
"""
Maps each Slither declaration kind to an integer ID used as node feature[0].

Ordering constraint: IDs must remain stable across dataset versions.
Adding a new entry at the end is safe (new ID, no shift); inserting in the
middle would renumber existing classes and invalidate all training data.

Special Function sub-kinds (FALLBACK, RECEIVE, CONSTRUCTOR) are detected
at extraction time and override the default FUNCTION(1) value — see
_build_node_features() in graph_extractor.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Visibility ordinal encoding
# ─────────────────────────────────────────────────────────────────────────────

VISIBILITY_MAP: dict[str, int] = {
    "public":   0,
    "external": 0,
    "internal": 1,
    "private":  2,
}
"""
Ordinal encoding for Solidity visibility qualifiers (node feature[1]).

Ordinal (not one-hot) so the model can learn that private > internal > public
on the open-to-closed axis without needing 3 extra dimensions.

public=0 is the default fallback for declarations without explicit visibility.
Values not present in this map map to 0 (public) via dict.get().
"""

# ─────────────────────────────────────────────────────────────────────────────
# Edge type vocabulary
# ─────────────────────────────────────────────────────────────────────────────

EDGE_TYPES: dict[str, int] = {
    "CALLS":    0,   # function → internally-called function
    "READS":    1,   # function → state variable it reads
    "WRITES":   2,   # function → state variable it writes
    "EMITS":    3,   # function → event it emits
    "INHERITS": 4,   # contract → parent contract (linearised MRO)
}
"""
Maps each semantic edge relation to an integer ID stored in graph.edge_attr.

Edges are directed and typed. GNNEncoder embeds these IDs via
nn.Embedding(NUM_EDGE_TYPES, gnn_edge_emb_dim) and feeds the result
into every GATConv layer (P0-B — added 2026-05-02).

Shape: graph.edge_attr must be a 1-D int64 tensor of shape [E] (PyG
convention). Pre-refactor .pt files produced by the old ast_extractor.py
stored shape [E, 1] — nn.Embedding will crash on that shape. Always run
validate_graph_dataset.py before training to confirm all files have [E] shape.
Re-extract with: python ml/src/data_extraction/ast_extractor.py --force
"""

# ─────────────────────────────────────────────────────────────────────────────
# Feature name registry
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: tuple[str, ...] = (
    "type_id",     # feature[0] — float(NODE_TYPES[kind])              node category
    "visibility",  # feature[1] — VISIBILITY_MAP ordinal 0-2           access control
    "pure",        # feature[2] — 1.0 if Function.pure                 no state I/O
    "view",        # feature[3] — 1.0 if Function.view                 read-only state
    "payable",     # feature[4] — 1.0 if Function.payable              Ether entry point
    "reentrant",   # feature[5] — 1.0 if Slither.is_reentrant          reentrancy flag
    "complexity",  # feature[6] — float(len(func.nodes)) CFG blocks    control-flow depth
    "loc",         # feature[7] — float(len(source_mapping.lines))     lines of code
)
"""
Human-readable labels for each node feature dimension.

Used by:
  - drift detection baseline scripts (compute_drift_baseline.py)
  - explainability / SHAP attribution tooling
  - test assertions in test_preprocessing.py

Index i in FEATURE_NAMES corresponds to column i in graph.x [N, NODE_FEATURE_DIM].
Non-Function nodes (state variables, events, modifiers, contract) receive 0.0
for features[2:6] — their structural role is captured by type_id alone.
"""

# Invariant check — catches accidental length divergence at import time, not
# at runtime deep inside a training loop.
assert len(FEATURE_NAMES) == NODE_FEATURE_DIM, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but NODE_FEATURE_DIM={NODE_FEATURE_DIM}. "
    "Update one to match the other."
)
