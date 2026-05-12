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
  2. Rebuild all token .pt files (only if tokenizer logic changed):
       python ml/scripts/tokenizer_v1_production.py --force
  3. Retrain the model from scratch:
       python ml/scripts/train.py
       (GNNEncoder reads in_channels=NODE_FEATURE_DIM at construction time)
  4. Increment FEATURE_SCHEMA_VERSION to invalidate all inference caches:
       FEATURE_SCHEMA_VERSION = "v3"  (next increment)

Skipping any of these steps will cause silent accuracy regression.

SCHEMA HISTORY
──────────────
v1  — 8 features: type_id, visibility, pure, view, payable, reentrant,
                  complexity, loc
      5 edge types: CALLS, READS, WRITES, EMITS, INHERITS
      8 node types: STATE_VAR…CONTRACT

v2  — 12 features: reentrant REMOVED; gas_intensity REMOVED; 5 new semantic
                   features added (return_ignored, call_target_typed,
                   in_unchecked, has_loop, external_call_count).
      7 edge types: + CONTAINS (function→cfg_node), + CONTROL_FLOW (cfg_node→cfg_node)
      13 node types: + 5 CFG subtypes (8–12: CALL, WRITE, READ, CHECK, OTHER)
      Rationale: v5 complete overhaul (2026-05-11). The model must learn
      structural vulnerability patterns from first principles.
      gas_intensity removed: it was f(complexity, has_loop, external_call_count) —
      a circular heuristic over features already in the vector at indices 5, 10, 11.
"""

from __future__ import annotations

import sys

# ─────────────────────────────────────────────────────────────────────────────
# Slither version assertion — hard failure at import, not a warning.
# An old Slither silently produces wrong in_unchecked features (NodeType.STARTUNCHECKED
# only available in >=0.9.3). Fail fast so the developer knows immediately.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import importlib.metadata as _importlib_metadata
    _ver_str = _importlib_metadata.version("slither-analyzer")
    _version = tuple(int(x) for x in _ver_str.split(".")[:3])
    if _version < (0, 9, 3):
        raise RuntimeError(
            f"slither-analyzer {_ver_str} is too old. "
            "v5 requires >=0.9.3 for NodeType.STARTUNCHECKED support. "
            "Upgrade: pip install 'slither-analyzer>=0.9.3,<0.11'"
        )
except importlib.metadata.PackageNotFoundError:
    pass  # slither not installed in this environment (e.g., inference-only deploy)


# ─────────────────────────────────────────────────────────────────────────────
# Schema version
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SCHEMA_VERSION: str = "v3"
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

NODE_FEATURE_DIM: int = 12
"""
Number of scalar features per graph node (v2 schema).

GNNEncoder is constructed with in_channels=NODE_FEATURE_DIM. Changing this
number requires a full graph re-extraction and model retrain.

Feature layout (v2 — 12 dims):
  [0]  type_id             — NODE_TYPES int (range 0–12)
  [1]  visibility          — VISIBILITY_MAP ordinal
  [2]  pure                — bool
  [3]  view                — bool
  [4]  payable             — bool
  [5]  complexity          — CFG block count
  [6]  loc                 — lines of code
  [7]  return_ignored      — NEW: 0.0 (captured) / 1.0 (discarded) / -1.0 (IR unavailable)
  [8]  call_target_typed   — NEW: 0.0 (raw addr) / 1.0 (typed) / -1.0 (source unavailable)
  [9]  in_unchecked        — NEW: bool
  [10] has_loop            — NEW: bool
  [11] external_call_count — NEW: float, log-normalized

Note: `reentrant` (Slither's own is_reentrant flag) was feature[5] in v1 and
is removed in v2 — keeping it gave the model a Slither-provided answer rather
than forcing it to detect reentrancy from structural patterns.

Note: `gas_intensity` was feature[12] in the intermediate v2 draft but is
removed in the final v2 schema — it was f(complexity, has_loop, external_call_count),
a circular heuristic over features already present at indices 5, 10, 11.
The GNN learns the combination itself and learns it better.
"""

NUM_NODE_TYPES: int = 13
"""Number of distinct node types (ids 0–12). Used by GNNEncoder for the node type embedding."""

NUM_EDGE_TYPES: int = 7
"""
Number of distinct edge-relation types (width of the EDGE_TYPES vocabulary).

Stored in graph.edge_attr as integer IDs in [0, NUM_EDGE_TYPES).
GNNEncoder embeds these via nn.Embedding(NUM_EDGE_TYPES, gnn_edge_emb_dim)
and adds the embedding to GATConv edge features.

v2 additions (ids 5 and 6):
  CONTAINS     — function node → its CFG_NODE children; makes the CFG subgraph
                 connected to the rest of the contract graph so GNN message
                 passing can aggregate function-level properties into statement
                 nodes and vice versa (Phase 1).
  CONTROL_FLOW — CFG_NODE → successor CFG_NODE; encodes intra-function
                 execution order, enabling the model to distinguish
                 "call before write" (reentrancy) from "write before call" (CEI)
                 via Phase 2 directed message passing.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Node type vocabulary
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPES: dict[str, int] = {
    # Declaration-level node types (v1 — ids 0–7, stable)
    "STATE_VAR":   0,
    "FUNCTION":    1,
    "MODIFIER":    2,
    "EVENT":       3,
    "FALLBACK":    4,
    "RECEIVE":     5,
    "CONSTRUCTOR": 6,
    "CONTRACT":    7,
    # CFG subtypes (v2 — ids 8–12, new in v2)
    # Distinct type_ids give the GNN different initial embeddings for different
    # statement roles. A CALL node (8) and WRITE node (9) start from different
    # representations; execution order is then encoded by directed CONTROL_FLOW edges.
    "CFG_NODE_CALL":   8,   # statement containing an external call
    "CFG_NODE_WRITE":  9,   # statement writing a state variable
    "CFG_NODE_READ":   10,  # statement reading a state variable
    "CFG_NODE_CHECK":  11,  # require / assert / if condition
    "CFG_NODE_OTHER":  12,  # all other statement types (synthetic nodes, etc.)
}
"""
Maps each Slither declaration kind to an integer ID used as node feature[0].

Ordering constraint: IDs must remain stable across dataset versions.
Adding a new entry at the end is safe (new ID, no shift); inserting in the
middle would renumber existing classes and invalidate all training data.

CFG subtypes (8–12) are new in v2. Each Slither FunctionNode within a function
becomes a separate graph node of one of these types, connected to its parent
function via a CONTAINS(5) edge and to its successors via CONTROL_FLOW(6) edges.

Priority for _cfg_node_type() when a single IR node spans multiple operations:
  1. CFG_NODE_CALL  (8) — external call present in IR (highest priority)
  2. CFG_NODE_WRITE (9) — state variable write present in IR
  3. CFG_NODE_READ  (10) — state variable read present in IR
  4. CFG_NODE_CHECK (11) — require / assert / if condition
  5. CFG_NODE_OTHER (12) — everything else, including synthetic nodes

Total node types: 13 (ids 0–12).
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
    "CALLS":        0,   # function → internally-called function
    "READS":        1,   # function → state variable it reads
    "WRITES":       2,   # function → state variable it writes
    "EMITS":        3,   # function → event it emits
    "INHERITS":     4,   # contract → parent contract (linearised MRO)
    # v2 additions — control-flow structure
    "CONTAINS":     5,   # function node → its CFG_NODE children (NEW in v2)
    "CONTROL_FLOW": 6,   # CFG_NODE → successor CFG_NODE, directed (NEW in v2)
}
"""
Maps each semantic edge relation to an integer ID stored in graph.edge_attr.

Edges are directed and typed. GNNEncoder embeds these IDs via
nn.Embedding(NUM_EDGE_TYPES, gnn_edge_emb_dim) and feeds the result
into each GATConv layer.

GNNEncoder uses three phases, each seeing a different subset of edge types:
  Phase 1 (Layers 1+2): types 0–5 (structural + CONTAINS forward)
  Phase 2 (Layer 3):    type  6  (CONTROL_FLOW only, directed)
  Phase 3 (Layer 4):    type  5  REVERSED (CFG_NODE → function; same embedding
                                  as forward CONTAINS — v5.0 known limitation)

Shape: graph.edge_attr must be a 1-D int64 tensor of shape [E] (PyG
convention). Pre-refactor .pt files produced by the old ast_extractor.py
stored shape [E, 1] — nn.Embedding will crash on that shape. Always run
validate_graph_dataset.py before training to confirm all files have [E] shape.
Re-extract with: python ml/src/data_extraction/ast_extractor.py --force
"""

# ─────────────────────────────────────────────────────────────────────────────
# Feature name registry (v2 — 12 dimensions)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: tuple[str, ...] = (
    "type_id",              # [0]  float(NODE_TYPES[kind])               node category (0-12)
    "visibility",           # [1]  VISIBILITY_MAP ordinal 0-2            access control
    "pure",                 # [2]  1.0 if Function.pure                  no state I/O
    "view",                 # [3]  1.0 if Function.view                  read-only state
    "payable",              # [4]  1.0 if Function.payable               Ether entry point
    # "reentrant" was [5] in v1 — REMOVED in v2 (Slither shortcut)
    "complexity",           # [5]  float(len(func.nodes))                CFG block count
    "loc",                  # [6]  float(len(source_mapping.lines))      lines of code
    "return_ignored",       # [7]  NEW: 0.0=captured / 1.0=discarded / -1.0=IR unavailable
    "call_target_typed",    # [8]  NEW: 0.0=raw addr / 1.0=typed / -1.0=source unavailable
    "in_unchecked",         # [9]  NEW: 1.0 if function contains unchecked{} block
    "has_loop",             # [10] NEW: 1.0 if function contains a loop
    "external_call_count",  # [11] NEW: float, log-normalized (log1p(n)/log1p(20), clamped [0,1])
    # "gas_intensity" was present in intermediate draft — REMOVED in final v2
    # It was f(complexity, has_loop, external_call_count) — a circular heuristic
    # over features already present at indices 5, 10, 11.
)
"""
Human-readable labels for each node feature dimension (v2 — 12 dims).

Used by:
  - drift detection baseline scripts (compute_drift_baseline.py)
  - explainability / SHAP attribution tooling
  - test assertions in test_preprocessing.py

Index i in FEATURE_NAMES corresponds to column i in graph.x [N, NODE_FEATURE_DIM].

Sentinel values:
  return_ignored [7]:    -1.0 when Slither IR unavailable (not assumed safe)
  call_target_typed [8]: -1.0 when source_mapping unavailable (not assumed safe)
  All other features: no sentinel (0.0 used for "not applicable" on non-Function nodes)

Non-Function nodes receive 0.0 for features [2:] except:
  - call_target_typed [8]: non-Function defaults to 1.0 (not applicable, safe default)
  - type_id [0], visibility [1], loc [6]: computed per declaration kind
CFG_NODE nodes receive features computed from the corresponding Slither FunctionNode.
CFG_NODE in_unchecked [9] is always 0.0 — never inherited from the parent function.
"""

# Invariant: length must match NODE_FEATURE_DIM. Caught at import time so any
# accidental divergence raises immediately rather than producing a silent shape
# mismatch deep inside a training loop.
assert len(FEATURE_NAMES) == NODE_FEATURE_DIM, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but NODE_FEATURE_DIM={NODE_FEATURE_DIM}. "
    "Update one to match the other."
)
assert len(EDGE_TYPES) == NUM_EDGE_TYPES, (
    f"EDGE_TYPES has {len(EDGE_TYPES)} entries but NUM_EDGE_TYPES={NUM_EDGE_TYPES}. "
    "Update one to match the other."
)
assert len(NODE_TYPES) == 13, (
    f"NODE_TYPES has {len(NODE_TYPES)} entries but expected 13 (ids 0-12). "
    "CFG subtypes 8-12 must all be present."
)
