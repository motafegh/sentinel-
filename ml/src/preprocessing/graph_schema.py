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
       FEATURE_SCHEMA_VERSION = "v9"  (next increment — currently v8)

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

v3  — 8 edge types: + REVERSE_CONTAINS(7) — runtime-only reverse of CONTAINS(5).
      Generated inside GNNEncoder Phase 3; never written to graph .pt files.
      NODE_FEATURE_DIM, node types, and all graph files are unchanged.
      Only the GNNEncoder embedding table gains one row (7→8).
      Rationale: Phase 1-A3 (2026-05-14). Fixes v5.0 limitation L2 (shared
      CONTAINS embedding for both directions prevented directional learning).

v4  — 12 features (same dim, changed semantics):
      [2]  pure → uses_block_globals: 1.0 if func reads block.timestamp/number/etc.
           Rationale: `pure` was almost always 0 and provided no discriminative power
           for vulnerability detection. SolidityVariableComposed (block.timestamp) is
           NOT in state_variables_read → no READS edge in graph → Timestamp class had
           zero direct feature signal. This feature gives Timestamp and TOD a signal.
      [6]  loc: raw line count → log1p(lines)/log1p(1000), normalised [0,1]
           Rationale: raw loc hit 2538 for CONTRACT node vs all other features in [0,1].
           This created 2538× scale imbalance in dot products. Log normalization fixes
           gradient magnitude imbalance at initialization.
      [11] external_call_count: now includes Transfer/Send IR ops (Slither classifies
           ETH transfer() and send() separately from LowLevelCall/HighLevelCall).
           Rationale: DoS via `payable(addr).transfer(share)` inside loops produced
           ext_calls=0.0 because Transfer is a different IR type. DoS loops are now
           visible.
      CFG node typing: Transfer/Send now classified as CFG_NODE_CALL (same fix).
      Rationale: analysis of all 20 manual test contracts revealed these 4 bugs were
      making MishandledException, UnusedReturn, Timestamp, and DoS nearly invisible.
      (2026-05-16 session)

v5  — 12 features (same dim, changed semantics — BUG FIXES from full audit):
      [5]  complexity: raw CFG block count → log1p(len(nodes))/log1p(100), normalised [0,1]
           Rationale: raw complexity hit 100+ for large contracts, creating scale
           dominance similar to the old raw-loc bug. Log normalization aligns it
           with other [0,1] features.
      [6]  loc (CFG nodes): raw line count → log1p(lines)/log1p(1000), normalised [0,1]
           Rationale: _build_cfg_node_features() was using raw loc while
           _build_node_features() already had log-normalization. This created
           a scale mismatch where CFG nodes had features 10-100× larger than
           declaration nodes. Now consistent.
      [7]  return_ignored: now detects .send() return-value discards (BUG-9 fix).
           Added Slither Send IR type to the isinstance check alongside
           LowLevelCall and HighLevelCall.
           Rationale: `addr.send(amount)` returns a bool indicating success/failure.
           Ignoring this return is a MishandledException vulnerability. The old
           code only checked LowLevelCall/HighLevelCall, missing Send entirely.
      Contract selection: "most functions" heuristic (52.6% accurate) replaced
           with "most_derived" composite heuristic (~92%+ accurate). Uses
           Slither's contract.inheritance to pick the contract that inherits from
           the most other in-file candidates. Fallback: last-defined (87.4%).
           Rationale: BUG-6 audit showed the old heuristic picked library contracts
           (StandardToken with 20+ functions) instead of the actual vulnerable
           implementation contract (ERC20Token with 3-5 overrides). 47.4% of
           multi-contract files had the wrong contract selected.
      (2026-05-17 session — full audit BUG-1,2,6,9 fixes)

v6  — 12 features (same dim — BUG-3 visibility fix, in-place patch applied):
      [1]  visibility: ordinal {public=0, internal=1, private=2} →
                       normalised {public=0.0, external=0.0, internal=0.5, private=1.0}
           Rationale: full data validation (2026-05-17) confirmed that 7,854 graphs
           (17.7% of dataset) had visibility=2 for private functions, which exceeded
           the declared [0,1] feature range. The GNN receives these as input dot-product
           operands; a value of 2 creates a 2× scale mismatch against all other [0,1]
           features in the first layer. The ordinal ordering is preserved
           (private=1.0 > internal=0.5 > public=0.0). Existing graphs are patched
           in-place by ml/scripts/patch_graph_features.py.
      BUG-1/2 in-place patch confirmed complete: all 44,470 graphs verified to have
           loc ∈ [0,1] and complexity ∈ [0,1] after patch_graph_features.py run.
      (2026-05-17 session — fresh full validation + in-place patch)

v7  — 11 features (in_unchecked dropped — BUG-L2):
      [9]  in_unchecked removed: dead for 87.9% of dataset (nearly always 0.0).
           Index shift: has_loop [10]→[9], external_call_count [11]→[10].
      EMITS(3) now fires: EventCall IR fallback added (BUG-H7).
      INHERITS(4) now fires: parent Contract nodes added (BUG-H8).
      CFG nodes inherit dims [1,3,4,5,9] from FUNCTION parent (BUG-C3).
      return_ignored uses lval.name not id(lval) for hashability (BUG-M1).
      OOR validation logged at extraction time (BUG-L4).
      REVERSE_CONTAINS(7) runtime edge type formalised in schema constants.
      Requires full re-extraction of all graph .pt files.
      (2026-05-18 session — all 27 pre-training bugs fixed, 41,522 graphs re-extracted)
"""

from __future__ import annotations

import sys
from enum import IntEnum

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
            "SENTINEL requires >=0.9.3 for NodeType.STARTUNCHECKED support. "
            "Upgrade: pip install 'slither-analyzer>=0.9.3,<0.11'"
        )
except importlib.metadata.PackageNotFoundError:
    pass  # slither not installed in this environment (e.g., inference-only deploy)


# ─────────────────────────────────────────────────────────────────────────────
# Schema version
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SCHEMA_VERSION: str = "v8"
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

NODE_FEATURE_DIM: int = 11
"""
Number of scalar features per graph node (v8 schema — 11 dims).

GNNEncoder is constructed with in_channels=NODE_FEATURE_DIM. Changing this
number requires a full graph re-extraction and model retrain.

Feature layout (v8 — 11 dims; dropped in_unchecked [was dim 9] from v6):
  [0]  type_id              — NODE_TYPES int (range 0–12), normalised /12.0
  [1]  visibility           — VISIBILITY_MAP ordinal
  [2]  uses_block_globals   — 1.0 if func reads block.timestamp/number/difficulty/basefee
  [3]  view                 — bool
  [4]  payable              — bool
  [5]  complexity           — log1p(CFG block count)/log1p(100), normalised [0,1]
  [6]  loc                  — log1p(lines)/log1p(1000), normalised [0,1]
  [7]  return_ignored       — 0.0 (captured) / 1.0 (discarded) / -1.0 (IR unavailable)
  [8]  call_target_typed    — 0.0 (raw addr) / 1.0 (typed) / -1.0 (source unavailable)
  [9]  has_loop             — bool  (was dim [10] in v6)
  [10] external_call_count  — float, log-normalized; includes Transfer/Send ops
                              (was dim [11] in v6)

Removed from v6:
  [9]  in_unchecked         — DEAD signal. 87.9% of BCCC dataset is Solidity 0.4.x
                              which predates the unchecked{} keyword (0.8.0+). Feature
                              was 0.0 for nearly all training samples. BUG-L2.

Note: `reentrant` (Slither's own is_reentrant flag) was feature[5] in v1 and
is removed in v2 — keeping it gave the model a Slither-provided answer rather
than forcing it to detect reentrancy from structural patterns.
"""

NUM_NODE_TYPES: int = 13
"""Number of distinct node types (ids 0–12). Used by GNNEncoder for the node type embedding."""

NUM_EDGE_TYPES: int = 11
"""
Number of distinct edge-relation types (width of the EDGE_TYPES vocabulary).

IDs 0–6 are stored in graph.edge_attr on disk. IDs 8–9 (CALL_ENTRY, RETURN_TO)
are the v8 ICFG-Lite additions — stored on disk in v8 graphs.
ID 7 (REVERSE_CONTAINS) is RUNTIME-ONLY — generated inside GNNEncoder Phase 3
by reversing CONTAINS(5) edges; NEVER written to .pt files on disk.

GNNEncoder embeds all 11 IDs via nn.Embedding(NUM_EDGE_TYPES, gnn_edge_emb_dim).

v2 additions (ids 5 and 6):
  CONTAINS     — function node → its CFG_NODE children (Phase 1).
  CONTROL_FLOW — CFG_NODE → successor CFG_NODE, intra-function (Phase 2).

Phase 1-A3 addition (id 7):
  REVERSE_CONTAINS — runtime-only; CFG_NODE → parent function (Phase 3).

v8 additions (ids 8–10) — ICFG-Lite and DEF_USE, stored on disk:
  CALL_ENTRY — calling CFG_NODE → ENTRYPOINT of the callee function.
               Enables cross-function CFG signal for reentrancy and CEI.
  RETURN_TO  — terminal CFG_NODE of callee → successor of the call site.
               Closes the inter-procedural control-flow loop.
  DEF_USE    — CFG_NODE defining a LocalVariable → CFG_NODE reading it.
               Intra-function data-flow: captures how values computed (by
               arithmetic ops, call returns) flow into checks and state writes.
               Key for integer overflow and return-value-ignored patterns.
CALL_ENTRY/RETURN_TO are routed through Phase 2 in GNNEncoder.
DEF_USE is also routed through Phase 2 (data-flow is execution-order-dependent).
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


class NodeType(IntEnum):
    """
    Typed aliases for NODE_TYPES integer IDs.

    Use these instead of raw integers when selecting or filtering node types in
    sentinel_model.py, predictor.py, or audit scripts.  The IntEnum values are
    identical to NODE_TYPES — they are derived from it at module load time, so
    they cannot drift.  Any future NODE_TYPES addition must also add a member here.

    Usage:
        STRUCTURAL_PREFIX_TYPES = {
            NodeType.FUNCTION, NodeType.MODIFIER, NodeType.CONSTRUCTOR,
            NodeType.FALLBACK,  NodeType.RECEIVE,
            NodeType.CFG_NODE_CALL, NodeType.CFG_NODE_WRITE, NodeType.CFG_NODE_CHECK,
        }
        eligible = [n for n in graph.node_type if n in STRUCTURAL_PREFIX_TYPES]
    """
    STATE_VAR      = NODE_TYPES["STATE_VAR"]        # 0
    FUNCTION       = NODE_TYPES["FUNCTION"]          # 1
    MODIFIER       = NODE_TYPES["MODIFIER"]          # 2
    EVENT          = NODE_TYPES["EVENT"]             # 3
    FALLBACK       = NODE_TYPES["FALLBACK"]          # 4
    RECEIVE        = NODE_TYPES["RECEIVE"]           # 5
    CONSTRUCTOR    = NODE_TYPES["CONSTRUCTOR"]       # 6
    CONTRACT       = NODE_TYPES["CONTRACT"]          # 7
    CFG_NODE_CALL  = NODE_TYPES["CFG_NODE_CALL"]    # 8
    CFG_NODE_WRITE = NODE_TYPES["CFG_NODE_WRITE"]   # 9
    CFG_NODE_READ  = NODE_TYPES["CFG_NODE_READ"]    # 10
    CFG_NODE_CHECK = NODE_TYPES["CFG_NODE_CHECK"]   # 11
    CFG_NODE_OTHER = NODE_TYPES["CFG_NODE_OTHER"]   # 12


STRUCTURAL_PREFIX_TYPES: frozenset[NodeType] = frozenset({
    NodeType.FUNCTION,
    NodeType.MODIFIER,
    NodeType.CONSTRUCTOR,
    NodeType.FALLBACK,
    NodeType.RECEIVE,
})
"""
Node types eligible for GNN prefix injection — Phase 1 (declaration-level only).

PRE-4 audit (41,576 training graphs, 2026-05-23):
  Declaration-level only: mean=20.3, P50=16, P95=47 → K=48 covers 95.5%.
  All-types (incl. CFG): mean=48.7, P95=122 → K=32 covers only 42.3% (unusable).

Why declaration-level only:
  After Phase 3 REVERSE_CONTAINS, FUNCTION nodes carry aggregated CFG signal
  from their child CFG_NODE_* children.  The transformer gets structural context
  from function-level nodes; CrossAttentionFusion provides CFG-level detail.
  CFG nodes inflate the prefix budget for minimal additional signal in Phase 1.

Phase 1B ablation (after Phase 1 results): add NodeType.CFG_NODE_CALL with K=64
to isolate whether explicit call-site visibility helps Reentrancy.

Priority ordering within eligible nodes (when contract has > K=48 eligible nodes):
  1. CONSTRUCTOR, FALLBACK, RECEIVE  — always first (≤ 3 per contract)
  2. MODIFIER                        — included next  (typically ≤ 5)
  3. FUNCTION                        — sorted by feature[10] descending (external_call_count)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Visibility ordinal encoding
# ─────────────────────────────────────────────────────────────────────────────

VISIBILITY_MAP: dict[str, float] = {
    "public":   0.0,
    "external": 0.0,
    "internal": 0.5,
    "private":  1.0,
}
"""
Normalised ordinal encoding for Solidity visibility qualifiers (node feature[1]).

Values are in [0, 1] and preserve the ordinal ordering:
    private (1.0) > internal (0.5) > public/external (0.0)

History:
    v5 and earlier used int encoding {public:0, internal:1, private:2}.
    private=2 exceeded the nominal [0,1] feature range and created a 2× scale
    imbalance in GNN dot products for 17.7% of graphs (7,854 / 44,470).

    Changed to float normalised encoding in schema v6 (2026-05-17).
    Existing graphs on disk were patched in-place by patch_graph_features.py.

public=0.0 is the default fallback for declarations without explicit visibility.
Values not present in this map map to 0.0 (public) via dict.get(..., 0.0).
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
    # v2 additions — intra-function control-flow structure
    "CONTAINS":          5,   # function node → its CFG_NODE children
    "CONTROL_FLOW":      6,   # CFG_NODE → successor CFG_NODE, directed
    # Phase 1-A3 — runtime-only reverse edge, NEVER on disk
    "REVERSE_CONTAINS":  7,   # CFG_NODE → parent function (GNNEncoder Phase 3 only)
    # v8 additions — ICFG-Lite cross-function edges, stored on disk
    "CALL_ENTRY":        8,   # calling CFG_NODE → ENTRYPOINT of callee function
    "RETURN_TO":         9,   # terminal CFG_NODE of callee → call-site successor
    # v8 addition — intra-function data-flow edges, stored on disk
    "DEF_USE":           10,  # CFG_NODE defining a LocalVariable → node reading it
}
"""
Maps each semantic edge relation to an integer ID stored in graph.edge_attr.

Edges are directed and typed. GNNEncoder embeds these IDs via
nn.Embedding(NUM_EDGE_TYPES, gnn_edge_emb_dim) and feeds the result
into each GATConv layer.

GNNEncoder uses three phases, each seeing a different subset of edge types:
  Phase 1 (Layers 1+2): types 0–5 (structural + CONTAINS forward)
  Phase 2 (Layers 3–5): types 6,8,9,10 (CONTROL_FLOW + CALL_ENTRY + RETURN_TO + DEF_USE; directed)
  Phase 3 (Layers 6+7+8): type 7  REVERSE_CONTAINS (CFG_NODE → function; runtime-only) + type 5 CONTAINS down (conv4c)

Shape: graph.edge_attr must be a 1-D int64 tensor of shape [E] (PyG
convention). Pre-refactor .pt files produced by the old ast_extractor.py
stored shape [E, 1] — nn.Embedding will crash on that shape. Always run
validate_graph_dataset.py before training to confirm all files have [E] shape.
Re-extract with: python ml/src/data_extraction/ast_extractor.py --force
"""

# ─────────────────────────────────────────────────────────────────────────────
# Feature name registry (v8 — 11 dimensions)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: tuple[str, ...] = (
    "type_id",              # [0]  float(NODE_TYPES[kind])/12.0           node category [0,1]
    "visibility",           # [1]  VISIBILITY_MAP ordinal 0-2             access control
    "uses_block_globals",   # [2]  1.0 if reads block.timestamp/number    Timestamp/TOD signal
    "view",                 # [3]  1.0 if Function.view                   read-only state
    "payable",              # [4]  1.0 if Function.payable                Ether entry point
    "complexity",           # [5]  log1p(len(func.nodes))/log1p(100)      normalised CFG block count
    "loc",                  # [6]  log1p(lines)/log1p(1000), [0,1]        normalised LoC
    "return_ignored",       # [7]  0.0=captured / 1.0=discarded / -1.0=IR unavailable
    "call_target_typed",    # [8]  0.0=raw addr / 1.0=typed / -1.0=source unavailable
    "has_loop",             # [9]  1.0 if function contains a loop        (was [10] in v6)
    "external_call_count",  # [10] log1p(n)/log1p(20), [0,1]; incl. Transfer/Send
                            #      (was [11] in v6)
)
"""
Human-readable labels for each node feature dimension (v8 — 11 dims).

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
# A1: Guard the normalization divisor used in graph_extractor.py and
# sentinel_model.py (_MAX_TYPE_ID = float(max(NODE_TYPES.values()))).
# If a new node type is ever appended, this assert fires immediately at import,
# forcing the developer to update every site that divides type_id by 12.
assert max(NODE_TYPES.values()) == 12, (
    "max(NODE_TYPES.values()) must equal 12. "
    "If you added a new node type, update _MAX_TYPE_ID normalization in "
    "graph_extractor.py (_MAX_TYPE_ID assert and decode-side NF-2 fix) and "
    "sentinel_model.py (_MAX_TYPE_ID assert) before re-extracting graphs."
)
