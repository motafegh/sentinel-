"""Schema constants for the SENTINEL graph representation — active schema: v9.

STUB = True: constants are copied by value from ml/src/preprocessing/graph_schema.py
(verified 2026-06-08). The real port happens in Stage 2 with a byte-identical regression
test. Stage 7 seam swap removes this stub and makes sentinel-ml import from here.

DO NOT change these constants without bumping FEATURE_SCHEMA_VERSION and updating
_schema_version_registry.json. Class order (NUM_CLASSES = 10, indices 0-9) is LOCKED —
changing it invalidates all existing checkpoints.
"""

# ── Sentinel flag ──────────────────────────────────────────────────────────────
STUB: bool = True  # Stage 7 removes this after regression test passes

# ── Schema version (bump on any structural change) ────────────────────────────
FEATURE_SCHEMA_VERSION: str = "v9"

# ── Dimensions ────────────────────────────────────────────────────────────────
NODE_FEATURE_DIM: int = 12    # was 11 in v8; feat[11] = in_unchecked_block added
NUM_NODE_TYPES: int = 14      # was 13 in v8; CFG_NODE_ARITH=13 added
NUM_EDGE_TYPES: int = 12      # was 11 in v8; EXTERNAL_CALL=11 self-loop added
_MAX_TYPE_ID: float = 13.0    # was 12.0 in v8
NUM_CLASSES: int = 10         # LOCKED — class order matches all existing checkpoints

# ── Feature names (index → name) ─────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    "node_type_norm",       # 0  normalised node type id
    "visibility",           # 1  function visibility (0=public/ext, 1=internal, 2=private)
    "uses_block_globals",   # 2  count of block.timestamp / block.number / now reads
    "external_call_count",  # 3  number of external calls in this node's scope
    "state_var_writes",     # 4  number of state variable write ops
    "contract_size_norm",   # 5  normalised contract line count
    "loc",                  # 6  raw line count of this function / node
    "return_ignored",       # 7  1.0 if a return value is silently dropped
    "call_target_typed",    # 8  1.0 if call target is typed (HighLevelCall), 0 = raw low-level
    "has_loop",             # 9  1.0 if this CFG node is inside a loop body
    "payable",              # 10 1.0 if the enclosing function is payable
    "in_unchecked_block",   # 11 NEW v9 — fraction of nodes in unchecked{} scope (pre-0.8 → 1.0)
]

# ── Node types (id → name, 14 entries, max id = 13) ──────────────────────────
NODE_TYPES: dict[int, str] = {
    0:  "STATE_VAR",
    1:  "FUNCTION",
    2:  "MODIFIER",
    3:  "EVENT",
    4:  "FALLBACK",
    5:  "RECEIVE",
    6:  "CONSTRUCTOR",
    7:  "CONTRACT",
    8:  "CFG_NODE_CALL",
    9:  "CFG_NODE_WRITE",
    10: "CFG_NODE_READ",
    11: "CFG_NODE_CHECK",
    12: "CFG_NODE_OTHER",
    13: "CFG_NODE_ARITH",  # NEW v9 — pure Binary arithmetic op nodes
}

# ── Edge types (id → name, 12 entries) ───────────────────────────────────────
EDGE_TYPES: dict[int, str] = {
    0:  "CONTAINS",
    1:  "CONTROL_FLOW",
    2:  "DEF_USE",
    3:  "CALL_ENTRY",
    4:  "RETURN_TO",
    5:  "STATE_READ",
    6:  "STATE_WRITE",
    7:  "INHERITANCE",
    8:  "MODIFIER_USE",
    9:  "EMITS",
    10: "REVERSE_CONTAINS",  # runtime-only; 0 on disk, built by GNNEncoder
    11: "EXTERNAL_CALL",     # NEW v9 — self-loop on cross-contract call nodes
}

# ── Function visibility map ───────────────────────────────────────────────────
VISIBILITY_MAP: dict[str, float] = {
    "public":   0.0,
    "external": 0.0,
    "internal": 1.0,
    "private":  2.0,
    "default":  0.0,
}

# ── Class order (LOCKED — matches all existing checkpoints) ──────────────────
CLASS_NAMES: list[str] = [
    "Reentrancy",           # 0
    "CallToUnknown",        # 1
    "Timestamp",            # 2
    "ExternalBug",          # 3
    "GasException",         # 4
    "DenialOfService",      # 5
    "IntegerUO",            # 6
    "UnusedReturn",         # 7
    "MishandledException",  # 8
    "NonVulnerable",        # 9
]
