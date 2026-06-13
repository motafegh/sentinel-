"""
graph_schema.py — SENTINEL graph feature schema (single source of truth, Stage 7B)

Stage 7B seam swap (2026-06-12): this file is now the canonical source of truth.
`ml/src/preprocessing/graph_schema.py` is now a thin re-export shim pointing here.
All model files (gnn_encoder.py, sentinel_model.py, predictor.py, etc.) continue to
work without modification — their imports still resolve via the shim.

Class order (CLASS_NAMES, NUM_CLASSES) is LOCKED — changing it invalidates existing
checkpoints. DO NOT reorder or add classes without bumping FEATURE_SCHEMA_VERSION.

WHY THIS FILE EXISTS
────────────────────
ml/src/data_extraction/ast_extractor.py (offline batch pipeline, ~41K training
graphs) and ml/src/inference/preprocess.py (online inference, one contract
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

  1. Rebuild all graph .pt files:
       sentinel-data export (re-run full pipeline)
  2. Rebuild all token .pt files (only if tokenizer logic changed):
       sentinel-data represent
  3. Retrain the model from scratch:
       python ml/scripts/train.py
       (GNNEncoder reads in_channels=NODE_FEATURE_DIM at construction time)
  4. Increment FEATURE_SCHEMA_VERSION to invalidate all inference caches:
       FEATURE_SCHEMA_VERSION = "v10"  (next increment — currently v9)

Skipping any of these steps will cause silent accuracy regression.

SCHEMA HISTORY (abbreviated — full history in ml/src/preprocessing/_backup_pre_seam_swap_2026-06-12/graph_schema.py)
v1–v8: see backup
v9   : NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, NUM_EDGE_TYPES=12
       Added: in_unchecked_block[11], CFG_NODE_ARITH(13), EXTERNAL_CALL(11)
       `now` keyword + library wrappers added to uses_block_globals[2]
"""

from __future__ import annotations

import sys
from enum import IntEnum

# ─────────────────────────────────────────────────────────────────────────────
# Slither version assertion — hard failure at import, not a warning.
# An old Slither silently produces wrong in_unchecked features.
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
except _importlib_metadata.PackageNotFoundError:
    pass  # slither not installed in this environment (e.g., inference-only deploy)


# ─────────────────────────────────────────────────────────────────────────────
# Schema version
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SCHEMA_VERSION: str = "v9"

# ─────────────────────────────────────────────────────────────────────────────
# Structural constants
# ─────────────────────────────────────────────────────────────────────────────

NODE_FEATURE_DIM: int = 12
NUM_NODE_TYPES: int = 14
NUM_EDGE_TYPES: int = 12

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
    # CFG subtypes (v2 — ids 8–12)
    "CFG_NODE_CALL":   8,
    "CFG_NODE_WRITE":  9,
    "CFG_NODE_READ":   10,
    "CFG_NODE_CHECK":  11,
    "CFG_NODE_OTHER":  12,
    # v9 addition (Fix #4) — arithmetic ops for IntegerUO signal
    "CFG_NODE_ARITH":  13,
}


class NodeType(IntEnum):
    """Typed aliases for NODE_TYPES integer IDs."""
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

# ─────────────────────────────────────────────────────────────────────────────
# Visibility ordinal encoding
# ─────────────────────────────────────────────────────────────────────────────

VISIBILITY_MAP: dict[str, float] = {
    "public":   0.0,
    "external": 0.0,
    "internal": 0.5,
    "private":  1.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Edge type vocabulary
# ─────────────────────────────────────────────────────────────────────────────

EDGE_TYPES: dict[str, int] = {
    "CALLS":             0,
    "READS":             1,
    "WRITES":            2,
    "EMITS":             3,
    "INHERITS":          4,
    "CONTAINS":          5,
    "CONTROL_FLOW":      6,
    "REVERSE_CONTAINS":  7,   # runtime-only, NEVER on disk
    "CALL_ENTRY":        8,
    "RETURN_TO":         9,
    "DEF_USE":           10,
    "EXTERNAL_CALL":     11,  # v9 addition (Fix #3)
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature name registry
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: tuple[str, ...] = (
    "type_id",              # [0]
    "visibility",           # [1]
    "uses_block_globals",   # [2]
    "view",                 # [3]
    "payable",              # [4]
    "complexity",           # [5]
    "loc",                  # [6]
    "return_ignored",       # [7]
    "call_target_typed",    # [8]
    "has_loop",             # [9]
    "external_call_count",  # [10]
    "in_unchecked_block",   # [11]
)

# ─────────────────────────────────────────────────────────────────────────────
# Class vocabulary (LOCKED — changing order invalidates checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    "CallToUnknown",              # 0
    "DenialOfService",            # 1
    "ExternalBug",                # 2
    "GasException",               # 3
    "IntegerUO",                  # 4
    "MishandledException",        # 5
    "Reentrancy",                 # 6
    "Timestamp",                  # 7
    "TransactionOrderDependence", # 8
    "UnusedReturn",               # 9
]
# NOTE: This is the LABELING order (matches trainer.py:105, the v9 checkpoint's
# class_names field, the v2 export's labels.parquet columns, and
# sentinel_data.labeling.schema.class_names()). Per ADR-0009 (Phase D, 2026-06-12),
# this file is the canonical source of truth for the 10-class vocabulary. The
# historical "representation order" (with NonVulnerable at index 9) was the
# pre-Run-7 ordering and is no longer used in production. The v9 best checkpoint
# (GCB-P1-Run9-v11-20260606_best.pt) was trained with this order; future
# refactors that import CLASS_NAMES from here will get the correct order.
NUM_CLASSES: int = len(CLASS_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# Derived constants
# ─────────────────────────────────────────────────────────────────────────────

_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 13.0 for v9

# ─────────────────────────────────────────────────────────────────────────────
# Invariant assertions — caught at import time
# ─────────────────────────────────────────────────────────────────────────────

assert len(FEATURE_NAMES) == NODE_FEATURE_DIM, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but NODE_FEATURE_DIM={NODE_FEATURE_DIM}."
)
assert len(EDGE_TYPES) == NUM_EDGE_TYPES, (
    f"EDGE_TYPES has {len(EDGE_TYPES)} entries but NUM_EDGE_TYPES={NUM_EDGE_TYPES}."
)
assert len(NODE_TYPES) == 14, (
    f"NODE_TYPES has {len(NODE_TYPES)} entries but expected 14 (ids 0-13, v9 schema)."
)
assert max(NODE_TYPES.values()) == 13, (
    "max(NODE_TYPES.values()) must equal 13 (v9 schema). "
    "Update _MAX_TYPE_ID normalization in graph_extractor.py and sentinel_model.py."
)


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "NODE_FEATURE_DIM",
    "NUM_NODE_TYPES",
    "NUM_EDGE_TYPES",
    "VISIBILITY_MAP",
    "NODE_TYPES",
    "EDGE_TYPES",
    "FEATURE_NAMES",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "_MAX_TYPE_ID",
    "NodeType",
    "STRUCTURAL_PREFIX_TYPES",
]
