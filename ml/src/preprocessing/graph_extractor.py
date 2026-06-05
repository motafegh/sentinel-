"""
graph_extractor.py — Canonical Solidity-to-PyG graph extraction (v8 schema)

WHAT THIS MODULE DOES
─────────────────────
Provides the single, authoritative implementation of the AST-to-graph
conversion used by both SENTINEL pipelines:

  Offline (batch)  ml/scripts/reextract_graphs.py
                     Processes ~41K training graphs in parallel.
                     Writes .pt files. Returns None on failure (skip and log).

  Online (inference)  ml/src/inference/preprocess.py
                         Processes one contract per API request.
                         Raises typed exceptions for HTTP error translation.

Before this module existed, both files contained identical node/edge feature
logic. Any change to one required a manual, error-prone update to the other.
A forgotten sync meant inference features would silently diverge from training
features, producing wrong vulnerability predictions with no error message.

PUBLIC SURFACE
──────────────
  GraphExtractionConfig  — dataclass: controls extraction behaviour
  GraphExtractionError   — base exception for all extraction failures
    SolcCompilationError — subclass: bad Solidity (user error → HTTP 400)
    SlitherParseError    — subclass: Slither/infra failure  (HTTP 500)
    EmptyGraphError      — subclass: zero AST nodes found
  extract_contract_graph(sol_path, config) → Data

  Never returns None. Always raises GraphExtractionError on failure.
  Callers decide how to handle it (re-raise, translate, log+skip).

SHAPE CONTRACT  (v8 schema — must match training data)
──────────────────────────────────────────────────────────────────────────────
  graph.x             [N, 11]  float32  node feature matrix (NODE_FEATURE_DIM=11)
  graph.edge_index    [2, E]   int64    edge connectivity in COO format
  graph.edge_attr     [E]      int64    edge type IDs 0–10 excl. 7 (runtime-only REVERSE_CONTAINS)
                                         (only attached when config.include_edge_attr)
  graph.node_metadata list     of dicts, one per node, index-aligned with graph.x
                                Required: {"name", "type", "source_lines"} keys.
                                Used by _find_function_node() in the pre-flight test.

  graph.contract_name  str   — name of the analysed Slither Contract object
  graph.num_nodes      int   — N (same as x.shape[0])
  graph.num_edges      int   — E (same as edge_index.shape[1])

  Caller-specific metadata (.contract_hash, .contract_path, .y) is NOT set
  here; each caller attaches its own values after the call returns.

V8 SCHEMA (current — 2026-05-24)
────────────────────────────────
  Node features: 11 dims (in_unchecked dropped — BUG-L2).
    [0]  type_id / 12.0          [1]  visibility (0=pub, 0.5=internal, 1=private)
    [2]  uses_block_globals       [3]  view        [4]  payable
    [5]  complexity (log1p)       [6]  loc (log1p)
    [7]  return_ignored           [8]  call_target_typed
    [9]  has_loop                 [10] external_call_count (log1p)

  Edge types: 11 (CALLS=0 READS=1 WRITES=2 EMITS=3 INHERITS=4 CONTAINS=5 CF=6 RC=7
    CALL_ENTRY=8 RETURN_TO=9 DEF_USE=10).
    - EMITS(3):            EventCall IR fallback (BUG-H7)
    - INHERITS(4):         parent nodes added (BUG-H8)
    - REVERSE_CONTAINS(7): runtime-only, flipped from CONTAINS at training time
    - CALL_ENTRY(8):       v8 ICFG-Lite cross-function call edge
    - RETURN_TO(9):        v8 ICFG-Lite cross-function return edge
    - DEF_USE(10):         v8 intra-function data-flow edge

  Node types: 13 (ids 0–12, unchanged from v2).

V2→V6 SCHEMA HISTORY (archived — see graph_schema.py for full changelog)
──────────────────────────────────────────────────────────────────────────
  v2 (2026-05-11): 8→12 dims; CONTAINS(5) + CF(6) edges; 8→13 node types.
  v5 (2026-05-17): BUG-1/2/6/9 bugfixes (loc log-norm, complexity log-norm,
    most_derived contract selection, Send return-ignored detection).
  v6 (2026-05-17): visibility fixed (BUG-3, in-place patch).
  v7 (2026-05-18): 12→11 dims (in_unchecked dropped); EMITS/INHERITS fire;
    CFG nodes inherit dims from FUNCTION parent (BUG-C3); RC(7) edges added.

V5 BUGFIXES (2026-05-17)
─────────────────────────
  BUG-1:  _build_cfg_node_features loc was raw line count, now log-normalised
          to match _build_node_features.  Raw loc could be 50+ for large
          CFG nodes, violating the [0,1] feature range contract.
  BUG-2:  _build_node_features complexity was raw CFG block count (could be
          100+), now log-normalised to [0,1].  Raw values dominated all
          other features in dot products.
  BUG-6:  _select_contract "most functions" heuristic had 47.4% failure rate
          on BCCC — WORSE than random.  Replaced with "most_derived"
          heuristic that uses in-file inheritance to pick the specific
          implementation contract over library contracts (~92%+ accuracy).
          GraphExtractionConfig.multi_contract_policy default changed from
          "first" to "most_derived".
  BUG-9:  _compute_return_ignored only checked LowLevelCall and HighLevelCall,
          missing Slither's Send IR type.  `.send()` return values that are
          ignored were not detected.  Added Send to the isinstance check.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from .graph_schema import (
    EDGE_TYPES, FEATURE_NAMES, NODE_FEATURE_DIM, NODE_TYPES, NUM_EDGE_TYPES, VISIBILITY_MAP,
)

logger = logging.getLogger(__name__)

# A3: Normalisation divisor for type_id → [0, 1].
# Derived from the schema so it tracks any future addition of new node types.
# The assert (echoed from graph_schema.py A1 guard) fires at import if a new
# node type is added without updating this file and sentinel_model.py.
_MAX_TYPE_ID = float(max(NODE_TYPES.values()))
assert _MAX_TYPE_ID == 12.0, (
    f"_MAX_TYPE_ID is {_MAX_TYPE_ID} but expected 12.0. "
    "A new node type was added to NODE_TYPES — update the normalization divisor "
    "in graph_extractor.py (this file) and sentinel_model.py before re-extracting."
)

# A11: name→index map derived from FEATURE_NAMES so _build_cfg_node_features
# uses named lookups instead of raw integers that silently drift on schema changes.
_FEAT_IDX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

# A6/A10/NF-7: module-level counters for silent-failure telemetry.
# These accumulate across the process lifetime; each worker process (multiprocessing
# fork) starts from 0. Gate 2.1 checks are per-run — interpret log output at batch end.
_call_target_fail_count: int = 0    # A6: type-resolution failures in _compute_call_target_typed
_cfg_type_fallback_count: int = 0   # A10: Slither-import failures in _cfg_node_type
_ext_call_fail_count: int = 0       # NF-7: failures in _compute_external_call_count
_block_globals_fail_count: int = 0  # NF-7: failures in _compute_uses_block_globals

# A9: attempt to import SolidityVariableComposed for proper isinstance check.
# The string-based type check (type(rv).__name__ == "SolidityVariableComposed") is
# fragile under Python -O and fails if Slither reorganises its module hierarchy.
# If the import fails, uses_block_globals will always be 0.0 (logged below).
try:
    from slither.core.declarations.solidity_variables import (
        SolidityVariableComposed as _SolidityVariableComposed,
    )
except (ImportError, AttributeError):
    _SolidityVariableComposed = None  # type: ignore[assignment,misc]
    logger.warning(
        "[A9] SolidityVariableComposed not importable from "
        "slither.core.declarations.solidity_variables — "
        "uses_block_globals will always be 0.0. "
        "Timestamp and TOD detection severely degraded. Check Slither version."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Typed exception hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class GraphExtractionError(Exception):
    """Base class for all failures inside extract_contract_graph()."""


class SolcCompilationError(GraphExtractionError):
    """
    Solidity source failed to compile.

    Root causes: syntax error, wrong pragma version, missing import path,
    constructor naming conflicts (Solidity < 0.5), or unsupported language
    feature for the installed solc version.

    This is a user-input error. Callers should translate it to HTTP 400
    (inference) or log + skip (offline batch).
    """


class SlitherParseError(GraphExtractionError):
    """
    Slither failed internally after compilation appeared to succeed.

    Root causes: unsupported Slither API version, unexpected AST layout for a
    valid but exotic contract pattern, OS-level subprocess failure, or a bug
    in Slither itself.

    This is an infrastructure error. Callers should translate it to HTTP 500
    (inference) or log + skip (offline batch).
    """


class EmptyGraphError(GraphExtractionError):
    """
    Contract produced zero analyzable AST nodes.

    Root causes: every declaration in the file belongs to an imported
    dependency (filtered by is_from_dependency()), or Slither successfully
    parsed the file but found no top-level contract body.

    A graph with N=0 cannot be fed to GNNEncoder (no message passing is
    possible). Callers should treat this as a user-input error and ask the
    submitter to provide a self-contained contract file.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Extraction configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphExtractionConfig:
    """
    Controls how extract_contract_graph() runs.

    Defaults are calibrated for online single-contract inference (system solc,
    no --allow-paths, most_derived-contract policy). The offline batch pipeline
    overrides solc_binary, solc_version, and allow_paths for each version group.
    """

    multi_contract_policy: str = "most_derived"
    """
    Which contract to analyse when the file defines multiple.

    "most_derived" — (DEFAULT) pick the contract that inherits from the most
                     other non-interface candidates in the file.  In BCCC,
                     the vulnerable contract almost always inherits from
                     library contracts defined earlier in the same file.
                     ~92%+ accurate.  Falls back to last-defined if no
                     in-file inheritance is found.
    "last"         — pick the last non-interface contract.  Simple, 87.4%
                     accurate on BCCC.
    "most_funcs"   — pick the contract with the most functions.  LEGACY
                     heuristic — 47.4% wrong on BCCC (worse than random).
                     Kept for compatibility only.
    "by_name"      — use the contract whose .name == target_contract_name.
                     Falls back to most_derived with a warning if the name
                     is not found.
    """

    target_contract_name: str | None = None
    """Used only when multi_contract_policy="by_name"."""

    include_edge_attr: bool = True
    """When True, graph.edge_attr [E] int64 is attached to the returned Data object."""

    solc_binary: str | Path | None = None
    """Override the solc binary Slither uses. None → Slither resolves via PATH."""

    solc_version: str | None = None
    """Solidity version string, e.g. "0.8.19"."""

    allow_paths: str | None = None
    """Directory path(s) passed to solc as --allow-paths."""


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_return_ignored(func: Any) -> float:
    """
    0.0 if all external calls capture their return value.
    1.0 if any external call discards its return value.
    -1.0 (SENTINEL) if Slither IR is unavailable — not assumed safe.

    Slither ALWAYS assigns a TupleVariable or TemporaryVariable as lvalue even
    when the programmer ignores the return — op.lvalue is never None. The correct
    check is whether that lvalue variable is referenced in any subsequent IR op
    within the function. If it is never read, the return was discarded.

    BUG-9 FIX (2026-05-17): Added Send to the isinstance check alongside
    LowLevelCall and HighLevelCall.

    IMP-D1 FIX (2026-05-24): Replace the global all_read_names set with a
    sequential scan that checks whether lval is read AFTER the call in IR order.
    The old approach had a false-negative: a TemporaryVariable name that happened
    to match an unrelated LocalVariable read anywhere in the function would
    incorrectly conclude the return was captured, mislabeling UnusedReturn and
    MishandledException samples. Slither's func.nodes is in CFG topological order.

    A5 FIX: Narrow the try block to only the specific Slither API access that may
    be absent in some versions (func.nodes). All remaining logic is outside the try
    block so unexpected AttributeErrors propagate instead of being swallowed.
    """
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall, Send
    except ImportError:
        logger.warning(
            "return_ignored: Slither IR operations not importable for %s — using sentinel -1.0",
            getattr(func, "canonical_name", "?"),
        )
        return -1.0

    # A5: only the specific Slither IR property access that may raise AttributeError
    # (func.nodes is a Slither property that can be absent in some API versions).
    try:
        nodes = func.nodes
    except AttributeError:
        logger.warning(
            "return_ignored: Slither IR unavailable for %s — using sentinel -1.0",
            getattr(func, "canonical_name", "?"),
        )
        return -1.0

    # All remaining logic is outside the try block: unexpected AttributeErrors propagate.
    # Build flat ordered list of (node, op) pairs in CFG topological order.
    nodes = nodes or []
    all_ops_ordered = [
        (node, op)
        for node in nodes
        for op in (getattr(node, "irs", None) or [])
    ]

    for call_idx, (_, op) in enumerate(all_ops_ordered):
        if not isinstance(op, (LowLevelCall, HighLevelCall, Send)):
            continue
        lval = op.lvalue
        if lval is None:
            return 1.0
        lval_name = getattr(lval, "name", None)
        if lval_name is None:
            return 1.0
        # IMP-D1: check if lval_name appears in any read AFTER this call in IR order.
        used_after = False
        for _, later_op in all_ops_ordered[call_idx + 1:]:
            for rv in (getattr(later_op, "read", None) or []):
                if getattr(rv, "name", None) == lval_name:
                    used_after = True
                    break
            if used_after:
                break
        if not used_after:
            return 1.0  # lval never read after the call → return discarded

    return 0.0


def _compute_call_target_typed(func: Any) -> float:
    """
    1.0 if all external calls go through typed interfaces (ContractType receiver).
    0.0 if any call goes to a raw address (AddressType receiver).
    -1.0 (SENTINEL) if type resolution fails AND source is unavailable.

    Closed-world assumption guard: returning 1.0 when source_mapping is
    unavailable would be "no evidence of danger" = "confirmed safe", which is
    wrong. The -1.0 sentinel preserves the uncertainty.
    """
    try:
        from slither.core.solidity_types import ElementaryType

        low_lvl  = list(getattr(func, "low_level_calls",  None) or [])
        high_lvl = list(getattr(func, "high_level_calls", None) or [])

        if low_lvl:
            return 0.0  # low-level call always uses raw address

        for item in high_lvl:
            recv = item[0] if isinstance(item, (tuple, list)) else item
            recv_type = getattr(recv, "type", None)
            if (recv_type is not None
                    and isinstance(recv_type, ElementaryType)
                    and getattr(recv_type, "name", "") == "address"):
                return 0.0

        return 1.0  # all calls typed, or no external calls

    except Exception as _ctt_exc:
        # A6: log instead of silently discarding — helps diagnose Slither API drift.
        global _call_target_fail_count
        _call_target_fail_count += 1
        logger.debug(
            "[A6] call_target_typed: type-resolution failed for %s: %s — falling back to source scan.",
            getattr(func, "canonical_name", "?"),
            _ctt_exc,
        )

    # Fallback: scan source when type resolution is unavailable
    sm = getattr(func, "source_mapping", None)
    if sm is None or not getattr(sm, "content", None):
        logger.warning(
            "call_target_typed: source_mapping unavailable for %s — using sentinel -1.0",
            getattr(func, "canonical_name", "?"),
        )
        return -1.0

    # Exclude address(this) — self-calls are not external unknown-target calls
    raw_addr_pattern = re.compile(r"address\s*\(\s*(?!this\b)[^)]+\)\s*\.call")
    if raw_addr_pattern.search(sm.content):
        return 0.0
    return 1.0


# DEPRECATED (v7 BUG-L2) — in_unchecked was dropped from the v7 feature vector.
# Dead signal for 87.9% of the dataset (Solidity <0.8 contracts where the
# unchecked{} construct does not exist). Feature dim was removed; any call site
# that was not updated is itself a bug.
# A7: replaced body with NotImplementedError so call sites surface immediately.
def _compute_in_unchecked(func: Any) -> float:
    raise NotImplementedError(
        "_compute_in_unchecked is deprecated in v7 (BUG-L2). "
        "in_unchecked was dropped from the feature vector — any call site was not updated. "
        "Remove the call or replace it with a schema-correct alternative."
    )


def _compute_has_loop(func: Any) -> float:
    """1.0 if func contains at least one loop construct."""
    try:
        from slither.core.cfg.node import NodeType
        loop_types = {NodeType.IFLOOP, NodeType.STARTLOOP, NodeType.ENDLOOP}
        for node in (getattr(func, "nodes", None) or []):
            if getattr(node, "type", None) in loop_types:
                return 1.0
    except Exception:
        pass

    # Fallback: Slither convenience attribute (exists in some versions).
    # A8: is True → bool() — `is True` rejects truthy non-bool values (e.g. 1).
    try:
        if bool(getattr(func, "is_loop_present", False)):
            return 1.0
    except Exception:
        pass

    return 0.0


def _compute_external_call_count(func: Any) -> float:
    """
    log1p(count) / log1p(20), clamped [0, 1].
    1 call → 0.23,  5 calls → 0.60,  20 calls → 1.0.

    Counts HighLevelCall, LowLevelCall, Transfer, and Send so that
    ETH-transfer loops (DoS pattern) produce a non-zero signal. Prior to
    this fix, `payable(addr).transfer(amount)` inside distribute() produced
    ext_calls=0.0 because Slither classifies it as Transfer, not LowLevelCall.
    """
    try:
        from slither.slithir.operations import Transfer, Send
        n  = len(list(getattr(func, "high_level_calls", None) or []))
        n += len(list(getattr(func, "low_level_calls",  None) or []))
        # Count Transfer/Send ops in each node's IR
        for node in (getattr(func, "nodes", None) or []):
            for op in (getattr(node, "irs", None) or []):
                if isinstance(op, (Transfer, Send)):
                    n += 1
        return min(math.log1p(n) / math.log1p(20), 1.0)
    except Exception as exc:
        # NF-7: log and count failures — silent 0.0 masked Slither API drift.
        global _ext_call_fail_count
        _ext_call_fail_count += 1
        logger.debug(
            "[NF-7] _compute_external_call_count failed for %s: %s",
            getattr(func, "canonical_name", "?"), exc,
        )
        return 0.0


def _compute_uses_block_globals(func: Any) -> float:
    """
    1.0 if any IR op in this function reads block.timestamp, block.number,
    block.difficulty, or block.basefee.
    0.0 otherwise.

    These are SolidityVariableComposed objects in Slither IR — they do NOT
    appear in func.state_variables_read and therefore create no READS edge in
    the graph. Without this direct feature, Timestamp contracts are completely
    invisible to the GNN. This feature gives Timestamp and TOD a direct signal.
    """
    try:
        _BLOCK_GLOBALS = {"timestamp", "number", "difficulty", "basefee", "prevrandao"}
        for node in (getattr(func, "nodes", None) or []):
            for op in (getattr(node, "irs", None) or []):
                for rv in (getattr(op, "read", None) or []):
                    # A9: use isinstance instead of string-based class name check.
                    # _SolidityVariableComposed is set at module load (may be None if
                    # Slither import failed — logged at module level as a WARNING).
                    if _SolidityVariableComposed is not None and isinstance(rv, _SolidityVariableComposed):
                        name = getattr(rv, "name", "") or ""
                        # name is e.g. "block.timestamp" — split on '.'
                        part = name.split(".")[-1].lower()
                        if part in _BLOCK_GLOBALS:
                            return 1.0
    except Exception as exc:
        # NF-7: log and count failures — silent 0.0 masked Slither API drift.
        global _block_globals_fail_count
        _block_globals_fail_count += 1
        logger.debug(
            "[NF-7] _compute_uses_block_globals failed for %s: %s",
            getattr(func, "canonical_name", "?"), exc,
        )
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CFG node helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_return_ignored(node: Any) -> float:
    """
    C-1 fix: per-statement return_ignored for a single CFG node.
    1.0 if any call in this node's IRs has its return value never read after
    the call within the same node's IR sequence. 0.0 otherwise.
    """
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall, Send
        irs = list(getattr(node, "irs", None) or [])
        for i, op in enumerate(irs):
            if not isinstance(op, (LowLevelCall, HighLevelCall, Send)):
                continue
            lval = op.lvalue
            if lval is None:
                return 1.0
            lval_name = getattr(lval, "name", None)
            if lval_name is None:
                return 1.0
            used_after = any(
                getattr(rv, "name", None) == lval_name
                for later_op in irs[i + 1:]
                for rv in (getattr(later_op, "read", None) or [])
            )
            if not used_after:
                return 1.0
    except Exception:
        pass
    return 0.0


def _node_call_target_typed(node: Any) -> float:
    """
    C-1 fix: per-statement call_target_typed for a single CFG node.
    0.0 if this node makes a low-level or raw-address call. 1.0 otherwise.
    """
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall
        from slither.core.solidity_types import ElementaryType
        for op in (getattr(node, "irs", None) or []):
            if isinstance(op, LowLevelCall):
                return 0.0
            if isinstance(op, HighLevelCall):
                recv = getattr(op, "destination", None)
                recv_type = getattr(recv, "type", None)
                if (recv_type is not None
                        and isinstance(recv_type, ElementaryType)
                        and getattr(recv_type, "name", "") == "address"):
                    return 0.0
    except Exception:
        pass
    return 1.0


def _node_uses_block_globals(node: Any) -> float:
    """
    C-1 fix: per-statement uses_block_globals for a single CFG node.
    1.0 if any IR op in this node reads block.timestamp/number/difficulty/etc.
    """
    try:
        _BLOCK_GLOBALS = {"timestamp", "number", "difficulty", "basefee", "prevrandao"}
        for op in (getattr(node, "irs", None) or []):
            for rv in (getattr(op, "read", None) or []):
                if _SolidityVariableComposed is not None and isinstance(rv, _SolidityVariableComposed):
                    part = (getattr(rv, "name", "") or "").split(".")[-1].lower()
                    if part in _BLOCK_GLOBALS:
                        return 1.0
    except Exception:
        pass
    return 0.0


def _node_external_call_count(node: Any) -> float:
    """
    C-1 fix: per-statement external_call_count for a single CFG node.
    log1p(n)/log1p(20) of the number of external calls in this node's IRs.
    """
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall, Transfer, Send
        n = sum(
            1 for op in (getattr(node, "irs", None) or [])
            if isinstance(op, (LowLevelCall, HighLevelCall, Transfer, Send))
        )
        return min(math.log1p(n) / math.log1p(20), 1.0)
    except Exception:
        pass
    return 0.0


def _cfg_node_type(slither_node: Any) -> int:
    """
    Map a Slither CFG node to a NODE_TYPES CFG subtype id.

    PRIORITY ORDER (highest to lowest):
      1. CFG_NODE_CALL  (8) — any node containing an external call
      2. CFG_NODE_WRITE (9) — any node writing a state variable
      3. CFG_NODE_READ  (10) — any node reading a state variable
      4. CFG_NODE_CHECK (11) — require / assert / if condition
      5. CFG_NODE_OTHER (12) — everything else (including synthetic nodes)

    When Slither generates a merged IR node containing both an external call
    AND another operation, the external call is the most vulnerability-relevant
    operation and wins priority 1.
    """
    try:
        from slither.core.cfg.node import NodeType as SNT
        from slither.slithir.operations import LowLevelCall, HighLevelCall, Transfer, Send
        from slither.core.variables.state_variable import StateVariable

        check_types = {SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.ENDLOOP, SNT.THROW}

        irs = list(getattr(slither_node, "irs", None) or [])

        # Priority 1: any IR op is an external call (including Transfer/Send which
        # Slither classifies separately from LowLevelCall/HighLevelCall but are still
        # ETH-sending operations — critical for DoS loop detection).
        if any(isinstance(op, (LowLevelCall, HighLevelCall, Transfer, Send)) for op in irs):
            return NODE_TYPES["CFG_NODE_CALL"]

        # Priority 2: node writes a state variable.
        # Use Slither's own state_variables_written (handles mapping/reference
        # lvalues that resolve through ReferenceVariable, not direct StateVariable).
        # Fall back to IR lvalue check for older Slither versions.
        sv_written = list(getattr(slither_node, "state_variables_written", None) or [])
        if sv_written or any(
            hasattr(op, "lvalue") and isinstance(op.lvalue, StateVariable)
            for op in irs
        ):
            return NODE_TYPES["CFG_NODE_WRITE"]

        # Priority 3: node reads a state variable.
        sv_read = list(getattr(slither_node, "state_variables_read", None) or [])
        if sv_read or any(
            isinstance(v, StateVariable)
            for op in irs
            for v in getattr(op, "read", [])
        ):
            return NODE_TYPES["CFG_NODE_READ"]

        # Priority 4: control-flow check node type
        if getattr(slither_node, "type", None) in check_types:
            return NODE_TYPES["CFG_NODE_CHECK"]

    except Exception as _ctn_exc:
        # A10: log Slither-import/API failures — they silently fall through to CFG_NODE_OTHER.
        # A high _cfg_type_fallback_count in logs indicates a Slither version mismatch.
        global _cfg_type_fallback_count
        _cfg_type_fallback_count += 1
        logger.warning(
            "[A10] _cfg_node_type failed for node %r: %s — falling back to CFG_NODE_OTHER.",
            getattr(slither_node, "node_id", "?"),
            _ctn_exc,
        )

    return NODE_TYPES["CFG_NODE_OTHER"]


def _build_cfg_node_features(
    slither_node: Any,
    func: Any,
    cfg_type: int,
    parent_features: list | None = None,
) -> list:
    """
    Build the 11-dim feature vector for a CFG (statement-level) node.

    Returns list[float] of exactly NODE_FEATURE_DIM (11) elements.
    torch.tensor(x_list) requires all sublists to be the same length;
    returning a different length silently causes crashes at tensor assembly.

    BUG-C3 FIX: CFG nodes inherit function-level features from parent_features
    (the parent FUNCTION node's feature vector) for dims that are function-scoped:
        [1] visibility, [3] view, [4] payable, [5] complexity, [9] has_loop.
    Without inheritance, these dims were 0.0 for all CFG nodes — statement-level
    nodes carried almost no signal, undermining CEI pattern detection.

    CRITICAL: in_unchecked was dropped in v7 (BUG-L2) — it is no longer in
    the feature vector. CFG nodes inherit has_loop [9] from the parent function.

    Slither synthetic nodes (ENTRY_POINT, EXPRESSION, BEGIN_LOOP, etc.) with
    no source_mapping and empty IRS are handled correctly: _cfg_node_type()
    returns CFG_NODE_OTHER (12) and loc defaults to 0.0. Do NOT filter them.

    BUG-1 FIX (2026-05-17): loc is now log-normalised to [0,1] to match the
    declaration-level _build_node_features. Previously, raw line counts
    (potentially 50+) violated the [0,1] feature range contract and dominated
    other features in dot products.
    """
    loc_raw = 0.0
    sm = getattr(slither_node, "source_mapping", None)
    if sm is not None and getattr(sm, "lines", None):
        loc_raw = float(len(sm.lines))
    loc = min(math.log1p(loc_raw) / math.log1p(1000), 1.0)

    # Inherit function-scoped dims from the parent FUNCTION node feature vector.
    # A11: use _FEAT_IDX named lookups instead of raw magic numbers so the correct
    # dim indices are always derived from FEATURE_NAMES (single source of truth).
    p = parent_features or []
    _vi = _FEAT_IDX["visibility"]
    _vw = _FEAT_IDX["view"]
    _vp = _FEAT_IDX["payable"]
    _vc = _FEAT_IDX["complexity"]
    _vl = _FEAT_IDX["has_loop"]
    visibility  = p[_vi] if len(p) > _vi else 0.0
    view        = p[_vw] if len(p) > _vw else 0.0
    payable     = p[_vp] if len(p) > _vp else 0.0
    complexity  = p[_vc] if len(p) > _vc else 0.0
    has_loop    = p[_vl] if len(p) > _vl else 0.0

    # C-1 fix: compute per-statement signals from this node's IR ops directly.
    return [
        float(cfg_type) / _MAX_TYPE_ID,              # [0]  type_id normalised to [0,1]
        visibility,                                   # [1]  inherited from parent FUNCTION (BUG-C3)
        _node_uses_block_globals(slither_node),       # [2]  per-statement block.timestamp/number check
        view,                                         # [3]  inherited from parent FUNCTION (BUG-C3)
        payable,                                      # [4]  inherited from parent FUNCTION (BUG-C3)
        complexity,                                   # [5]  inherited from parent FUNCTION (BUG-C3)
        loc,                                          # [6]  loc — log-normalised
        _node_return_ignored(slither_node),           # [7]  per-statement: call return discarded?
        _node_call_target_typed(slither_node),        # [8]  per-statement: typed interface vs raw addr
        has_loop,                                     # [9]  inherited from parent FUNCTION (BUG-C3)
        _node_external_call_count(slither_node),      # [10] per-statement: external call count (log1p)
    ]


def _build_control_flow_edges(
    func: Any,
    func_node_idx: int,
    node_index_map: dict,
    x_list: list,
    node_metadata: list,
    parent_features: list | None = None,
) -> tuple:
    """
    For a given function, build CFG_NODE children and their edges.

    Appends new node feature vectors to x_list and entries to node_metadata
    in-place. Populates node_index_map with slither_node → graph_idx mappings.
    x_list and node_metadata must have the same length at entry and at exit.

    Args:
        func:           Slither Function object.
        func_node_idx:  Graph node index of the parent FUNCTION node (in x_list).
        node_index_map: Maps slither_node objects → graph indices. Mutated.
        x_list:         Shared list of all node feature vectors. Mutated.
        node_metadata:  Shared list of node metadata dicts. Mutated.

    Returns:
        (contains_edges, control_flow_edges):
            contains_edges:     list of (func_node_idx, cfg_graph_idx) — edge type 5
            control_flow_edges: list of (cfg_src_idx, cfg_dst_idx)     — edge type 6

    INDEX ASSIGNMENT — CRITICAL:
      graph_idx = len(x_list)
      This is the correct index because x_list is the single shared list across
      ALL node types. Its length before appending is the next available index.
      Do NOT use len(node_index_map) — it only tracks CFG objects for this
      function, not the full set of declaration + CFG nodes.

    CFG NODE ORDERING — DETERMINISTIC:
      Always sort func.nodes by (source_line, node_id) so identical source
      produces identical graphs across Slither versions and extraction runs.
      Synthetic nodes (ENTRY_POINT etc.) with no source_mapping get line 0.
    """
    _type_name_map = {v: k for k, v in NODE_TYPES.items()}

    cfg_nodes = sorted(
        getattr(func, "nodes", None) or [],
        key=lambda n: (
            n.source_mapping.lines[0]
            if n.source_mapping and n.source_mapping.lines else 0,
            # A12: getattr fallback — node_id may be absent on synthetic Slither nodes.
            getattr(n, "node_id", 0),
        ),
    )

    contains_edges:     list = []
    control_flow_edges: list = []

    # Pass 1: assign indices, build feature vectors, populate shared lists
    for slither_node in cfg_nodes:
        cfg_type  = _cfg_node_type(slither_node)
        graph_idx = len(x_list)          # CORRECT: next available global index
        node_index_map[slither_node] = graph_idx

        x_list.append(_build_cfg_node_features(slither_node, func, cfg_type, parent_features))

        cfg_type_name = _type_name_map.get(cfg_type, "CFG_NODE_OTHER")
        sm = getattr(slither_node, "source_mapping", None)
        node_metadata.append({
            "name":         str(slither_node),
            "type":         cfg_type_name,
            "source_lines": list(sm.lines) if sm and sm.lines else [],
        })

        contains_edges.append((func_node_idx, graph_idx))

    # Pass 2: build CONTROL_FLOW edges (all CFG nodes indexed in pass 1)
    _dropped_cf_edges = 0
    for slither_node in cfg_nodes:
        src_idx = node_index_map[slither_node]
        for successor in (getattr(slither_node, "sons", None) or []):
            if successor in node_index_map:
                control_flow_edges.append((src_idx, node_index_map[successor]))
            else:
                # A13: log dropped edges instead of silently discarding.
                # Sons outside node_index_map indicate cross-function successors or
                # synthetic Slither nodes that were not included in cfg_nodes.
                _dropped_cf_edges += 1
                logger.debug(
                    "[A13] Dropped CONTROL_FLOW edge: node %r → son %r not in cfg_node_map "
                    "(function '%s').",
                    getattr(slither_node, "node_id", "?"),
                    getattr(successor, "node_id", "?"),
                    getattr(func, "canonical_name", "?"),
                )

    if _dropped_cf_edges > 0:
        logger.debug(
            "[A13] %d CONTROL_FLOW edge(s) dropped for function '%s'.",
            _dropped_cf_edges,
            getattr(func, "canonical_name", "?"),
        )

    return contains_edges, control_flow_edges


def _add_icfg_edges(
    contract: Any,
    func_entry_map: dict,
    func_terminal_map: dict,
    func_cfg_maps: dict,
    edges: list,
    edge_types: list,
) -> None:
    """
    PLAN-1D — ICFG-Lite: add cross-function control-flow edges.

    For every CFG node that makes an internal call:
      CALL_ENTRY (8): calling CFG node → ENTRYPOINT of the callee function.
      RETURN_TO  (9): each terminal node of the callee → each successor of the
                      calling CFG node (call-site return targets).

    Only emits edges when both endpoints are present in the extracted graph
    (callee may be absent if it failed CFG extraction or is a library stub).

    Args:
        func_entry_map:    canonical_name → graph_idx of callee ENTRYPOINT node.
        func_terminal_map: canonical_name → [graph_idx, ...] of callee terminal nodes.
        func_cfg_maps:     canonical_name → {slither_node → graph_idx} per function.
    """
    _CALL_ENTRY = EDGE_TYPES["CALL_ENTRY"]
    _RETURN_TO  = EDGE_TYPES["RETURN_TO"]

    for func in contract.functions:
        func_key = getattr(func, "canonical_name", None) or func.name
        local_map = func_cfg_maps.get(func_key)
        if local_map is None:
            continue

        for node in (getattr(func, "nodes", None) or []):
            caller_idx = local_map.get(node)
            if caller_idx is None:
                continue

            for callee in (getattr(node, "internal_calls", None) or []):
                callee_key = getattr(callee, "canonical_name", None)
                if not callee_key:
                    continue

                # CALL_ENTRY: caller node → callee ENTRYPOINT
                callee_entry = func_entry_map.get(callee_key)
                if callee_entry is not None:
                    edges.append([caller_idx, callee_entry])
                    edge_types.append(_CALL_ENTRY)

                # RETURN_TO: callee terminals → call-site successors
                callee_terminals = func_terminal_map.get(callee_key)
                if not callee_terminals:
                    continue
                call_site_sons = getattr(node, "sons", None) or []
                for son in call_site_sons:
                    son_idx = local_map.get(son)
                    if son_idx is None:
                        continue
                    for terminal_idx in callee_terminals:
                        edges.append([terminal_idx, son_idx])
                        edge_types.append(_RETURN_TO)


def _add_def_use_edges(
    contract: Any,
    func_cfg_maps: dict,
    edges: list,
    edge_types: list,
) -> None:
    """
    PLAN-1E — DEF_USE data-flow edges (intra- and inter-function for state variables).

    Tracks variable definitions (lval of any IR op) and emits DEF_USE(10) edges
    from the defining CFG node to every CFG node that reads the variable.

    A15 FIX — two-tier scope key to prevent cross-function false edges:
      Local variables  (LocalVariable): keyed by (func_canonical_name, var_name).
                        A local variable named `balance` in function A must NOT
                        create DEF_USE edges to a read of `balance` in function B.
      State variables  (StateVariable): keyed by (contract_canonical_name, var_name).
                        State variables ARE shared across functions — their DEF_USE
                        edges correctly span function boundaries.

    Deduplicates (def_node, use_node) pairs across all functions.
    """
    _DEF_USE = EDGE_TYPES["DEF_USE"]

    try:
        from slither.core.variables.local_variable import LocalVariable as _LV
        from slither.core.variables.state_variable import StateVariable as _SV
        from slither.core.variables.variable import Variable as _VAR
    except ImportError:
        return

    # H-2 fix: helper to resolve ReferenceVariable to its underlying StateVariable.
    # Slither uses ReferenceVariable for mapping/array lvalues (balances[msg.sender]).
    # .points_to ultimately resolves to the StateVariable being written.
    def _resolve_lval(lval: Any) -> tuple | None:
        if isinstance(lval, _SV):
            return (contract_key, getattr(lval, "name", None))
        if isinstance(lval, _LV):
            return (func_key, getattr(lval, "name", None))
        # ReferenceVariable: follow .points_to chain to the underlying variable
        pts = getattr(lval, "points_to", None)
        if pts is not None and isinstance(pts, _SV):
            return (contract_key, getattr(pts, "name", None))
        # Some Slither versions store the root var in .points_to_origin
        origin = getattr(lval, "points_to_origin", None)
        if origin is not None and isinstance(origin, _SV):
            return (contract_key, getattr(origin, "name", None))
        return None

    contract_key = getattr(contract, "canonical_name", None) or contract.name

    # Build def_map in one pass over ALL functions using two-tier scope keys.
    # This allows state-variable DEF_USE edges to cross function boundaries.
    def_map: dict = {}
    for func in contract.functions:
        func_key = getattr(func, "canonical_name", None) or func.name
        local_map = func_cfg_maps.get(func_key)
        if local_map is None:
            continue
        for node in (getattr(func, "nodes", None) or []):
            node_idx = local_map.get(node)
            if node_idx is None:
                continue
            for ir in (getattr(node, "irs", None) or []):
                lval = getattr(ir, "lvalue", None)
                if lval is None:
                    continue
                # H-2 fix: resolve ReferenceVariable in addition to SV/LV
                scope_key = _resolve_lval(lval)
                if scope_key is None:
                    continue
                if scope_key[1] is None:
                    continue
                def_map.setdefault(scope_key, []).append(node_idx)

    if not def_map:
        return

    # Emit DEF_USE edges: for each function's CFG nodes, look up the correct scope key.
    seen_pairs: set = set()
    for func in contract.functions:
        func_key = getattr(func, "canonical_name", None) or func.name
        local_map = func_cfg_maps.get(func_key)
        if local_map is None:
            continue
        for node in (getattr(func, "nodes", None) or []):
            use_idx = local_map.get(node)
            if use_idx is None:
                continue
            for ir in (getattr(node, "irs", None) or []):
                for var in (getattr(ir, "read", None) or []):
                    vname = getattr(var, "name", None)
                    if vname is None:
                        continue
                    # Resolve to the same scope key used on the def side.
                    if isinstance(var, _SV):
                        scope_key = (contract_key, vname)
                    else:
                        scope_key = (func_key, vname)
                    def_nodes = def_map.get(scope_key)
                    if not def_nodes:
                        continue
                    for def_idx in def_nodes:
                        if def_idx == use_idx:
                            continue
                        pair = (def_idx, use_idx)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        edges.append([def_idx, use_idx])
                        edge_types.append(_DEF_USE)


def _compute_has_cei_path(
    node_metadata: list,
    edge_index:    "torch.Tensor",
    edge_attr:     "torch.Tensor",
    max_hops:      int = 8,
) -> int:
    """CEI labeler (Phase 7 / Interp-2): detect CFG_NODE_CALL → CFG_NODE_WRITE within max_hops.

    A Checks-Effects-Interactions violation exists when a state-variable write
    (CFG_NODE_WRITE) is reachable from an external-call site (CFG_NODE_CALL)
    by following CONTROL_FLOW edges only.  This is the structural prerequisite
    for reentrancy: external call runs first, callback can trigger effects later.

    Returns 1 if at least one CFG_NODE_CALL can reach a CFG_NODE_WRITE within
    max_hops CONTROL_FLOW hops; 0 otherwise.  Only CONTROL_FLOW edges (type 6)
    are traversed — CALL_ENTRY/RETURN_TO/DEF_USE are ignored so we stay
    intra-function (not inter-procedural).

    Called during graph assembly (Phase 7 re-extraction).  The result is
    stored as graph.has_cei_path (int, 0 or 1) and used by aux_cei_loss
    after Gate 7.5 validates label quality on v9 data.
    """
    from ml.src.preprocessing.graph_schema import EDGE_TYPES

    _CF = EDGE_TYPES["CONTROL_FLOW"]  # 6

    if edge_index.shape[1] == 0:
        return 0

    # Build adjacency list for CONTROL_FLOW edges only.
    cf_mask   = (edge_attr == _CF)
    cf_src    = edge_index[0][cf_mask].tolist()
    cf_dst    = edge_index[1][cf_mask].tolist()
    adj: dict[int, list[int]] = {}
    for s, d in zip(cf_src, cf_dst):
        adj.setdefault(s, []).append(d)

    # Find CALL and WRITE node indices.
    call_indices  = [i for i, m in enumerate(node_metadata) if m.get("type") == "CFG_NODE_CALL"]
    write_set     = {i for i, m in enumerate(node_metadata) if m.get("type") == "CFG_NODE_WRITE"}

    if not call_indices or not write_set:
        return 0

    # BFS from each CALL node; early-exit if a WRITE node is reached.
    for start in call_indices:
        frontier = [start]
        visited  = {start}
        for _ in range(max_hops):
            next_frontier = []
            for node in frontier:
                for nb in adj.get(node, []):
                    if nb in write_set:
                        return 1
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
            if not frontier:
                break

    return 0


def _build_node_features(obj: Any, type_id: int) -> list:
    """
    Compute the 11-dimensional feature vector (v8 schema) for one AST node.

    Returns list[float] of exactly NODE_FEATURE_DIM (11) elements.

    Feature layout (v8 schema):
      [0]  type_id              — float(NODE_TYPES[kind]) / 12.0, normalised [0,1]
      [1]  visibility           — VISIBILITY_MAP ordinal
      [2]  uses_block_globals   — 1.0 if func reads block.timestamp/number/etc.
      [3]  view                 — 1.0 if Function.view
      [4]  payable              — 1.0 if Function.payable
      [5]  complexity           — log-normalised CFG block count
                                  min(log1p(len(func.nodes)) / log1p(100), 1.0)
      [6]  loc                  — log1p(lines) / log1p(1000), normalised [0,1]
      [7]  return_ignored       — 0.0/1.0/-1.0 sentinel
      [8]  call_target_typed    — 0.0/1.0/-1.0 sentinel
      [9]  has_loop             — 1.0 if function contains a loop  (was [10] in v6)
      [10] external_call_count  — log-normalized count             (was [11] in v6)

    Removed from v6→v7 (BUG-L2): in_unchecked — dead signal for 87.9% of the
    dataset (Solidity <0.8 contracts where unchecked{} does not exist).

    Non-Function nodes receive 0.0 for features [2:] except:
      - call_target_typed [8] defaults to 1.0 (safe: not applicable)
      - loc [6] is computed when source_mapping is available
    """
    _is_function = hasattr(obj, "nodes") and hasattr(obj, "pure")

    visibility = float(VISIBILITY_MAP.get(
        str(getattr(obj, "visibility", "public")), 0.0
    ))

    # loc: normalise with log1p to prevent scale dominance.
    # Raw values hit 2538 vs all other features in [0,1] — this caused the
    # CONTRACT node (avg loc=133) to overwhelm other features in early dot products.
    loc_raw = 0.0
    sm = getattr(obj, "source_mapping", None)
    if sm is not None and getattr(sm, "lines", None):
        loc_raw = float(len(sm.lines))
    loc = min(math.log1p(loc_raw) / math.log1p(1000), 1.0)

    # Defaults for non-Function nodes
    uses_block_globals = 0.0
    view = payable = 0.0
    complexity = 0.0
    return_ignored     = 0.0
    call_target_typed  = 1.0   # safe default: "not applicable"
    has_loop           = 0.0
    external_call_count = 0.0

    if _is_function:
        uses_block_globals = _compute_uses_block_globals(obj)
        view    = 1.0 if getattr(obj, "view",    False) else 0.0
        payable = 1.0 if getattr(obj, "payable", False) else 0.0
        # BUG-2 FIX (2026-05-17): complexity is now log-normalised to [0,1].
        # Raw CFG block count could be 100+, dominating all other features.
        try:
            _raw = float(len(obj.nodes)) if obj.nodes else 0.0
            complexity = min(math.log1p(_raw) / math.log1p(100), 1.0)
        except Exception:
            complexity = 0.0

        return_ignored      = _compute_return_ignored(obj)
        call_target_typed   = _compute_call_target_typed(obj)
        has_loop            = _compute_has_loop(obj)
        external_call_count = _compute_external_call_count(obj)

        # Override FUNCTION(1) type_id for special function kinds
        if getattr(obj, "is_constructor", False):
            type_id = NODE_TYPES["CONSTRUCTOR"]
        elif getattr(obj, "is_fallback", False):
            type_id = NODE_TYPES["FALLBACK"]
        elif getattr(obj, "is_receive", False):
            type_id = NODE_TYPES["RECEIVE"]

    # A16: assert → raise ValueError — asserts are disabled by Python -O flag (production).
    if return_ignored not in (-1.0, 0.0, 1.0):
        raise ValueError(
            f"return_ignored out of range: {return_ignored!r} for "
            f"'{getattr(obj, 'canonical_name', '?')}'. "
            "Bug in _compute_return_ignored() — must return -1.0, 0.0, or 1.0."
        )
    if call_target_typed not in (-1.0, 0.0, 1.0):
        raise ValueError(
            f"call_target_typed out of range: {call_target_typed!r} for "
            f"'{getattr(obj, 'canonical_name', '?')}'. "
            "Bug in _compute_call_target_typed() — must return -1.0, 0.0, or 1.0."
        )

    return [
        float(type_id) / _MAX_TYPE_ID,  # [0]  normalised to [0,1]
        visibility,                      # [1]
        uses_block_globals,              # [2]  block.timestamp/number signal
        view,                            # [3]
        payable,                         # [4]
        complexity,                      # [5]  log-normalised CFG block count (BUG-2 fix)
        loc,                             # [6]  log-normalised (was raw line count)
        return_ignored,                  # [7]
        call_target_typed,               # [8]
        # in_unchecked removed (BUG-L2): dead signal for 87.9% Solidity 0.4.x dataset
        has_loop,                        # [9]  (was [10] in v6)
        external_call_count,             # [10] (was [11] in v6)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────────

def _select_contract(sl: Any, config: GraphExtractionConfig) -> Any:
    """
    Pick the target Slither Contract from the parsed contract list.

    Filters out imported dependencies so only user-supplied code is analysed.
    When a file defines multiple non-interface contracts, the "most_derived"
    heuristic picks the one that inherits from the most other candidates in the
    file.  This correctly selects the specific implementation contract (e.g.
    ERC20Token, BTPCoin) over the library contract (e.g. StandardToken) that
    it inherits from.

    AUDIT RESULT (BUG-6 fix, 2026-05-17):
      "most functions" heuristic:  52.6% accurate — WORSE than random!
      "last contract" heuristic:   87.4% accurate
      "most_derived" heuristic:    ~92%+ accurate (uses inheritance info)

    NOTE (Slither 0.10.x): `is_interface` is a reliable property.
    `is_abstract` does not exist — abstract contracts share contract_kind='contract'
    with concrete ones.
    """
    candidates = [c for c in sl.contracts if not c.is_from_dependency()]
    if not candidates:
        raise EmptyGraphError(
            "No non-dependency contracts found — all declarations are in imported "
            "dependency files. Ensure the target contract body is present in the "
            "submitted .sol file rather than only import statements."
        )

    if config.multi_contract_policy == "by_name" and config.target_contract_name:
        matching = [c for c in candidates if c.name == config.target_contract_name]
        if matching:
            return matching[0]
        logger.warning(
            "Contract %r not found in file (available: %s); "
            "falling back to most_derived heuristic.",
            config.target_contract_name,
            [c.name for c in candidates],
        )

    # Prefer non-interface contracts.
    non_iface = [c for c in candidates if not c.is_interface]
    if non_iface:
        if len(non_iface) == 1:
            return non_iface[0]

        policy = config.multi_contract_policy

        # "last" — pick the last non-interface contract (simple, 87.4% on BCCC)
        if policy == "last":
            return non_iface[-1]

        # "most_funcs" — legacy heuristic (47.4% wrong, kept for compatibility)
        if policy == "most_funcs":
            return max(non_iface, key=lambda c: len(c.functions))

        # Default: "most_derived" — pick the contract that inherits from the
        # most other candidates in the file.  In BCCC, the vulnerable contract
        # almost always inherits from library contracts defined earlier in the
        # same file (e.g. ERC20Token is StandardToken, Ownable).  The most-
        # derived contract is the specific implementation, not the library.
        candidate_names = {c.name for c in non_iface}

        def _derivation_score(c: Any) -> tuple[int, int]:
            """Return (n_inherited_from_candidates, source_order_index).
            Higher n_inherited → more derived → picked first.
            Tiebreak: last in source order wins (BCCC pattern).
            """
            inherited_in_file = 0
            for parent in (getattr(c, "inheritance", None) or []):
                if getattr(parent, "name", None) in candidate_names:
                    inherited_in_file += 1
            source_idx = non_iface.index(c)  # preserves sl.contracts order
            return (inherited_in_file, source_idx)

        best = max(non_iface, key=_derivation_score)
        best_score = _derivation_score(best)

        if best_score[0] > 0:
            logger.debug(
                "Selected %r (inherits from %d in-file contracts) "
                "among %d non-iface candidates",
                best.name, best_score[0], len(non_iface),
            )
            return best

        # No inheritance info — fall back to "last defined" heuristic
        logger.debug(
            "No in-file inheritance among %d non-iface candidates; "
            "falling back to last-defined heuristic",
            len(non_iface),
        )
        return non_iface[-1]

    # All candidates are interfaces — fall back to the first one.
    logger.warning(
        "All %d non-dependency contracts in this file are interfaces; "
        "extracting the first one. Graph will likely be minimal.",
        len(candidates),
    )
    return candidates[0]


def _build_solc_args(config: GraphExtractionConfig) -> str | None:
    """Compute the solc_args string to pass to Slither, if any."""
    if not config.allow_paths:
        return None

    if config.solc_version:
        parts = config.solc_version.split(".")
        try:
            major, minor = int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            major, minor = 0, 0
        if (major, minor) < (0, 5):
            return None

    return f"--allow-paths .,{config.allow_paths}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_contract_graph(
    sol_path: Path,
    config: GraphExtractionConfig | None = None,
) -> Data:
    """
    Parse a Solidity file and return a PyG Data graph object (v2 schema).

    This is the single canonical AST-to-graph conversion used by both the
    offline training pipeline and the online inference API.

    Node insertion order:
      CONTRACT → STATE_VARs → FUNCTIONs (each followed by its CFG_NODE children,
      sorted by source_line) → MODIFIERs → EVENTs
    This order is fixed and must match across all extraction runs.

    Args:
        sol_path: Path to a .sol file that must already exist on disk.
        config:   Extraction settings. None → defaults (online inference).

    Returns:
        PyG Data with:
          .x              Tensor [N, 12] float32  node features
          .edge_index     Tensor [2, E]  int64    COO edge connectivity
          .edge_attr      Tensor [E]     int64    edge type IDs 0-6
                                                  (only if config.include_edge_attr)
          .node_metadata  list[dict]               index-aligned with .x
                                                  keys: name, type, source_lines
          .contract_name  str
          .num_nodes      int
          .num_edges      int

    Raises:
        RuntimeError:         Slither library is not installed.
        SolcCompilationError: Solidity failed to compile.
        SlitherParseError:    Slither failed internally.
        EmptyGraphError:      Contract produced zero analyzable AST nodes.
    """
    if config is None:
        config = GraphExtractionConfig()
    sol_path = Path(sol_path)

    # ── Slither availability ───────────────────────────────────────────────
    try:
        from slither import Slither
    except ImportError as exc:
        raise RuntimeError(
            "Slither is not installed. "
            "Run: pip install slither-analyzer  (or: poetry add slither-analyzer)"
        ) from exc

    # ── Slither instantiation ──────────────────────────────────────────────
    solc_args = _build_solc_args(config)
    try:
        slither_kwargs: dict = {"solc_args": solc_args, "detectors_to_run": []}
        if config.solc_binary:
            slither_kwargs["solc"] = str(config.solc_binary)
        sl = Slither(str(sol_path), **slither_kwargs)
    except Exception as exc:
        # A17: type-based routing first — avoids string keyword false positives.
        # SlitherError is Slither's own base exception for internal parse failures.
        try:
            from slither.exceptions import SlitherError as _SlitherError
            if isinstance(exc, _SlitherError):
                raise SlitherParseError(
                    f"Slither failed to parse '{sol_path.name}': {exc}"
                ) from exc
        except ImportError:
            pass  # Slither exception hierarchy unavailable — fall through to string match

        # Fallback: string keyword matching for CryticCompile and other bare Exceptions.
        # TODO: replace with isinstance checks on CryticCompile's exception hierarchy
        #       once crytic_compile.errors is confirmed stable across versions.
        exc_lower = str(exc).lower()
        if any(kw in exc_lower for kw in ("compil", "syntax", "invalid solidity", "parsing", "solc")):
            raise SolcCompilationError(
                f"Solidity compilation failed for '{sol_path.name}': {exc}"
            ) from exc
        raise SlitherParseError(
            f"Slither failed to parse '{sol_path.name}': {exc}"
        ) from exc

    # ── Contract selection ─────────────────────────────────────────────────
    contract = _select_contract(sl, config)

    # ── Shared node lists (x_list and node_metadata are index-aligned) ─────
    # x_list:        grows as declaration nodes and CFG nodes are added.
    # node_metadata: parallel list of dicts; same length as x_list at all times.
    x_list:        list = []
    node_metadata: list = []

    # node_map: canonical_name → index in x_list (declaration nodes only)
    node_map: dict[str, int] = {}

    _type_name_map = {v: k for k, v in NODE_TYPES.items()}

    def _add_node(obj: Any, initial_type_id: int, override_key: str | None = None) -> int | None:
        """Add one declaration node. Returns the assigned index, or None if duplicate.

        override_key: when provided, use this string as the node_map key instead of
        obj.canonical_name. Used by the NF-10 synthetic-key path to preserve overriding
        function nodes without mutating the Slither object.
        """
        key: str = override_key or getattr(obj, "canonical_name", None) or obj.name
        if key in node_map:
            return None
        idx = len(x_list)
        node_map[key] = idx
        x_list.append(_build_node_features(obj, initial_type_id))

        # Determine the actual type_id after _build_node_features may override it.
        # x_list[-1][0] is normalised (float(type_id)/_MAX_TYPE_ID), so reverse-normalise.
        # NF-2: was hardcoded * 12 — now uses _MAX_TYPE_ID so it stays correct if
        # the schema ever gains a new node type (mirrors the A3/A32 fix).
        actual_type_id = int(round(x_list[-1][0] * _MAX_TYPE_ID))
        actual_type_name = _type_name_map.get(actual_type_id, "FUNCTION")
        sm = getattr(obj, "source_mapping", None)
        node_metadata.append({
            "name":         override_key or getattr(obj, "canonical_name", None) or getattr(obj, "name", str(obj)),
            "type":         actual_type_name,
            "source_lines": list(sm.lines) if sm and getattr(sm, "lines", None) else [],
        })
        return idx

    # ── Add declaration nodes (fixed insertion order) ──────────────────────
    # ⚠  This order must remain stable — node indices flow into edge_index.
    _add_node(contract, NODE_TYPES["CONTRACT"])
    # BUG-H8: add inherited parent contracts as CONTRACT nodes so that
    # INHERITS edges (added in the edge section below) can resolve their indices.
    # Slither's contract.functions already includes inherited functions, so we
    # don't re-add functions here — just the parent CONTRACT node itself.
    for parent in (getattr(contract, "inheritance", None) or []):
        _add_node(parent, NODE_TYPES["CONTRACT"])
    for var   in contract.state_variables: _add_node(var,   NODE_TYPES["STATE_VAR"])
    # BUG-H9: MODIFIER and EVENT nodes must be registered BEFORE edge creation runs.
    # _add_edge() silently drops any edge whose destination isn't yet in node_map.
    # Original code registered these after the edge loop, causing all EMITS edges
    # (function→event) and any CALLS edges targeting modifiers to be dropped.
    for mod   in contract.modifiers: _add_node(mod,   NODE_TYPES["MODIFIER"])
    for event in contract.events:    _add_node(event, NODE_TYPES["EVENT"])

    # For functions: add function node first, then immediately add its CFG children
    # so CFG nodes follow their parent function in x_list (cleaner indexing).
    edges:      list = []
    edge_types: list = []

    _cfg_failure_count = 0
    _icfg_failure_count = 0   # A18: counts ICFG map construction failures (Gate 2.1)
    _duplicate_func_count = 0  # NF-10: counts duplicate functions given synthetic keys
    _func_total = len(contract.functions)

    # PLAN-1C: accumulated across functions — needed by _add_icfg_edges after the loop.
    _func_entry_map:    dict = {}   # canonical_name → graph_idx of ENTRYPOINT node
    _func_terminal_map: dict = {}   # canonical_name → [graph_idx of terminal nodes]
    _func_cfg_maps:     dict = {}   # canonical_name → {slither_node → graph_idx}

    for func_index, func in enumerate(contract.functions):
        fn_idx = _add_node(func, NODE_TYPES["FUNCTION"])
        if fn_idx is None:
            # NF-10: duplicate canonical_name (overriding/overloaded function).
            # Assign a synthetic key so the overriding function's CFG is preserved —
            # overrides often introduce vulnerabilities and must not be silently merged
            # into the original function's node.
            fn_key = getattr(func, "canonical_name", None) or func.name
            synthetic_key = f"{fn_key}__override__{func_index}"
            fn_idx = _add_node(func, NODE_TYPES["FUNCTION"], override_key=synthetic_key)
            if fn_idx is not None:
                _duplicate_func_count += 1
                logger.debug(
                    "[NF-10] Duplicate function '%s' in '%s' — synthetic key '%s' (idx %d).",
                    fn_key, contract.name, synthetic_key, fn_idx,
                )
            else:
                # Synthetic key also collides — degenerate case, skip this override.
                logger.warning(
                    "[NF-10] Synthetic key '%s' also duplicated in '%s' — skipping CFG.",
                    synthetic_key, contract.name,
                )
                continue

        # CFG nodes for this function, appended immediately after it
        try:
            cfg_node_map: dict = {}  # slither_node → graph_idx, scoped per function
            contains_edges, control_flow_edges = _build_control_flow_edges(
                func, fn_idx, cfg_node_map, x_list, node_metadata,
                parent_features=x_list[fn_idx],  # BUG-C3: propagate function features
            )
            for src, dst in contains_edges:
                edges.append([src, dst])
                edge_types.append(EDGE_TYPES["CONTAINS"])
            for src, dst in control_flow_edges:
                edges.append([src, dst])
                edge_types.append(EDGE_TYPES["CONTROL_FLOW"])

            # PLAN-1C: record per-function maps for ICFG edge construction.
            func_key = getattr(func, "canonical_name", None) or func.name
            _func_cfg_maps[func_key] = cfg_node_map
            try:
                from slither.core.cfg.node import NodeType as _SNT
                func_nodes = getattr(func, "nodes", None) or []
                for _n in func_nodes:
                    if _n.type == _SNT.ENTRYPOINT and _n in cfg_node_map:
                        _func_entry_map[func_key] = cfg_node_map[_n]
                        break
                # A14: exclude THROW (revert/require-failure) and RETURN terminal nodes.
                # Only normal-exit terminals (fall-through, no successors except due to
                # revert) should generate RETURN_TO edges — THROW/RETURN unwind the stack
                # and do NOT transfer control back to the call site.
                _revert_types: frozenset = frozenset(filter(None, (
                    getattr(_SNT, "THROW", None),
                    getattr(_SNT, "RETURN", None),
                )))
                _func_terminal_map[func_key] = [
                    cfg_node_map[_n]
                    for _n in func_nodes
                    if _n in cfg_node_map
                    and not (getattr(_n, "sons", None) or [])
                    and getattr(_n, "type", None) not in _revert_types
                ]
            except Exception as exc:
                # A18: log and count ICFG map failures — Gate 2.1 checks this count.
                _icfg_failure_count += 1
                logger.error(
                    "[A18] ICFG map construction failed for '%s' in '%s': %s",
                    func_key, contract.name, exc,
                )

        except Exception as exc:
            _cfg_failure_count += 1
            logger.warning(
                "CFG extraction failed for function '%s' in '%s': %s — "
                "CONTAINS/CONTROL_FLOW edges for this function omitted.",
                getattr(func, "canonical_name", "?"),
                contract.name,
                exc,
            )

    # PLAN-1D: add ICFG-Lite cross-function edges (CALL_ENTRY + RETURN_TO).
    try:
        _add_icfg_edges(
            contract, _func_entry_map, _func_terminal_map, _func_cfg_maps,
            edges, edge_types,
        )
    except Exception as exc:
        logger.warning(
            "ICFG edge extraction failed for '%s': %s — CALL_ENTRY/RETURN_TO omitted.",
            contract.name, exc,
        )

    # PLAN-1E: add intra-function DEF_USE data-flow edges.
    try:
        _add_def_use_edges(contract, _func_cfg_maps, edges, edge_types)
    except Exception as exc:
        logger.warning(
            "DEF_USE edge extraction failed for '%s': %s — DEF_USE edges omitted.",
            contract.name, exc,
        )

    # Warn if CFG extraction failures exceeded 5% of functions in this contract.
    # Single-statement or synthetic functions raising here is benign; a high rate
    # signals a Slither version mismatch or corrupt source.
    if _cfg_failure_count > 0 and _func_total > 0:
        failure_rate = _cfg_failure_count / _func_total
        log_fn = logger.error if failure_rate > 0.05 else logger.debug
        log_fn(
            "CFG extraction: %d/%d functions failed in '%s' (%.0f%%)%s",
            _cfg_failure_count,
            _func_total,
            contract.name,
            failure_rate * 100,
            " — exceeds 5% threshold, investigate Slither version or source" if failure_rate > 0.05 else "",
        )

    if not x_list:
        raise EmptyGraphError(
            f"Contract '{contract.name}' produced zero graph nodes after filtering. "
            "Slither parsed the file but found no analyzable declarations."
        )

    # ── Feature tensor + dimension guard ──────────────────────────────────
    x = torch.tensor(x_list, dtype=torch.float)   # [N, NODE_FEATURE_DIM]
    if x.shape[1] != NODE_FEATURE_DIM:
        raise SlitherParseError(
            f"Node feature dimension mismatch for '{contract.name}': "
            f"expected {NODE_FEATURE_DIM}, got {x.shape[1]}. "
            "Bug in _build_node_features() or _build_cfg_node_features() — "
            f"each must return exactly {NODE_FEATURE_DIM} floats."
        )

    # BUG-L4: validate feature ranges at extraction time — catch OOR values
    # before they corrupt the training cache. Log a warning (don't raise) so
    # a single bad contract doesn't abort a full extraction run.
    oor_mask = (x < -1.0) | (x > 1.0)
    if oor_mask.any():
        oor_nodes, oor_dims = oor_mask.nonzero(as_tuple=True)
        logger.warning(
            "OOR features in '%s': %d cells across nodes %s dims %s — "
            "feature values outside [-1, 1] may destabilise training.",
            contract.name,
            int(oor_mask.sum()),
            oor_nodes.unique().tolist()[:5],
            oor_dims.unique().tolist(),
        )

    # node_metadata must stay index-aligned with x_list.
    # A4: assert → raise ValueError — asserts are stripped by Python -O (production).
    if len(node_metadata) != x.shape[0]:
        raise ValueError(
            f"node_metadata length {len(node_metadata)} ≠ x.shape[0] {x.shape[0]} "
            f"for '{contract.name}'. This is a bug — _add_node() and "
            "_build_control_flow_edges() must always append to both lists together."
        )

    # ── E2 / Interp-3: Contract-size normalisation (Timestamp size shortcut fix) ──
    # complexity (dim 5) is log1p(CFG block count)/log1p(100) — an absolute measure.
    # Large contracts have many functions, each with moderate complexity, so the
    # raw mean is size-correlated. Normalise by contract total node count to make
    # complexity a relative (within-contract) signal.
    # Formula: complexity_norm[i] = complexity[i] * (1 + log(x.shape[0])) / (1 + x.shape[0])
    # This rescales: small contracts keep high relative complexity; large contracts
    # get complexity scaled down proportional to their size.
    _CONTRACT_SIZE = float(x.shape[0])
    if _CONTRACT_SIZE > 1.0:
        import math as _math
        _size_factor = _math.log1p(_CONTRACT_SIZE) / _CONTRACT_SIZE
        x[:, 5] = x[:, 5] * _size_factor

    # ── Declaration-level edges ────────────────────────────────────────────
    # NF-1: build short_name → canonical_name map for EMITS fallback path.
    # Slither's EventCall IR stores the short name (e.g. "Transfer") but node_map
    # keys are canonical names (e.g. "ERC20.Transfer(address,address,uint256)").
    # For Solidity <0.4.21 contracts (no `emit` keyword), the fallback path fires;
    # without this translation, all EMITS edges from the fallback are silently dropped.
    _event_name_map: dict[str, str] = {}
    for _evt in (getattr(contract, "events", None) or []):
        _short = getattr(_evt, "name", None)
        _canon = getattr(_evt, "canonical_name", None) or _short
        if _short and _canon:
            _event_name_map[_short] = _canon

    # NF-11: track dropped edge counts per edge type for the post-edge summary log.
    _edge_drop_counts: dict[int, int] = {}

    def _add_edge(src_key: str, dst_key: str, etype: int) -> None:
        si = node_map.get(src_key)
        di = node_map.get(dst_key)
        if si is not None and di is not None:
            edges.append([si, di])
            edge_types.append(etype)
        else:
            # NF-11: count dropped edges by type instead of silently discarding.
            _edge_drop_counts[etype] = _edge_drop_counts.get(etype, 0) + 1
            logger.debug(
                "[NF-11] Dropped edge type=%d: src '%s'%s dst '%s'%s",
                etype,
                src_key, "" if si is not None else " (missing)",
                dst_key, "" if di is not None else " (missing)",
            )

    for func in contract.functions:
        fn = getattr(func, "canonical_name", None) or func.name

        for call in (getattr(func, "internal_calls", None) or []):
            if hasattr(call, "canonical_name"):
                _add_edge(fn, call.canonical_name, EDGE_TYPES["CALLS"])

        for var in (getattr(func, "state_variables_read", None) or []):
            _add_edge(fn, var.canonical_name, EDGE_TYPES["READS"])
        for var in (getattr(func, "state_variables_written", None) or []):
            _add_edge(fn, var.canonical_name, EDGE_TYPES["WRITES"])

        # BUG-H7: events_emitted is unreliable for Solidity <0.4.21 (no `emit`
        # keyword). Fall back to scanning IR ops for EventCall objects, which
        # Slither populates even for old-style "Transfer(...)" event emission.
        emitted: set[str] = set()
        if hasattr(func, "events_emitted"):
            try:
                for evt in func.events_emitted:
                    key = getattr(evt, "canonical_name", None) or getattr(evt, "name", None)
                    if key:
                        emitted.add(key)
            except Exception:
                pass
        if not emitted:
            try:
                from slither.slithir.operations import EventCall as _EventCall
                for node in (getattr(func, "nodes", None) or []):
                    for ir in (getattr(node, "irs", None) or []):
                        if isinstance(ir, _EventCall):
                            _short_key = getattr(ir, "name", None)
                            if _short_key:
                                # NF-1: translate short event name to canonical name so
                                # the key matches node_map (keyed by canonical_name).
                                # For pre-0.4.21 contracts, _short_key is e.g. "Transfer"
                                # but node_map has "ERC20.Transfer(address,...)".
                                emitted.add(_event_name_map.get(_short_key, _short_key))
            except Exception:
                pass
        for key in emitted:
            _add_edge(fn, key, EDGE_TYPES["EMITS"])

    # BUG-H8: Use canonical_name for both src and dst so they match node_map keys.
    try:
        contract_key = getattr(contract, "canonical_name", None) or contract.name
        for parent in (getattr(contract, "inheritance", None) or []):
            parent_key = getattr(parent, "canonical_name", None) or parent.name
            _add_edge(contract_key, parent_key, EDGE_TYPES["INHERITS"])
    except Exception:
        pass

    # ── Per-contract extraction summary logs ──────────────────────────────
    if _icfg_failure_count > 0:
        logger.warning(
            "[A18] ICFG map construction failed for %d function(s) in '%s' — "
            "CALL_ENTRY/RETURN_TO edges may be incomplete.",
            _icfg_failure_count, contract.name,
        )
    if _duplicate_func_count > 0:
        logger.info(
            "[NF-10] %d duplicate function(s) in '%s' given synthetic node keys — "
            "CFG extracted for overriding functions.",
            _duplicate_func_count, contract.name,
        )
    # NF-11: log per-type edge drop summary
    if _edge_drop_counts:
        _type_name_map_local = {v: k for k, v in EDGE_TYPES.items()}
        _drop_summary = ", ".join(
            f"{_type_name_map_local.get(et, str(et))}: {cnt}"
            for et, cnt in sorted(_edge_drop_counts.items())
        )
        logger.debug(
            "[NF-11] Edge drop summary for '%s': %s",
            contract.name, _drop_summary,
        )

    # ── Assemble PyG Data object ───────────────────────────────────────────
    if edges:
        edge_index = torch.tensor(edges,      dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros(0,      dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index)
    if config.include_edge_attr:
        graph.edge_attr = edge_attr

    # [Phase 7 / Interp-2] CEI path label — stored unconditionally so v9 cache
    # always carries this field.  0 = no CEI violation detected; 1 = present.
    # Activated in trainer.py only after Gate 7.5 passes (label quality check).
    graph.has_cei_path = _compute_has_cei_path(node_metadata, edge_index, edge_attr)

    graph.node_metadata  = node_metadata
    graph.contract_name  = contract.name
    graph.num_nodes      = int(x.shape[0])
    graph.num_edges      = len(edges)

    return graph
