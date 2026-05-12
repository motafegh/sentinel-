"""
graph_extractor.py — Canonical Solidity-to-PyG graph extraction (v2 schema)

WHAT THIS MODULE DOES
─────────────────────
Provides the single, authoritative implementation of the AST-to-graph
conversion used by both SENTINEL pipelines:

  Offline (batch)  ml/src/data_extraction/ast_extractor.py
                     Processes ~68K training contracts in parallel.
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

SHAPE CONTRACT  (v2 schema — must match training data)
──────────────────────────────────────────────────────────────────────────────
  graph.x             [N, 12]  float32  node feature matrix (NODE_FEATURE_DIM=12)
  graph.edge_index    [2, E]   int64    edge connectivity in COO format
  graph.edge_attr     [E]      int64    edge type IDs 0-6 (1-D per PyG convention)
                                         (only attached when config.include_edge_attr)
  graph.node_metadata list     of dicts, one per node, index-aligned with graph.x
                                Required: {"name", "type", "source_lines"} keys.
                                Used by _find_function_node() in the pre-flight test.

  graph.contract_name  str   — name of the analysed Slither Contract object
  graph.num_nodes      int   — N (same as x.shape[0])
  graph.num_edges      int   — E (same as edge_index.shape[1])

  Caller-specific metadata (.contract_hash, .contract_path, .y) is NOT set
  here; each caller attaches its own values after the call returns.

V2 SCHEMA CHANGES (2026-05-11)
───────────────────────────────
  Node features: 8 → 12 dims.
    - `reentrant` (Slither.is_reentrant) removed: Slither shortcut.
    - `gas_intensity` removed: circular heuristic over features already in vector.
    - 5 new features added: return_ignored (sentinel -1.0), call_target_typed
      (sentinel -1.0), in_unchecked, has_loop, external_call_count.

  Edge types: 5 → 7.
    - CONTAINS(5):     function → its CFG_NODE children (new)
    - CONTROL_FLOW(6): CFG_NODE → successor CFG_NODE, directed (new)

  Node types: 8 → 13.
    - CFG_NODE_CALL(8):  statement containing an external call
    - CFG_NODE_WRITE(9): statement writing a state variable
    - CFG_NODE_READ(10): statement reading a state variable
    - CFG_NODE_CHECK(11): require / assert / if condition
    - CFG_NODE_OTHER(12): all other statement types (including synthetic nodes)

  graph.node_metadata: new attribute — parallel list of metadata dicts,
    one entry per node, required for pre-flight embedding-separation test.

  These changes require a full re-extraction of all ~68K graph .pt files.
  See graph_schema.py CHANGE POLICY.
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

from .graph_schema import EDGE_TYPES, NODE_FEATURE_DIM, NODE_TYPES, VISIBILITY_MAP

logger = logging.getLogger(__name__)


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
    no --allow-paths, first-contract policy). The offline batch pipeline
    overrides solc_binary, solc_version, and allow_paths for each version group.
    """

    multi_contract_policy: str = "first"
    """
    Which contract to analyse when the file defines multiple.

    "first"   — use contracts[0], the first non-dependency contract.
    "by_name" — use the contract whose .name == target_contract_name.
                Falls back to "first" with a warning if the name is not found.
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

    No regex fallback: regex over multi-line source is unreliable for this
    feature (multi-line assignments, self-calls). The -1.0 sentinel gives the
    GNN a distinct embedding for "unknown" vs "safe" vs "discarded".
    """
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall
        for op in func.slithir_operations:
            if isinstance(op, (LowLevelCall, HighLevelCall)):
                if op.lvalue is None:
                    return 1.0
        return 0.0
    except AttributeError:
        logger.warning(
            "return_ignored: Slither IR unavailable for %s — using sentinel -1.0",
            getattr(func, "canonical_name", "?"),
        )
        return -1.0


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

    except Exception:
        pass  # type resolution failed; fall through to source scan

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


def _compute_in_unchecked(func: Any) -> float:
    """1.0 if func body contains an unchecked{} arithmetic block."""
    try:
        from slither.core.cfg.node import NodeType
        for node in (getattr(func, "nodes", None) or []):
            # STARTUNCHECKED available in Slither >=0.9.3 (enforced at import in graph_schema.py)
            if getattr(node, "type", None) == NodeType.STARTUNCHECKED:
                return 1.0
    except AttributeError:
        pass  # NodeType.STARTUNCHECKED absent — fall through to regex

    try:
        sm = getattr(func, "source_mapping", None)
        content = getattr(sm, "content", "") or ""
        # Covers: "unchecked {", "unchecked{", "unchecked\n{" (all valid Solidity)
        if re.search(r"\bunchecked\s*\{", content):
            return 1.0
    except Exception:
        pass

    return 0.0


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

    # Fallback: Slither convenience attribute (exists in some versions)
    try:
        if getattr(func, "is_loop_present", None) is True:
            return 1.0
    except Exception:
        pass

    return 0.0


def _compute_external_call_count(func: Any) -> float:
    """
    log1p(count) / log1p(20), clamped [0, 1].
    1 call → 0.23,  5 calls → 0.60,  20 calls → 1.0.
    """
    try:
        n  = len(list(getattr(func, "high_level_calls", None) or []))
        n += len(list(getattr(func, "low_level_calls",  None) or []))
        return min(math.log1p(n) / math.log1p(20), 1.0)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CFG node helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        from slither.slithir.operations import LowLevelCall, HighLevelCall
        from slither.core.variables.state_variable import StateVariable

        check_types = {SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.ENDLOOP, SNT.THROW}

        irs = list(getattr(slither_node, "irs", None) or [])

        # Priority 1: any IR op is an external call
        if any(isinstance(op, (LowLevelCall, HighLevelCall)) for op in irs):
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

    except Exception:
        pass

    return NODE_TYPES["CFG_NODE_OTHER"]


def _build_cfg_node_features(slither_node: Any, func: Any, cfg_type: int) -> list:
    """
    Build the 12-dim feature vector for a CFG (statement-level) node.

    Returns list[float] of exactly NODE_FEATURE_DIM (12) elements.
    torch.tensor(x_list) requires all sublists to be the same length;
    returning a different length silently causes crashes at tensor assembly.

    CRITICAL: in_unchecked [9] is ALWAYS 0.0 — never inherited from the parent
    function's flag. If a function has any unchecked block, ALL its child CFG
    nodes would get 1.0 including statements OUTSIDE the unchecked scope,
    creating false positives for IntegerUO. The function-level node carries
    this signal and the GNN propagates it via Phase 1 CONTAINS edges.

    Slither synthetic nodes (ENTRY_POINT, EXPRESSION, BEGIN_LOOP, etc.) with
    no source_mapping and empty IRS are handled correctly: _cfg_node_type()
    returns CFG_NODE_OTHER (12) and loc defaults to 0.0. Do NOT filter them.
    """
    loc = 0.0
    sm = getattr(slither_node, "source_mapping", None)
    if sm is not None and getattr(sm, "lines", None):
        loc = float(len(sm.lines))

    return [
        float(cfg_type) / 12.0,  # [0]  type_id normalised to [0,1] (raw 8–12 / 12)
        0.0,              # [1]  visibility — not applicable
        0.0,              # [2]  pure — not applicable
        0.0,              # [3]  view — not applicable
        0.0,              # [4]  payable — not applicable
        0.0,              # [5]  complexity — function-level metric
        loc,              # [6]  loc — lines of this statement's source span
        0.0,              # [7]  return_ignored — not per-statement in v5.0
        1.0,              # [8]  call_target_typed — default safe (not applicable)
        0.0,              # [9]  in_unchecked — NEVER inherited from parent func
        0.0,              # [10] has_loop — not applicable at statement level
        0.0,              # [11] external_call_count — not applicable
    ]


def _build_control_flow_edges(
    func: Any,
    func_node_idx: int,
    node_index_map: dict,
    x_list: list,
    node_metadata: list,
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
            n.node_id,
        ),
    )

    contains_edges:     list = []
    control_flow_edges: list = []

    # Pass 1: assign indices, build feature vectors, populate shared lists
    for slither_node in cfg_nodes:
        cfg_type  = _cfg_node_type(slither_node)
        graph_idx = len(x_list)          # CORRECT: next available global index
        node_index_map[slither_node] = graph_idx

        x_list.append(_build_cfg_node_features(slither_node, func, cfg_type))

        cfg_type_name = _type_name_map.get(cfg_type, "CFG_NODE_OTHER")
        sm = getattr(slither_node, "source_mapping", None)
        node_metadata.append({
            "name":         str(slither_node),
            "type":         cfg_type_name,
            "source_lines": list(sm.lines) if sm and sm.lines else [],
        })

        contains_edges.append((func_node_idx, graph_idx))

    # Pass 2: build CONTROL_FLOW edges (all CFG nodes indexed in pass 1)
    for slither_node in cfg_nodes:
        src_idx = node_index_map[slither_node]
        for successor in (getattr(slither_node, "sons", None) or []):
            if successor in node_index_map:
                control_flow_edges.append((src_idx, node_index_map[successor]))

    return contains_edges, control_flow_edges


def _build_node_features(obj: Any, type_id: int) -> list:
    """
    Compute the 12-dimensional feature vector (v2 schema) for one AST node.

    Returns list[float] of exactly NODE_FEATURE_DIM (12) elements.

    Feature layout:
      [0]  type_id             — float(NODE_TYPES[kind])
      [1]  visibility          — VISIBILITY_MAP ordinal 0-2
      [2]  pure                — 1.0 if Function.pure
      [3]  view                — 1.0 if Function.view
      [4]  payable             — 1.0 if Function.payable
      [5]  complexity          — float(len(func.nodes)) CFG block count
      [6]  loc                 — float(len(source_mapping.lines))
      [7]  return_ignored      — 0.0/1.0/-1.0 sentinel
      [8]  call_target_typed   — 0.0/1.0/-1.0 sentinel
      [9]  in_unchecked        — 1.0 if body contains unchecked{} block
      [10] has_loop            — 1.0 if function contains a loop
      [11] external_call_count — log-normalized count of external calls

    Non-Function nodes receive 0.0 for features [2:] except:
      - call_target_typed [8] defaults to 1.0 (safe: not applicable)
      - loc [6] is computed when source_mapping is available
    """
    _is_function = hasattr(obj, "nodes") and hasattr(obj, "pure")

    visibility = float(VISIBILITY_MAP.get(
        str(getattr(obj, "visibility", "public")), 0
    ))
    loc = 0.0
    sm = getattr(obj, "source_mapping", None)
    if sm is not None and getattr(sm, "lines", None):
        loc = float(len(sm.lines))

    # Defaults for non-Function nodes
    pure = view = payable = 0.0
    complexity = 0.0
    return_ignored     = 0.0
    call_target_typed  = 1.0   # safe default: "not applicable"
    in_unchecked       = 0.0
    has_loop           = 0.0
    external_call_count = 0.0

    if _is_function:
        pure    = 1.0 if getattr(obj, "pure",    False) else 0.0
        view    = 1.0 if getattr(obj, "view",    False) else 0.0
        payable = 1.0 if getattr(obj, "payable", False) else 0.0
        try:
            complexity = float(len(obj.nodes)) if obj.nodes else 0.0
        except Exception:
            complexity = 0.0

        return_ignored      = _compute_return_ignored(obj)
        call_target_typed   = _compute_call_target_typed(obj)
        in_unchecked        = _compute_in_unchecked(obj)
        has_loop            = _compute_has_loop(obj)
        external_call_count = _compute_external_call_count(obj)

        # Override FUNCTION(1) type_id for special function kinds
        if getattr(obj, "is_constructor", False):
            type_id = NODE_TYPES["CONSTRUCTOR"]
        elif getattr(obj, "is_fallback", False):
            type_id = NODE_TYPES["FALLBACK"]
        elif getattr(obj, "is_receive", False):
            type_id = NODE_TYPES["RECEIVE"]

    return [
        float(type_id) / 12.0,  # normalised to [0,1] (raw 0–12 / 12)
        visibility,
        pure,
        view,
        payable,
        complexity,
        loc,
        return_ignored,
        call_target_typed,
        in_unchecked,
        has_loop,
        external_call_count,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────────

def _select_contract(sl: Any, config: GraphExtractionConfig) -> Any:
    """
    Pick the target Slither Contract from the parsed contract list.

    Filters out imported dependencies so only user-supplied code is analysed.
    When a file defines interfaces before the main contract (common in protocol
    contracts), interfaces are skipped and the concrete contract with the most
    functions is preferred.  This avoids extracting a ghost graph (2 nodes,
    0 edges) from an interface whose function bodies are empty.

    NOTE (Slither 0.10.x): `is_interface` is a reliable property.
    `is_abstract` does not exist — abstract contracts share contract_kind='contract'
    with concrete ones.  The len(functions) sort handles abstract vs concrete
    implicitly since abstract functions have no body and produce fewer nodes.
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
            "Contract %r not found in file (available: %s); falling back to first.",
            config.target_contract_name,
            [c.name for c in candidates],
        )

    # Prefer non-interface contracts.  If multiple remain, pick the one with the
    # most functions — this naturally selects concrete over abstract and the
    # richest contract when multiple implementations are defined in one file.
    non_iface = [c for c in candidates if not c.is_interface]
    if non_iface:
        return max(non_iface, key=lambda c: len(c.functions))

    # All candidates are interfaces (e.g. a pure-interface file) — fall back to
    # the first one so extraction still proceeds rather than silently failing.
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

    def _add_node(obj: Any, initial_type_id: int) -> int | None:
        """Add one declaration node. Returns the assigned index, or None if duplicate."""
        key: str = getattr(obj, "canonical_name", None) or obj.name
        if key in node_map:
            return None
        idx = len(x_list)
        node_map[key] = idx
        x_list.append(_build_node_features(obj, initial_type_id))

        # Determine the actual type_id after _build_node_features may override it
        actual_type_id = int(x_list[-1][0])
        actual_type_name = _type_name_map.get(actual_type_id, "FUNCTION")
        sm = getattr(obj, "source_mapping", None)
        node_metadata.append({
            "name":         getattr(obj, "canonical_name", None) or getattr(obj, "name", str(obj)),
            "type":         actual_type_name,
            "source_lines": list(sm.lines) if sm and getattr(sm, "lines", None) else [],
        })
        return idx

    # ── Add declaration nodes (fixed insertion order) ──────────────────────
    # ⚠  This order must remain stable — node indices flow into edge_index.
    _add_node(contract, NODE_TYPES["CONTRACT"])
    for var   in contract.state_variables: _add_node(var,   NODE_TYPES["STATE_VAR"])

    # For functions: add function node first, then immediately add its CFG children
    # so CFG nodes follow their parent function in x_list (cleaner indexing).
    edges:      list = []
    edge_types: list = []

    _cfg_failure_count = 0
    _func_total = len(contract.functions)

    for func in contract.functions:
        fn_idx = _add_node(func, NODE_TYPES["FUNCTION"])
        if fn_idx is None:
            # Duplicate function name — still need to find its index for CFG edges
            fn_key = getattr(func, "canonical_name", None) or func.name
            fn_idx = node_map.get(fn_key)
            if fn_idx is None:
                continue

        # CFG nodes for this function, appended immediately after it
        try:
            cfg_node_map: dict = {}  # slither_node → graph_idx, scoped per function
            contains_edges, control_flow_edges = _build_control_flow_edges(
                func, fn_idx, cfg_node_map, x_list, node_metadata
            )
            for src, dst in contains_edges:
                edges.append([src, dst])
                edge_types.append(EDGE_TYPES["CONTAINS"])
            for src, dst in control_flow_edges:
                edges.append([src, dst])
                edge_types.append(EDGE_TYPES["CONTROL_FLOW"])
        except Exception as exc:
            _cfg_failure_count += 1
            logger.warning(
                "CFG extraction failed for function '%s' in '%s': %s — "
                "CONTAINS/CONTROL_FLOW edges for this function omitted.",
                getattr(func, "canonical_name", "?"),
                contract.name,
                exc,
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

    for mod   in contract.modifiers: _add_node(mod,   NODE_TYPES["MODIFIER"])
    for event in contract.events:    _add_node(event, NODE_TYPES["EVENT"])

    if not x_list:
        raise EmptyGraphError(
            f"Contract '{contract.name}' produced zero graph nodes after filtering. "
            "Slither parsed the file but found no analyzable declarations."
        )

    # ── Feature tensor + dimension guard ──────────────────────────────────
    x = torch.tensor(x_list, dtype=torch.float)   # [N, 12]
    if x.shape[1] != NODE_FEATURE_DIM:
        raise SlitherParseError(
            f"Node feature dimension mismatch for '{contract.name}': "
            f"expected {NODE_FEATURE_DIM}, got {x.shape[1]}. "
            "Bug in _build_node_features() or _build_cfg_node_features() — "
            f"each must return exactly {NODE_FEATURE_DIM} floats."
        )

    # node_metadata must stay index-aligned with x_list
    assert len(node_metadata) == x.shape[0], (
        f"node_metadata length {len(node_metadata)} ≠ x.shape[0] {x.shape[0]} "
        f"for '{contract.name}'. This is a bug — _add_node() and "
        "_build_control_flow_edges() must always append to both lists together."
    )

    # ── Declaration-level edges ────────────────────────────────────────────
    def _add_edge(src_key: str, dst_key: str, etype: int) -> None:
        si = node_map.get(src_key)
        di = node_map.get(dst_key)
        if si is not None and di is not None:
            edges.append([si, di])
            edge_types.append(etype)

    for func in contract.functions:
        fn = getattr(func, "canonical_name", None) or func.name

        for call in (getattr(func, "internal_calls", None) or []):
            if hasattr(call, "canonical_name"):
                _add_edge(fn, call.canonical_name, EDGE_TYPES["CALLS"])

        for var in (getattr(func, "state_variables_read", None) or []):
            _add_edge(fn, var.canonical_name, EDGE_TYPES["READS"])
        for var in (getattr(func, "state_variables_written", None) or []):
            _add_edge(fn, var.canonical_name, EDGE_TYPES["WRITES"])

        if hasattr(func, "events_emitted"):
            try:
                for evt in func.events_emitted:
                    _add_edge(fn, evt.canonical_name, EDGE_TYPES["EMITS"])
            except Exception:
                pass

    try:
        for parent in (getattr(contract, "inheritance", None) or []):
            _add_edge(contract.name, parent.name, EDGE_TYPES["INHERITS"])
    except Exception:
        pass

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

    graph.node_metadata  = node_metadata
    graph.contract_name  = contract.name
    graph.num_nodes      = int(x.shape[0])
    graph.num_edges      = len(edges)

    return graph
