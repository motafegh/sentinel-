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
  graph.x           [N, 13] float32  node feature matrix (NODE_FEATURE_DIM=13)
  graph.edge_index  [2, E]  int64    edge connectivity in COO format
  graph.edge_attr   [E]     int64    edge type IDs 0-6 — 1-D per PyG convention
                                     (only attached when config.include_edge_attr)

  graph.contract_name  str   — name of the analysed Slither Contract object
  graph.num_nodes      int   — N (same as x.shape[0])
  graph.num_edges      int   — E (same as edge_index.shape[1])

  Caller-specific metadata (.contract_hash, .contract_path, .y) is NOT set
  here; each caller attaches its own values after the call returns.

V2 SCHEMA CHANGES (2026-05-10)
───────────────────────────────
  Node features: 8 → 13 dims.
    - `reentrant` (Slither.is_reentrant) removed: it gave the model Slither's
      answer rather than making it learn from structure.
    - 6 semantic features added: return_ignored, call_target_typed, in_unchecked,
      has_loop, gas_intensity, external_call_count. See graph_schema.FEATURE_NAMES.

  Edge types: 5 → 7.
    - CONTAINS(5):     function → its CFG_NODE children (new)
    - CONTROL_FLOW(6): CFG_NODE → successor CFG_NODE (new)

  Node types: 8 → 9.
    - CFG_NODE(8): intra-function control-flow basic block (new)

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

    Example — offline batch use:
        config = GraphExtractionConfig(
            solc_binary=get_solc_binary(version),
            solc_version=version,
            allow_paths=str(project_root),
        )

    Example — online inference (single contract, known name):
        config = GraphExtractionConfig(
            multi_contract_policy="by_name",
            target_contract_name="MyToken",
        )
    """

    multi_contract_policy: str = "first"
    """
    Which contract to analyse when the file defines multiple.

    "first"   — use contracts[0], the first non-dependency contract.
                This matches the policy used during offline training and is
                the safe default for API requests where the contract name is
                not known in advance.
    "by_name" — use the contract whose .name == target_contract_name.
                Falls back to "first" with a warning if the name is not found.
    """

    target_contract_name: str | None = None
    """Used only when multi_contract_policy="by_name"."""

    include_edge_attr: bool = True
    """
    When True, graph.edge_attr [E] int64 is attached to the returned Data object.

    GNNEncoder currently uses only edge_index (type-agnostic GAT message
    passing). Edge attributes are retained for forward compatibility with
    GATv2/RGAT architectures. Disable for memory-constrained environments where
    the edge type information is not needed.
    """

    solc_binary: str | Path | None = None
    """
    Override the solc binary Slither uses. None → Slither resolves via PATH.

    The offline batch pipeline uses solc-select or a pinned venv binary to
    ensure each contract is compiled with a solc version matching its pragma.
    Online inference uses the system solc and accepts the resulting minor
    version mismatch (contracts are validated client-side before submission).
    """

    solc_version: str | None = None
    """
    Solidity version string, e.g. "0.8.19".

    When provided alongside allow_paths, this field determines whether
    --allow-paths is injected into the solc arguments: the flag requires
    solc >= 0.5.0. None → version check skipped, allow_paths used as-is.
    """

    allow_paths: str | None = None
    """
    Directory path(s) passed to solc as --allow-paths.

    Required in offline mode where contracts import local OpenZeppelin or
    other on-disk libraries that live outside the contract's own directory.
    None → no --allow-paths argument is injected.

    Example value: "/home/user/sentinel-" (project root containing node_modules).
    The extractor prepends ".," so the full argument becomes "--allow-paths .,<value>".
    """


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_return_ignored(func: Any) -> float:
    """
    1.0 if any low/high-level call in func discards its return value.

    PRIMARY: iterate func.slithir_operations; check .lvalue on call operations.
    FALLBACK: regex scan on source text (less precise, logged as WARNING).
    """
    # Primary path: Slither IR
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall
        ops = list(getattr(func, "slithir_operations", None) or [])
        for op in ops:
            if isinstance(op, (LowLevelCall, HighLevelCall)):
                if getattr(op, "lvalue", None) is None:
                    return 1.0
        return 0.0
    except Exception:
        pass

    # Fallback: text-based — bare `.call{` or `.call(` not preceded by `=` or `,`
    try:
        src = getattr(getattr(func, "source_mapping", None), "content", "") or ""
        # Matches bare `.call(` / `.call{value` not on the right-hand side of an assignment
        if re.search(r"(?<![=,(])\s*\.call[\({]", src):
            logger.warning(
                "return_ignored fallback (regex) for %s — Slither IR unavailable.",
                getattr(func, "canonical_name", "?"),
            )
            return 1.0
    except Exception:
        pass

    return 0.0


def _compute_call_target_typed(func: Any) -> float:
    """
    1.0 if all external calls in func go through typed interfaces (not raw address).

    PRIMARY: inspect func.high_level_calls / func.low_level_calls receiver types.
    FALLBACK: regex — raw `address(...).call` pattern implies untyped.
    """
    # Primary path: Slither type analysis
    try:
        from slither.core.solidity_types import AddressType

        low_lvl = list(getattr(func, "low_level_calls", None) or [])
        if low_lvl:
            # Any low-level call goes to a raw address by definition
            return 0.0

        high_lvl = list(getattr(func, "high_level_calls", None) or [])
        for item in high_lvl:
            # item is (contract_called, function_called) tuple
            recv = item[0] if isinstance(item, (tuple, list)) else item
            recv_type = getattr(recv, "type", None)
            if recv_type is not None and isinstance(recv_type, AddressType):
                return 0.0
        return 1.0  # all calls typed (or no external calls)
    except Exception:
        pass

    # Fallback: raw address call pattern in source
    try:
        src = getattr(getattr(func, "source_mapping", None), "content", "") or ""
        if re.search(r"address\s*\([^)]+\)\s*\.\s*call|address\b.*\.call", src):
            logger.warning(
                "call_target_typed fallback (regex) for %s.",
                getattr(func, "canonical_name", "?"),
            )
            return 0.0
    except Exception:
        pass

    return 1.0  # default safe: no evidence of raw-address calls


def _compute_in_unchecked(func: Any) -> float:
    """1.0 if func body contains an unchecked{} arithmetic block."""
    # Primary path: Slither NodeType
    try:
        from slither.core.cfg.node import NodeType
        nodes = list(getattr(func, "nodes", None) or [])
        for node in nodes:
            nt = getattr(node, "type", None)
            # UNCHECKED_BEGIN was added in Slither ≥ 0.9.3;
            # fall through to regex if the attribute doesn't exist on this version.
            unchecked_begin = getattr(NodeType, "UNCHECKED_BEGIN", None)
            if unchecked_begin is not None and nt == unchecked_begin:
                return 1.0
    except Exception:
        pass

    # Fallback: literal "unchecked" in source (reliable for Solidity 0.8+)
    try:
        src = getattr(getattr(func, "source_mapping", None), "content", "") or ""
        if "unchecked" in src:
            return 1.0
    except Exception:
        pass

    return 0.0


def _compute_has_loop(func: Any) -> float:
    """1.0 if func contains at least one loop construct."""
    # Primary: check func.nodes for loop-related NodeType values
    try:
        from slither.core.cfg.node import NodeType
        loop_types = {NodeType.IFLOOP, NodeType.STARTLOOP, NodeType.ENDLOOP}
        nodes = list(getattr(func, "nodes", None) or [])
        for node in nodes:
            if getattr(node, "type", None) in loop_types:
                return 1.0
    except Exception:
        pass

    # Fallback: Slither convenience attribute (exists in some versions)
    try:
        if getattr(func, "is_loop_present", False):
            return 1.0
    except Exception:
        pass

    # Fallback: regex — for / while / do loops in source
    try:
        src = getattr(getattr(func, "source_mapping", None), "content", "") or ""
        if re.search(r"\b(for|while|do)\b\s*[\({]", src):
            return 1.0
    except Exception:
        pass

    return 0.0


def _compute_gas_intensity(func: Any, has_loop: float) -> float:
    """
    Float [0, 1] heuristic for gas-expensive operations in func.

    Components:
      - complexity (CFG node count) / 50.0  clamped [0, 1]
      - +0.3 if has_loop (loops are always expensive)
      - +0.3 if any external call present
    Clamped to [0, 1] before return.
    """
    try:
        complexity = float(len(func.nodes)) if getattr(func, "nodes", None) else 0.0
    except Exception:
        complexity = 0.0

    score = min(complexity / 50.0, 1.0)
    if has_loop > 0.0:
        score += 0.3
    try:
        ext_calls = (
            list(getattr(func, "high_level_calls", None) or [])
            + list(getattr(func, "low_level_calls", None) or [])
        )
        if ext_calls:
            score += 0.3
    except Exception:
        pass

    return min(score, 1.0)


def _compute_external_call_count(func: Any) -> float:
    """log1p(total external calls) so the model sees a smooth signal."""
    try:
        n = len(list(getattr(func, "high_level_calls", None) or []))
        n += len(list(getattr(func, "low_level_calls", None) or []))
        return math.log1p(n)
    except Exception:
        return 0.0


def _build_node_features(obj: Any, type_id: int) -> list[float]:
    """
    Compute the 13-dimensional feature vector (v2 schema) for one AST node.

    See graph_schema.FEATURE_NAMES for the full index-to-name mapping.

    Feature layout:
      [0]  type_id             — float(NODE_TYPES[kind])
      [1]  visibility          — VISIBILITY_MAP ordinal 0-2 (default 0)
      [2]  pure                — 1.0 if Function.pure
      [3]  view                — 1.0 if Function.view
      [4]  payable             — 1.0 if Function.payable
      [5]  complexity          — float(len(func.nodes)) CFG block count
      [6]  loc                 — float(len(source_mapping.lines))
      [7]  return_ignored      — 1.0 if any call return value discarded
      [8]  call_target_typed   — 1.0 if all external calls go to typed interfaces
      [9]  in_unchecked        — 1.0 if body contains unchecked{} block
      [10] has_loop            — 1.0 if function contains a loop
      [11] gas_intensity       — float [0,1] gas expense heuristic
      [12] external_call_count — log1p(count of external calls)

    Non-Function nodes (STATE_VAR, EVENT, MODIFIER, CONTRACT) receive 0.0
    for all function-specific features, except:
      - call_target_typed [8] defaults to 1.0 (safe: not applicable)
      - loc [6] is computed when source_mapping is available

    Args:
        obj:     A Slither AST declaration (Contract, Function, StateVariable, …).
        type_id: Initial node type ID. Overridden internally for CONSTRUCTOR,
                 FALLBACK, and RECEIVE function kinds.

    Returns:
        List of exactly NODE_FEATURE_DIM (13) floats.
    """
    from slither.core.declarations import Function  # lazy: Slither is optional dep

    visibility = float(VISIBILITY_MAP.get(str(getattr(obj, "visibility", "public")), 0))
    loc = 0.0

    src = getattr(obj, "source_mapping", None)
    if src is not None:
        lines = getattr(src, "lines", None)
        if lines:
            loc = float(len(lines) if isinstance(lines, list) else lines)

    # Default values for function-specific features (used for non-Function nodes)
    pure = view = payable = 0.0
    complexity = 0.0
    return_ignored = 0.0
    call_target_typed = 1.0  # safe default: "all calls typed" when not applicable
    in_unchecked = 0.0
    has_loop = 0.0
    gas_intensity = 0.0
    external_call_count = 0.0

    if isinstance(obj, Function):
        pure    = 1.0 if obj.pure    else 0.0
        view    = 1.0 if obj.view    else 0.0
        payable = 1.0 if obj.payable else 0.0
        try:
            complexity = float(len(obj.nodes)) if obj.nodes else 0.0
        except Exception:
            complexity = 0.0

        # New semantic features (v2)
        return_ignored      = _compute_return_ignored(obj)
        call_target_typed   = _compute_call_target_typed(obj)
        in_unchecked        = _compute_in_unchecked(obj)
        has_loop            = _compute_has_loop(obj)
        gas_intensity       = _compute_gas_intensity(obj, has_loop)
        external_call_count = _compute_external_call_count(obj)

        # Override FUNCTION(1) type_id for special function kinds
        if obj.is_constructor:
            type_id = NODE_TYPES["CONSTRUCTOR"]
        elif obj.is_fallback:
            type_id = NODE_TYPES["FALLBACK"]
        elif obj.is_receive:
            type_id = NODE_TYPES["RECEIVE"]

    return [
        float(type_id),
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
        gas_intensity,
        external_call_count,
    ]


def _build_cfg_node_features(fn_node: Any, parent_func: Any) -> list[float]:
    """
    Compute the 13-dimensional feature vector for a CFG_NODE (basic block).

    CFG_NODE nodes represent individual Slither FunctionNode basic blocks
    within a function. Their features are computed at the block level where
    possible, falling back to parent-function values otherwise.

    Feature layout mirrors _build_node_features (same 13-dim schema):
      [0]  type_id = NODE_TYPES["CFG_NODE"]  (always 8)
      [1]  visibility = parent function visibility
      [2-4] pure/view/payable = 0.0 (not applicable at block level)
      [5]  complexity = 0.0 (block-level; overall function complexity is on func node)
      [6]  loc = lines in this basic block's source mapping
      [7]  return_ignored = 1.0 if this block has an ignored-return call op
      [8]  call_target_typed = 1.0 if all calls in this block go to typed interfaces
      [9]  in_unchecked = 1.0 if this block is inside an unchecked{} scope
      [10] has_loop = 1.0 if this block is a loop-related node type
      [11] gas_intensity = 0.0 (heuristic computed at function level)
      [12] external_call_count = log1p(external calls in this block)
    """
    from slither.core.declarations import Function  # noqa: F401 — ensures Slither is present

    type_id    = NODE_TYPES["CFG_NODE"]
    visibility = float(VISIBILITY_MAP.get(
        str(getattr(parent_func, "visibility", "public")), 0
    ))

    # Block-level loc
    loc = 0.0
    try:
        src = getattr(fn_node, "source_mapping", None)
        if src is not None:
            lines = getattr(src, "lines", None)
            if lines:
                loc = float(len(lines) if isinstance(lines, list) else lines)
    except Exception:
        pass

    # return_ignored: any IR op in this block with no lvalue
    return_ignored = 0.0
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall
        for op in (getattr(fn_node, "irs", None) or []):
            if isinstance(op, (LowLevelCall, HighLevelCall)):
                if getattr(op, "lvalue", None) is None:
                    return_ignored = 1.0
                    break
    except Exception:
        pass

    # call_target_typed: any low-level call in this block → 0.0
    call_target_typed = 1.0
    try:
        from slither.slithir.operations import LowLevelCall
        for op in (getattr(fn_node, "irs", None) or []):
            if isinstance(op, LowLevelCall):
                call_target_typed = 0.0
                break
    except Exception:
        pass

    # in_unchecked: check if this node is inside an unchecked scope
    in_unchecked = 0.0
    try:
        from slither.core.cfg.node import NodeType
        unchecked_begin = getattr(NodeType, "UNCHECKED_BEGIN", None)
        if unchecked_begin is not None and getattr(fn_node, "type", None) == unchecked_begin:
            in_unchecked = 1.0
        # Also check the node's source for the "unchecked" keyword as fallback
        if in_unchecked == 0.0:
            src_content = getattr(getattr(fn_node, "source_mapping", None), "content", "") or ""
            if "unchecked" in src_content:
                in_unchecked = 1.0
    except Exception:
        pass

    # has_loop: node type is a loop-related type
    has_loop = 0.0
    try:
        from slither.core.cfg.node import NodeType
        loop_types = {NodeType.IFLOOP, NodeType.STARTLOOP, NodeType.ENDLOOP}
        if getattr(fn_node, "type", None) in loop_types:
            has_loop = 1.0
    except Exception:
        pass

    # external_call_count at block level
    external_call_count = 0.0
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall
        n = sum(
            1 for op in (getattr(fn_node, "irs", None) or [])
            if isinstance(op, (LowLevelCall, HighLevelCall))
        )
        external_call_count = math.log1p(n)
    except Exception:
        pass

    return [
        float(type_id),
        visibility,
        0.0,           # pure  — not applicable at block level
        0.0,           # view  — not applicable at block level
        0.0,           # payable — not applicable at block level
        0.0,           # complexity — meaningful at function level, not block level
        loc,
        return_ignored,
        call_target_typed,
        in_unchecked,
        has_loop,
        0.0,           # gas_intensity — heuristic computed at function level
        external_call_count,
    ]


def _select_contract(sl: Any, config: GraphExtractionConfig) -> Any:
    """
    Pick the target Slither Contract from the parsed contract list.

    Filters out imported dependencies (OpenZeppelin, forge-std, etc.) so only
    user-supplied code is analysed — the same policy applied during offline
    training data construction.

    Args:
        sl:     A Slither instance whose .contracts list has been populated.
        config: Extraction config controlling multi-contract policy.

    Returns:
        The selected Slither Contract object.

    Raises:
        EmptyGraphError: if no non-dependency contracts exist in the file,
                         or if policy="by_name" and name is not found (after
                         fallback to first is exhausted due to empty list).
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
            "falling back to first contract %r.",
            config.target_contract_name,
            [c.name for c in candidates],
            candidates[0].name,
        )

    return candidates[0]


def _build_solc_args(config: GraphExtractionConfig) -> str | None:
    """
    Compute the solc_args string to pass to Slither, if any.

    --allow-paths is required in offline batch mode when contracts use local
    import paths (e.g. node_modules/). It was introduced in solc 0.5.0, so we
    skip it for older versions to avoid a solc startup error.

    Returns None when no arguments should be passed (online inference default).
    """
    if not config.allow_paths:
        return None

    # If we know the version, only inject --allow-paths for solc >= 0.5.0.
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
    Parse a Solidity file and return a PyG Data graph object.

    This is the single canonical AST-to-graph conversion used by both the
    offline training pipeline and the online inference API.  See module docstring
    for the full rationale.

    Node insertion order (CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs)
    is fixed and must match the original ast_extractor.py exactly.  Node indices
    appear in edge_index; reordering them would invalidate all training data.

    Args:
        sol_path: Path to a .sol file that must already exist on disk.
                  (For in-memory source strings, the caller writes a temp file
                  first — see ContractPreprocessor.process_source().)
        config:   Extraction settings. None → GraphExtractionConfig() defaults,
                  which are calibrated for online single-contract inference.

    Returns:
        PyG Data with the following attributes (v2 schema):
          .x              Tensor [N, 13] float32  node features
          .edge_index     Tensor [2, E]  int64    COO edge connectivity
          .edge_attr      Tensor [E]     int64    edge type IDs 0-6
                                                  (only if config.include_edge_attr)
          .contract_name  str            name of the analysed contract
          .num_nodes      int            N (declaration nodes + CFG_NODE nodes)
          .num_edges      int            E

        Caller-specific attributes (.contract_hash, .contract_path, .y) are NOT
        set here; attach them after the call returns.

    Raises:
        RuntimeError:         Slither library is not installed.
        SolcCompilationError: Solidity failed to compile (user-input error).
        SlitherParseError:    Slither failed internally (infrastructure error).
        EmptyGraphError:      Contract produced zero analyzable AST nodes.
    """
    if config is None:
        config = GraphExtractionConfig()

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
        # Distinguish compilation failures (user error) from Slither internals (infra).
        # Keywords sourced from solc's actual stderr output patterns.
        if any(kw in exc_lower for kw in ("compil", "syntax", "invalid solidity", "parsing", "solc")):
            raise SolcCompilationError(
                f"Solidity compilation failed for '{sol_path.name}': {exc}"
            ) from exc
        raise SlitherParseError(
            f"Slither failed to parse '{sol_path.name}': {exc}"
        ) from exc

    # ── Contract selection ─────────────────────────────────────────────────
    # Raises EmptyGraphError if no non-dependency contracts are found.
    contract = _select_contract(sl, config)

    # ── Node feature extraction ────────────────────────────────────────────
    # node_map: canonical_name → index in node_features_list.
    # Duplicate declarations are silently skipped (same canonical_name).
    node_features_list: list[list[float]] = []
    node_map: dict[str, int] = {}

    def _add_node(obj: Any, initial_type_id: int) -> None:
        # Contract objects expose only .name; all other declarations have .canonical_name.
        key: str = getattr(obj, "canonical_name", None) or obj.name
        if key not in node_map:
            node_map[key] = len(node_features_list)
            node_features_list.append(_build_node_features(obj, initial_type_id))

    # ⚠  Insertion order must match original ast_extractor.py exactly.
    # Node indices flow into edge_index; changing order shifts indices and
    # corrupts all edges, requiring a full dataset rebuild.
    _add_node(contract, NODE_TYPES["CONTRACT"])
    for var   in contract.state_variables: _add_node(var,   NODE_TYPES["STATE_VAR"])
    for func  in contract.functions:       _add_node(func,  NODE_TYPES["FUNCTION"])
    for mod   in contract.modifiers:       _add_node(mod,   NODE_TYPES["MODIFIER"])
    for event in contract.events:          _add_node(event, NODE_TYPES["EVENT"])

    if not node_features_list:
        raise EmptyGraphError(
            f"Contract '{contract.name}' produced zero graph nodes after filtering. "
            "Slither parsed the file but found no analyzable declarations."
        )

    # ── Feature tensor + dimension guard ──────────────────────────────────
    x = torch.tensor(node_features_list, dtype=torch.float)  # [N, 13]

    # Validate here — a dim mismatch surfaces at a meaningful boundary rather
    # than crashing deep inside GATConv with a cryptic matrix multiplication error.
    if x.shape[1] != NODE_FEATURE_DIM:
        raise SlitherParseError(
            f"Node feature dimension mismatch for '{contract.name}': "
            f"expected {NODE_FEATURE_DIM}, got {x.shape[1]}. "
            "This is a bug in _build_node_features() — each call must return "
            f"exactly {NODE_FEATURE_DIM} values."
        )

    # ── Declaration-level edge extraction ─────────────────────────────────
    edges:      list[list[int]] = []
    edge_types: list[int]       = []

    def _add_edge(src: str, dst: str, etype: int) -> None:
        """Add a directed edge if both endpoints exist in node_map."""
        si, di = node_map.get(src), node_map.get(dst)
        if si is not None and di is not None:
            edges.append([si, di])
            edge_types.append(etype)

    for func in contract.functions:
        fn = func.canonical_name

        # CALLS: func → each function it calls internally
        for call in func.internal_calls:
            if hasattr(call, "canonical_name"):
                _add_edge(fn, call.canonical_name, EDGE_TYPES["CALLS"])

        # READS / WRITES: func → state variables it accesses
        for var in func.state_variables_read:
            _add_edge(fn, var.canonical_name, EDGE_TYPES["READS"])
        for var in func.state_variables_written:
            _add_edge(fn, var.canonical_name, EDGE_TYPES["WRITES"])

        # EMITS: func → events it fires.
        # events_emitted is absent in older Slither releases — guard defensively.
        if hasattr(func, "events_emitted"):
            try:
                for evt in func.events_emitted:
                    _add_edge(fn, evt.canonical_name, EDGE_TYPES["EMITS"])
            except Exception:
                pass  # skip silently — missing event edges are non-critical

    # INHERITS: contract → each parent in the linearised MRO
    try:
        for parent in contract.inheritance:
            _add_edge(contract.name, parent.name, EDGE_TYPES["INHERITS"])
    except Exception:
        pass  # skip silently — older Slither versions may not expose .inheritance

    # ── CFG node extraction + CONTAINS / CONTROL_FLOW edges ───────────────
    # Each Slither FunctionNode (basic block) within each function becomes a
    # separate CFG_NODE graph node.  Two edge types connect CFG nodes:
    #
    #   CONTAINS(5):     function_node → cfg_node  (ownership)
    #   CONTROL_FLOW(6): cfg_node_a   → cfg_node_b (execution successor)
    #
    # Without CONTAINS, the CFG subgraph is disconnected from the rest of the
    # contract graph and GNN message passing cannot propagate function-level
    # properties (payable, visibility, external_call_count) into statement nodes.
    # Without CONTROL_FLOW, the model cannot distinguish "call before write" from
    # "write before call" — both produce identical graphs without ordered edges.
    cfg_features: list[list[float]] = []
    cfg_offset = len(node_features_list)  # CFG_NODE indices start here

    # cfg_key_to_local: maps cfg_key → index within cfg_features list
    cfg_key_to_local: dict[str, int] = {}

    try:
        for func in contract.functions:
            fn_key = func.canonical_name
            fn_idx = node_map.get(fn_key)
            if fn_idx is None:
                continue  # function not in node_map (e.g. inherited from dep)

            fn_nodes = list(getattr(func, "nodes", None) or [])
            if not fn_nodes:
                continue

            for fn_node in fn_nodes:
                node_id = getattr(fn_node, "node_id", None)
                if node_id is None:
                    continue

                cfg_key = f"{fn_key}::cfg::{node_id}"
                if cfg_key in cfg_key_to_local:
                    continue  # dedup (shouldn't happen but guard anyway)

                local_idx = len(cfg_features)
                cfg_key_to_local[cfg_key] = local_idx
                cfg_features.append(_build_cfg_node_features(fn_node, func))

                # CONTAINS: function_node → this cfg_node
                global_cfg_idx = cfg_offset + local_idx
                edges.append([fn_idx, global_cfg_idx])
                edge_types.append(EDGE_TYPES["CONTAINS"])

            # CONTROL_FLOW: cfg_node → successor cfg_node (execution order)
            for fn_node in fn_nodes:
                node_id = getattr(fn_node, "node_id", None)
                if node_id is None:
                    continue
                src_key = f"{fn_key}::cfg::{node_id}"
                src_local = cfg_key_to_local.get(src_key)
                if src_local is None:
                    continue

                sons = list(getattr(fn_node, "sons", None) or [])
                for son in sons:
                    son_id = getattr(son, "node_id", None)
                    if son_id is None:
                        continue
                    dst_key = f"{fn_key}::cfg::{son_id}"
                    dst_local = cfg_key_to_local.get(dst_key)
                    if dst_local is None:
                        continue
                    edges.append([
                        cfg_offset + src_local,
                        cfg_offset + dst_local,
                    ])
                    edge_types.append(EDGE_TYPES["CONTROL_FLOW"])

    except Exception as exc:
        # CFG extraction is best-effort; a failure here should not crash the
        # overall extraction. Log and continue with whatever nodes were built.
        logger.warning(
            "CFG extraction partially failed for '%s': %s — "
            "some CONTROL_FLOW edges may be missing.",
            contract.name,
            exc,
        )

    # ── Merge declaration nodes and CFG nodes ──────────────────────────────
    if cfg_features:
        cfg_tensor = torch.tensor(cfg_features, dtype=torch.float)
        x = torch.cat([x, cfg_tensor], dim=0)  # [N + C, 13]

    total_nodes = int(x.shape[0])

    # ── Assemble PyG Data object ───────────────────────────────────────────
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
        edge_attr  = torch.tensor(edge_types, dtype=torch.long)              # [E]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros(0, dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index)
    if config.include_edge_attr:
        graph.edge_attr = edge_attr

    graph.contract_name = contract.name
    graph.num_nodes     = total_nodes
    graph.num_edges     = len(edges)

    return graph
