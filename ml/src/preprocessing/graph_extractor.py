"""
graph_extractor.py — Canonical Solidity-to-PyG graph extraction

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

SHAPE CONTRACT  (must match training data — do not change without retraining)
──────────────────────────────────────────────────────────────────────────────
  graph.x           [N, 8]  float32  node feature matrix (NODE_FEATURE_DIM=8)
  graph.edge_index  [2, E]  int64    edge connectivity in COO format
  graph.edge_attr   [E]     int64    edge type IDs — 1-D per PyG convention
                                     (only attached when config.include_edge_attr)

  graph.contract_name  str   — name of the analysed Slither Contract object
  graph.num_nodes      int   — N (same as x.shape[0])
  graph.num_edges      int   — E (same as edge_index.shape[1])

  Caller-specific metadata (.contract_hash, .contract_path, .y) is NOT set
  here; each caller attaches its own values after the call returns.

EDGE ATTR SHAPE NOTE
────────────────────
Pre-refactor ast_extractor.py stored edge_attr with shape [E, 1].
This module uses shape [E] (PyG 1-D convention for scalar attributes).
GNNEncoder ignores edge_attr entirely, so both shapes are safe for the current
model. The inconsistency between old .pt files and new ones is documented and
harmless. When GATv2/RGAT support is added, a migration step will be needed.
"""

from __future__ import annotations

import logging
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

def _build_node_features(obj: Any, type_id: int) -> list[float]:
    """
    Compute the 8-dimensional feature vector for one AST node.

    ⚠  REPLICATION CONSTRAINT — DO NOT CHANGE WITHOUT RETRAINING ⚠
    ─────────────────────────────────────────────────────────────────
    This function replicates node_features() from the pre-refactor
    ml/src/data_extraction/ast_extractor.py (ASTExtractorV4) EXACTLY.
    That script built all ~68K training .pt files. The checkpoint has
    GNNEncoder(in_channels=8) trained on those precise float vectors.

    Any change here — including a reordering, a new feature, or a different
    visibility default — requires a full dataset rebuild and model retrain.
    See graph_schema.py CHANGE POLICY for the required steps.

    Feature layout (see graph_schema.FEATURE_NAMES for the name-indexed view):

      Index  Name         Value
      ─────  ──────────── ──────────────────────────────────────────────────
        0    type_id      float(NODE_TYPES[kind])  — overridden for special
                          Function sub-kinds (CONSTRUCTOR, FALLBACK, RECEIVE)
        1    visibility   VISIBILITY_MAP ordinal 0–2; default 0 (public)
        2    pure         1.0 if Function.pure, else 0.0
        3    view         1.0 if Function.view, else 0.0
        4    payable      1.0 if Function.payable, else 0.0
        5    reentrant    1.0 if Slither marks is_reentrant, else 0.0
        6    complexity   float(len(func.nodes)) — CFG basic-block count
        7    loc          float(len(source_mapping.lines)) — lines of code

    Non-Function nodes (STATE_VAR, EVENT, MODIFIER, CONTRACT) receive 0.0
    for indices 2–6; their role in the graph is captured by type_id alone.

    Args:
        obj:     A Slither AST declaration object (Contract, Function, etc.).
        type_id: Initial node type ID from NODE_TYPES. May be overridden
                 internally for CONSTRUCTOR / FALLBACK / RECEIVE functions.

    Returns:
        List of exactly NODE_FEATURE_DIM (8) floats.
    """
    from slither.core.declarations import Function  # lazy: Slither is optional dep

    visibility = float(VISIBILITY_MAP.get(str(getattr(obj, "visibility", "public")), 0))
    pure = view = payable = reentrant = complexity = loc = 0.0

    src = getattr(obj, "source_mapping", None)
    if src is not None:
        lines = getattr(src, "lines", None)
        if lines:
            loc = float(len(lines) if isinstance(lines, list) else lines)

    if isinstance(obj, Function):
        pure      = 1.0 if obj.pure    else 0.0
        view      = 1.0 if obj.view    else 0.0
        payable   = 1.0 if obj.payable else 0.0
        reentrant = 1.0 if getattr(obj, "is_reentrant", False) else 0.0
        try:
            complexity = float(len(obj.nodes)) if obj.nodes else 0.0
        except Exception:
            complexity = 0.0

        # Special function kinds override the caller-supplied FUNCTION(1) type_id.
        # This must stay in sync with ast_extractor.py — both apply the same override.
        if obj.is_constructor:
            type_id = NODE_TYPES["CONSTRUCTOR"]
        elif obj.is_fallback:
            type_id = NODE_TYPES["FALLBACK"]
        elif obj.is_receive:
            type_id = NODE_TYPES["RECEIVE"]

    return [float(type_id), visibility, pure, view, payable, reentrant, complexity, loc]


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
        PyG Data with the following attributes:
          .x              Tensor [N, 8]  float32  node features
          .edge_index     Tensor [2, E]  int64    COO edge connectivity
          .edge_attr      Tensor [E]     int64    edge type IDs
                                                  (only if config.include_edge_attr)
          .contract_name  str            name of the analysed contract
          .num_nodes      int            N
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
    x = torch.tensor(node_features_list, dtype=torch.float)  # [N, 8]

    # Validate here — a dim mismatch surfaces at a meaningful boundary rather
    # than crashing deep inside GATConv with a cryptic matrix multiplication error.
    if x.shape[1] != NODE_FEATURE_DIM:
        raise SlitherParseError(
            f"Node feature dimension mismatch for '{contract.name}': "
            f"expected {NODE_FEATURE_DIM}, got {x.shape[1]}. "
            "This is a bug in _build_node_features() — each call must return "
            f"exactly {NODE_FEATURE_DIM} values."
        )

    # ── Edge extraction ────────────────────────────────────────────────────
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
    graph.num_nodes     = int(x.shape[0])
    graph.num_edges     = len(edges)

    return graph
