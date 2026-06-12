"""Standalone CFG builder for SENTINEL v2 — opt-in via ``--emit-cfg``.

Produces a normalized per-function control-flow graph (CFG) as a JSON-
serializable ``CfgArtifact``.  The CFG is a *standalone* artifact alongside
the PyG graph (``<sha256>.pt``); it is NOT merged into the training graph.
Its primary consumer is Stage 6's ``complexity_proxy_risk`` detector, which
needs a flat CFG view to spot loops, dead code, and deeply nested branches.

Public API
----------
build_cfg(sol_path, config) -> CfgArtifact | None

Data structures
---------------
CfgNode    — one Slither CFG node, normalised.
CfgEdge    — one control-flow edge between two CfgNode indices.
CfgFunction — per-function CFG (nodes + edges).
CfgArtifact — top-level result; JSON-serialisable via dataclasses.asdict().

The CFG is keyed on ``canonical_name`` (Slither's stable cross-run identifier)
so that Stage 6 can join on the function name without additional mapping.

Design decisions
----------------
- Uses Slither's ``func.nodes`` + ``node.sons`` directly — same IR that
  ``graph_extractor.py`` already imports, so no new Slither API surface.
- Node types use the same ``NODE_TYPES`` string keys as ``graph_schema.py``
  for consistency, but stored as strings (not ints) in the artifact so it is
  self-describing.
- This module is INTENTIONALLY opt-in; calling ``build_cfg`` starts Slither
  which adds ~0.5 s per contract.  The orchestrator only calls it when
  ``emit_cfg=True`` (default: False).
- Loop detection uses DFS-back-edge counting — O(N+E) per function.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
#  Data structures                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class CfgNode:
    """One normalized Slither CFG node.

    Attributes:
        index: Positional index within the function's sorted node list.
        type: Category string (e.g. ``"CFG_NODE_WRITE"``, ``"CFG_NODE_CALL"``).
        source_lines: Slither source-mapping line numbers (may be empty).
        expression: Human-readable label from ``str(slither_node)``.
    """
    index: int
    type: str                    # e.g. "CFG_NODE_WRITE", "CFG_NODE_OTHER"
    source_lines: list[int]      # Slither source_mapping.lines (may be empty)
    expression: str              # str(slither_node) — human-readable label


@dataclass
class CfgEdge:
    """A single control-flow edge between two ``CfgNode`` indices."""
    src: int
    dst: int


@dataclass
class CfgFunction:
    """Per-function CFG: normalized nodes, edges, and structural metrics.

    Attributes:
        canonical_name: Slither's stable cross-run function identifier.
        nodes: Sorted list of ``CfgNode`` objects.
        edges: Control-flow edges between node indices.
        num_loops: Number of back-edges detected via DFS (loop count).
        max_depth: Longest path from the entrypoint to any terminal node.
    """
    canonical_name: str
    nodes: list[CfgNode]
    edges: list[CfgEdge]
    num_loops: int               # number of back-edges (DFS)
    max_depth: int               # longest path from ENTRYPOINT to any terminal


@dataclass
class CfgArtifact:
    """Top-level CFG result for a single contract; JSON-serializable via ``asdict()``.

    Attributes:
        sha256: Content hash of the source file (from Stage 1 meta.json).
        source: Dataset source label (e.g. ``"solidifi"``).
        solc_version: Solidity compiler version used for Slither analysis.
        schema_version: ``FEATURE_SCHEMA_VERSION`` at the time of extraction.
        extractor_version: ``EXTRACTOR_VERSION`` at the time of extraction.
        functions: Per-function CFGs (empty on Slither failure).
        error: Error message if Slither parsing failed; ``None`` on success.
    """
    sha256: str
    source: str
    solc_version: str
    schema_version: str
    extractor_version: str
    functions: list[CfgFunction] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary via ``dataclasses.asdict``."""
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _cfg_node_type_str(slither_node: Any) -> str:
    """Map a Slither node to a NODE_TYPES string key."""
    try:
        from slither.slithir.operations import LowLevelCall, HighLevelCall, Transfer, Send
        from slither.core.cfg.node import NodeType as SNT
        from slither.core.variables.state_variable import StateVariable

        irs = list(getattr(slither_node, "irs", None) or [])

        if any(isinstance(op, (LowLevelCall, HighLevelCall, Transfer, Send)) for op in irs):
            return "CFG_NODE_CALL"
        if getattr(slither_node, "state_variables_written", None):
            return "CFG_NODE_WRITE"
        if getattr(slither_node, "state_variables_read", None):
            return "CFG_NODE_READ"

        nt = getattr(slither_node, "type", None)
        if nt in (getattr(SNT, "IF", None), getattr(SNT, "IFLOOP", None)):
            return "CFG_NODE_CHECK"

        try:
            from slither.slithir.operations import Binary, BinaryType
            if any(
                isinstance(op, Binary) and op.type in (
                    BinaryType.ADDITION, BinaryType.SUBTRACTION,
                    BinaryType.MULTIPLICATION, BinaryType.DIVISION,
                    BinaryType.MODULO, BinaryType.POWER,
                    BinaryType.LEFT_SHIFT, BinaryType.RIGHT_SHIFT,
                )
                for op in irs
            ):
                return "CFG_NODE_ARITH"
        except Exception:
            pass
    except Exception:
        pass
    return "CFG_NODE_OTHER"


def _count_back_edges(nodes: list[Any]) -> int:
    """Count back-edges (loops) via DFS on Slither node graph."""
    visited: set[int] = set()
    stack: set[int] = set()
    back_edges = 0

    def dfs(n: Any) -> None:
        nonlocal back_edges
        nid = id(n)
        if nid in stack:
            back_edges += 1
            return
        if nid in visited:
            return
        visited.add(nid)
        stack.add(nid)
        for son in (getattr(n, "sons", None) or []):
            dfs(son)
        stack.discard(nid)

    for n in nodes:
        if n not in visited:
            dfs(n)
    return back_edges


def _max_depth_from_entry(entry_node: Any) -> int:
    """BFS longest path from entry_node (DAG approximation — ignores back-edges)."""
    from collections import deque
    if entry_node is None:
        return 0
    visited: set[int] = set()
    queue: deque = deque([(entry_node, 0)])
    max_d = 0
    while queue:
        node, depth = queue.popleft()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        max_d = max(max_d, depth)
        for son in (getattr(node, "sons", None) or []):
            if id(son) not in visited:
                queue.append((son, depth + 1))
    return max_d


def _build_function_cfg(func: Any) -> CfgFunction:
    """Build a normalized ``CfgFunction`` from a Slither function object.

    Nodes are sorted by source line then node ID for deterministic ordering.
    Back-edge and depth metrics are computed via DFS/BFS on the Slither graph.
    """
    nodes_raw = sorted(
        getattr(func, "nodes", None) or [],
        key=lambda n: (
            n.source_mapping.lines[0]
            if n.source_mapping and n.source_mapping.lines else 0,
            getattr(n, "node_id", 0),
        ),
    )

    # Assign indices
    node_to_idx: dict[int, int] = {id(n): i for i, n in enumerate(nodes_raw)}

    cfg_nodes = [
        CfgNode(
            index=i,
            type=_cfg_node_type_str(n),
            source_lines=list(n.source_mapping.lines) if n.source_mapping and n.source_mapping.lines else [],
            expression=str(n)[:120],
        )
        for i, n in enumerate(nodes_raw)
    ]

    cfg_edges = []
    for n in nodes_raw:
        src = node_to_idx[id(n)]
        for son in (getattr(n, "sons", None) or []):
            dst = node_to_idx.get(id(son))
            if dst is not None:
                cfg_edges.append(CfgEdge(src=src, dst=dst))

    # Entry node for depth calculation
    try:
        from slither.core.cfg.node import NodeType as SNT
        entry = next((n for n in nodes_raw if n.type == SNT.ENTRYPOINT), None)
    except Exception:
        entry = nodes_raw[0] if nodes_raw else None

    return CfgFunction(
        canonical_name=getattr(func, "canonical_name", None) or func.name,
        nodes=cfg_nodes,
        edges=cfg_edges,
        num_loops=_count_back_edges(nodes_raw),
        max_depth=_max_depth_from_entry(entry),
    )


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def build_cfg(
    sol_path: str | Path,
    config: Any,
    sha256: str = "",
    source: str = "",
) -> CfgArtifact:
    """Build a normalized per-function CFG for every contract in ``sol_path``.

    Args:
        sol_path: Path to a preprocessed ``.sol`` file.
        config:   ``GraphExtractionConfig`` (provides ``solc_binary`` +
                  ``solc_version`` + ``allow_paths``).
        sha256:   Content hash of the source file (from Stage 1 meta.json).
        source:   Dataset source label (e.g. ``"solidifi"``).

    Returns:
        ``CfgArtifact`` — JSON-serialisable.  On Slither failure, the
        ``error`` field is set and ``functions`` is empty.
    """
    from ml.src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION
    from sentinel_data.representation.orchestrator import EXTRACTOR_VERSION

    artifact = CfgArtifact(
        sha256=sha256,
        source=source,
        solc_version=getattr(config, "solc_version", ""),
        schema_version=FEATURE_SCHEMA_VERSION,
        extractor_version=EXTRACTOR_VERSION,
    )

    try:
        from slither import Slither
        solc_bin  = str(getattr(config, "solc_binary", "solc"))
        allow_paths = [str(p) for p in (getattr(config, "allow_paths", None) or [])]
        slither_kwargs: dict[str, Any] = {"solc": solc_bin, "solc_args": ""}
        if allow_paths:
            slither_kwargs["solc_args"] = f"--allow-paths {','.join(allow_paths)}"

        sl = Slither(str(sol_path), **slither_kwargs)

        for contract in sl.contracts:
            for func in (list(contract.functions) + list(getattr(contract, "modifiers", []))):
                nodes = getattr(func, "nodes", None) or []
                if not nodes:
                    continue
                artifact.functions.append(_build_function_cfg(func))

    except Exception as exc:
        artifact.error = str(exc)

    return artifact
