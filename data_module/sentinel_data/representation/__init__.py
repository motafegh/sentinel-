"""representation — canonical Solidity graph extraction and schema surfaces.

Stage 0 (2026-06-08): shipped a stub with hard-coded v9 constants + 3 latent
bugs (dict direction reversed, list instead of tuple). The stub is replaced
in Stage 2 (2026-06-10) by thin-adapter re-exports. Stage 7 completed the seam
swap so ``sentinel_data.representation`` is the canonical implementation used
by ML consumers.

R4 Phase-8 keeps historical v9 extraction immutable while V10 evolves through
explicit extractor identities. The V10 extraction guard installed here is
version-aware: it activates deterministic CFG WRITE supplementation only when
``GraphExtractionConfig.graph_schema_version == "v10"`` and leaves v9 calls on
the historical classifier.
"""

from sentinel_data.representation.graph_schema import (
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_DIM,
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
    _MAX_TYPE_ID,
    NUM_CLASSES,
    VISIBILITY_MAP,
    NODE_TYPES,
    EDGE_TYPES,
    FEATURE_NAMES,
    CLASS_NAMES,
    NodeType,
    STRUCTURAL_PREFIX_TYPES,
)
from sentinel_data.representation import graph_extractor as _graph_extractor
from sentinel_data.representation.v10_cfg_determinism import (
    install_v10_extraction_guard,
)

# Install once after the canonical extractor module has loaded. The wrapper is
# inert for v9 and restores the underlying CFG classifier after each V10 call.
install_v10_extraction_guard()

extract_contract_graph = _graph_extractor.extract_contract_graph
GraphExtractionConfig = _graph_extractor.GraphExtractionConfig
GraphExtractionError = _graph_extractor.GraphExtractionError
SolcCompilationError = _graph_extractor.SolcCompilationError
SlitherParseError = _graph_extractor.SlitherParseError
EmptyGraphError = _graph_extractor.EmptyGraphError

__all__ = [
    # Schema
    "FEATURE_SCHEMA_VERSION",
    "NODE_FEATURE_DIM",
    "NUM_NODE_TYPES",
    "NUM_EDGE_TYPES",
    "_MAX_TYPE_ID",
    "NUM_CLASSES",
    "VISIBILITY_MAP",
    "NODE_TYPES",
    "EDGE_TYPES",
    "FEATURE_NAMES",
    "CLASS_NAMES",
    "NodeType",
    "STRUCTURAL_PREFIX_TYPES",
    # Extractor
    "extract_contract_graph",
    "GraphExtractionConfig",
    "GraphExtractionError",
    "SolcCompilationError",
    "SlitherParseError",
    "EmptyGraphError",
]
