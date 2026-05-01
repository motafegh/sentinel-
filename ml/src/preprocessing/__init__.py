"""
ml.src.preprocessing — Shared Solidity graph preprocessing package.

Exports the canonical constants and extraction function used by both the
offline batch pipeline (ast_extractor.py) and the online inference API
(preprocess.py).  Importing from this package guarantees both pipelines
use identical feature engineering.
"""

from .graph_schema import (
    EDGE_TYPES,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_DIM,
    NODE_TYPES,
    NUM_EDGE_TYPES,
    VISIBILITY_MAP,
)
from .graph_extractor import (
    EmptyGraphError,
    GraphExtractionConfig,
    GraphExtractionError,
    SlitherParseError,
    SolcCompilationError,
    extract_contract_graph,
)

__all__ = [
    # schema constants
    "EDGE_TYPES",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "NODE_FEATURE_DIM",
    "NODE_TYPES",
    "NUM_EDGE_TYPES",
    "VISIBILITY_MAP",
    # extractor
    "EmptyGraphError",
    "GraphExtractionConfig",
    "GraphExtractionError",
    "SlitherParseError",
    "SolcCompilationError",
    "extract_contract_graph",
]
