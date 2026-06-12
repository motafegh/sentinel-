"""representation — thin adapters over `ml/src/preprocessing/`.

Stage 0 (2026-06-08): shipped a stub with hard-coded v9 constants + 3 latent
bugs (dict direction reversed, list instead of tuple). The stub is replaced
in Stage 2 (2026-06-10) by thin-adapter re-exports from `ml/`. The Stage 0
`STUB = True` flag is gone.

Stage 2: thin-adapter ports of `graph_schema.py` and `graph_extractor.py`.
The byte-identical regression test gates the port.
Stage 7: seam swap — sentinel-ml rebinds its import to this package;
        `ml/src/preprocessing/` is removed from the import path.

The thin-adapter pattern means there is ONE source of truth (`ml/`) and TWO
import names (`ml.src.preprocessing.X` and `sentinel_data.representation.X`).
Both resolve to the same Python object. Bug fixes apply once, in `ml/`.
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
from sentinel_data.representation.graph_extractor import (
    extract_contract_graph,
    GraphExtractionConfig,
    GraphExtractionError,
    SolcCompilationError,
    SlitherParseError,
    EmptyGraphError,
)

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
