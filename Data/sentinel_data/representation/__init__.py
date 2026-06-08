"""representation — graph schema constants + graph/token extraction stubs.

Stage 0: constants are live v9 values (copied by value from ml/src/preprocessing/graph_schema.py).
Stage 2: graph_schema.py and graph_extractor.py are ported from ml/ with a byte-identical
         regression test gating the port.
Stage 7: seam swap removes the stub flag and wires sentinel-ml to import from this path.
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
    STUB,
)

__all__ = [
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
    "STUB",
]
