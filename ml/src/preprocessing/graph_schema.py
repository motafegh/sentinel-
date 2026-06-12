# Stage 7B seam swap (2026-06-12): sentinel_data.representation.graph_schema is now the
# source of truth. This file is kept as a thin re-export shim for backward compatibility
# with model files (gnn_encoder.py, sentinel_model.py, predictor.py, training_logger.py,
# cache.py, preprocess.py, graph_extractor.py) — they continue to work without changes.
#
# To change schema constants: edit sentinel_data/representation/graph_schema.py, then
# bump FEATURE_SCHEMA_VERSION and re-export here if needed.
from sentinel_data.representation.graph_schema import (  # noqa: F401
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_DIM,
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
    VISIBILITY_MAP,
    NODE_TYPES,
    EDGE_TYPES,
    FEATURE_NAMES,
    CLASS_NAMES,
    NUM_CLASSES,
    _MAX_TYPE_ID,
    NodeType,
    STRUCTURAL_PREFIX_TYPES,
)
