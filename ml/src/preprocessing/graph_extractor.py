# Stage 7B seam swap (2026-06-15): sentinel_data.representation.graph_extractor is now the
# source of truth. This file is kept as a thin re-export shim for backward compatibility
# with model files (preprocess.py, and any other callers using the ml.src.preprocessing path)
# — they continue to work without changes.
#
# __getattr__ proxies any attribute lookup (including private names) to the canonical module,
# so code that does `getattr(graph_extractor, "_private_fn")` still works transparently.
from __future__ import annotations

from sentinel_data.representation.graph_extractor import (  # noqa: F401
    EmptyGraphError,
    GraphExtractionConfig,
    GraphExtractionError,
    SlitherParseError,
    SolcCompilationError,
    extract_contract_graph,
)
from sentinel_data.representation import graph_extractor as _canonical  # noqa: F401


def __getattr__(name: str):
    """Proxy any attribute not explicitly imported here to the canonical module."""
    try:
        return getattr(_canonical, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
