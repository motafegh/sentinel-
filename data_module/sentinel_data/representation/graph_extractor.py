"""Graph extractor — thin adapter over `ml/src/preprocessing/graph_extractor.py`.

Stage 2 (2026-06-10) thin-adapter port.

The real implementation lives at `ml/src/preprocessing/graph_extractor.py`.
This file re-exports every public symbol from there. Stage 7 deletes this
file and rebinds the active training pipeline to import from
`sentinel_data.representation.graph_extractor` directly.

Why thin-adapter:
  - Zero code duplication. Bug fixes in `ml/` propagate to the new path
    automatically.
  - The Stage 7 seam swap is a 1-line change (delete this file) instead
    of a multi-file refactor.
  - The byte-identical-output guarantee is trivially true — same object,
    different import name.

What's NOT here (deferred):
  - The v1 `ast_extractor.py` parquet-orchestrator is NOT ported. The v2
    orchestrator is a NEW file (`sentinel_data/representation/orchestrator.py`,
    task 2.4 in the Stage 2 plan) that reads Stage 1's preprocessed output.
  - The CFG / PDG / call-graph / opcode builders are NOT here. Only the
    CFG builder ships in Stage 2 (task 2.7); the others are v3.1.

Lazy import support: `__getattr__` falls back to lazy import if `ml/` is
not on the Python path. This lets `sentinel-data` be installed as a
standalone PyPI package in the future.
"""

from __future__ import annotations

from typing import Any

_LIVE_EXTRACTOR_MODULE = "ml.src.preprocessing.graph_extractor"
_LIVE_EXTRACTOR_ATTRS = (
    # Public functions
    "extract_contract_graph",
    # Configuration / error types
    "GraphExtractionConfig",
    "GraphExtractionError",
    "SolcCompilationError",
    "SlitherParseError",
    "EmptyGraphError",
)


def __getattr__(name: str) -> Any:
    """Lazy re-export for sentinel-data standalone install support."""
    if name in _LIVE_EXTRACTOR_ATTRS:
        try:
            import importlib
            mod = importlib.import_module(_LIVE_EXTRACTOR_MODULE)
        except ImportError as e:
            raise ImportError(
                f"sentinel_data.representation.graph_extractor.{name} requires the "
                f"`ml` package (from SENTINEL's `ml/` directory). Install it or "
                f"add it to PYTHONPATH. Original error: {e}"
            ) from e
        return getattr(mod, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available: {sorted(_LIVE_EXTRACTOR_ATTRS)}"
    )


# Eager re-export — see graph_schema.py for the rationale.
from ml.src.preprocessing.graph_extractor import (  # noqa: E402
    extract_contract_graph,
    GraphExtractionConfig,
    GraphExtractionError,
    SolcCompilationError,
    SlitherParseError,
    EmptyGraphError,
)


__all__ = list(_LIVE_EXTRACTOR_ATTRS)
