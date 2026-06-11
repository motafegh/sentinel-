"""Tokenizer — thin adapter over `ml/src/data_extraction/windowed_tokenizer.py`.

Stage 2 (2026-06-10) thin-adapter port (Task 2.5).
Corrected (2026-06-11): previous version pointed at the wrong module.

The v2 tokenizer uses GraphCodeBERT (microsoft/graphcodebert-base) with a
sliding-window approach (stride=256, up to 4 windows of 512 tokens each),
producing [max_windows, 512] tensors. This matches the existing v1 training
data in ml/data/tokens_windowed/ and the model's dual_path_dataset.py, which
expects either [512] (single-window legacy) or [W, 512] (multi-window).

The WRONG module (ml/src/data_extraction/tokenizer.py) was previously used.
That module uses microsoft/codebert-base and outputs (512,) single-window
tensors — wrong model, wrong shape. It is kept for v1 batch-script use only.

The CORRECT module (ml/src/data_extraction/windowed_tokenizer.py) exports:
  - tokenize_windowed_contract(contract_path, max_windows) → dict | None
  - init_worker()
  - TOKENIZER_MODEL, WINDOW_SIZE, STRIDE, MAX_WINDOWS

Why thin-adapter:
  - Zero code duplication. Bug fixes in ml/ propagate to the new path.
  - The Stage 7 seam swap becomes a 1-line change.
  - The byte-identical guarantee: same graphcodebert-base model + same params.
"""

from __future__ import annotations

from typing import Any

_LIVE_TOKENIZER_MODULE = "ml.src.data_extraction.windowed_tokenizer"
_LIVE_TOKENIZER_ATTRS = (
    "tokenize_windowed_contract",
    "init_worker",
    "TOKENIZER_MODEL",
    "WINDOW_SIZE",
    "STRIDE",
    "MAX_WINDOWS",
)


def __getattr__(name: str) -> Any:
    """Lazy re-export for sentinel-data standalone install support."""
    if name in _LIVE_TOKENIZER_ATTRS:
        try:
            import importlib
            mod = importlib.import_module(_LIVE_TOKENIZER_MODULE)
        except ImportError as e:
            raise ImportError(
                f"sentinel_data.representation.tokenizer.{name} requires the "
                f"`ml` package (from SENTINEL's `ml/` directory). Install it or "
                f"add it to PYTHONPATH. Original error: {e}"
            ) from e
        return getattr(mod, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available: {sorted(_LIVE_TOKENIZER_ATTRS)}"
    )


# Eager re-export — see graph_schema.py for the rationale.
from ml.src.data_extraction.windowed_tokenizer import (  # noqa: E402
    tokenize_windowed_contract,
    init_worker,
    TOKENIZER_MODEL,
    WINDOW_SIZE,
    STRIDE,
    MAX_WINDOWS,
)


__all__ = list(_LIVE_TOKENIZER_ATTRS)
