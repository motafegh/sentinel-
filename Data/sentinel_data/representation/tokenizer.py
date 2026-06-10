"""Tokenizer — thin adapter over `ml/src/data_extraction/tokenizer.py`.

Stage 2 (2026-06-10) Day 2 thin-adapter port (Task 2.5).

The per-file tokenization function lives at
`ml/src/data_extraction/tokenizer.py`. This file re-exports the
function-level surface (constants, init_worker, tokenize_single_contract,
save_token_file). The v1 parquet-orchestrator (process_batch_with_checkpoint)
is NOT ported — the v2 orchestrator (`sentinel_data/representation/orchestrator.py`)
handles batching using Stage 1's preprocessed output.

Why thin-adapter:
  - The v1 tokenizer uses `get_contract_hash` from `ml/src/utils/hash_utils.py`
    which is MD5. The v2 build drops MD5 in favor of SHA-256 from Stage 1.
    The thin adapter preserves the function unchanged; the v2 orchestrator
    computes SHA-256 separately and overwrites the `contract_hash` field.
  - Zero code duplication. Bug fixes in ml/ propagate to the new path.
  - The byte-identical tokenization guarantee is trivially true (same
    CodeBERT model + same params).

For the FULL set of symbols exported here, see `_LIVE_TOKENIZER_ATTRS`
below. The lazy __getattr__ falls back to lazy import if ml/ is not on
the Python path (e.g. when sentinel-data is installed as a standalone
PyPI package).
"""

from __future__ import annotations

from typing import Any

_LIVE_TOKENIZER_MODULE = "ml.src.data_extraction.tokenizer"
_LIVE_TOKENIZER_ATTRS = (
    # Per-file function (the value of the v1 tokenizer)
    "tokenize_single_contract",
    # Worker init (must be called per-process)
    "init_worker",
    # Save function (per-file)
    "save_token_file",
    # Module-level constants
    "TOKENIZER_MODEL",
    "MAX_LENGTH",
    "PADDING",
    "TRUNCATION",
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
from ml.src.data_extraction.tokenizer import (  # noqa: E402
    tokenize_single_contract,
    init_worker,
    save_token_file,
    TOKENIZER_MODEL,
    MAX_LENGTH,
    PADDING,
    TRUNCATION,
)


__all__ = list(_LIVE_TOKENIZER_ATTRS)
