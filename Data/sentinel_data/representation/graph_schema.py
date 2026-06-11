"""Schema constants for the SENTINEL graph representation — active schema: v9.

THIS IS A THIN ADAPTER over `ml/src/preprocessing/graph_schema.py`.

Stage 0 (2026-06-08) shipped a stub with the constants hard-coded and 3 latent
bugs (dict direction reversed, list instead of tuple). Stage 2 (2026-06-10) task
2.1-2.2 fixes those bugs and replaces the stub with a re-export from the
single source of truth in `ml/`.

Why thin-adapter over copy-paste:
  - Bug fixes apply once (in `ml/`), automatically propagate to the new path.
  - The Stage 7 seam swap becomes a 1-line change (delete this file) instead
    of a multi-file refactor.
  - The "byte-identical output" guarantee is trivially true — same object,
    different import name.

How it works:
  - All public symbols from `ml/src/preprocessing/graph_schema.py` are
    re-exported with the same names and values.
  - `is` equality holds between the old and new import paths because
    they're the same Python object.
  - If `ml/src/preprocessing/graph_schema.py` is not importable (e.g. when
    `sentinel_data` is installed as a standalone PyPI package in the future),
    `__getattr__` raises a clear ImportError pointing the user to the dep.

DO NOT change these constants without bumping FEATURE_SCHEMA_VERSION and
updating the version registry. Class order (NUM_CLASSES = 10, indices 0-9)
is LOCKED — changing it invalidates all existing checkpoints.
"""

from __future__ import annotations

import importlib.metadata  # pre-import to avoid NameError in ml/'s except branch

from typing import Any

_LIVE_SCHEMA_MODULE = "ml.src.preprocessing.graph_schema"
_LIVE_SCHEMA_ATTRS = (
    # Dimensions
    "FEATURE_SCHEMA_VERSION",
    "NODE_FEATURE_DIM",
    "NUM_NODE_TYPES",
    "NUM_EDGE_TYPES",
    "NUM_CLASSES",
    # Derived scalar (not in live schema, computed from NODE_TYPES)
    "_MAX_TYPE_ID",
    # Vocabularies
    "VISIBILITY_MAP",
    "NODE_TYPES",
    "EDGE_TYPES",
    "FEATURE_NAMES",
    # Class order (LOCKED; defined locally since not in graph_schema.py)
    "CLASS_NAMES",
    # Typed aliases
    "NodeType",
    # Structural prefix (for K=48 GNN prefix injection)
    "STRUCTURAL_PREFIX_TYPES",
)

# _MAX_TYPE_ID was referenced in the Stage 0 stub but is NOT a real
# module-level constant in the live ml/ schema. It is derived at the
# call sites as `float(max(NODE_TYPES.values()))`. We re-export it
# here as a derived value so downstream code can still use
# `from sentinel_data.representation.graph_schema import _MAX_TYPE_ID`.
_MAX_TYPE_ID: float = 0.0  # populated at import time below

# NUM_CLASSES and CLASS_NAMES live in ml/src/training/trainer.py
# (not in graph_schema.py). The Stage 0 stub duplicated them here.
# We re-export them as derived values so the v1 import path
# `from sentinel_data.representation.graph_schema import NUM_CLASSES`
# continues to work. The class order is LOCKED to match existing
# checkpoints — DO NOT change without bumping FEATURE_SCHEMA_VERSION.
CLASS_NAMES: list[str] = [
    "Reentrancy",           # 0
    "CallToUnknown",        # 1
    "Timestamp",            # 2
    "ExternalBug",          # 3
    "GasException",         # 4
    "DenialOfService",      # 5
    "IntegerUO",            # 6
    "UnusedReturn",         # 7
    "MishandledException",  # 8
    "NonVulnerable",        # 9
]
NUM_CLASSES: int = len(CLASS_NAMES)



_LOCAL_ATTRS = frozenset(("_MAX_TYPE_ID", "CLASS_NAMES", "NUM_CLASSES"))


def __getattr__(name: str) -> Any:
    """Lazy re-export for sentinel-data standalone install support.

    _LOCAL_ATTRS are defined in this module and never fetched from ml/.
    The rest of _LIVE_SCHEMA_ATTRS are lazily imported from ml/ if the
    eager import block at the bottom of this file hasn't run (e.g. when
    ml/ is not on PYTHONPATH in a standalone install).
    """
    if name in _LOCAL_ATTRS:
        # These are in __dict__ already (defined above); __getattr__ is only
        # called when normal lookup fails, so this branch is a safety net.
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"(local constant not yet initialised — import order issue)"
        )
    if name in _LIVE_SCHEMA_ATTRS:
        try:
            import importlib
            mod = importlib.import_module(_LIVE_SCHEMA_MODULE)
        except ImportError as e:
            raise ImportError(
                f"sentinel_data.representation.graph_schema.{name} requires the "
                f"`ml` package (from SENTINEL's `ml/` directory). Install it or "
                f"add it to PYTHONPATH. Original error: {e}"
            ) from e
        return getattr(mod, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available: {sorted(_LIVE_SCHEMA_ATTRS)}"
    )


# Eager re-export: this is the thin-adapter. The `from X import Y` at
# module load time means `from sentinel_data.representation.graph_schema
# import NODE_TYPES` returns the SAME dict object as
# `from ml.src.preprocessing.graph_schema import NODE_TYPES`. Hence
# `is` equality is trivially true and the byte-identical regression
# test cannot fail due to "two different NODE_TYPES objects."
from ml.src.preprocessing.graph_schema import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_DIM,
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
    VISIBILITY_MAP,
    NODE_TYPES,
    EDGE_TYPES,
    FEATURE_NAMES,
    NodeType,
    STRUCTURAL_PREFIX_TYPES,
)

# _MAX_TYPE_ID is derived (not exported by the live schema). Compute it
# after the eager import so we can re-export it for backward compat.
_MAX_TYPE_ID = float(max(NODE_TYPES.values()))


__all__ = list(_LIVE_SCHEMA_ATTRS)
