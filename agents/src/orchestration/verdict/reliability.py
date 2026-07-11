"""
reliability.py — Per-(source, class) reliability lookup (P2 placeholder → P3 learned).

P3 (B-3 / D-C) fits these from a confusion matrix on the labeled benchmark.
When `configs/reliability_v1.yaml` exists and its `schema_version` matches,
this module returns the L3 data-derived fitted values. When it does not
exist or has a different schema version, it falls back to the L1 hand-set
`accuracy_weights` from `verdicts_default.yaml` (P2 placeholder).

The ML weight is still scaled by `ml_weight_scale` regardless of source
(L1 or L3) so the consensus engine's discount behaviour is preserved.
This matches the existing pattern in `consensus.py::get_weights`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# L3 (P3 fitted) config — path is relative to the agents/ project root.
# The build_reliability_matrix.py / reliability_fit.py scripts write
# `configs/reliability_v1.yaml`; this loader reads it when present.
L3_RELIABILITY_PATH: str = os.getenv(
    "SENTINEL_RELIABILITY_CONFIG",
    "configs/reliability_v1.yaml",
)
L3_EXPECTED_SCHEMA: str = "1"


def _load_l3_table() -> dict[tuple[str, str], float] | None:
    """
    Try to load the L3 data-derived reliability table from
    `configs/reliability_v1.yaml`.

    Returns:
        dict[(source, cls), weight] on success, or None if the file is
        missing, has a different schema version, or is malformed.
        Malformed YAML falls back to L1 — never raises (per Rule 5C: a
        missing or broken fitted config does not silently produce the
        wrong number; it falls back to the documented L1 prior, and the
        caller can detect the fallback by comparing the returned dict
        against a re-loaded L1).
    """
    path = Path(L3_RELIABILITY_PATH)
    if not path.is_file():
        return None
    try:
        doc = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("schema_version") != L3_EXPECTED_SCHEMA:
        return None
    table_raw = doc.get("table")
    if not isinstance(table_raw, dict):
        return None
    out: dict[tuple[str, str], float] = {}
    for source, by_cls in table_raw.items():
        if not isinstance(by_cls, dict):
            continue
        for cls, val in by_cls.items():
            try:
                out[(source, cls)] = float(val)
            except (TypeError, ValueError):
                continue
    return out


def load_reliability(config=None) -> dict[tuple[str, str], float]:
    """
    Build the reliability lookup table: {(source, class): weight}.

    Resolution order:
        1. `configs/reliability_v1.yaml` (L3, fitted) — if present + valid.
        2. `verdicts_default.yaml::consensus.accuracy_weights` (L1, hand-set).
           ML weight is scaled by `ml_weight_scale` (consensus discount).
        3. `verdicts_default.yaml::consensus.default_weights` (L0 fallback
           for unknown (source, class) pairs).

    Rule 5C: missing or malformed L3 config falls back to L1 (the
    documented prior). The L1 value is the same value that was in effect
    before P3 — never a fabrication. A caller can detect the fallback
    by re-loading the L3 path (None returned) and surfacing a `tool_status`
    entry if needed.
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    scale = config.consensus.ml_weight_scale
    acc = config.consensus.accuracy_weights
    defaults = config.consensus.default_weights

    # 1) Try L3 fitted table.
    l3 = _load_l3_table()
    if l3 is not None:
        # Apply ml_weight_scale to ML values for consistency with the
        # consensus engine's discount pattern (the L3 matrix builder does
        # not apply the scale — it's a downstream discount for ML).
        return {
            (src, cls): round(val * scale, 4) if src == "ml" else round(val, 4)
            for (src, cls), val in l3.items()
        }

    # 2) Fall back to L1 hand-set values.
    table: dict[tuple[str, str], float] = {}
    all_classes = set(acc.keys())
    for cls in all_classes:
        w = acc.get(cls, defaults)
        table[("ml", cls)] = round(w["ml"] * scale, 4)
        table[("slither", cls)] = w["slither"]
        table[("aderyn", cls)] = w["aderyn"]
    return table


def get_reliability(source: str, cls: str, config=None) -> float:
    """Single lookup with a fallback to the default weight."""
    table = load_reliability(config)

    if (source, cls) in table:
        return table[(source, cls)]

    # Fallback: use default weights from config
    if config is None:
        from src.config import get_config
        config = get_config()

    defaults = config.consensus.default_weights
    scale = config.consensus.ml_weight_scale
    if source == "ml":
        return round(defaults.get("ml", 0.60) * scale, 4)
    if source in defaults:
        return defaults[source]
    # Hard fallback for unknown sources (RAG, debate, etc.)
    _source_defaults = {"rag": 0.50, "debate": 0.55, "quick_screen": 0.40}
    return _source_defaults.get(source, 0.50)
