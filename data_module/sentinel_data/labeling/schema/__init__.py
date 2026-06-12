"""Canonical 10-class taxonomy loader."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_TAXONOMY_PATH = Path(__file__).parent / "taxonomy.yaml"


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    """Load and return the taxonomy YAML (cached after first call)."""
    with open(_TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def class_names() -> list[str]:
    """Return the 10 class names in locked index order (index 0–9)."""
    return [c["name"] for c in load_taxonomy()["classes"]]


def class_index(name: str) -> int:
    """Return the integer index for a class name. Raises KeyError if unknown."""
    names = class_names()
    try:
        return names.index(name)
    except ValueError:
        raise KeyError(f"Unknown class '{name}'. Valid classes: {names}")
