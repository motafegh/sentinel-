"""
loader.py — Eager, fail-fast config loader.

Reads YAML → validates via Pydantic → caches singleton.
No hot-reload (proposal §10.1 reproducibility).

Resolution order:
  1. SENTINEL_CONFIG env var (path to a .yaml file)
  2. Default path: configs/verdicts_default.yaml relative to agents/
  3. If neither exists → SentinelConfig() with all defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import SentinelConfig

_CONFIG: SentinelConfig | None = None


def _resolve_config_path() -> Path | None:
    env_path = os.getenv("SENTINEL_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(
            f"SENTINEL_CONFIG={env_path} points to a non-existent file"
        )
    default = Path(__file__).parent.parent.parent / "configs" / "verdicts_default.yaml"
    return default if default.is_file() else None


def load_config(path: str | Path | None = None) -> SentinelConfig:
    raw: dict[str, Any]
    if path is not None:
        p = Path(path) if isinstance(path, str) else path
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return SentinelConfig(**raw)
    resolved = _resolve_config_path()
    if resolved is not None:
        with open(resolved, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return SentinelConfig(**raw)
    return SentinelConfig()


def get_config() -> SentinelConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reload_config(path: str | Path | None = None) -> SentinelConfig:
    """Force-reload (for tests). Resets the singleton."""
    global _CONFIG
    _CONFIG = load_config(path)
    return _CONFIG
