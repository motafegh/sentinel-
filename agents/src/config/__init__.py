"""
config — Decision-number externalisation (P1, 2026-06-23).

YAML-backed, Pydantic-validated, eagerly loaded on first access.
Singleton per process; no hot-reload (proposal §10.1).

Usage:
    from src.config import get_config
    cfg = get_config()
    print(cfg.consensus.confirmed_band)
"""

from __future__ import annotations

from src.config.loader import get_config, SentinelConfig

__all__ = ["get_config", "SentinelConfig"]
