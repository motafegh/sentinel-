"""Schema and extractor version registry for SENTINEL v2.

The versioner records the current (schema_version, extractor_version) pair
in ``data/representations/_version_registry.json``.  On each run the
orchestrator checks whether the registry matches the live schema; if not, it
evicts stale cache entries before starting extraction.

This prevents the "silent mix of versions" failure mode that occurred in
Run 8 (v8 graphs mixed with v9 graphs in the same training dataset).

Registry format
---------------
{
  "schema_version":    "<FEATURE_SCHEMA_VERSION>",
  "extractor_version": "<EXTRACTOR_VERSION>",
  "updated_at":        "<ISO-8601 timestamp>"
}

Public surface
--------------
read_registry(representations_root)                         -> dict
write_registry(representations_root, schema_v, extractor_v) -> None
check_and_evict(representations_root, source, schema_v, extractor_v) -> int
current_versions()                                          -> tuple[str, str]
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("sentinel_data.versioner")

_REGISTRY_FILENAME = "_version_registry.json"


def read_registry(representations_root: Path) -> dict:
    """Load the version registry; return {} if absent or unreadable."""
    p = representations_root / _REGISTRY_FILENAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def write_registry(
    representations_root: Path,
    schema_version: str,
    extractor_version: str,
) -> None:
    """Write the current versions to the registry file."""
    representations_root.mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version":    schema_version,
        "extractor_version": extractor_version,
        "updated_at":        datetime.now(timezone.utc).isoformat(),
    }
    p = representations_root / _REGISTRY_FILENAME
    p.write_text(json.dumps(registry, indent=2))
    log.debug("Version registry updated: schema=%s extractor=%s", schema_version, extractor_version)


def check_and_evict(
    representations_root: Path,
    source: str,
    schema_version: str,
    extractor_version: str,
) -> int:
    """Check registry; evict stale source-level cache entries if version changed.

    Returns the number of evicted cache entries.
    """
    from sentinel_data.representation.cache_manager import evict_stale

    registry = read_registry(representations_root)
    reg_schema    = registry.get("schema_version")
    reg_extractor = registry.get("extractor_version")

    source_dir = representations_root / source
    evicted = 0

    if reg_schema != schema_version or reg_extractor != extractor_version:
        log.info(
            "Version mismatch detected for source '%s': "
            "registry=(%s, %s) live=(%s, %s) — evicting stale cache entries.",
            source, reg_schema, reg_extractor, schema_version, extractor_version,
        )
        evicted = evict_stale(source_dir, schema_version, extractor_version)

    return evicted


def current_versions() -> tuple[str, str]:
    """Return (FEATURE_SCHEMA_VERSION, EXTRACTOR_VERSION) from live modules."""
    from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION
    from sentinel_data.representation.orchestrator import EXTRACTOR_VERSION
    return FEATURE_SCHEMA_VERSION, EXTRACTOR_VERSION
