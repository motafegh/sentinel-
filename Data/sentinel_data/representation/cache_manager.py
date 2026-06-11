"""Content-addressed representation cache for SENTINEL v2.

The cache key is (sha256_of_source, schema_version, extractor_version).
A change to any of the three values invalidates the cache for that file.

Cache layout
------------
data/representations/<source>/
    <sha256>.pt          — PyG Data graph
    <sha256>.tokens.pt   — windowed GraphCodeBERT tokens
    <sha256>.rep.json    — sidecar with provenance + versions

The sidecar is the canonical cache record.  ``is_cached()`` reads it and
compares versions.  ``invalidate()`` removes all three files.

Public surface
--------------
is_cached(output_dir, sha256, schema_version, extractor_version) -> bool
record_cache_hit(output_dir, sha256)                              -> None
invalidate(output_dir, sha256)                                    -> None
list_cached_sha256s(output_dir)                                   -> list[str]
stale_entries(output_dir, schema_version, extractor_version)      -> list[str]
evict_stale(output_dir, schema_version, extractor_version)        -> int
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("sentinel_data.cache_manager")


def is_cached(
    output_dir: Path,
    sha256: str,
    schema_version: str,
    extractor_version: str,
) -> bool:
    """Return True iff the sidecar exists with matching versions."""
    rep_path = output_dir / f"{sha256}.rep.json"
    if not rep_path.exists():
        return False
    try:
        sidecar = json.loads(rep_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        sidecar.get("schema_version") == schema_version
        and sidecar.get("extractor_version") == extractor_version
        and (output_dir / f"{sha256}.pt").exists()
        and (output_dir / f"{sha256}.tokens.pt").exists()
    )


def record_cache_hit(output_dir: Path, sha256: str) -> None:
    """Mark a sidecar as accessed (no-op in this implementation).

    Placeholder for LRU / access-time tracking in v3.1.
    """


def invalidate(output_dir: Path, sha256: str) -> None:
    """Remove graph + tokens + sidecar for one sha256."""
    for ext in (".pt", ".tokens.pt", ".rep.json"):
        p = output_dir / f"{sha256}{ext}"
        if p.exists():
            p.unlink()
            log.debug("Invalidated %s", p.name)


def list_cached_sha256s(output_dir: Path) -> list[str]:
    """Return sha256s with all three files present (graph + tokens + sidecar)."""
    if not output_dir.exists():
        return []
    sidecars = list(output_dir.glob("*.rep.json"))
    result = []
    for s in sidecars:
        sha = s.stem.removesuffix(".rep") if s.stem.endswith(".rep") else s.stem
        if (output_dir / f"{sha}.pt").exists() and (output_dir / f"{sha}.tokens.pt").exists():
            result.append(sha)
    return result


def stale_entries(
    output_dir: Path,
    schema_version: str,
    extractor_version: str,
) -> list[str]:
    """Return sha256s whose sidecar has a mismatched schema or extractor version."""
    if not output_dir.exists():
        return []
    stale = []
    for rep_path in output_dir.glob("*.rep.json"):
        sha = rep_path.stem
        try:
            sidecar = json.loads(rep_path.read_text())
        except (json.JSONDecodeError, OSError):
            stale.append(sha)
            continue
        if (
            sidecar.get("schema_version") != schema_version
            or sidecar.get("extractor_version") != extractor_version
        ):
            stale.append(sha)
    return stale


def evict_stale(
    output_dir: Path,
    schema_version: str,
    extractor_version: str,
) -> int:
    """Remove all stale entries; return count of evicted files."""
    to_evict = stale_entries(output_dir, schema_version, extractor_version)
    for sha in to_evict:
        invalidate(output_dir, sha)
    log.info("Evicted %d stale cache entries from %s", len(to_evict), output_dir)
    return len(to_evict)
