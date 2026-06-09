"""Ingestion service — orchestrates connector + manifest for one or all enabled sources."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from sentinel_data.ingestion.connectors import ConnectorError, SourceConfig, get_connector
from sentinel_data.ingestion.manifest import IngestionManifest, build_file_records


def _all_sources(cfg: dict) -> dict[str, dict]:
    """Merge sources_critical_path + sources_additive into one flat dict."""
    out: dict = {}
    out.update(cfg.get("sources_critical_path") or {})
    out.update(cfg.get("sources_additive") or {})
    out.update(cfg.get("sources") or {})
    return out


def _enabled_sources(cfg: dict) -> dict[str, dict]:
    return {k: v for k, v in _all_sources(cfg).items() if v.get("enabled")}


def _source_config(name: str, entry: dict) -> SourceConfig:
    return SourceConfig(
        name=name,
        connector=entry.get("connector", "git"),
        url=entry.get("url", ""),
        pin=entry.get("pin", ""),
        hf_dataset=entry.get("hf_dataset", ""),
        zenodo_record=entry.get("zenodo_record", ""),
        description=entry.get("description", ""),
        extra={k: v for k, v in entry.items()
               if k not in ("enabled", "tier", "tier_subtype", "connector",
                            "url", "pin", "hf_dataset", "zenodo_record",
                            "description", "crosswalk")},
    )


def ingest_source(
    name: str,
    cfg: dict,
    data_dir: Path,
    dry_run: bool = False,
) -> IngestionManifest | None:
    """Pull one source and write its ingestion manifest.

    Returns the manifest on success, None on dry_run.
    Raises ConnectorError on failure.
    """
    sources = _enabled_sources(cfg)
    if name not in sources:
        all_s = _all_sources(cfg)
        if name in all_s:
            raise ConnectorError(f"Source '{name}' exists but is not enabled in config.yaml")
        raise ConnectorError(f"Source '{name}' not found in config.yaml")

    entry = sources[name]
    source_cfg = _source_config(name, entry)
    raw_dir = data_dir / "raw" / name

    if dry_run:
        print(f"  would pull  : {source_cfg.url or '(manual)'}")
        print(f"  pin         : {source_cfg.pin or 'HEAD'}")
        print(f"  destination : {raw_dir}")
        return None

    connector = get_connector(source_cfg.connector)
    result = connector.pull(source_cfg, raw_dir)

    manifest = IngestionManifest(
        source=name,
        connector=source_cfg.connector,
        url=source_cfg.url,
        pin=source_cfg.pin,
        resolved_pin=result.resolved_pin,
        fetched_at=result.fetched_at,
        duration_s=result.duration_s,
        contract_count=len(result.sol_files),
        files=build_file_records(result.sol_files, raw_dir),
    )
    manifest.save(raw_dir / "ingestion_manifest.json")
    return manifest


def ingest_all(
    cfg: dict,
    data_dir: Path,
    dry_run: bool = False,
) -> list[IngestionManifest]:
    sources = _enabled_sources(cfg)
    manifests = []
    for name in sources:
        print(f"\n[ingest] {name}")
        m = ingest_source(name, cfg, data_dir, dry_run=dry_run)
        if m:
            manifests.append(m)
            print(f"  contracts   : {m.contract_count}")
            print(f"  resolved    : {m.resolved_pin[:12]}")
    return manifests
