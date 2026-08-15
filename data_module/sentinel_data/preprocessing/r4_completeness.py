"""Fail-closed completeness checks for repaired preprocessing outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import PREPROCESSING_ARTIFACT_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_complete_preprocessed_source(source: str, directory: Path) -> dict[str, Any]:
    """Load one repaired manifest and reject partial or inconsistent builds."""

    path = Path(directory) / "repaired_preprocessing_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing repaired preprocessing manifest for {source}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source") != source:
        raise ValueError(f"repaired preprocessing source mismatch for {source}")
    if value.get("preprocessing_artifact_version") != PREPROCESSING_ARTIFACT_VERSION:
        raise ValueError(f"repaired preprocessing version mismatch for {source}")
    total = int(value.get("manifest_records_total", -1))
    requested = int(value.get("records_requested", -1))
    prepared = int(value.get("records_prepared", -1))
    dropped = int(value.get("records_dropped", -1))
    if value.get("raw_manifest_verification_passed") is not True:
        raise ValueError(f"raw manifest was not verified for {source}")
    if value.get("complete_source_build") is not True:
        raise ValueError(
            f"repaired preprocessing is incomplete for {source}: "
            f"requested={requested} manifest_total={total}"
        )
    if total < 1 or requested != total or prepared + dropped != total:
        raise ValueError(
            f"repaired preprocessing reconciliation failed for {source}: "
            f"total={total} requested={requested} prepared={prepared} dropped={dropped}"
        )
    artifact_count = len(list(Path(directory).glob("*.meta.json")))
    if artifact_count != int(value.get("artifacts_written", -1)):
        raise ValueError(
            f"repaired preprocessing artifact count mismatch for {source}: "
            f"manifest={value.get('artifacts_written')} physical={artifact_count}"
        )
    return {**value, "manifest_path": path, "manifest_sha256": _sha256(path)}


def require_complete_preprocessed_sources(
    source_dirs: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    return {
        source: require_complete_preprocessed_source(source, directory)
        for source, directory in sorted(source_dirs.items())
    }


__all__ = [
    "require_complete_preprocessed_source",
    "require_complete_preprocessed_sources",
]
