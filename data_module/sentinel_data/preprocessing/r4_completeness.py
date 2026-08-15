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


def require_complete_representation_source(
    source: str,
    directory: Path,
    *,
    expected_preprocessing_manifest_sha256: str,
) -> dict[str, Any]:
    """Reject partial, failed, or physically incomplete representation output."""

    directory = Path(directory)
    path = directory / "repaired_representation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing repaired representation manifest for {source}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    total = int(value.get("preprocessed_artifacts_total", -1))
    requested = int(value.get("contracts_requested", -1))
    written = int(value.get("representations_written", -1))
    failed = int(value.get("representations_failed", -1))
    if value.get("source") != source:
        raise ValueError(f"repaired representation source mismatch for {source}")
    if value.get("complete_representation_build") is not True:
        raise ValueError(f"repaired representation build is partial for {source}")
    if requested != total or written + failed != total or failed != 0:
        raise ValueError(
            f"repaired representation reconciliation failed for {source}: "
            f"total={total} requested={requested} written={written} failed={failed}"
        )
    if value.get("preprocessing_manifest_sha256") != expected_preprocessing_manifest_sha256:
        raise ValueError(f"representation/preprocessing binding mismatch for {source}")
    physical = {
        "graphs": len([p for p in directory.glob("*.pt") if not p.name.endswith(".tokens.pt")]),
        "tokens": len(list(directory.glob("*.tokens.pt"))),
        "sidecars": len(list(directory.glob("*.rep.json"))),
    }
    if any(count != written for count in physical.values()):
        raise ValueError(
            f"repaired representation physical count mismatch for {source}: "
            f"manifest={written} physical={physical}"
        )
    return {**value, "manifest_path": path, "manifest_sha256": _sha256(path)}


def require_complete_representation_sources(
    representation_root: Path,
    preprocessing_manifests: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not preprocessing_manifests:
        raise ValueError("grouping has no bound preprocessing manifests")
    return {
        source: require_complete_representation_source(
            source,
            Path(representation_root) / source,
            expected_preprocessing_manifest_sha256=str(value["manifest_sha256"]),
        )
        for source, value in sorted(preprocessing_manifests.items())
    }


__all__ = [
    "require_complete_preprocessed_source",
    "require_complete_preprocessed_sources",
    "require_complete_representation_source",
    "require_complete_representation_sources",
]
