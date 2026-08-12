"""Publication helpers for DATA vNext report-to-manifest bindings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bind_semantic_validation_report(export_dir: Path, report_path: Path) -> dict[str, Any]:
    """Bind a successful pre-local semantic validation report into manifest.json."""
    export_dir = Path(export_dir)
    report_path = Path(report_path)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = _load_json(report_path)
    if report.get("passed") is not True:
        raise ValueError("cannot publish a failed semantic validation report")
    if report.get("require_representation_binding") is not False:
        raise ValueError("semantic validation report must be pre-local-binding")

    manifest = _load_json(manifest_path)
    if manifest.get("status") != "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING":
        raise ValueError(f"unexpected manifest status before semantic publication: {manifest.get('status')!r}")
    manifest["semantic_validation_report"] = {
        "path": report_path.name,
        "sha256": _sha256(report_path),
        "bytes": report_path.stat().st_size,
        "status": "VALIDATED_SEMANTICS",
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def bind_final_g7_validation_report(export_dir: Path, report_path: Path) -> dict[str, Any]:
    """Bind the successful post-representation final G7 validation report."""
    export_dir = Path(export_dir)
    report_path = Path(report_path)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = _load_json(report_path)
    if report.get("passed") is not True or report.get("require_representation_binding") is not True:
        raise ValueError("final G7 validation report must be a successful representation-bound validation")

    manifest = _load_json(manifest_path)
    if manifest.get("status") != "VALIDATED_G7_CANDIDATE":
        raise ValueError("manifest must already bind a successful local representation report")
    if not isinstance(manifest.get("representation_binding_report"), dict):
        raise ValueError("representation binding must exist before final G7 report publication")
    manifest["g7_validation_report"] = {
        "path": report_path.name,
        "sha256": _sha256(report_path),
        "bytes": report_path.stat().st_size,
        "status": "VALIDATED_G7",
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def _verify_bound_json(
    export_dir: Path,
    meta: dict[str, Any],
    *,
    label: str,
    checked: dict[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    path = export_dir / str(meta.get("path", ""))
    if not path.is_file():
        errors.append(f"{label}_missing")
        return None
    actual = _sha256(path)
    checked[label] = {"path": path.name, "sha256": actual, "bytes": path.stat().st_size}
    if actual != meta.get("sha256"):
        errors.append(f"{label}_hash_mismatch")
    if path.stat().st_size != int(meta.get("bytes", -1)):
        errors.append(f"{label}_size_mismatch")
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        errors.append(f"{label}_invalid_json")
        return None


def verify_publication_bindings(export_dir: Path) -> dict[str, Any]:
    """Verify report bindings currently declared by the v2 manifest."""
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    errors: list[str] = []
    checked: dict[str, Any] = {}

    semantic = manifest.get("semantic_validation_report")
    if not isinstance(semantic, dict):
        errors.append("semantic_validation_report_not_bound")
    else:
        report = _verify_bound_json(
            export_dir, semantic,
            label="semantic_validation_report", checked=checked, errors=errors
        )
        if report is not None:
            if report.get("passed") is not True or report.get("require_representation_binding") is not False:
                errors.append("semantic_validation_report_content_invalid")

    representation = manifest.get("representation_binding_report")
    if representation is not None:
        if not isinstance(representation, dict):
            errors.append("representation_binding_report_invalid_metadata")
        else:
            report = _verify_bound_json(
                export_dir, representation,
                label="representation_binding_report", checked=checked, errors=errors
            )
            if report is not None:
                if report.get("status") != "VALIDATED_LOCAL_G7" or report.get("passed") is not True:
                    errors.append("representation_binding_report_content_invalid")
                if report.get("binding_digest_sha256") != representation.get("binding_digest_sha256"):
                    errors.append("representation_binding_digest_mismatch")

    g7 = manifest.get("g7_validation_report")
    if g7 is not None:
        if not isinstance(g7, dict):
            errors.append("g7_validation_report_invalid_metadata")
        else:
            report = _verify_bound_json(
                export_dir, g7,
                label="g7_validation_report", checked=checked, errors=errors
            )
            if report is not None:
                if report.get("passed") is not True or report.get("require_representation_binding") is not True:
                    errors.append("g7_validation_report_content_invalid")

    if manifest.get("status") == "VALIDATED_G7_CANDIDATE":
        if not isinstance(representation, dict):
            errors.append("candidate_manifest_missing_representation_binding")
        if not isinstance(g7, dict):
            errors.append("candidate_manifest_missing_final_g7_report")

    return {
        "schema": "sentinel-data-vnext-publication-binding-report-v1",
        "passed": not errors,
        "errors": errors,
        "checked": checked,
        "manifest_status": manifest.get("status"),
    }


__all__ = [
    "bind_final_g7_validation_report",
    "bind_semantic_validation_report",
    "verify_publication_bindings",
]
