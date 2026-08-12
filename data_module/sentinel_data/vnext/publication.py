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


def bind_semantic_validation_report(export_dir: Path, report_path: Path) -> dict[str, Any]:
    """Bind a successful pre-local semantic validation report into manifest.json."""
    export_dir = Path(export_dir)
    report_path = Path(report_path)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("cannot publish a failed semantic validation report")
    if report.get("require_representation_binding") is not False:
        raise ValueError("semantic validation report must be pre-local-binding")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING":
        raise ValueError(f"unexpected manifest status before semantic publication: {manifest.get('status')!r}")
    manifest["semantic_validation_report"] = {
        "path": report_path.name,
        "sha256": _sha256(report_path),
        "bytes": report_path.stat().st_size,
        "status": "VALIDATED_SEMANTICS",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_publication_bindings(export_dir: Path) -> dict[str, Any]:
    """Verify report bindings currently declared by the v2 manifest."""
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked: dict[str, Any] = {}

    semantic = manifest.get("semantic_validation_report")
    if not isinstance(semantic, dict):
        errors.append("semantic_validation_report_not_bound")
    else:
        path = export_dir / str(semantic.get("path", ""))
        if not path.is_file():
            errors.append("semantic_validation_report_missing")
        else:
            actual = _sha256(path)
            checked["semantic_validation_report"] = {
                "path": path.name,
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
            if actual != semantic.get("sha256"):
                errors.append("semantic_validation_report_hash_mismatch")
            if path.stat().st_size != int(semantic.get("bytes", -1)):
                errors.append("semantic_validation_report_size_mismatch")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("passed") is not True or report.get("require_representation_binding") is not False:
                errors.append("semantic_validation_report_content_invalid")

    representation = manifest.get("representation_binding_report")
    if representation is not None:
        if not isinstance(representation, dict):
            errors.append("representation_binding_report_invalid_metadata")
        else:
            path = export_dir / str(representation.get("path", ""))
            if not path.is_file():
                errors.append("representation_binding_report_missing")
            else:
                actual = _sha256(path)
                checked["representation_binding_report"] = {
                    "path": path.name,
                    "sha256": actual,
                    "bytes": path.stat().st_size,
                }
                if actual != representation.get("sha256"):
                    errors.append("representation_binding_report_hash_mismatch")
                if path.stat().st_size != int(representation.get("bytes", -1)):
                    errors.append("representation_binding_report_size_mismatch")
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("status") != "VALIDATED_LOCAL_G7" or report.get("passed") is not True:
                    errors.append("representation_binding_report_content_invalid")
                if report.get("binding_digest_sha256") != representation.get("binding_digest_sha256"):
                    errors.append("representation_binding_digest_mismatch")

    return {
        "schema": "sentinel-data-vnext-publication-binding-report-v1",
        "passed": not errors,
        "errors": errors,
        "checked": checked,
        "manifest_status": manifest.get("status"),
    }


__all__ = ["bind_semantic_validation_report", "verify_publication_bindings"]
