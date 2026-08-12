from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel_data.vnext.loader import VNextExport
from sentinel_data.vnext.publication import (
    bind_final_g7_validation_report,
    bind_semantic_validation_report,
    verify_publication_bindings,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_loader_rejects_historical_v1_without_fallback(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {
        "dataset_version": "historical",
        "export_schema_version": "v1",
        "graph_schema_version": "v9",
        "historical_artifacts_mutated": False,
    })
    with pytest.raises(ValueError, match="requires export_schema_version='v2'"):
        VNextExport(tmp_path)


def test_loader_accepts_explicit_v2_surface(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {
        "dataset_version": "sentinel-r4-vnext-v1",
        "export_schema_version": "v2",
        "graph_schema_version": "v9",
        "status": "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING",
        "historical_artifacts_mutated": False,
        "population": {"contracts": 22493, "contract_class_rows": 224930},
        "role_contract_counts": {"TRAIN_UNLABELED": 22493},
    })
    export = VNextExport(tmp_path)
    assert export.manifest["export_schema_version"] == "v2"
    assert "sentinel-r4-vnext-v1" in repr(export)


def test_semantic_validation_report_is_hash_bound(tmp_path: Path) -> None:
    manifest = {
        "dataset_version": "sentinel-r4-vnext-v1",
        "export_schema_version": "v2",
        "graph_schema_version": "v9",
        "status": "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING",
        "historical_artifacts_mutated": False,
        "semantic_validation_report": None,
        "representation_binding_report": None,
    }
    write_json(tmp_path / "manifest.json", manifest)
    report_path = tmp_path / "validation_report.json"
    write_json(report_path, {
        "schema": "sentinel-data-vnext-validation-report-v1",
        "passed": True,
        "require_representation_binding": False,
        "errors": [],
    })

    bound = bind_semantic_validation_report(tmp_path, report_path)
    assert bound["semantic_validation_report"]["path"] == "validation_report.json"
    check = verify_publication_bindings(tmp_path)
    assert check["passed"] is True
    assert "semantic_validation_report" in check["checked"]

    report_path.write_text(report_path.read_text() + "\n")
    tampered = verify_publication_bindings(tmp_path)
    assert tampered["passed"] is False
    assert "semantic_validation_report_hash_mismatch" in tampered["errors"]


def test_final_g7_report_promotes_only_intermediate_manifest(tmp_path: Path) -> None:
    semantic_path = tmp_path / "validation_report.json"
    rep_path = tmp_path / "representation_binding_report.json"
    g7_path = tmp_path / "g7_validation_report.json"
    write_json(semantic_path, {
        "passed": True,
        "require_representation_binding": False,
        "errors": [],
    })
    write_json(rep_path, {
        "status": "VALIDATED_LOCAL_G7",
        "passed": True,
        "binding_digest_sha256": "a" * 64,
    })
    write_json(g7_path, {
        "passed": True,
        "require_representation_binding": True,
        "errors": [],
    })

    import hashlib
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    write_json(tmp_path / "manifest.json", {
        "dataset_version": "sentinel-r4-vnext-v1",
        "export_schema_version": "v2",
        "graph_schema_version": "v9",
        "status": "REPRESENTATIONS_VALIDATED_G7_PENDING_FINAL",
        "historical_artifacts_mutated": False,
        "semantic_validation_report": {
            "path": semantic_path.name,
            "sha256": sha(semantic_path),
            "bytes": semantic_path.stat().st_size,
            "status": "VALIDATED_SEMANTICS",
        },
        "representation_binding_report": {
            "path": rep_path.name,
            "sha256": sha(rep_path),
            "bytes": rep_path.stat().st_size,
            "status": "VALIDATED_LOCAL_G7",
            "binding_digest_sha256": "a" * 64,
        },
        "g7_validation_report": None,
    })

    intermediate = verify_publication_bindings(tmp_path)
    assert intermediate["passed"] is True
    promoted = bind_final_g7_validation_report(tmp_path, g7_path)
    assert promoted["status"] == "VALIDATED_G7_CANDIDATE"
    assert promoted["g7_validation_report"]["status"] == "VALIDATED_G7"
    final = verify_publication_bindings(tmp_path)
    assert final["passed"] is True
    assert set(final["checked"]) == {
        "semantic_validation_report",
        "representation_binding_report",
        "g7_validation_report",
    }
