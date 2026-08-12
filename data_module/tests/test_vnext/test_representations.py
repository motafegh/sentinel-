from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sentinel_data.vnext.representations import (
    bind_representation_report,
    verify_local_representations,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_overlay(tmp_path: Path) -> tuple[Path, Path, str]:
    export = tmp_path / "export"
    reps = tmp_path / "representations"
    export.mkdir()
    required_id = "a" * 64
    excluded_id = "b" * 64
    table = pa.Table.from_pylist([
        {
            "contract_id": required_id,
            "source": "dive",
            "role": "TRAIN_UNLABELED",
            "representation_required": True,
        },
        {
            "contract_id": excluded_id,
            "source": "dive",
            "role": "EXCLUDED",
            "representation_required": False,
        },
    ])
    pq.write_table(table, export / "ml_targets.parquet")
    write_json(export / "representation_requirements.json", {
        "graph_schema_version": "v9",
        "required_contracts": 1,
        "excluded_contracts": 1,
    })
    write_json(export / "manifest.json", {
        "dataset_version": "sentinel-r4-vnext-v1",
        "export_schema_version": "v2",
        "graph_schema_version": "v9",
        "status": "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING",
        "historical_artifacts_mutated": False,
        "representation_binding_report": None,
    })
    return export, reps, required_id


def create_valid_representation(reps: Path, cid: str) -> None:
    source_dir = reps / "dive"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{cid}.pt").write_bytes(b"graph-bytes")
    (source_dir / f"{cid}.tokens.pt").write_bytes(b"token-bytes")
    write_json(source_dir / f"{cid}.rep.json", {
        "sha256": cid,
        "source": "dive",
        "schema_version": "v9",
        "extractor_version": "v2.1-windowed-gcb",
    })


def test_valid_local_representation_population_binds(tmp_path: Path) -> None:
    export, reps, cid = make_overlay(tmp_path)
    create_valid_representation(reps, cid)
    report_path = export / "representation_binding_report.json"
    report = verify_local_representations(export, reps, report_path=report_path)
    assert report["passed"] is True
    assert report["status"] == "VALIDATED_LOCAL_G7"
    assert report["required_contracts"] == 1
    assert report["checked_contracts"] == 1
    assert report["checked_files"] == 3
    assert report["binding_digest_sha256"]
    assert report["extractor_version_counts"] == {"v2.1-windowed-gcb": 1}

    manifest = bind_representation_report(export, report_path)
    assert manifest["status"] == "VALIDATED_G7_CANDIDATE"
    assert manifest["representation_binding_report"]["binding_digest_sha256"] == report["binding_digest_sha256"]


def test_missing_token_file_fails_closed(tmp_path: Path) -> None:
    export, reps, cid = make_overlay(tmp_path)
    create_valid_representation(reps, cid)
    (reps / "dive" / f"{cid}.tokens.pt").unlink()
    report = verify_local_representations(export, reps)
    assert report["passed"] is False
    assert report["status"] == "FAILED_LOCAL_G7"
    assert report["missing_files_total"] == 1
    assert report["checked_contracts"] == 0


def test_sidecar_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    export, reps, cid = make_overlay(tmp_path)
    create_valid_representation(reps, cid)
    sidecar = reps / "dive" / f"{cid}.rep.json"
    write_json(sidecar, {
        "sha256": "c" * 64,
        "source": "dive",
        "schema_version": "v9",
        "extractor_version": "v2.1-windowed-gcb",
    })
    report = verify_local_representations(export, reps)
    assert report["passed"] is False
    assert report["mismatch_total"] == 1
    assert "contract_id_mismatch" in report["mismatches"][0]["reason"]
