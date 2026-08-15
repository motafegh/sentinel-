"""Tests for byte-exact raw and full-population repaired DATA gates."""

from __future__ import annotations

import hashlib
import json

import pytest

from sentinel_data.preprocessing.r4_completeness import (
    require_complete_preprocessed_source,
)
from sentinel_data.preprocessing.r4_raw_verifier import verify_manifest_source
from sentinel_data.preprocessing.r4_versions import PREPROCESSING_ARTIFACT_VERSION


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_raw_verifier_accepts_manifest_symlink_within_allowed_repository(tmp_path):
    repository = tmp_path / "repository"
    raw = repository / "raw"
    staging = repository / "staging"
    raw.mkdir(parents=True)
    staging.mkdir()
    payload = b"contract Vault {}\n"
    (staging / "Vault.sol").write_bytes(payload)
    (raw / "repo").symlink_to(staging, target_is_directory=True)
    manifest = raw / "ingestion_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "contract_count": 1,
                "files": [
                    {
                        "path": "repo/Vault.sol",
                        "size_bytes": len(payload),
                        "sha256": _sha(payload),
                    }
                ],
            }
        )
    )

    report = verify_manifest_source(
        "fixture",
        raw,
        manifest,
        allowed_resolved_roots=(repository,),
    )
    assert report["passed"] is True
    assert report["manifest_records"] == 1


def test_raw_verifier_rejects_lexical_traversal(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = b"contract Vault {}\n"
    (tmp_path / "Vault.sol").write_bytes(payload)
    manifest = raw / "ingestion_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "../Vault.sol",
                        "size_bytes": len(payload),
                        "sha256": _sha(payload),
                    }
                ]
            }
        )
    )
    report = verify_manifest_source(
        "fixture", raw, manifest, allowed_resolved_roots=(tmp_path,)
    )
    assert report["passed"] is False
    assert report["errors"][0]["reason"] == "invalid_or_traversing_manifest_path"


def test_completeness_gate_rejects_limit_build(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "a.meta.json").write_text("{}")
    (directory / "repaired_preprocessing_manifest.json").write_text(
        json.dumps(
            {
                "source": "fixture",
                "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
                "manifest_records_total": 10,
                "records_requested": 1,
                "records_prepared": 1,
                "records_dropped": 0,
                "artifacts_written": 1,
                "complete_source_build": False,
                "raw_manifest_verification_passed": True,
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        require_complete_preprocessed_source("fixture", directory)


def test_completeness_gate_accepts_reconciled_full_build(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "a.meta.json").write_text("{}")
    (directory / "repaired_preprocessing_manifest.json").write_text(
        json.dumps(
            {
                "source": "fixture",
                "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
                "manifest_records_total": 2,
                "records_requested": 2,
                "records_prepared": 2,
                "records_dropped": 0,
                "artifacts_written": 1,
                "complete_source_build": True,
                "raw_manifest_verification_passed": True,
            }
        )
    )
    report = require_complete_preprocessed_source("fixture", directory)
    assert report["manifest_sha256"]
