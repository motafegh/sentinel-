"""Tests for explicit repaired-R4 Slither compatibility recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from sentinel_data.preprocessing.r4_versions import PREPROCESSING_ARTIFACT_VERSION
from sentinel_data.representation import r4_compatibility as compatibility
from sentinel_data.representation import r4_orchestrator as orchestrator
from sentinel_data.representation.graph_extractor import SlitherParseError


def test_constant_array_fold_is_narrow_and_layout_preserving():
    source = """contract Example {
  uint8 constant stageTotal = 11;
  Rule[20+1] public rules;
  uint8[stageTotal * 3] bonus;
  function read(uint i) public { bonus[i + 1]; }
}
"""
    transformed, replacements = compatibility.fold_constant_array_lengths(source)

    assert "Rule[21  ] public rules" in transformed
    assert "uint8[33            ] bonus" in transformed
    assert "bonus[i + 1]" in transformed
    assert len(transformed.encode()) == len(source.encode())
    assert transformed.count("\n") == source.count("\n")
    assert [(row["expression"], row["value"]) for row in replacements] == [
        ("20+1", 21),
        ("stageTotal * 3", 33),
    ]


def test_compatibility_uses_parse_only_after_full_analysis_failure(
    tmp_path, monkeypatch
):
    sol = tmp_path / "fixture.sol"
    sol.write_text("contract Vault {}\n")

    def fake_extract(path, config):
        if not config.slither_skip_analyze:
            raise SlitherParseError("Slither IR defect")
        return SimpleNamespace(contract_name="Vault", num_nodes=1, num_edges=0)

    monkeypatch.setattr(compatibility, "extract_contract_graph", fake_extract)
    result = compatibility.extract_components_with_compatibility(
        sol,
        ("Vault",),
        solc_binary=None,
        solc_version="0.4.18",
    )

    assert result.mode == compatibility.PARSE_ONLY
    assert result.analysis_degraded is True
    assert result.source_transform is None
    assert result.actual_targets == ("Vault",)
    assert result.fallback_errors[0]["error_type"] == "SlitherParseError"


def test_compatibility_forwards_v10_schema(tmp_path, monkeypatch):
    sol = tmp_path / "fixture.sol"
    sol.write_text("contract Vault {}\n")
    observed: list[str] = []

    def fake_extract(path, config):
        observed.append(config.graph_schema_version)
        return SimpleNamespace(contract_name="Vault", num_nodes=1, num_edges=0)

    monkeypatch.setattr(compatibility, "extract_contract_graph", fake_extract)
    compatibility.extract_components_with_compatibility(
        sol,
        ("Vault",),
        solc_binary=None,
        solc_version="0.4.18",
        graph_schema_version="v10",
    )
    assert observed == ["v10"]


def test_compatibility_folds_not_constant_array_length(tmp_path, monkeypatch):
    sol = tmp_path / "fixture.sol"
    sol.write_text("contract Vault { uint8[20+1] values; }\n")

    class NotConstant(Exception):
        pass

    def fake_extract(path, config):
        value = Path(path).read_text()
        if "[20+1]" in value:
            cause = NotConstant("cannot fold")
            wrapped = SlitherParseError("")
            wrapped.__cause__ = cause
            raise wrapped
        return SimpleNamespace(contract_name="Vault", num_nodes=1, num_edges=0)

    monkeypatch.setattr(compatibility, "extract_contract_graph", fake_extract)
    result = compatibility.extract_components_with_compatibility(
        sol,
        ("Vault",),
        solc_binary=None,
        solc_version="0.4.24",
    )

    assert result.mode == compatibility.FULL_ANALYSIS_CONSTANT_FOLD
    assert result.analysis_degraded is False
    assert result.source_transform["replacements"] == [
        {"line": 1, "expression": "20+1", "value": 21}
    ]
    assert result.source_transform["byte_length_preserved"] is True


def test_failed_tail_recovery_reuses_success_and_retries_only_failure(
    tmp_path, monkeypatch
):
    preprocessed = tmp_path / "preprocessed"
    attempt = tmp_path / "attempt"
    output = tmp_path / "output"
    preprocessed.mkdir()
    attempt.mkdir()
    accepted = "a" * 64
    failed = "b" * 64

    preprocessing_manifest = {
        "source": "fixture",
        "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
        "manifest_records_total": 2,
        "records_requested": 2,
        "records_prepared": 2,
        "records_dropped": 0,
        "artifacts_written": 2,
        "complete_source_build": True,
        "raw_manifest_verification_passed": True,
    }
    manifest_path = preprocessed / "repaired_preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(preprocessing_manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for artifact in (accepted, failed):
        (preprocessed / f"{artifact}.meta.json").write_text(
            json.dumps(
                {
                    "sha256": artifact,
                    "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
                }
            )
        )
        (preprocessed / f"{artifact}.sol").write_text("contract Vault {}\n")

    for suffix in (".pt", ".tokens.pt", ".rep.json"):
        (attempt / f"{accepted}{suffix}").write_bytes(b"accepted" + suffix.encode())
    (attempt / "repaired_representation_manifest.json").write_text(
        json.dumps(
            {
                "source": "fixture",
                "complete_representation_build": True,
                "contracts_requested": 2,
                "preprocessed_artifacts_total": 2,
                "representations_written": 1,
                "representations_failed": 1,
                "preprocessing_manifest_sha256": manifest_sha,
            }
        )
    )
    (attempt / "representation_failures.jsonl").write_text(
        json.dumps({"meta_path": f"{failed}.meta.json", "error": "fixture"}) + "\n"
    )

    retried: list[str] = []

    def fake_extract(source, sol_path, meta, output_dir):
        retried.append(meta["sha256"])
        for suffix in (".pt", ".tokens.pt", ".rep.json"):
            (output_dir / f"{meta['sha256']}{suffix}").write_bytes(b"recovered")
        return {
            "graph_extraction_mode": compatibility.PARSE_ONLY,
            "graph_analysis_degraded": True,
            "graph_source_transform_applied": False,
        }

    monkeypatch.setattr(orchestrator, "_extract_one", fake_extract)
    result = orchestrator.recover_failed_representations(
        "fixture", preprocessed, attempt, output, n_workers=1
    )

    assert retried == [failed]
    assert result.representations_written == 2
    assert result.representations_failed == 0
    final_manifest = json.loads(
        (output / "repaired_representation_manifest.json").read_text()
    )
    assert final_manifest["recovery"]["reused_representations"] == 1
    assert final_manifest["recovery"]["retried_representations"] == 1
    assert final_manifest["graph_analysis_degraded_total"] == 1
    assert (output / f"{accepted}.pt").read_bytes() == b"accepted.pt"
