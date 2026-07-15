"""R0.0 tests for immutable before/after evidence closure rules."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.r0_evidence.environment as environment_module
from scripts.r0_evidence.cli import (
    capture_probe,
    create_environment_manifest,
    validate_command_manifest,
)
from scripts.r0_evidence.environment import probe_environment, validate_environment_manifest
from scripts.r0_evidence.matrix import MATRIX_ROWS, matrix_manifest
from scripts.r0_evidence.model import (
    load_evidence_artifacts,
    redact_text,
    sha256_file,
    validate_coverage,
    validate_record,
)

BASELINE = "1" * 40
CANDIDATE = "2" * 40


def _clean_git_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("evidence fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Sentinel Tests",
            "-c",
            "user.email=tests@sentinel.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=workspace,
        check=True,
    )
    return workspace


def _record(row_id: str, phase: str, *, comparison_key: str = "a" * 64) -> dict:
    after = phase == "after"
    return {
        "schema_version": "1",
        "kind": "r0_evidence_record",
        "record_id": f"{row_id}:{phase}",
        "matrix_row_id": row_id,
        "phase": phase,
        "baseline_commit": BASELINE,
        "candidate_commit": CANDIDATE if after else None,
        "comparison_key": comparison_key,
        "probe": {
            "probe_id": "probe",
            "contract_version": "1",
            "argv_template": ["python", "probe.py"],
            "resolved_argv": ["python", "probe.py"],
            "cwd": ".",
        },
        "environment_manifest": {
            "path": "environment.json",
            "sha256": "3" * 64,
            "environment_contract": "linux-x86_64-python-3.12",
        },
        "execution": {
            "started_at": "2026-07-14T00:00:00+00:00",
            "finished_at": "2026-07-14T00:00:01+00:00",
            "exit_code": 0,
            "stdout_sha256": "4" * 64,
            "stderr_sha256": "5" * 64,
            "stdout": "{}",
            "stderr": "",
        },
        "outcome": {
            "status": "pass" if after else "fail",
            "invariant_passed": after,
            "assertions": [{"name": "invariant", "passed": after, "detail": "fixture"}],
        },
        "test_references": ["tests/test_fixture.py::test_invariant"],
        "review": {
            "status": "accepted",
            "reviewer": "reviewer",
            "decided_at": "2026-07-14T00:00:02+00:00",
        },
    }


def _complete_records() -> list[dict]:
    records = []
    for row in MATRIX_ROWS:
        records.extend((_record(row.row_id, "before"), _record(row.row_id, "after")))
    return records


def test_matrix_has_exactly_the_eight_approved_r0_rows() -> None:
    manifest = matrix_manifest()
    assert len(MATRIX_ROWS) == 8
    assert len({row.row_id for row in MATRIX_ROWS}) == 8
    assert len(manifest["rows"]) == 8


def test_structurally_valid_record_has_no_errors() -> None:
    assert validate_record(_record(MATRIX_ROWS[0].row_id, "before")) == []


def test_missing_fields_are_reported_together() -> None:
    errors = validate_record({"schema_version": "1"})
    assert errors
    assert errors[0].startswith("missing fields:")
    assert "record_id" in errors[0]
    assert "review" in errors[0]


def test_complete_accepted_comparable_pairs_close_all_rows() -> None:
    report = validate_coverage(_complete_records())
    assert report["complete"] is True
    assert all(row["complete"] for row in report["rows"])


def test_after_like_noncanonical_artifact_blocks_closure(tmp_path: Path) -> None:
    path = tmp_path / "2026-07-15_SYSTEM_R0-PROOF-IDENTITY_after_r0-4.json"
    path.write_text(
        json.dumps({"row_id": "R0-PROOF-IDENTITY", "status": "pass"}),
        encoding="utf-8",
    )
    records, invalid_artifacts = load_evidence_artifacts(tmp_path)
    report = validate_coverage(
        [*_complete_records(), *records], invalid_artifacts=invalid_artifacts
    )
    assert report["complete"] is False
    assert report["invalid_artifacts"] == [
        {
            "path": path.name,
            "error": "after-like artifact is not a canonical r0_evidence_record",
        }
    ]


def test_unreadable_json_is_not_silently_skipped(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    records, invalid_artifacts = load_evidence_artifacts(tmp_path)
    assert records == []
    assert invalid_artifacts[0]["path"] == path.name
    assert invalid_artifacts[0]["error"].startswith("unreadable JSON:")


def test_after_only_cannot_close_a_row() -> None:
    records = _complete_records()
    records = [
        record
        for record in records
        if not (record["matrix_row_id"] == MATRIX_ROWS[0].row_id and record["phase"] == "before")
    ]
    report = validate_coverage(records)
    first = next(row for row in report["rows"] if row["row_id"] == MATRIX_ROWS[0].row_id)
    assert report["complete"] is False
    assert "missing before record" in first["issues"]


def test_changed_probe_contract_cannot_close_a_row() -> None:
    records = _complete_records()
    after = next(
        record
        for record in records
        if record["matrix_row_id"] == MATRIX_ROWS[0].row_id and record["phase"] == "after"
    )
    after["comparison_key"] = "b" * 64
    report = validate_coverage(records)
    first = next(row for row in report["rows"] if row["row_id"] == MATRIX_ROWS[0].row_id)
    assert "before/after comparison_key mismatch" in first["issues"]


def test_pending_review_cannot_close_a_row() -> None:
    records = _complete_records()
    after = next(record for record in records if record["phase"] == "after")
    after["review"] = {"status": "pending", "reviewer": None, "decided_at": None}
    report = validate_coverage(records)
    row = next(item for item in report["rows"] if item["row_id"] == after["matrix_row_id"])
    assert "after record lacks accepted reviewer decision" in row["issues"]


def test_nonpassing_after_probe_cannot_close_a_row() -> None:
    records = _complete_records()
    after = next(record for record in records if record["phase"] == "after")
    after["outcome"]["invariant_passed"] = False
    after["outcome"]["status"] = "fail"
    report = validate_coverage(records)
    row = next(item for item in report["rows"] if item["row_id"] == after["matrix_row_id"])
    assert "after record does not prove the invariant" in row["issues"]


def test_after_record_requires_candidate_commit() -> None:
    record = _record(MATRIX_ROWS[0].row_id, "after")
    record["candidate_commit"] = None
    assert "after record requires a full candidate_commit" in validate_record(record)


def test_test_reference_is_mandatory() -> None:
    record = _record(MATRIX_ROWS[0].row_id, "before")
    record["test_references"] = []
    assert "test_references must be a non-empty string list" in validate_record(record)


def test_record_digests_must_be_full_sha256_values() -> None:
    record = _record(MATRIX_ROWS[0].row_id, "before")
    record["comparison_key"] = "short"
    record["execution"]["stdout_sha256"] = "not-a-hash"
    errors = validate_record(record)
    assert "comparison_key must be a lowercase SHA-256" in errors
    assert "execution.stdout_sha256 must be a lowercase SHA-256" in errors


def test_redaction_removes_credentials_and_local_paths(tmp_path: Path) -> None:
    text = f"token=abc123 password:secret {tmp_path} https://user:pass@example.test/path"
    redacted = redact_text(text, workspace=tmp_path)
    assert "abc123" not in redacted
    assert "secret" not in redacted
    assert "user:pass" not in redacted
    assert str(tmp_path) not in redacted
    assert "<REDACTED>" in redacted
    assert "<WORKSPACE>" in redacted


def test_environment_manifest_never_copies_process_environment(monkeypatch) -> None:
    workspace = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("R0_TEST_SECRET", "must-not-appear")
    manifest = create_environment_manifest(workspace)
    serialized = json.dumps(manifest)
    assert "R0_TEST_SECRET" not in serialized
    assert "must-not-appear" not in serialized
    assert manifest["environment_contract"]
    assert manifest["comparison_fingerprint"]
    assert "harness_python" in manifest["runtimes"]
    assert validate_environment_manifest(manifest) == []
    assert manifest["workspace_commit"]


def test_environment_manifest_detects_comparison_material_tampering() -> None:
    workspace = Path(__file__).resolve().parents[2]
    manifest = create_environment_manifest(workspace)
    manifest["lockfiles"][0]["sha256"] = "0" * 64
    assert (
        "environment comparison_fingerprint does not match its material"
        in validate_environment_manifest(manifest)
    )


def test_environment_manifest_rejects_malformed_identity_and_policy() -> None:
    workspace = Path(__file__).resolve().parents[2]
    manifest = create_environment_manifest(workspace)
    manifest["runtimes"]["harness_python"]["packages_sha256"] = "short"
    manifest["probe_environment_policy"]["inherited_keys"].append("HOME")
    errors = validate_environment_manifest(manifest)
    assert "environment runtime identities are invalid" in errors
    assert "environment probe policy does not match the supported policy" in errors


def test_runtime_identity_is_remeasured_instead_of_cached(monkeypatch) -> None:
    inventories = iter(
        [
            {"implementation": "CPython", "version": "3.12.3", "packages": [["one", "1"]]},
            {"implementation": "CPython", "version": "3.12.3", "packages": [["two", "2"]]},
        ]
    )
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(next(inventories)), stderr=""
        )

    monkeypatch.setattr(environment_module.subprocess, "run", fake_run)
    first = environment_module.runtime_identity(sys.executable)
    second = environment_module.runtime_identity(sys.executable)
    assert len(calls) == 2
    assert first["packages_sha256"] != second["packages_sha256"]


def test_probe_environment_drops_secrets_and_uses_an_isolated_home(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SENTINEL_OPERATOR_KEY", "must-not-cross-boundary")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-boundary")
    environment = probe_environment(tmp_path)
    assert "SENTINEL_OPERATOR_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert environment["HOME"] == str(tmp_path.resolve())
    assert environment["PYTHONHASHSEED"] == "0"


def test_capture_records_failed_baseline_without_marking_it_complete(tmp_path: Path) -> None:
    workspace = _clean_git_workspace(tmp_path)
    environment = create_environment_manifest(workspace)
    command_manifest = tmp_path / "commands.json"
    command_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "kind": "r0_command_manifest",
                "baseline_commit": environment["workspace_commit"],
                "probes": [
                    {
                        "probe_id": "synthetic" if index == 0 else f"unused-{index}",
                        "matrix_row_id": row.row_id,
                        "contract_version": "1",
                        "argv": (
                            [
                                "{python}",
                                "-c",
                                (
                                    "import json; print(json.dumps({"
                                    "'status':'fail','invariant_passed':False,"
                                    "'assertions':[{'name':'unsafe','passed':False,"
                                    "'detail':'baseline'}]}))"
                                ),
                            ]
                            if index == 0
                            else ["{python}", "-c", "raise SystemExit('not selected')"]
                        ),
                        "cwd": ".",
                        "test_references": ["tests/r0/test_evidence_harness.py"],
                    }
                    for index, row in enumerate(MATRIX_ROWS)
                ],
            }
        ),
        encoding="utf-8",
    )
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    output = tmp_path / "record.json"
    record = capture_probe(
        command_manifest_path=command_manifest,
        probe_id="synthetic",
        phase="before",
        workspace=workspace,
        environment_manifest_path=environment_path,
        output=output,
        variables={},
    )
    assert record["outcome"]["status"] == "fail"
    assert record["outcome"]["invariant_passed"] is False
    assert record["review"]["status"] == "pending"
    assert output.is_file()
    assert validate_record(record) == []


def test_capture_rejects_an_undeclared_command_placeholder(tmp_path: Path) -> None:
    workspace = _clean_git_workspace(tmp_path)
    environment = create_environment_manifest(workspace)
    command_manifest = tmp_path / "commands.json"
    command_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "kind": "r0_command_manifest",
                "baseline_commit": environment["workspace_commit"],
                "probes": [
                    {
                        "probe_id": "synthetic" if index == 0 else f"unused-{index}",
                        "matrix_row_id": row.row_id,
                        "contract_version": "1",
                        "argv": (
                            ["{python}", "{undeclared}"]
                            if index == 0
                            else ["{python}", "-c", "raise SystemExit('not selected')"]
                        ),
                        "cwd": ".",
                        "test_references": ["tests/r0/test_evidence_harness.py"],
                    }
                    for index, row in enumerate(MATRIX_ROWS)
                ],
            }
        ),
        encoding="utf-8",
    )
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")

    try:
        capture_probe(
            command_manifest_path=command_manifest,
            probe_id="synthetic",
            phase="before",
            workspace=workspace,
            environment_manifest_path=environment_path,
            output=tmp_path / "record.json",
            variables={},
        )
    except ValueError as exc:
        assert str(exc) == "Unknown command placeholder {undeclared}"
    else:
        raise AssertionError("undeclared command placeholder was accepted")


def test_committed_command_manifest_covers_each_matrix_row_once() -> None:
    workspace = Path(__file__).resolve().parents[2]
    path = (
        workspace
        / "docs/changes/system-engineering/2026-07-14_SYSTEM_R0_EVIDENCE_containment"
        / "2026-07-14_SYSTEM_R0_EVIDENCE_command_manifest_v2.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert validate_command_manifest(manifest) == []
    assert manifest["comparison_contract_version"] == "2"
    assert set(manifest["runtime_bindings"]) == {"agents_python", "data_python", "python"}
    for relative, expected in manifest["fixture_sha256"].items():
        assert sha256_file(workspace / relative) == expected


def test_committed_matrix_manifest_matches_the_code_owner() -> None:
    workspace = Path(__file__).resolve().parents[2]
    path = (
        workspace
        / "docs/changes/system-engineering/2026-07-14_SYSTEM_R0_EVIDENCE_containment"
        / "2026-07-14_SYSTEM_R0_EVIDENCE_matrix_rows.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == matrix_manifest()


def test_baseline_manifest_binds_each_expected_failing_record() -> None:
    workspace = Path(__file__).resolve().parents[2]
    package = (
        workspace / "docs/changes/system-engineering/2026-07-14_SYSTEM_R0_EVIDENCE_containment"
    )
    manifest = json.loads(
        (package / "2026-07-14_SYSTEM_R0_EVIDENCE_baseline_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["candidate_commit"] is None
    assert manifest["closure_status"] == "open"
    assert manifest["dirty_state_guard"]["workspace_dirty"] is False
    assert {item["matrix_row_id"] for item in manifest["evidence_records"]} == {
        row.row_id for row in MATRIX_ROWS
    }

    bound = [
        manifest["command_manifest"],
        manifest["matrix_manifest"],
        manifest["environment_manifest"],
        *manifest["evidence_records"],
    ]
    for item in bound:
        path = package / item["path"]
        assert path.is_file()
        assert sha256_file(path) == item["sha256"]
    for item in manifest["evidence_records"]:
        record = json.loads((package / item["path"]).read_text(encoding="utf-8"))
        assert record["matrix_row_id"] == item["matrix_row_id"]
        assert record["outcome"]["status"] == "fail"
        assert record["outcome"]["invariant_passed"] is False


def test_baseline_v2_manifest_binds_the_full_environment_series() -> None:
    workspace = Path(__file__).resolve().parents[2]
    package = (
        workspace / "docs/changes/system-engineering/2026-07-14_SYSTEM_R0_EVIDENCE_containment"
    )
    manifest = json.loads(
        (package / "2026-07-14_SYSTEM_R0_EVIDENCE_baseline_manifest_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["series_id"] == "r0-baseline-2"
    assert manifest["comparison_contract_version"] == "2"
    assert manifest["environment_manifest"]["comparison_fingerprint"]
    assert {item["matrix_row_id"] for item in manifest["evidence_records"]} == {
        row.row_id for row in MATRIX_ROWS
    }
    for item in [
        manifest["command_manifest"],
        manifest["matrix_manifest"],
        manifest["environment_manifest"],
        *manifest["evidence_records"],
    ]:
        assert sha256_file(package / item["path"]) == item["sha256"]
    fingerprints = set()
    for item in manifest["evidence_records"]:
        record = json.loads((package / item["path"]).read_text(encoding="utf-8"))
        fingerprints.add(record["environment_manifest"]["comparison_fingerprint"])
        assert validate_record(record) == []
        assert record["outcome"]["invariant_passed"] is False
    assert fingerprints == {manifest["environment_manifest"]["comparison_fingerprint"]}
