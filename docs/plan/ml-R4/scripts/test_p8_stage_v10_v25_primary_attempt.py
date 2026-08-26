from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "p8_stage_v10_v25_primary_attempt.py"
SPEC = importlib.util.spec_from_file_location("p8_stage_v10_v25_primary_attempt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def _write_triple(root: Path, source: str, contract_id: str, *, slither: str = "0.10.0") -> None:
    directory = root / source
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{contract_id}.pt").write_bytes(b"graph:" + contract_id.encode())
    (directory / f"{contract_id}.tokens.pt").write_bytes(b"tokens:" + contract_id.encode())
    sidecar = {
        "sha256": contract_id,
        "source": source,
        "schema_version": stage.V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": stage.V10_REPRESENTATION_EXTRACTOR_VERSION,
        "token_lineage": "accepted_v9_byte_copy",
        "graph_extraction_mode": "slither_full_analysis",
        "graph_analysis_degraded": False,
        "unclassified_call_ir": [],
        "call_mapping_errors": [],
        "classified_call_ir_counts": {},
        "emitted_call_edge_counts": {},
        "slither_runtime": {
            "slither_analyzer": slither,
            "runtime_role": "primary",
            "required_for_physical_acceptance": stage.V10_PRIMARY_SLITHER_VERSION,
        },
    }
    (directory / f"{contract_id}.rep.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exception_id = "e" * 64
    normal_id = "a" * 64
    monkeypatch.setattr(
        stage,
        "V10_SLITHER_RUNTIME_EXCEPTIONS",
        {exception_id: "0.11.5"},
    )

    accepted = tmp_path / "accepted"
    attempt = tmp_path / "attempt" / stage.V10_REPRESENTATION_ROOT_NAME
    output = tmp_path / "final" / stage.V10_REPRESENTATION_ROOT_NAME

    _write_triple(accepted, "dive", normal_id)
    _write_triple(accepted, "dive", exception_id)
    _write_triple(attempt, "dive", normal_id)

    # Candidate tokens must be byte-identical to accepted V9 tokens.
    (attempt / "dive" / f"{normal_id}.tokens.pt").write_bytes(
        (accepted / "dive" / f"{normal_id}.tokens.pt").read_bytes()
    )
    (attempt / "dive" / "representation_failures.jsonl").write_text(
        json.dumps({"meta_path": f"{exception_id}.meta.json", "error": "runtime"}) + "\n",
        encoding="utf-8",
    )
    (attempt / "dive" / "repaired_representation_manifest.json").write_text(
        json.dumps({"source": "dive"}), encoding="utf-8"
    )
    return normal_id, exception_id, accepted, attempt, output


def test_primary_stage_accepts_exact_declared_exception_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normal_id, exception_id, accepted, attempt, output = _fixture(tmp_path, monkeypatch)
    report = stage.stage_primary_attempt(
        SimpleNamespace(
            primary_attempt_root=attempt,
            accepted_v9_root=accepted,
            output_root=output,
        )
    )

    assert report["passed"] is True
    assert report["accepted_v9_contracts"] == 2
    assert report["staged_primary_contracts"] == 1
    assert report["runtime_exception_contracts"] == 1
    assert report["missing_runtime_exception_identities"] == [f"dive/{exception_id}"]
    assert (output / "dive" / f"{normal_id}.rep.json").is_file()
    assert not (output / "dive" / f"{exception_id}.rep.json").exists()


def test_primary_stage_rejects_unexpected_missing_primary_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _normal_id, _exception_id, accepted, attempt, output = _fixture(tmp_path, monkeypatch)
    for path in list((attempt / "dive").glob("a" * 64 + ".*")):
        path.unlink()
    with pytest.raises(ValueError, match="primary attempt population mismatch"):
        stage.stage_primary_attempt(
            SimpleNamespace(
                primary_attempt_root=attempt,
                accepted_v9_root=accepted,
                output_root=output,
            )
        )


def test_primary_stage_rejects_failure_set_outside_declared_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normal_id, _exception_id, accepted, attempt, output = _fixture(tmp_path, monkeypatch)
    failure_file = attempt / "dive" / "representation_failures.jsonl"
    failure_file.write_text(
        json.dumps({"meta_path": f"{normal_id}.meta.json", "error": "wrong"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failure set does not exactly match"):
        stage.stage_primary_attempt(
            SimpleNamespace(
                primary_attempt_root=attempt,
                accepted_v9_root=accepted,
                output_root=output,
            )
        )


def test_primary_stage_rejects_non_primary_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normal_id, _exception_id, accepted, attempt, output = _fixture(tmp_path, monkeypatch)
    sidecar_path = attempt / "dive" / f"{normal_id}.rep.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["slither_runtime"]["slither_analyzer"] = "0.11.5"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="not generated under primary Slither"):
        stage.stage_primary_attempt(
            SimpleNamespace(
                primary_attempt_root=attempt,
                accepted_v9_root=accepted,
                output_root=output,
            )
        )
