from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).with_name("p8_generate_v10_v25_primary_attempt.py")
SCRIPT_DIR = str(SCRIPT.parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location(
    "p8_generate_v10_v25_primary_attempt",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_population(tmp_path: Path) -> tuple[Path, Path, str, str]:
    accepted = tmp_path / "accepted"
    preprocessed = tmp_path / "preprocessed"
    source = "dive"
    (accepted / source).mkdir(parents=True)
    (preprocessed / source).mkdir(parents=True)

    exception_id = next(iter(MODULE.V10_SLITHER_RUNTIME_EXCEPTIONS))
    ordinary_id = "a" * 64
    assert ordinary_id != exception_id

    for contract_id in (ordinary_id, exception_id):
        (accepted / source / f"{contract_id}.rep.json").write_text(
            json.dumps({"sha256": contract_id, "source": source}),
            encoding="utf-8",
        )
        (preprocessed / source / f"{contract_id}.meta.json").write_text(
            json.dumps({"sha256": contract_id}),
            encoding="utf-8",
        )
        (preprocessed / source / f"{contract_id}.sol").write_text(
            "contract Fixture {}\n",
            encoding="utf-8",
        )
    return accepted, preprocessed, ordinary_id, exception_id


def test_exception_key_resolves_uniquely(tmp_path: Path) -> None:
    accepted, _, _, exception_id = _write_population(tmp_path)
    inventory = MODULE._inventory_sidecars(accepted)

    assert MODULE._resolve_exception_keys(inventory) == {("dive", exception_id)}


def test_deferred_failure_records_required_runtime() -> None:
    exception_id, required = next(iter(MODULE.V10_SLITHER_RUNTIME_EXCEPTIONS.items()))

    row = MODULE._deferred_failure(exception_id)

    assert row["meta_path"] == f"{exception_id}.meta.json"
    assert row["error_type"] == "IdentityBoundRuntimeDeferred"
    assert required in row["error"]


def test_primary_attempt_never_invokes_exception_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, preprocessed, ordinary_id, exception_id = _write_population(tmp_path)
    output = tmp_path / "attempt" / MODULE.V10_REPRESENTATION_ROOT_NAME

    monkeypatch.setattr(
        MODULE,
        "_require_primary_runtime",
        lambda: {
            "slither_analyzer": "0.10.0",
            "crytic_compile": "0.3.11",
            "runtime_role": "primary",
        },
    )

    observed_worker_ids: list[str] = []

    def fake_run(worker_args, workers):
        assert workers == 2
        rows = []
        for args in worker_args:
            meta_path = Path(args[1])
            contract_id = meta_path.name.removesuffix(".meta.json")
            observed_worker_ids.append(contract_id)
            output_dir = Path(args[3])
            (output_dir / f"{contract_id}.rep.json").write_text(
                "{}\n", encoding="utf-8"
            )
            rows.append(
                (
                    True,
                    {"graph_extraction_mode": "slither_full_analysis"},
                    None,
                )
            )
        return rows

    monkeypatch.setattr(MODULE, "_run_workers", fake_run)

    report = MODULE.build_primary_attempt(
        SimpleNamespace(
            workers=2,
            output_root=output,
            accepted_v9_root=accepted,
            preprocessed_root=preprocessed,
        )
    )

    assert report["passed"] is True
    assert report["representations_written"] == 1
    assert report["runtime_exceptions_deferred"] == 1
    assert report["observed_attempt_contracts"] == 1
    assert observed_worker_ids == [ordinary_id]
    assert exception_id not in observed_worker_ids
    failures = [
        json.loads(line)
        for line in (
            output / "dive" / "representation_failures.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert failures == [MODULE._deferred_failure(exception_id)]


def test_unexpected_ordinary_failure_keeps_attempt_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, preprocessed, ordinary_id, _ = _write_population(tmp_path)
    output = tmp_path / "attempt" / MODULE.V10_REPRESENTATION_ROOT_NAME

    monkeypatch.setattr(MODULE, "_require_primary_runtime", lambda: {})
    monkeypatch.setattr(
        MODULE,
        "_run_workers",
        lambda worker_args, workers: [
            (
                False,
                None,
                {
                    "meta_path": f"{ordinary_id}.meta.json",
                    "error_type": "RuntimeError",
                    "error": "fixture failure",
                },
            )
            for _ in worker_args
        ],
    )

    report = MODULE.build_primary_attempt(
        SimpleNamespace(
            workers=1,
            output_root=output,
            accepted_v9_root=accepted,
            preprocessed_root=preprocessed,
        )
    )

    assert report["passed"] is False
    assert report["unexpected_failures_total"] == 1
    assert report["representations_written"] == 0


def test_population_mismatch_fails_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, preprocessed, ordinary_id, _ = _write_population(tmp_path)
    (preprocessed / "dive" / f"{ordinary_id}.meta.json").unlink()
    (preprocessed / "dive" / f"{ordinary_id}.sol").unlink()

    monkeypatch.setattr(MODULE, "_require_primary_runtime", lambda: {})

    with pytest.raises(ValueError, match="accepted/preprocessed population mismatch"):
        MODULE.build_primary_attempt(
            SimpleNamespace(
                workers=1,
                output_root=(
                    tmp_path / "attempt" / MODULE.V10_REPRESENTATION_ROOT_NAME
                ),
                accepted_v9_root=accepted,
                preprocessed_root=preprocessed,
            )
        )
