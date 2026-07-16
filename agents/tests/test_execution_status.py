"""Tests for the canonical dependency execution-status contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.contracts.execution import (
    ExecutionState,
    ExecutionStatus,
    availability_label,
    bind_status,
    failure_status,
    mock_status,
    require_eligible_payload,
    status_allows_evidence,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.src.inference.execution_status import bind_live_result as bind_ml_live_result


def test_live_success_binds_input_and_output_and_is_eligible() -> None:
    result = bind_status(
        {"label": "safe", "probabilities": {"Reentrancy": 0.1}},
        dependency="module1-inference",
        input_payload={"source_code": "contract Safe {}"},
        duration_ms=12.5,
    )
    status = require_eligible_payload(result, purpose="evaluation")
    assert status.status is ExecutionState.SUCCEEDED
    assert status_allows_evidence(result["execution_status"])
    assert availability_label(result["execution_status"]) == "live"


def test_mutated_result_fails_digest_binding() -> None:
    result = bind_status(
        {"label": "safe"},
        dependency="module1-inference",
        input_payload={"source_code": "contract Safe {}"},
        duration_ms=1,
    )
    result["label"] = "confirmed_vulnerable"
    with pytest.raises(ValueError, match="digest"):
        require_eligible_payload(result, purpose="proof")


def test_mismatched_input_fails_digest_binding() -> None:
    result = bind_status(
        {"label": "safe"},
        dependency="module1-inference",
        input_payload={"source_code": "contract A {}"},
        duration_ms=1,
    )
    with pytest.raises(ValueError, match="input digest"):
        require_eligible_payload(
            result,
            purpose="proof",
            input_payload={"source_code": "contract B {}"},
        )


def test_explicit_mock_is_visible_but_never_eligible() -> None:
    result = mock_status(
        {"label": "safe", "probabilities": {"Reentrancy": 0.1}},
        dependency="module1-inference",
        input_payload={"source_code": "contract Safe {}"},
    )
    assert result["execution_status"]["status"] == "MOCK"
    assert availability_label(result["execution_status"]) == "mock"
    assert not status_allows_evidence(result["execution_status"])
    with pytest.raises(ValueError, match="MOCK"):
        require_eligible_payload(result, purpose="finality")


def test_unavailable_status_has_no_output_digest_or_prediction() -> None:
    status = failure_status(
        ExecutionState.UNAVAILABLE,
        dependency="module1-inference",
        reason_code="connection_refused",
        detail="connection refused",
        attempted=True,
        input_payload={"source_code": "contract Safe {}"},
        duration_ms=2,
    )
    assert status["ran"] is False
    assert status["output_digest"] is None
    assert availability_label(status) == "unavailable"


def test_invalid_success_and_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionStatus.model_validate(
            {
                "schema_version": "1",
                "status": "SUCCEEDED",
                "attempted": False,
                "ran": True,
                "reason_code": "completed",
                "dependency": "module1",
                "provenance": "live",
                "duration_ms": 1,
                "attempt": 1,
                "unknown": True,
            }
        )


def test_ml_service_wire_output_matches_canonical_validator() -> None:
    result = bind_ml_live_result(
        {"fusion_embedding": [0.0] * 128, "model_hash": "a" * 64},
        dependency="module1-fusion",
        input_payload={"source_code": "contract C {}"},
        duration_ms=1,
    )
    status = require_eligible_payload(
        result,
        purpose="submission",
        input_payload={"source_code": "contract C {}"},
    )
    assert status.dependency == "module1-fusion"
