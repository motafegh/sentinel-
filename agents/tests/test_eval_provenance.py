from __future__ import annotations

import pytest
from src.contracts.execution import bind_status, mock_status
from src.eval.provenance import EvaluationInputError, validate_report_ml_provenance


def _report(ml_result: dict) -> dict:
    return {
        "ml_result": ml_result,
        "tool_status": {"ml": ml_result.get("execution_status")},
    }


def test_live_bound_result_is_accepted() -> None:
    result = bind_status(
        {"probabilities": {"Reentrancy": 0.8}},
        dependency="module1-inference",
        input_payload={"source_code": "contract C {}"},
        duration_ms=1,
    )
    validate_report_ml_provenance(_report(result), report_path="report.json")


@pytest.mark.parametrize("case", ["missing", "mock", "mutated", "mismatched"])
def test_ineligible_ml_report_is_rejected(case: str) -> None:
    live = bind_status(
        {"probabilities": {"Reentrancy": 0.8}},
        dependency="module1-inference",
        input_payload={"source_code": "contract C {}"},
        duration_ms=1,
    )
    report = _report(live)

    if case == "missing":
        report = {"ml_result": {"probabilities": {"Reentrancy": 0.8}}}
    elif case == "mock":
        mocked = mock_status(
            {"probabilities": {"Reentrancy": 0.8}},
            dependency="module1-inference",
            input_payload={"source_code": "contract C {}"},
        )
        report = _report(mocked)
    elif case == "mutated":
        report["ml_result"]["probabilities"]["Reentrancy"] = 0.01
    else:
        other = bind_status(
            {"probabilities": {"Reentrancy": 0.1}},
            dependency="module1-inference",
            input_payload={"source_code": "contract C {}"},
            duration_ms=1,
        )
        report["tool_status"]["ml"] = other["execution_status"]

    with pytest.raises(EvaluationInputError):
        validate_report_ml_provenance(report, report_path="report.json")
