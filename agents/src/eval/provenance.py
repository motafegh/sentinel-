"""Fail-closed provenance validation for evaluation inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.contracts.execution import parse_status, require_eligible_payload


class EvaluationInputError(ValueError):
    """Raised when a report is unsafe to include in measured evaluation."""


def validate_report_ml_provenance(
    report: Mapping[str, Any],
    *,
    report_path: Path | str,
) -> None:
    """Require one live, digest-bound ML result consistently carried by the report."""

    location = str(report_path)
    ml_result = report.get("ml_result")
    if not isinstance(ml_result, Mapping):
        raise EvaluationInputError(f"{location}: ml_result is missing or malformed")

    try:
        result_status = require_eligible_payload(ml_result, purpose="evaluation")
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{location}: ineligible ML result: {exc}") from exc

    final_report = report.get("final_report") or {}
    tool_status = report.get("tool_status") or final_report.get("tool_status") or {}
    if not isinstance(tool_status, Mapping):
        raise EvaluationInputError(f"{location}: tool_status is missing or malformed")

    try:
        carried_status = parse_status(tool_status.get("ml"))
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{location}: tool_status.ml is missing or malformed") from exc

    if carried_status != result_status:
        raise EvaluationInputError(f"{location}: ml_result and tool_status.ml provenance disagree")


__all__ = ["EvaluationInputError", "validate_report_ml_provenance"]
