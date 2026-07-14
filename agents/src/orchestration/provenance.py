"""Canonical ML provenance checks shared by orchestration consumers."""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.execution import require_eligible_payload


def eligible_ml_result(state: Mapping[str, Any], *, purpose: str) -> dict[str, Any]:
    """Return ML output only when live provenance and payload bindings verify."""

    result = state.get("ml_result")
    if not isinstance(result, dict):
        return {}
    contract_code = state.get("contract_code")
    input_payload = {"source_code": contract_code} if isinstance(contract_code, str) else None
    try:
        require_eligible_payload(
            result,
            purpose=purpose,
            input_payload=input_payload,
        )
    except (TypeError, ValueError):
        return {}
    return result


__all__ = ["eligible_ml_result"]
