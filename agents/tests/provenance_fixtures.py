"""Test-only builders for canonical live dependency results."""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.execution import bind_status


def live_ml_result(
    payload: Mapping[str, Any],
    *,
    contract_code: str = "fixture",
) -> dict[str, Any]:
    material = {key: value for key, value in payload.items() if key != "execution_status"}
    return bind_status(
        material,
        dependency="module1-inference",
        input_payload={"source_code": contract_code},
        duration_ms=1,
    )


def live_ml_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    payload = result.get("ml_result")
    if not isinstance(payload, Mapping):
        raise TypeError("live_ml_state requires a mapping at ml_result")
    contract_code = result.get("contract_code", "fixture")
    result["ml_result"] = live_ml_result(payload, contract_code=str(contract_code))
    return result


def live_audit_result(
    payload: Mapping[str, Any],
    *,
    operation: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    return bind_status(
        payload,
        dependency="audit-registry",
        input_payload={"operation": operation, "arguments": dict(arguments)},
        duration_ms=1,
        clean=not bool(payload.get("records")),
    )


__all__ = ["live_audit_result", "live_ml_result", "live_ml_state"]
