"""Fail-closed Module 1 HTTP bridge and explicit development mock."""

from __future__ import annotations

import time
from typing import Any

import httpx
from src.contracts.execution import ExecutionState, bind_status, failure_status, mock_status

DEPENDENCY = "module1-inference"


def _failed_result(
    state: ExecutionState,
    *,
    reason_code: str,
    detail: str,
    attempted: bool,
    input_payload: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "error": "inference_unavailable",
        "detail": detail,
        "execution_status": failure_status(
            state,
            dependency=DEPENDENCY,
            reason_code=reason_code,
            detail=detail,
            attempted=attempted,
            input_payload=input_payload,
            duration_ms=(time.monotonic() - started) * 1000,
        ),
    }


def _validate_prediction(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("prediction response must be a JSON object")
    label = payload.get("label")
    probabilities = payload.get("probabilities")
    model_hash = payload.get("model_hash")
    if not isinstance(label, str) or not label:
        raise ValueError("prediction response requires a non-empty label")
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("prediction response requires a probability map")
    if not isinstance(model_hash, str) or not model_hash:
        raise ValueError("prediction response requires model_hash provenance")
    if "execution_status" in payload:
        raise ValueError("Module 1 response cannot override bridge execution status")
    return payload


async def call_inference_api(
    contract_code: str,
    *,
    client: httpx.AsyncClient | None,
    module_url: str,
    timeout_s: float,
    mock_mode: bool,
) -> dict[str, Any]:
    """Return a bound live prediction or explicit terminal failure, never fallback mock."""

    input_payload = {"source_code": contract_code}
    started = time.monotonic()
    if mock_mode:
        return mock_prediction(contract_code)
    if client is None:
        return _failed_result(
            ExecutionState.UNAVAILABLE,
            reason_code="client_not_initialized",
            detail="shared inference HTTP client is not initialized",
            attempted=False,
            input_payload=input_payload,
            started=started,
        )

    try:
        response = await client.post(f"{module_url}/predict", json=input_payload)
        response.raise_for_status()
        prediction = _validate_prediction(response.json())
    except httpx.TimeoutException as exc:
        return _failed_result(
            ExecutionState.UNAVAILABLE,
            reason_code="timeout",
            detail=f"Module 1 timed out after {timeout_s:g}s: {exc}",
            attempted=True,
            input_payload=input_payload,
            started=started,
        )
    except httpx.HTTPStatusError as exc:
        detail = f"Module 1 returned HTTP {exc.response.status_code}"
        return _failed_result(
            ExecutionState.FAILED,
            reason_code="http_status",
            detail=detail,
            attempted=True,
            input_payload=input_payload,
            started=started,
        )
    except httpx.RequestError as exc:
        return _failed_result(
            ExecutionState.UNAVAILABLE,
            reason_code="request_error",
            detail=f"Module 1 is unreachable: {type(exc).__name__}: {exc}",
            attempted=True,
            input_payload=input_payload,
            started=started,
        )
    except (TypeError, ValueError) as exc:
        return _failed_result(
            ExecutionState.FAILED,
            reason_code="malformed_response",
            detail=str(exc),
            attempted=True,
            input_payload=input_payload,
            started=started,
        )

    return bind_status(
        prediction,
        dependency=DEPENDENCY,
        input_payload=input_payload,
        duration_ms=(time.monotonic() - started) * 1000,
    )


def mock_prediction(contract_code: str) -> dict[str, Any]:
    """Return a deterministic development-only prediction carrying MOCK provenance."""

    code_lower = contract_code.lower()
    has_reentrancy_pattern = "call.value" in code_lower or "transfer(" in code_lower
    class_names = [
        "Reentrancy",
        "IntegerUO",
        "GasException",
        "Timestamp",
        "TransactionOrderDependence",
        "ExternalBug",
        "CallToUnknown",
        "MishandledException",
        "UnusedReturn",
        "DenialOfService",
    ]
    probabilities: dict[str, float] = {
        "Reentrancy": 0.72 if has_reentrancy_pattern else 0.08,
        "IntegerUO": 0.54 if has_reentrancy_pattern else 0.12,
        "GasException": 0.18,
        "Timestamp": 0.31 if has_reentrancy_pattern else 0.14,
        "TransactionOrderDependence": 0.09,
        "ExternalBug": 0.14,
        "CallToUnknown": 0.07,
        "MishandledException": 0.22,
        "UnusedReturn": 0.19,
        "DenialOfService": 0.06,
    }
    confirmed = _tier(probabilities, minimum=0.55, maximum=None, name="CONFIRMED")
    suspicious = _tier(probabilities, minimum=0.25, maximum=0.55, name="SUSPICIOUS")
    label = "confirmed_vulnerable" if confirmed else "suspicious" if suspicious else "safe"
    payload = {
        "label": label,
        "probabilities": probabilities,
        "confirmed": confirmed,
        "suspicious": suspicious,
        "vulnerabilities": [
            {"vulnerability_class": item["vulnerability_class"], "probability": item["probability"]}
            for item in confirmed
        ],
        "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
        "thresholds": [0.5] * len(class_names),
        "truncated": False,
        "windows_used": 1,
        "num_nodes": 42,
        "num_edges": 58,
        "model_hash": "mock_model_hash_" + "0" * 46,
    }
    return mock_status(
        payload,
        dependency=DEPENDENCY,
        input_payload={"source_code": contract_code},
    )


def _tier(
    probabilities: dict[str, float],
    *,
    minimum: float,
    maximum: float | None,
    name: str,
) -> list[dict[str, Any]]:
    selected = [
        {"vulnerability_class": cls, "probability": probability, "tier": name}
        for cls, probability in probabilities.items()
        if probability >= minimum and (maximum is None or probability < maximum)
    ]
    return sorted(selected, key=lambda item: item["probability"], reverse=True)


__all__ = ["DEPENDENCY", "call_inference_api", "mock_prediction"]
