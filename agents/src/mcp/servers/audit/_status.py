"""Canonical AuditRegistry result status and mutable readiness publication."""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.execution import ExecutionState, bind_status, failure_status, mock_status

DEPENDENCY = "audit-registry"


def _publish(status: dict[str, Any]) -> None:
    from src.mcp.servers import audit_server as audit_shim

    audit_shim._execution_status = status


def live_result(
    payload: Mapping[str, Any],
    *,
    operation: str,
    arguments: Mapping[str, Any],
    duration_ms: float,
    clean: bool = False,
) -> dict[str, Any]:
    result = bind_status(
        payload,
        dependency=DEPENDENCY,
        input_payload={"operation": operation, "arguments": dict(arguments)},
        duration_ms=duration_ms,
        clean=clean,
    )
    _publish(result["execution_status"])
    return result


def explicit_mock_result(
    payload: Mapping[str, Any],
    *,
    operation: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    result = mock_status(
        payload,
        dependency=DEPENDENCY,
        input_payload={"operation": operation, "arguments": dict(arguments)},
    )
    _publish(result["execution_status"])
    return result


def unavailable_result(
    payload: Mapping[str, Any],
    *,
    operation: str,
    arguments: Mapping[str, Any],
    reason_code: str,
    detail: str,
    attempted: bool,
    duration_ms: float = 0,
) -> dict[str, Any]:
    result = dict(payload)
    status = failure_status(
        ExecutionState.UNAVAILABLE,
        dependency=DEPENDENCY,
        reason_code=reason_code,
        detail=detail,
        attempted=attempted,
        input_payload={"operation": operation, "arguments": dict(arguments)},
        duration_ms=duration_ms,
    )
    result["execution_status"] = status
    _publish(status)
    return result


def failed_result(
    payload: Mapping[str, Any],
    *,
    operation: str,
    arguments: Mapping[str, Any],
    reason_code: str,
    detail: str,
    attempted: bool,
    duration_ms: float = 0,
) -> dict[str, Any]:
    result = dict(payload)
    status = failure_status(
        ExecutionState.FAILED,
        dependency=DEPENDENCY,
        reason_code=reason_code,
        detail=detail,
        attempted=attempted,
        input_payload={"operation": operation, "arguments": dict(arguments)},
        duration_ms=duration_ms,
    )
    result["execution_status"] = status
    _publish(status)
    return result


__all__ = [
    "DEPENDENCY",
    "explicit_mock_result",
    "failed_result",
    "live_result",
    "unavailable_result",
]
