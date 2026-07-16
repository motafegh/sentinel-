"""Canonical dependency execution status, provenance, and eligibility rules."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionState(str, Enum):
    """Exhaustive terminal states for a selected dependency or tool attempt."""

    SUCCEEDED = "SUCCEEDED"
    CLEAN = "CLEAN"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    SKIPPED_POLICY = "SKIPPED_POLICY"
    UNAVAILABLE = "UNAVAILABLE"
    MOCK = "MOCK"


class ExecutionStatus(BaseModel):
    """Strict, serializable status carried with every dependency result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    status: ExecutionState
    attempted: bool
    ran: bool
    reason_code: str = Field(min_length=1, pattern=r"^[a-z0-9_.-]+$")
    detail: str = ""
    dependency: str = Field(min_length=1)
    provenance: Literal["live", "mock", "policy"]
    input_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    duration_ms: float = Field(ge=0)
    attempt: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_semantics(self) -> "ExecutionStatus":
        if self.status in {ExecutionState.SUCCEEDED, ExecutionState.CLEAN}:
            if not self.attempted or not self.ran or self.provenance != "live":
                raise ValueError("successful/clean status requires attempted live execution")
            if self.input_digest is None or self.output_digest is None:
                raise ValueError("successful/clean status requires bound input and output digests")
        elif self.status is ExecutionState.MOCK:
            if not self.ran or self.provenance != "mock" or self.output_digest is None:
                raise ValueError(
                    "mock status requires mock provenance, ran=true, and output digest"
                )
        elif self.status is ExecutionState.SKIPPED_POLICY:
            if self.attempted or self.ran or self.provenance != "policy":
                raise ValueError(
                    "policy skip requires attempted=false, ran=false, policy provenance"
                )
        elif self.ran and self.status in {ExecutionState.FAILED, ExecutionState.UNAVAILABLE}:
            raise ValueError("failed/unavailable status cannot claim ran=true")
        return self


def digest_payload(value: Any) -> str:
    """Return the canonical SHA-256 for a JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _result_material(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "execution_status"}


def bind_status(
    payload: Mapping[str, Any],
    *,
    dependency: str,
    input_payload: Any,
    duration_ms: float,
    clean: bool = False,
    attempt: int = 1,
) -> dict[str, Any]:
    """Attach a successful live status cryptographically bound to input and output."""

    result = dict(payload)
    status = ExecutionStatus(
        status=ExecutionState.CLEAN if clean else ExecutionState.SUCCEEDED,
        attempted=True,
        ran=True,
        reason_code="clean" if clean else "completed",
        dependency=dependency,
        provenance="live",
        input_digest=digest_payload(input_payload),
        output_digest=digest_payload(result),
        duration_ms=duration_ms,
        attempt=attempt,
    )
    result["execution_status"] = status.model_dump(mode="json")
    return result


def mock_status(
    payload: Mapping[str, Any],
    *,
    dependency: str,
    input_payload: Any,
    duration_ms: float = 0,
) -> dict[str, Any]:
    """Attach explicit mock provenance; mock output is never evidence eligible."""

    result = dict(payload)
    status = ExecutionStatus(
        status=ExecutionState.MOCK,
        attempted=False,
        ran=True,
        reason_code="explicit_mock",
        dependency=dependency,
        provenance="mock",
        input_digest=digest_payload(input_payload),
        output_digest=digest_payload(result),
        duration_ms=duration_ms,
        attempt=1,
    )
    result["execution_status"] = status.model_dump(mode="json")
    return result


def failure_status(
    state: ExecutionState,
    *,
    dependency: str,
    reason_code: str,
    detail: str,
    attempted: bool,
    input_payload: Any | None = None,
    duration_ms: float = 0,
    attempt: int = 1,
) -> dict[str, Any]:
    """Build a terminal non-success status without prediction/evidence fields."""

    if state not in {
        ExecutionState.DEGRADED,
        ExecutionState.FAILED,
        ExecutionState.SKIPPED_POLICY,
        ExecutionState.UNAVAILABLE,
    }:
        raise ValueError(f"failure_status does not accept {state.value}")
    status = ExecutionStatus(
        status=state,
        attempted=attempted,
        ran=state is ExecutionState.DEGRADED,
        reason_code=reason_code,
        detail=detail,
        dependency=dependency,
        provenance="policy" if state is ExecutionState.SKIPPED_POLICY else "live",
        input_digest=digest_payload(input_payload) if input_payload is not None else None,
        output_digest=None,
        duration_ms=duration_ms,
        attempt=attempt,
    )
    return status.model_dump(mode="json")


def parse_status(value: Any) -> ExecutionStatus:
    """Parse an untrusted status value using the strict canonical schema."""

    return ExecutionStatus.model_validate(value)


def status_allows_evidence(value: Any) -> bool:
    """Return true only for complete, live, successful dependency output."""

    try:
        status = parse_status(value)
    except (TypeError, ValueError):
        return False
    return status.status in {ExecutionState.SUCCEEDED, ExecutionState.CLEAN}


def require_eligible_payload(
    payload: Mapping[str, Any],
    *,
    purpose: str,
    input_payload: Any | None = None,
) -> ExecutionStatus:
    """Reject missing, malformed, mock, degraded, or mutated result provenance."""

    status = parse_status(payload.get("execution_status"))
    if not status_allows_evidence(status):
        raise ValueError(f"{purpose} requires live successful execution, got {status.status.value}")
    if input_payload is not None and status.input_digest != digest_payload(input_payload):
        raise ValueError(f"{purpose} input digest does not match execution status")
    actual_digest = digest_payload(_result_material(payload))
    if status.output_digest != actual_digest:
        raise ValueError(f"{purpose} result digest does not match execution status")
    return status


def availability_label(value: Any) -> str:
    """Map canonical dependency status to the public readiness vocabulary."""

    try:
        status = parse_status(value).status
    except (TypeError, ValueError):
        return "unavailable"
    if status in {ExecutionState.SUCCEEDED, ExecutionState.CLEAN}:
        return "live"
    if status is ExecutionState.MOCK:
        return "mock"
    if status is ExecutionState.DEGRADED:
        return "degraded"
    return "unavailable"


__all__ = [
    "ExecutionState",
    "ExecutionStatus",
    "availability_label",
    "bind_status",
    "digest_payload",
    "failure_status",
    "mock_status",
    "parse_status",
    "require_eligible_payload",
    "status_allows_evidence",
]
