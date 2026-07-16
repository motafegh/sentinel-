"""Shared runtime contracts used across SENTINEL AGENTS boundaries."""

from src.contracts.execution import (
    ExecutionState,
    ExecutionStatus,
    availability_label,
    bind_status,
    failure_status,
    mock_status,
    parse_status,
    require_eligible_payload,
    status_allows_evidence,
)
from src.contracts.submission import SUBMISSION_SCHEMA_VERSION, normalize_submission

__all__ = [
    "ExecutionState",
    "ExecutionStatus",
    "availability_label",
    "bind_status",
    "failure_status",
    "mock_status",
    "parse_status",
    "require_eligible_payload",
    "status_allows_evidence",
    "SUBMISSION_SCHEMA_VERSION",
    "normalize_submission",
]
