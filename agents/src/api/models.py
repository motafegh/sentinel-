"""
Pydantic schemas for the SENTINEL audit gateway (WS6a / C.1, 2026-06-22).

Lifecycle:
    1. Client POSTs an `AuditRequest` (contract code + optional address).
    2. Server returns a `JobResponse` with `status="queued"` and a `job_id`.
    3. Client polls `GET /audit/{job_id}` → `JobResponse` (status may be
       queued | running | completed | failed) until completed.
    4. When status == "completed", `JobResponse.report` carries the audit
       report (the `final_report` dict from the LangGraph output).

Sizing limits:
    - `MAX_CONTRACT_CHARS = 200_000` — hard cap on `contract_code` length.
      Contracts above this are rejected with 422 (Pydantic validator).

These limits are intentionally generous: real-world contracts are typically
<50 KB; the cap allows very large audited files (e.g. sprawling proxy
implementations) but blocks abuse (a 1 GB POST would DoS the job store).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Sizing limits ────────────────────────────────────────────────────────
MAX_CONTRACT_CHARS = 200_000
MAX_AUDIT_TIMEOUT_S = 3_600.0  # matches timeouts.UNBOUNDED_TIMEOUT_S
DEFAULT_AUDIT_TIMEOUT_S = 300.0  # matches run_real_audit.py default

# A *very* loose Solidity smell test — used to reject obviously non-Solidity
# submissions early. Not a parser: real syntax validation happens in
# quick_screen (Slither) and static_analysis (Aderyn) nodes. We just want
# to catch the "user pasted the wrong file" case.
_SOLIDITY_HINTS = re.compile(
    r"\b(pragma|contract|library|interface|function|event|mapping|address|uint|bytes)\b",
    re.IGNORECASE,
)


# ── Request ──────────────────────────────────────────────────────────────
class AuditRequest(BaseModel):
    """Request body for `POST /audit`."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "contract_code": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\ncontract Vault {}\n",
                "contract_address": "0xdeadbeef",
                "audit_timeout_s": 300.0,
                "metadata": {"source": "manual-paste"},
            }
        },
    )

    contract_code: str = Field(
        ..., min_length=1, max_length=MAX_CONTRACT_CHARS,
        description="Solidity source code to audit. Capped at "
                    f"{MAX_CONTRACT_CHARS:,} chars to prevent DoS.",
    )
    contract_address: str | None = Field(
        default=None,
        description="On-chain address. Optional — if omitted, a deterministic "
                    "0x... address is derived from a hash of contract_code.",
    )
    audit_timeout_s: float = Field(
        default=DEFAULT_AUDIT_TIMEOUT_S,
        gt=0, le=MAX_AUDIT_TIMEOUT_S,
        description="Per-audit wall-clock timeout in seconds. Default 300s. "
                    f"Max {MAX_AUDIT_TIMEOUT_S:.0f}s.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata stored alongside the job. NOT used by "
                    "the audit pipeline — purely for caller bookkeeping.",
    )

    @field_validator("contract_code")
    @classmethod
    def _looks_like_solidity(cls, v: str) -> str:
        """Warn (not reject) if the source doesn't look like Solidity.

        We do NOT reject: some legitimate inputs (e.g. raw bytecode in
        `0x...` form for Gigahorse analysis, or partial snippets) won't
        contain the Solidity keywords. The static-analysis nodes will
        report "not Solidity" themselves. But we do reject if the input
        is *clearly* binary (looks like a compiled artifact with a long
        string of non-printable characters).
        """
        non_printable = sum(1 for c in v if not c.isprintable() and c not in "\n\r\t")
        if len(v) > 0 and non_printable / len(v) > 0.05:
            raise ValueError(
                "contract_code contains >5% non-printable characters "
                "(looks like binary, not source code)"
            )
        return v


# ── Response ─────────────────────────────────────────────────────────────
class JobResponse(BaseModel):
    """Response body for `GET /audit/{job_id}` (and the immediate POST ack)."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., description="Server-assigned UUID for this job.")
    status: str = Field(
        ..., description="One of: queued, running, completed, failed.",
    )
    submitted_at: str = Field(..., description="ISO-8601 UTC timestamp.")
    started_at: str | None = Field(
        default=None, description="ISO-8601 UTC timestamp; null until status=running.",
    )
    finished_at: str | None = Field(
        default=None, description="ISO-8601 UTC timestamp; null until status=completed or failed.",
    )
    contract_address: str | None = Field(
        default=None, description="Echoed from the request (or generated).",
    )
    audit_timeout_s: float = Field(
        default=300.0,
        description="Per-audit wall-clock timeout in seconds (echoed from request).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Echoed from request.metadata — caller bookkeeping only.",
    )
    error: str | None = Field(
        default=None, description="Failure reason; null unless status=failed.",
    )
    report: dict[str, Any] | None = Field(
        default=None, description="Audit report dict (final_report) — present "
                                  "only when status=completed.",
    )


class ServiceHealth(BaseModel):
    """Health snapshot of one upstream service the gateway depends on."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="'ok' or 'degraded'.")
    gateway: str = Field(..., description="Gateway version string.")
    jobs: dict[str, int] = Field(
        ..., description="Live job counts: {queued, running, completed, failed}.",
    )
    services: list[ServiceHealth] = Field(
        default_factory=list,
        description="Upstream health checks (ml_api, mcp_*, lm_studio). "
                    "Empty list if --no-services was used at startup.",
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str | None = None
    job_id: str | None = None
