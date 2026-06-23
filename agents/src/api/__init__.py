"""
SENTINEL API gateway (WS6a / Phase C.1, 2026-06-22).

Public HTTP surface for the SENTINEL audit pipeline.

Exports (all deferred — see `__getattr__` below to avoid circular imports
when submodules import each other in the wrong order):
    - `models`           — Pydantic request/response schemas.
    - `job_store`        — In-memory job record store.
    - `gateway`          — FastAPI app + background runner.
    - `run`              — Convenience entry point: `python -m src.api.gateway`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.gateway import create_app, run
    from src.api.job_store import JobRecord, JobStatus, JobStore
    from src.api.models import (
        AuditRequest,
        ErrorResponse,
        HealthResponse,
        JobResponse,
        ServiceHealth,
    )


# PEP 562 — defer submodule imports so circular references (gateway ↔ job_store)
# don't break eager `from src.api import models` imports.
def __getattr__(name: str):
    if name in ("AuditRequest", "ErrorResponse", "HealthResponse", "JobResponse", "ServiceHealth"):
        from src.api import models
        return getattr(models, name)
    if name in ("JobRecord", "JobStore"):
        from src.api import job_store
        return getattr(job_store, name)
    if name == "JobStatus":
        # Re-exported from job_store for convenience (it's a str enum).
        from src.api.job_store import JobStatus
        return JobStatus
    if name in ("create_app", "run"):
        from src.api import gateway
        return getattr(gateway, name)
    if name in ("gateway", "job_store", "models"):
        import importlib
        return importlib.import_module(f"src.api.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AuditRequest",
    "ErrorResponse",
    "HealthResponse",
    "JobRecord",
    "JobResponse",
    "JobStatus",
    "JobStore",
    "ServiceHealth",
    "create_app",
    "gateway",
    "job_store",
    "models",
    "run",
]
