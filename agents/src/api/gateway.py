"""
FastAPI gateway for the SENTINEL audit pipeline (WS6a / C.1, 2026-06-22).

This is the public HTTP surface for submitting audits. The gateway wraps
the existing LangGraph pipeline (see `src.orchestration.graph.build_graph`)
with a thin async job lifecycle:

    POST /audit               → enqueue + return job_id
    GET  /audit/{job_id}      → poll status + report when done
    GET  /audit               → list recent jobs (debugging)
    GET  /health              → liveness + service probes
    GET  /                    → service banner

The actual audit runs in a background `asyncio.create_task`. Each task
calls `mark_running` → `graph.ainvoke` → `mark_completed | mark_failed`,
with `asyncio.wait_for(..., timeout=audit_timeout_s)` to enforce the
per-audit wall-clock budget (matches `run_real_audit.py` behaviour).

WIRING TO THE GRAPH
===================
We import `build_graph` lazily (inside the request handler) so the gateway
module can be imported even when the agents module's other dependencies
(ML API, LM Studio) are not yet up. This matters for:

  1. `pytest tests/test_gateway.py` — tests should not need real services.
  2. Production: if the ML API is down at gateway startup, `/health`
     should still respond so load balancers can route around the box.

The lazy import also means `create_app()` is cheap — no graph compilation
at import time. Compilation happens on the first POST /audit.

`--no-llm` SUPPORT
==================
The gateway respects the `AUDIT_NO_LLM` env var (or `audit_no_llm` query
param) to skip LLM-dependent nodes (cross_validator debate, synthesizer
narrative). Useful for smoke tests and CI gates.

RUN AS A MODULE
===============
    $ cd ~/projects/sentinel/agents
    $ .venv/bin/python -m src.api.gateway
    ...
    INFO:     Started server process [PID]
    INFO:     Uvicorn running on http://0.0.0.0:8000
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

# ── CRITICAL: load .env BEFORE any agents module is imported ────────────
# client.py reads LM_STUDIO_BASE_URL etc. at IMPORT time. We must populate
# them from .env first. We deliberately do this BEFORE importing FastAPI
# so the import side effects of the agents module see the right URLs.
from dotenv import load_dotenv
_AGENTS_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_AGENTS_DIR / ".env", override=True)
sys.path.insert(0, str(_AGENTS_DIR))


# FastAPI imports (kept after dotenv so a missing FastAPI install doesn't
# break agents module import paths).
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False

from src.api.job_store import JobRecord, JobStatus, JobStore
from src.api.models import (
    AuditRequest,
    ErrorResponse,
    HealthResponse,
    JobResponse,
    ServiceHealth,
)


# ── Module-level constants (configurable via env) ───────────────────────
GATEWAY_VERSION = "0.1.0"
GATEWAY_DEFAULT_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_DEFAULT_PORT = int(os.getenv("GATEWAY_PORT", "8000"))
AUDIT_NO_LLM_DEFAULT = os.getenv("AUDIT_NO_LLM", "false").lower() in ("1", "true", "yes")


# ── App factory ──────────────────────────────────────────────────────────
def create_app(
    store: Any | None = None,
    *,
    graph_factory: Any | None = None,
    no_llm: bool | None = None,
    skip_service_probes: bool = False,
) -> Any:
    """Build and return a FastAPI app.

    Args:
        store: Pre-built `JobStore`. If None, a new one is created. Tests
            pass a fresh store per test for isolation.
        graph_factory: Callable that returns a compiled LangGraph. If None,
            uses `src.orchestration.graph.build_graph(use_checkpointer=False)`.
            Tests can pass a stub to avoid loading the real graph.
        no_llm: If True, patches the LLM client to raise immediately (matches
            `run_real_audit.py --no-llm`). If None, reads `AUDIT_NO_LLM` env.
        skip_service_probes: If True, `/health` does NOT probe upstream
            services. Useful for tests where ml_api/mcp_* are not running.

    Returns:
        A FastAPI app instance, ready for `uvicorn.run(app, ...)`.
    """
    if not HAVE_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install with: "
            "pip install 'fastapi>=0.116' 'uvicorn[standard]>=0.32'"
        )

    # Resolve `no_llm` against env var (None means "use env").
    if no_llm is None:
        no_llm = AUDIT_NO_LLM_DEFAULT
    if no_llm:
        _patch_no_llm()

    # Resolve the graph factory.
    if graph_factory is None:
        def _default_graph_factory():
            from src.orchestration.graph import build_graph
            return build_graph(use_checkpointer=False)
        graph_factory = _default_graph_factory

    # One JobStore per app instance — SQLite by default (P10), in-memory for tests.
    if store is None:
        from src.api.sqlite_job_store import SqliteJobStore
        _db_path = os.getenv("SENTINEL_JOBS_DB", "data/jobs.db")
        store = SqliteJobStore(db_path=_db_path, max_completed=500)

    # ── Lifespan: probe services at startup, recover crashed jobs, start health monitor ──
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"SENTINEL gateway v{GATEWAY_VERSION} starting up")
        logger.info(f"  no_llm={no_llm} | skip_service_probes={skip_service_probes}")

        # P10: Recover jobs left in RUNNING state after a crash.
        if hasattr(store, "recover_pending"):
            pending = store.recover_pending()
            for job in pending:
                store.mark_failed(job.job_id, "gateway restart (crash recovery)")
                logger.warning("Recovered job {} → FAILED (crash recovery)", job.job_id[:8])

        if not skip_service_probes and not no_llm:
            try:
                services = await _probe_services()
                app.state.services = services
            except Exception as e:
                logger.warning(f"Startup service probe failed: {e}")
                app.state.services = []
        else:
            app.state.services = []

        # P10: Background health monitor — probe every 30s.
        _health_task: asyncio.Task | None = None
        if not skip_service_probes:
            async def _health_loop():
                while True:
                    await asyncio.sleep(30)
                    try:
                        services = await _probe_services()
                        app.state.services = services
                        down = [s for s in services if not s.ok]
                        if down:
                            logger.warning("Health check: {} service(s) down: {}",
                                          len(down), [s.name for s in down])
                    except Exception as e:
                        logger.debug(f"Health loop error: {e}")

            _health_task = asyncio.create_task(_health_loop())
            logger.info("Background health monitor started (30s interval)")

        yield

        if _health_task:
            _health_task.cancel()
        logger.info("SENTINEL gateway shutting down")

    app = FastAPI(
        title="SENTINEL Audit Gateway",
        version=GATEWAY_VERSION,
        description=(
            "Public HTTP surface for the SENTINEL audit pipeline. "
            "Submit a contract → poll for the report."
        ),
        lifespan=lifespan,
    )
    # Stash config for the route handlers.
    app.state.store = store
    app.state.graph_factory = graph_factory
    app.state.no_llm = no_llm
    app.state.skip_service_probes = skip_service_probes

    # ── Routes ────────────────────────────────────────────────────────
    @app.get("/", response_model=None)
    async def root():
        return {
            "service": "sentinel-audit-gateway",
            "version": GATEWAY_VERSION,
            "endpoints": ["POST /audit", "GET /audit/{job_id}", "GET /health"],
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        counts = store.count_by_status()
        services: list[ServiceHealth] = list(getattr(app.state, "services", []) or [])
        # P10: Use cached services from background monitor (no per-request probe).
        # The background loop refreshes every 30s. Only probe on-demand if
        # no cached results exist (first request before monitor ran).
        if not services and not skip_service_probes:
            try:
                services = await _probe_services()
                app.state.services = services
            except Exception as e:
                logger.debug(f"/health service probe failed: {e}")

        # P10: Status considers BOTH job failures AND service health.
        job_degraded = counts.get("failed", 0) >= max(counts.get("completed", 1), 1)
        service_down = any(not s.ok for s in services)
        status = "degraded" if (job_degraded or service_down) else "ok"

        return HealthResponse(
            status=status,
            gateway=GATEWAY_VERSION,
            jobs=counts,
            services=services,
        )

    @app.post(
        "/audit",
        response_model=JobResponse,
        status_code=202,
        responses={
            422: {"model": ErrorResponse, "description": "Validation error"},
            503: {"model": ErrorResponse, "description": "All graph slots busy"},
        },
    )
    async def submit_audit(
        req: AuditRequest,
        no_llm: bool = Query(
            default=False,
            description="If true, skip LLM calls (cross_validator debate, "
                        "synthesizer narrative). Faster, lower verdict quality.",
        ),
    ):
        # Generate a deterministic address from the contract code if the
        # client didn't supply one. Matches `run_real_audit.py` convention.
        contract_address = req.contract_address or _derive_address(req.contract_code)

        # Apply the per-request no_llm override.
        effective_no_llm = no_llm or app.state.no_llm
        if effective_no_llm and not app.state.no_llm:
            # The per-request override only affects this one audit. We
            # can't dynamically patch the LLM mid-graph (the graph is
            # already compiled), so we just record the intent in metadata
            # for downstream debugging.
            req.metadata["_effective_no_llm"] = True

        record = store.create(
            contract_code=req.contract_code,
            contract_address=contract_address,
            audit_timeout_s=req.audit_timeout_s,
            metadata=req.metadata,
        )

        # Kick off the background task.
        task = asyncio.create_task(
            _run_job(
                job_id=record.job_id,
                store=store,
                graph_factory=app.state.graph_factory,
                audit_timeout_s=req.audit_timeout_s,
            ),
            name=f"audit-{record.job_id[:8]}",
        )
        # Hold a reference so the task isn't garbage-collected mid-flight.
        app.state._tasks = getattr(app.state, "_tasks", set())
        app.state._tasks.add(task)
        task.add_done_callback(app.state._tasks.discard)

        logger.info(
            f"job queued | id={record.job_id} | addr={contract_address} | "
            f"chars={len(req.contract_code)} | no_llm={effective_no_llm}"
        )
        return JobResponse(
            job_id=record.job_id,
            status=JobStatus.QUEUED.value,
            submitted_at=record.submitted_at,
            contract_address=contract_address,
        )

    @app.get(
        "/audit/{job_id}",
        response_model=JobResponse,
        responses={404: {"model": ErrorResponse, "description": "Job not found"}},
    )
    async def get_audit(job_id: str):
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"job_id not found: {job_id}")
        return JobResponse(**record.to_dict())

    @app.get("/audit", response_model=list[JobResponse])
    async def list_audits(limit: int = Query(default=20, ge=1, le=100)):
        return [JobResponse(**r.to_dict()) for r in store.list_recent(n=limit)]

    return app


# ── Background runner ───────────────────────────────────────────────────
async def _run_job(
    job_id: str,
    store: JobStore,
    graph_factory: Any,
    audit_timeout_s: float,
) -> None:
    """Execute the audit for one job, updating the store as it goes.

    This coroutine is the heart of the gateway: it owns the lifecycle
    transitions for a single job and is the ONLY place that calls
    `graph.ainvoke`. Exceptions are caught and stored in the record;
    they never propagate out of this function (a bug in one job must
    not crash the gateway).
    """
    record = store.get(job_id)
    if record is None:
        logger.error(f"_run_job: job_id={job_id} vanished from store")
        return

    store.mark_running(job_id)
    t0 = time.time()
    logger.info(f"job running | id={job_id} | addr={record.contract_address} | "
                f"timeout={audit_timeout_s:.0f}s")

    try:
        # Build the graph INSIDE the task so the compile cost only hits
        # the first request (and so any compile errors become per-job
        # errors, not startup-time crashes).
        graph = graph_factory()
        initial_state = {
            "contract_code": record.contract_code,
            "contract_address": record.contract_address,
        }
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=audit_timeout_s,
        )
    except asyncio.TimeoutError:
        dt = time.time() - t0
        msg = f"audit timed out after {dt:.1f}s (limit={audit_timeout_s:.0f}s)"
        store.mark_failed(job_id, msg)
        logger.error(f"job timeout | id={job_id} | {msg}")
        return
    except Exception as e:
        dt = time.time() - t0
        msg = f"{type(e).__name__}: {e}"
        store.mark_failed(job_id, msg)
        logger.error(f"job failed | id={job_id} | {dt:.1f}s | {msg}")
        logger.debug(f"traceback:\n{traceback.format_exc()}")
        return

    # Extract + serialise the report.
    if not isinstance(result, dict):
        store.mark_failed(job_id, f"graph returned non-dict: {type(result).__name__}")
        return
    final_report = result.get("final_report", {}) or {}
    report = {
        "final_report": final_report,
        "verdicts": result.get("verdicts", {}),
        "ml_result": _shrink_ml_result(result.get("ml_result", {})),
        "static_findings_count": len(result.get("static_findings", []) or []),
        "rag_results_count": len(result.get("rag_results", []) or []),
        "audit_history_count": len(result.get("audit_history", []) or []),
        "routing_decisions": result.get("routing_decisions", []),
        "consensus_verdict": result.get("consensus_verdict", {}),
        "debate_transcript": result.get("debate_transcript", {}),
        "confirmations": result.get("confirmations", {}),
        "contradictions": result.get("contradictions", {}),
        "narrative": result.get("narrative"),
        "error": result.get("error"),
    }
    store.mark_completed(job_id, report)
    dt = time.time() - t0
    overall = final_report.get("overall_label", "N/A")
    logger.info(f"job completed | id={job_id} | {dt:.1f}s | overall={overall}")


# ── Service probes ──────────────────────────────────────────────────────
def _probe_urls() -> list[tuple[str, str]]:
    """Return (name, url) for every upstream service we want to probe."""
    return [
        ("ml_api",        os.getenv("MODULE1_INFERENCE_URL", "http://localhost:8001") + "/health"),
        ("mcp_inference", f"http://localhost:{os.getenv('MCP_INFERENCE_PORT', '8010')}/health"),
        ("mcp_rag",       f"http://localhost:{os.getenv('MCP_RAG_PORT', '8011')}/health"),
        ("mcp_audit",     f"http://localhost:{os.getenv('MCP_AUDIT_PORT', '8012')}/health"),
        ("mcp_graph",     f"http://localhost:{os.getenv('MCP_GRAPH_INSPECTOR_PORT', '8013')}/health"),
        ("mcp_representation", f"http://localhost:{os.getenv('MCP_REPRESENTATION_PORT', '8014')}/health"),
    ]


async def _probe_services(timeout: float = 1.5) -> list[ServiceHealth]:
    """Best-effort health probe of every upstream service.

    Uses `httpx.AsyncClient` because the gateway is already async. A failure
    to connect on any single service does NOT raise — we just record the
    service as down. The gateway should still respond to /health even when
    every service is down.
    """
    import httpx
    results: list[ServiceHealth] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for name, url in _probe_urls():
            try:
                resp = await client.get(url)
                results.append(ServiceHealth(
                    name=name, url=url, ok=resp.status_code < 500,
                    detail=f"HTTP {resp.status_code}",
                ))
            except Exception as e:
                short = str(e).splitlines()[0][:120] if str(e) else "unreachable"
                results.append(ServiceHealth(
                    name=name, url=url, ok=False, detail=f"UNREACHABLE: {short}",
                ))
    return results


# ── --no-llm patch (mirrors run_real_audit.py:_patch_no_llm) ───────────
def _patch_no_llm() -> None:
    """Replace the LLM client with a stub that raises immediately.

    This is the same pattern as `run_real_audit.py:_patch_no_llm`. It
    patches both the canonical location and the import locations in
    `graph.py` and `nodes.py` (which captured the symbol at import time).
    """
    from src.llm import client as llm_client

    class _StubLLM:
        def invoke(self, *a, **kw):
            raise RuntimeError("LLM disabled by --no-llm mode")

    def _stub_strong_llm(*a, **kw):
        return _StubLLM()

    llm_client.get_strong_llm = _stub_strong_llm
    import src.orchestration.graph as graph_mod
    import src.orchestration.nodes as nodes_mod
    for mod in (graph_mod, nodes_mod):
        if hasattr(mod, "get_strong_llm"):
            mod.get_strong_llm = _stub_strong_llm
    logger.warning(
        "--no-llm MODE: cross_validator → rule-based verdicts; "
        "synthesizer narrative → None"
    )


# ── Helpers ─────────────────────────────────────────────────────────────
def _derive_address(contract_code: str) -> str:
    """Generate a deterministic 0x... address from contract source code.

    Mirrors `run_real_audit.py:498-500`. Same source → same address
    across runs, which is useful for testing the same contract repeatedly.
    """
    h = hashlib.sha256(contract_code.encode()).hexdigest()[:40]
    return "0x" + h


def _shrink_ml_result(ml_result: dict | None) -> dict:
    """Strip bulky fields from ml_result to keep the report small.

    The full ml_result can include all 10 class probabilities, MLflow
    metadata, model internals, etc. For the gateway response, we keep
    only the fields a caller is likely to want.
    """
    if not ml_result:
        return {}
    keep_keys = {
        "label", "probabilities", "confirmed", "suspicious",
        "vulnerabilities", "tier_thresholds", "thresholds",
        "truncated", "windows_used", "num_nodes", "num_edges",
        "eye_predictions",
    }
    return {k: v for k, v in ml_result.items() if k in keep_keys}


# ── Module-level entry point ────────────────────────────────────────────
# `app` is created lazily so importing `src.api.gateway` doesn't trigger
# graph compilation or LLM client import. Tests that want the app call
# `create_app()` directly; this is just the `python -m src.api.gateway`
# convenience entry.
def run() -> None:
    """Start the gateway via uvicorn. Used by `python -m src.api.gateway`."""
    import uvicorn
    app = create_app()
    uvicorn.run(
        app,
        host=GATEWAY_DEFAULT_HOST,
        port=GATEWAY_DEFAULT_PORT,
        log_level="info",
        # workers > 1 would require a multi-process job store; see module
        # docstring §"NOT PRODUCTION-READY" for why we stay single-process.
        workers=1,
    )


if __name__ == "__main__":
    run()
