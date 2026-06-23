# WS6a C.1 — FastAPI Gateway (2026-06-22)

## Summary

Built the public HTTP surface for the SENTINEL audit pipeline at
`agents/src/api/`. Users can now submit contracts and poll for results
over HTTP instead of running the CLI directly.

## What was built

**`agents/src/api/`** — new package, 4 files, ~770 lines:
- `__init__.py` (~70 lines) — PEP 562 lazy exports
- `models.py` (~140 lines) — Pydantic schemas + validators
- `job_store.py` (~200 lines) — Thread-safe in-memory job record store
- `gateway.py` (~360 lines) — FastAPI app + endpoints + background runner

**`agents/tests/test_gateway.py`** — 44 new tests (8 model + 12 store + 3
address + 18 E2E + 3 exports). All pass.

**`agents/pyproject.toml`** — added `fastapi`, `uvicorn[standard]`, pinned
`sse-starlette=2.1.3` for compatibility.

## Endpoints

| Method | Path | Status | What |
|--------|------|--------|------|
| POST   | `/audit` | 202 | Submit contract, returns `job_id` + `status="queued"` |
| GET    | `/audit/{job_id}` | 200/404 | Poll status + report (when done) |
| GET    | `/audit` | 200 | List recent jobs (debugging) |
| GET    | `/health` | 200 | Liveness + job counts + service probes |
| GET    | `/` | 200 | Service banner |
| GET    | `/docs` | 200 | Swagger UI (auto) |
| GET    | `/openapi.json` | 200 | OpenAPI schema (auto) |

## Lifecycle (POST /audit → GET /audit/{job_id})

```
client POST /audit
  ↓ (Pydantic validates)
  → store.create(contract_code, ...) → returns JobRecord(status=QUEUED)
  → asyncio.create_task(_run_job(job_id, ...))  (background)
  → 202 Accepted + {job_id, status: "queued", submitted_at, contract_address}

background _run_job (parallel to client polling):
  → store.mark_running  (status=RUNNING, started_at=set)
  → graph = graph_factory()       (lazy — only compiled on first job)
  → result = await asyncio.wait_for(graph.ainvoke(state), timeout=audit_timeout_s)
  → store.mark_completed(result)  (status=COMPLETED, finished_at=set, report=result)
  OR
  → store.mark_failed(error)      (status=FAILED, finished_at=set, error=msg)

client GET /audit/{job_id}
  → 200 + JobResponse (status, timestamps, error, report)
  → 404 if job_id not found
```

## Key design decisions

1. **In-memory job store** — `OrderedDict` with `max_completed=100` eviction.
   Fine for dev, single-process. Marked in module docstring as
   "swap to SQLite/REDIS for production". The `JobStore` interface
   (`get`/`create`/`mark_*`/`list_recent`) is small enough that a swap
   is mechanical.

2. **Lazy graph import** — `graph_factory()` is called inside `_run_job`,
   not at app creation. This means `import src.api.gateway` is cheap
   (no graph compilation, no LM Studio import). It also means
   `create_app()` works when the ML API is down — the gateway can still
   serve `/health` and the first `/audit` will fail gracefully (not at
   startup, but per-job).

3. **PEP 562 lazy `__getattr__` in `__init__.py`** — defers submodule
   imports so circular references (gateway ↔ job_store) don't break
   eager `from src.api import models` imports.

4. **Background task held in `app.state._tasks`** — `asyncio.create_task`
   schedules a coroutine but if you don't keep a reference, the task
   can be garbage-collected mid-flight. The set + `add_done_callback`
   pattern prevents this.

5. **Strict state machine in JobStore** — `mark_completed` requires
   `status == RUNNING`. This catches bugs in callers (e.g. "I forgot to
   call mark_running") instead of silently marking an unrun job as done.

6. **`asyncio.wait_for` enforces per-audit timeout** — matches
   `run_real_audit.py:asyncio.wait_for(..., timeout=timeout_s)`. On
   timeout, the job is marked failed with `"timed out after Ns"` and
   the task returns cleanly (no `CancelledError` propagating out).

7. **Exceptions caught in `_run_job`** — graph crashes become
   `status="failed"` + `error="..."`, never crash the server. Verified
   by test_graph_exception_does_not_kill_gateway: one job crashes,
   next job still completes.

8. **Pydantic `extra="forbid"` on request** — clients can't slip extra
   fields past us. Verified by test_validation_extra_field_rejected.

9. **Validators on contract_code** — reject empty (Pydantic min_length),
   too long (>200K chars), or binary (>5% non-printable). All cheap
   to evaluate; full Solidity syntax validation happens in
   `quick_screen` (Slither) and `static_analysis` (Aderyn) nodes.

## Catching real bugs during development

1. **starlette/sse-starlette version conflict** — FastAPI 0.116 caps
   starlette<0.49.0; sse-starlette 3.3.4 requires starlette>=0.49.1.
   Pinning sse-starlette to 2.1.3 (no upper bound) makes everything
   work. Higher FastAPI versions (0.118+) still cap starlette<0.49.

2. **TestClient without `with` context manager** — `asyncio.create_task`
   creates tasks on a loop that doesn't get scheduled while the test
   client is making sync calls. Using `with TestClient(app) as client:`
   properly enters the lifespan, which makes the tasks progress.
   (Caught by `test_full_lifecycle_success` initially failing — job
   stayed in "running" forever until we wrapped with `with`.)

3. **`mark_completed` from QUEUED is a no-op** — caught in early
   test of eviction logic. The state machine is strict: COMPLETED
   requires RUNNING. Test was wrong, code is right.

## Test count

- **440 pass** (up from 396 = +44 new gateway tests)
- 3 pre-existing solc failures (`PATH` doesn't include `.venv/bin/`)
  — confirmed same 3 from before this slice by running with
  `PATH=.venv/bin:$PATH` (4 pass, 0 fail).
- No regressions from this slice.

## Smoke test (real graph, no-llm mode)

```python
app = create_app(store=JobStore(), no_llm=True, skip_service_probes=True)
with TestClient(app) as client:
    contract = open("test_contracts/vulnerable_reentrant.sol").read()
    resp = client.post("/audit", json={
        "contract_code": contract,
        "contract_address": "0xREENTRANT_VAULT",
        "audit_timeout_s": 60.0,
    })
    job_id = resp.json()["job_id"]
    # poll until done → 1.2s
    # final: status=completed, all 13 nodes ran, elapsed=0.3s for graph
```

Without ML API running, ML result is empty (verdict=SAFE, no findings)
— expected behavior when services are down. Flow itself is correct.

## How to run

```bash
cd agents
.venv/bin/python -m src.api.gateway
# INFO:     Uvicorn running on http://0.0.0.0:8000
# OpenAPI UI: http://0.0.0.0:8000/docs
```

Environment variables:
- `GATEWAY_HOST` (default `0.0.0.0`)
- `GATEWAY_PORT` (default `8000`)
- `AUDIT_NO_LLM` (`true`/`false`, default `false`) — global --no-llm

## What's NOT in this slice

- No auth
- No persistent job store (SQLite/REDIS)
- No streaming/SSE
- No CORS
- No rate limiting
- No `/benchmark` endpoint (would use `src/eval.Benchmark`)

These are all explicit "future work" items in the plan doc
(`docs/plan/agents/2026-06-17-extended-capability/03_PHASE_C_EXECUTION_PLAN.md`).
C.3 (guards) and C.4 (monitoring) wrap around this gateway.

## Files

**Created (6 files, ~1500 lines total):**
- `agents/src/api/__init__.py`
- `agents/src/api/models.py`
- `agents/src/api/job_store.py`
- `agents/src/api/gateway.py`
- `agents/tests/test_gateway.py` (44 tests)
- `~/.claude/scratch/ws6a_c1_gateway_20260622.md` (working memory)

**Modified (1 file):**
- `agents/pyproject.toml` — added fastapi, uvicorn, sse-starlette pin
