# Plan: Doc 07 — Gateway Production Hardening: SQLite, Health, Crash Recovery

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/07_gateway_production.md`
**Session:** 4 of 5
**Prerequisite docs:** Doc 01 (Pipeline), Doc 05 (MCP)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that the pipeline runs as a LangGraph with `graph.ainvoke()`. The gateway wraps this — it accepts HTTP requests, creates jobs, runs the graph in background tasks, and returns results. The gateway is the entry point (`POST /audit`).

**From Doc 05 (MCP):** You learned that 5 MCP servers run on ports 8010-8014. The gateway's `/health` endpoint probes these servers to check if they're alive. The `_probe_services()` function does this.

**Connection to this doc:** This doc explains the gateway's production hardening (P10): how jobs persist across restarts (SQLite), how health monitoring works (background probe every 30s), and how crash recovery works (orphaned RUNNING jobs marked FAILED).

**Key concepts carried forward:** `POST /audit` → `graph.ainvoke()`, `_probe_services()`, 5 MCP server ports, fail-soft principle.

---

## Step 1: Read source files

- [ ] `agents/src/api/gateway.py` (513 lines) — `create_app()`, `lifespan()` with health loop + crash recovery, routes (`POST /audit`, `GET /audit/{id}`, `GET /health`), `_probe_services()`, `_health_loop()`, `_run_job()`
- [ ] `agents/src/api/sqlite_job_store.py` (~200 lines) — `SqliteJobStore` class, `_SCHEMA`, `_connect()`, `_row_to_record()`, `create()`, `mark_running()`, `mark_completed()`, `mark_failed()`, `get()`, `count_by_status()`, `list_recent()`, `recover_pending()`, `_evict_completed()`
- [ ] `agents/src/api/job_store.py` (234 lines) — old in-memory `JobStore` (for comparison), `JobStatus` enum, `JobRecord` dataclass
- [ ] `agents/src/api/models.py` (175 lines) — `AuditRequest`, `JobResponse`, `HealthResponse`, `ServiceHealth`, `ErrorResponse` Pydantic models

## Step 2: Read tests

- [ ] `agents/tests/test_p10_gateway.py` — 13 tests:
  - `TestSqliteJobStore`: create_and_get, mark_running, mark_completed, mark_failed, persistence_across_instances, count_by_status, list_recent, recover_pending, eviction, metadata_persisted, report_with_complex_json
  - `TestGatewayHealthStatus`: health_response_includes_services, health_response_ok_when_all_up

## Step 3: Write sections

- [ ] **TL;DR:** SQLite JobStore replaces in-memory (jobs survive restart), background health monitor probes 6 services every 30s, crash recovery marks orphaned RUNNING jobs as FAILED, `/health` status considers both job failures AND service health
- [ ] **The Problem:** In-memory `OrderedDict` JobStore loses all jobs on gateway restart. `/health` probes 6 services on every request (up to 9s latency). No crash recovery — RUNNING jobs left orphaned after restart. `/health` status ignores service health
- [ ] **How We Arrived at This Design:** invariant (jobs survive restart) → constraint (single-host, no extra infra) → simplest persistence (SQLite, same interface as in-memory) → stress-test (100 concurrent audits) → measure (write contention threshold)
- [ ] **The Solution:** Job lifecycle diagram:
  ```
  POST /audit → SqliteJobStore.create() → status=QUEUED
    → asyncio.create_task(_run_job())
      → store.mark_running() → status=RUNNING
      → graph.ainvoke()
      → store.mark_completed() → status=COMPLETED + report
      (or) store.mark_failed() → status=FAILED + error
  ```
  Health monitor flow:
  ```
  lifespan startup → _health_loop() asyncio task
    → every 30s: _probe_services() → app.state.services (cached)
    → /health uses cached results (no per-request probe)
  ```
  Crash recovery:
  ```
  lifespan startup → store.recover_pending()
    → finds jobs where status=RUNNING
    → store.mark_failed(job_id, "gateway restart (crash recovery)")
  ```
- [ ] **Key Code:**
  - `SqliteJobStore` class (sqlite_job_store.py) — same 7-method interface as `JobStore`: `create`, `mark_running`, `mark_completed`, `mark_failed`, `get`, `count_by_status`, `list_recent`
  - `_SCHEMA` (sqlite_job_store.py) — single `jobs` table, JSON columns for `report` and `metadata`, indexes on `status` and `submitted_at`
  - `recover_pending()` (sqlite_job_store.py) — `SELECT * FROM jobs WHERE status = 'running'` → gateway marks them FAILED
  - `_health_loop()` (gateway.py lifespan) — `asyncio.create_task` running `while True: await asyncio.sleep(30); services = await _probe_services()`
  - Health status logic: `job_degraded = failed >= max(completed, 1); service_down = any(not s.ok for s in services); status = "degraded" if (job_degraded or service_down) else "ok"`
  - `JobRecord` dataclass (job_store.py) — `job_id`, `status`, `submitted_at`, `contract_code`, `contract_address`, `audit_timeout_s`, `metadata`, `started_at`, `finished_at`, `report`, `error`
- [ ] **Design Decision:** SQLite vs Redis vs Postgres (tradeoff table: infrastructure, persistence, multi-host, write throughput, team familiarity)
- [ ] **Technology Choice:** SQLite (5-question framework: category, alternatives, why SQLite, when Postgres/Redis, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ In-memory store forever — "it's just a prototype." Breaks: restart loses everything, can't debug past jobs, no crash recovery. Right: SQLite from day 1 for single-host
  - ❌ Per-request health probes — "probe on every /health call." Breaks: 6 services × 1.5s timeout = 9s worst-case latency. Right: background cache loop every 30s
- [ ] **Mistakes & Fixes:**
  - In-memory `JobStore` lost all jobs on restart. The docstring literally said "NOT PRODUCTION-READY." Fix: `SqliteJobStore` with same interface — swap was mechanical because interface was clean (7 methods)
  - `/health` probed 6 services on EVERY request. Up to 9s latency per request. Fix: `_health_loop()` background task probes every 30s, `/health` uses cached `app.state.services`
  - `/health` status ignored service health — a dead ML API didn't flip status to "degraded." Fix: `any(not s.ok for s in services)` in status logic
  - No crash recovery — jobs left in RUNNING state after gateway crash. Fix: `recover_pending()` on startup finds them and marks FAILED
  - Type error: `store: JobStore | None` didn't accept `SqliteJobStore`. Fix: change to `store: Any | None`
- [ ] **What Would Break Without This:** Remove SQLite → restart loses all jobs. Remove health monitor → `/health` takes 9s per request. Remove crash recovery → orphaned RUNNING jobs forever. Remove eviction → database grows unbounded
- [ ] **At Scale:** 61 contracts (current) / 610 / 6,100 / 61,000 — SQLite write contention at ~100 concurrent audits. Migration to Postgres when needed (same interface)
- [ ] **Try It Yourself:**
  ```
  cd agents && source .venv/bin/activate
  python -m pytest tests/test_p10_gateway.py -v
  python3 -c "
  from src.api.sqlite_job_store import SqliteJobStore
  import tempfile, os
  with tempfile.TemporaryDirectory() as td:
      s = SqliteJobStore(os.path.join(td, 'test.db'))
      r = s.create('contract code', '0xTEST', 300.0)
      print(f'Created: {r.job_id[:8]} status={r.status.value}')
      s.mark_running(r.job_id)
      s.mark_completed(r.job_id, {'verdict': 'CONFIRMED'})
      r2 = s.get(r.job_id)
      print(f'Retrieved: status={r2.status.value} report={r2.report}')
  "
  ```
- [ ] **Limitations:** Single-host only (no multi-process), no job cancellation API, no TTL on completed jobs, sequential health probes (not concurrent), no webhook/Slack alerting on service down
- [ ] **Transferable Patterns:** (1) Mechanical interface swap — design clean interface first, swap implementation later (2) Background health monitoring — cache expensive probes (3) Crash recovery — find orphans, mark them failed. Each with interview story + when wrong.

## Step 4: Verify

- [ ] Open `sqlite_job_store.py` and verify `_SCHEMA` has `jobs` table with correct columns
- [ ] Open `gateway.py` and verify `_health_loop` exists in `lifespan()` with `asyncio.create_task`
- [ ] Verify `recover_pending()` is called in `lifespan()` before the `yield`
- [ ] Confirm health status logic uses `any(not s.ok for s in services)`
- [ ] Confirm test count: 13 tests in `test_p10_gateway.py`
- [ ] Verify `store` parameter type in `create_app()` is `Any | None` (not `JobStore | None`)
