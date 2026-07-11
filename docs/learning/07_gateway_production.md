# 07. Gateway Production Hardening: SQLite, Health, Crash Recovery

> **Prerequisites:** [01. The Audit Pipeline] — `graph.ainvoke()` is what the gateway calls. [05. MCP Architecture] — 5 MCP servers on ports 8010-8014, `/health` probes them.
> **Next:** [08. Evaluation Framework] covers how the eval benchmark calls the gateway to run 83-contract evaluations.
> **Cross-ref:** [04. Reproducibility] — the gateway respects `SENTINEL_DETERMINISTIC` mode.
> **Scope:** This doc covers the FastAPI gateway: the job lifecycle (POST /audit → background task → GET /audit/{id}), SQLite job persistence, the background health monitor, crash recovery, and the `/health` status logic. It does NOT cover the pipeline itself (see Doc 01) or the MCP servers (see Doc 05).
> **TL;DR:** The gateway is the HTTP entry point (`POST /audit`). It accepts contract source, creates a job in SQLite, kicks off a background `asyncio.create_task` that calls `graph.ainvoke()`, and returns a `job_id` immediately (HTTP 202). The client polls `GET /audit/{job_id}` until `status=completed`. P10 replaced the in-memory `OrderedDict` job store with `SqliteJobStore` — jobs now survive gateway restarts. A background health monitor probes 6 upstream services every 30s and caches the results in `app.state.services` — `/health` responds in <1ms instead of up to 9s (6 services × 1.5s timeout). On startup, `recover_pending()` finds jobs left in RUNNING state (from a crash) and marks them FAILED — no orphaned jobs. The `/health` status considers both job failures AND service health: `status = "degraded"` if `failed >= completed` OR any service is down.

---

## The Problem: An In-Memory Store Is Not Production-Ready

### What was wrong before P10

The original gateway (WS6a, 2026-06-22) used an in-memory `OrderedDict` as the job store. The docstring literally said "NOT PRODUCTION-READY." Three problems:

1. **Restart loses everything.** If the gateway crashes (OOM, power failure, deploy), all jobs — queued, running, completed — vanish. A client polling `GET /audit/{job_id}` gets 404. There's no way to recover the audit result.

2. **`/health` probes on every request.** The original `/health` endpoint called `_probe_services()` synchronously on every request. With 6 services × 1.5s timeout each, a single `/health` call could take up to 9 seconds (if all services were down). Load balancers probing `/health` every 5s would timeout.

3. **No crash recovery.** If the gateway crashes while a job is RUNNING, the job stays in RUNNING state forever (in the in-memory dict, it's lost — but even with persistence, without recovery logic, it would be orphaned). The client polls forever and never gets a result.

### Teaching: why not just use Redis from the start?

You might think: "If in-memory isn't production-ready, just use Redis. It's the standard for job queues." But Redis adds infrastructure:
- A Redis server process to run and monitor
- A Redis client library to install and configure
- Network latency (Redis is a separate process, even on localhost)
- Operational complexity (Redis persistence config, eviction policy, memory limits)

For a single-host deployment with one developer, SQLite is simpler: it's a file (`data/jobs.db`), no server process, no client library beyond Python's built-in `sqlite3`. The same 7-method interface (`create`, `mark_running`, `mark_completed`, `mark_failed`, `get`, `count_by_status`, `list_recent`) works. The swap was mechanical — `SqliteJobStore` is a drop-in replacement.

**The reasoning:** choose the simplest persistence that solves the *current* problem. The current problem is "restart loses jobs." SQLite solves that. Redis would solve it too, but adds infrastructure you don't need yet. When you need multi-host (multiple gateway processes sharing a job queue), *then* Redis is the right choice. The key is designing the interface so the swap is mechanical — and it was, because the in-memory `JobStore` had a clean 7-method interface.

---

## How We Arrived at This Design

> **How to read this section:** Each step shows the question, *how to reason about it*, and the chain of logic.

### Step 1 — Identify the invariant

**The question:** What must always be true about the gateway, even if it crashes?

| Candidate property | If violated → | Verdict |
|---|---|---|
| Jobs survive gateway restart | Client loses audit result → must re-submit → wasted compute | **Invariant** |
| `/health` responds fast (<1s) | Load balancer times out → takes gateway out of rotation → no audits accepted | **Invariant** |
| Orphaned RUNNING jobs are detected | Client polls forever → liveness failure | **Invariant** |
| `/health` reflects service health | Dead ML API → status still "ok" → load balancer routes to dead box | **Invariant** |

### Step 2 — Identify the constraints

**Constraint A: Single-host deployment.**
- *Why:* SENTINEL runs on one machine (RTX 3070, 64GB RAM). No Kubernetes, no container orchestration.
- *What this forces:* SQLite (file-based, no server process) instead of Redis/Postgres (server-based, needs infrastructure). The migration to Postgres happens only when you have multiple gateway hosts.

**Constraint B: The job store interface must be swappable.**
- *Why:* The in-memory `JobStore` was already deployed. The swap to `SqliteJobStore` must not change any gateway code — only the store implementation.
- *What this forces:* `SqliteJobStore` implements the same 7 methods as `JobStore`. The gateway's `create_app(store: Any | None = None)` accepts any store that implements the interface. Duck typing.

**Constraint C: The health monitor must not block requests.**
- *Why:* `/health` is called by load balancers, monitoring systems, and humans. If it takes 9s, load balancers remove the gateway from rotation.
- *What this forces:* Health probing happens in a background `asyncio.create_task` (`_health_loop`), not in the request handler. `/health` reads cached results from `app.state.services`.

### Step 3 — Eliminate alternatives

| Approach | How it breaks | When it breaks | Eliminate? |
|---|---|---|---|
| **In-memory `OrderedDict`** | Restart loses all jobs | On any crash/restart | **Yes** — breaks the invariant |
| **Redis** | Adds infrastructure (server process, client lib, network) | When you have one host and no Redis expertise | **Yes** — overkill for current scale |
| **Postgres** | Adds infrastructure (server, schema management, connection pool) | When you have one host and don't need SQL queries | **Yes** — overkill |
| **SQLite** | Single-writer (write lock serializes) | At ~100 concurrent writes | **No** — current concurrency is 1-5 |

**The reasoning:** In-memory breaks on restart (invariant violation). Redis and Postgres don't break — but they add infrastructure cost (server process, client library, network latency, operational monitoring) that isn't justified at current scale (one host, 1-5 concurrent audits). SQLite is a file — zero infrastructure, zero network, built into Python. Its limitation (single-writer write lock) triggers at ~100 concurrent writes, which is 20x above current load. The interface is designed so the swap to Postgres/Redis is mechanical when the trigger hits.

### Step 4 — Stress-test (crash recovery)

**The test:** Kill the gateway while a job is RUNNING. Restart the gateway. What happens?

**Without `recover_pending()`:** the job stays in RUNNING state forever. The SQLite row has `status='running'`. The client polls `GET /audit/{job_id}` and sees `status='running'` indefinitely. No one knows the job is dead.

**With `recover_pending()`:** on startup, the gateway calls `store.recover_pending()`, which does `SELECT * FROM jobs WHERE status = 'running'`. For each orphaned job, the gateway calls `store.mark_failed(job_id, "gateway restart (crash recovery)")`. The client polls and sees `status='failed'` with the crash recovery message. The human reviewer knows the job died in a crash, not in a code bug.

### Step 5 — Measure

**Before P10:**
- `/health` latency: up to 9s (6 services × 1.5s timeout, sequential, on every request)
- Jobs after restart: 0 (all lost)
- Orphaned RUNNING jobs: ∞ (never detected)

**After P10:**
- `/health` latency: <1ms (reads cached `app.state.services`)
- Jobs after restart: 100% (SQLite file survives)
- Orphaned RUNNING jobs: 0 (all detected and marked FAILED on startup)
- 13 new tests pass (`test_p10_gateway.py`)

> **The method, summarized:** (1) Find invariants — jobs survive restart, `/health` is fast, orphans are detected. (2) Find constraints — single-host, swappable interface, non-blocking health. (3) Eliminate alternatives — in-memory breaks on restart, Redis/Postgres are overkill. (4) Stress-test crash recovery — `recover_pending()` on startup. (5) Measure — before/after latency and recovery metrics.

---

## The Solution

### The job lifecycle

```
Client                        Gateway                          Pipeline
──────                        ───────                          ────────

POST /audit ──────────→  store.create() ──→ status=QUEUED
  (contract_code)            │
                             ├─ asyncio.create_task(_run_job())
                             │
  ← 202 job_id  ←────────────┘
                                    │
                              store.mark_running() ──→ status=RUNNING
                                    │
                              graph.ainvoke(initial_state)
                                    │                          │
                                    │                    ┌─────┴─────┐
                                    │                    │ 14 nodes  │
                                    │                    │ run audit  │
                                    │                    └─────┬─────┘
                                    │                          │
                              ┌─────┴──────────────────────────┘
                              │
                        success? 
                         ├─ yes → store.mark_completed(report) → status=COMPLETED
                         └─ no  → store.mark_failed(error)    → status=FAILED

GET /audit/{job_id} ──→ store.get(job_id) ──→ JobResponse (status + report)
  (poll until completed)
```

**Teaching: why `asyncio.create_task` and not `await`?** The `POST /audit` handler returns immediately with HTTP 202 (Accepted) and a `job_id`. The actual audit runs in a background task. If the handler `await`ed the audit, the HTTP request would hang for 60s (deep-path audit time) — the client would timeout, and the gateway's event loop would be blocked (no other requests could be processed). `asyncio.create_task` schedules the audit as a coroutine that runs concurrently with other requests. The event loop multiplexes between the audit task and incoming HTTP requests.

### The health monitor

```
lifespan startup
    │
    ├─ store.recover_pending() ──→ orphaned RUNNING jobs → FAILED
    │
    ├─ _probe_services() ──→ initial probe (6 services, 1.5s timeout each)
    │                    ──→ app.state.services (cached)
    │
    └─ asyncio.create_task(_health_loop())
            │
            └─ while True:
                   await asyncio.sleep(30)          # probe every 30s
                   services = await _probe_services()
                   app.state.services = services     # update cache
                   down = [s for s in services if not s.ok]
                   if down:
                       logger.warning(...)
```

**Teaching: why 30s?** The health monitor probes every 30 seconds. This is a tradeoff:
- Too fast (1s): 6 HTTP requests per second to upstream services — wasteful, and if a service is slow, the probes pile up.
- Too slow (5min): a dead service is invisible for up to 5 minutes — the load balancer routes to a dead box.
- 30s: a dead service is detected within 30s (acceptable for a security oracle, not a trading system). 6 probes per 30s = 0.2 requests/second per service (negligible overhead).

### The `/health` status logic

```python
job_degraded = counts.get("failed", 0) >= max(counts.get("completed", 1), 1)
service_down = any(not s.ok for s in services)
status = "degraded" if (job_degraded or service_down) else "ok"
```

**Teaching: why both job failures AND service health?** Before P10, `/health` status only considered job counts — if `failed >= completed`, status was "degraded." But if the ML API was down (all audits failing with "ML unavailable"), the status was "degraded" (correct) — but if the ML API was down and no audits had been submitted yet, the status was "ok" (wrong — the gateway can't function without the ML API). The fix: check service health too. If any service is down, status is "degraded" regardless of job counts.

**The `max(completed, 1)` guard:** prevents division-by-zero when `completed=0` (fresh gateway, no audits yet). `max(0, 1) = 1`, so `failed >= 1` triggers "degraded" only if there's at least one failure. On a fresh gateway with zero audits and all services up, status is "ok."

## Key Code

### The `SqliteJobStore` class — persistent job storage

The complete job store — same interface as the in-memory version, backed by SQLite:

```python
# sqlite_job_store.py:35-51, 54-76
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    status           TEXT NOT NULL,
    submitted_at     TEXT NOT NULL,
    contract_code    TEXT NOT NULL,
    contract_address TEXT NOT NULL,
    audit_timeout_s  REAL NOT NULL,
    metadata         TEXT NOT NULL DEFAULT '{}',
    started_at       TEXT,
    finished_at      TEXT,
    report           TEXT,
    error            TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs(submitted_at);
"""

class SqliteJobStore:
    def __init__(self, db_path: str | Path = "data/jobs.db", max_completed: int = 500):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.max_completed = max_completed

        with self._lock:
            conn = self._connect()
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
```

Why this matters: three design decisions:

1. **`check_same_thread=False`:** SQLite's default is to only allow access from the thread that created the connection. The gateway runs on asyncio (single thread), but the `threading.Lock` is for safety in non-async tests. `check_same_thread=False` allows any thread to use the connection — the lock serializes access.

2. **`threading.Lock`:** SQLite writes are serialized by the lock. This means only one thread can write at a time. For the gateway's workload (1-5 concurrent audits, each writing once per state transition), this is fine — the lock is held for microseconds (the SQLite write), never across `graph.ainvoke()` (the 60-second audit).

3. **Two indexes:** `idx_jobs_status` for `WHERE status = 'running'` (crash recovery) and `count_by_status()`. `idx_jobs_submitted` for `ORDER BY submitted_at DESC` (list_recent). Without indexes, these queries scan the entire table — slow when the table grows to thousands of rows.

### The `create` method — insert a new job

```python
# sqlite_job_store.py:93-127
def create(self, contract_code, contract_address, audit_timeout_s, metadata=None) -> JobRecord:
    record = JobRecord(
        job_id=str(uuid.uuid4()),
        status=JobStatus.QUEUED,
        submitted_at=_utcnow_iso(),
        contract_code=contract_code,
        contract_address=contract_address,
        audit_timeout_s=audit_timeout_s,
        metadata=metadata or {},
    )
    with self._lock:
        conn = self._connect()
        conn.execute(
            """INSERT INTO jobs (job_id, status, submitted_at, contract_code,
               contract_address, audit_timeout_s, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (record.job_id, record.status.value, record.submitted_at,
             record.contract_code, record.contract_address,
             record.audit_timeout_s, json.dumps(record.metadata)),
        )
        conn.commit()
        conn.close()
    return record
```

Why this matters: `metadata` is stored as `json.dumps(record.metadata)` — a JSON string in a TEXT column. This is the standard pattern for storing dicts in SQLite: serialize to JSON on write, deserialize on read (`json.loads(row["metadata"])` in `_row_to_record`). The alternative (one column per metadata key) would require schema changes when metadata keys change. JSON-in-TEXT is schemaless — any metadata dict works.

### The `mark_running` / `mark_completed` / `mark_failed` methods — state transitions

```python
# sqlite_job_store.py:129-175
def mark_running(self, job_id: str) -> None:
    with self._lock:
        conn = self._connect()
        conn.execute(
            """UPDATE jobs SET status = ?, started_at = ?
               WHERE job_id = ? AND status = ?""",
            (JobStatus.RUNNING.value, _utcnow_iso(), job_id, JobStatus.QUEUED.value),
        )
        conn.commit(); conn.close()

def mark_completed(self, job_id: str, report: dict[str, Any]) -> None:
    with self._lock:
        conn = self._connect()
        conn.execute(
            """UPDATE jobs SET status = ?, finished_at = ?, report = ?
               WHERE job_id = ? AND status = ?""",
            (JobStatus.COMPLETED.value, _utcnow_iso(),
             json.dumps(report, default=str), job_id, JobStatus.RUNNING.value),
        )
        conn.commit(); conn.close()
    self._evict_completed()

def mark_failed(self, job_id: str, error: str) -> None:
    with self._lock:
        conn = self._connect()
        conn.execute(
            """UPDATE jobs SET status = ?, finished_at = ?, error = ?
               WHERE job_id = ? AND status IN (?, ?)""",
            (JobStatus.FAILED.value, _utcnow_iso(), error,
             job_id, JobStatus.QUEUED.value, JobStatus.RUNNING.value),
        )
        conn.commit(); conn.close()
    self._evict_completed()
```

Why this matters: the `WHERE ... AND status = ?` clause is a **state machine guard**. `mark_running` only transitions `QUEUED → RUNNING` — it won't transition an already-COMPLETED job back to RUNNING. `mark_completed` only transitions `RUNNING → COMPLETED` — it won't complete a job that was never started. `mark_failed` accepts both `QUEUED → FAILED` and `RUNNING → FAILED` (a job can fail before it starts or during execution). This prevents invalid state transitions at the database level — even if two concurrent tasks try to complete the same job, only one succeeds (the other's WHERE clause doesn't match).

**Teaching: the `json.dumps(report, default=str)` in `mark_completed`:** the `default=str` parameter tells `json.dumps` to call `str()` on any object it can't serialize. This handles `datetime` objects, `dataclass` instances, and other non-JSON-native types in the report dict. Without it, a report containing a `datetime` would crash with `TypeError: Object of type datetime is not JSON serializable`.

### The `recover_pending` method — crash recovery

```python
# sqlite_job_store.py:242-252
def recover_pending(self) -> list[JobRecord]:
    """Recover jobs left in RUNNING state after a crash. Returns them so
    the gateway can mark them as FAILED (the graph state is lost)."""
    with self._lock:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = ?",
            (JobStatus.RUNNING.value,),
        ).fetchall()
        conn.close()
    return [self._row_to_record(row) for row in rows]
```

Why this matters: this is called once at gateway startup (`gateway.py:155-159`). It finds all jobs that were RUNNING when the gateway crashed. The gateway then calls `mark_failed(job_id, "gateway restart (crash recovery)")` for each one. Without this, orphaned RUNNING jobs would stay in RUNNING forever — the client would poll indefinitely.

**Teaching: why not resume the audit instead of failing it?** The graph state (LangGraph's `AuditState`) is in-memory — it's lost when the gateway crashes. Even though `SqliteSaver` checkpoints the graph state to a separate SQLite file (`data/checkpoints.db`), the gateway doesn't currently resume from the checkpoint — it marks the job as FAILED. This is a known limitation (see Doc 01's Limitations). Resuming would require: (a) finding the checkpoint by `thread_id` (the job_id), (b) rebuilding the graph with the same `SqliteSaver`, (c) calling `graph.ainvoke(None, config={"configurable": {"thread_id": job_id}})` to resume. This is P10+ work.

### The eviction method — bounded database growth

```python
# sqlite_job_store.py:217-240
def _evict_completed(self) -> None:
    with self._lock:
        conn = self._connect()
        row = conn.execute(
            """SELECT COUNT(*) as cnt FROM jobs
               WHERE status IN (?, ?)""",
            (JobStatus.COMPLETED.value, JobStatus.FAILED.value),
        ).fetchone()
        terminal_count = row["cnt"] if row else 0
        excess = terminal_count - self.max_completed
        if excess <= 0:
            conn.close()
            return
        conn.execute(
            """DELETE FROM jobs WHERE job_id IN (
               SELECT job_id FROM jobs
               WHERE status IN (?, ?)
               ORDER BY submitted_at ASC
               LIMIT ?
            )""",
            (JobStatus.COMPLETED.value, JobStatus.FAILED.value, excess),
        )
        conn.commit(); conn.close()
```

Why this matters: without eviction, the database grows unbounded — every audit adds a row, and rows are never deleted. At 100 audits/day, the database reaches 36,500 rows/year — SQLite handles this fine (it can handle millions of rows), but the `list_recent` query would slow down (more rows to sort). Eviction keeps the terminal job count at `max_completed=500` — when a new job completes and the count exceeds 500, the oldest COMPLETED/FAILED job is deleted. QUEUED and RUNNING jobs are never evicted (they might still be needed).

### The `_run_job` function — the heart of the gateway

```python
# gateway.py:328-402
async def _run_job(job_id, store, graph_factory, audit_timeout_s):
    record = store.get(job_id)
    if record is None:
        return

    store.mark_running(job_id)

    try:
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
        store.mark_failed(job_id, f"audit timed out after {audit_timeout_s:.0f}s")
        return
    except Exception as e:
        store.mark_failed(job_id, f"{type(e).__name__}: {e}")
        return

    final_report = result.get("final_report", {})
    report = {
        "final_report": final_report,
        "verdicts": result.get("verdicts", {}),
        "ml_result": _shrink_ml_result(result.get("ml_result", {})),
        # ... elided: static_findings_count, rag_results_count, etc.
    }
    store.mark_completed(job_id, report)
```

Why this matters: five patterns:

1. **`asyncio.wait_for(graph.ainvoke(...), timeout=...)`:** enforces the per-audit wall-clock budget. If the audit takes longer than `audit_timeout_s` (default 300s), `wait_for` raises `asyncio.TimeoutError`, which is caught and stored as a job failure. Without this, a hung audit (e.g., Slither in an infinite loop) would run forever, consuming a background task slot.

2. **Exception isolation:** `except Exception as e` catches *any* exception from the graph and stores it in the job record. A bug in one audit (e.g., `KeyError` in a node) doesn't crash the gateway — it fails that one job and the gateway continues processing other jobs. **Teaching: a background task that crashes takes down the event loop. Every `asyncio.create_task` must have a top-level try/except that never lets exceptions propagate.**

3. **`graph_factory()` inside the task:** the graph is compiled *inside* the background task, not at gateway startup. This means: (a) the first audit pays the compile cost (~2s), subsequent audits reuse the cached graph (LangGraph caches the compiled graph), (b) if graph compilation fails (e.g., missing dependency), the error is per-job, not per-startup — the gateway still starts and `/health` still responds.

4. **`_shrink_ml_result(result.get("ml_result", {}))`:** strips bulky fields from `ml_result` before storing in the report. The full `ml_result` can include all 10 class probabilities, MLflow metadata, model internals. The shrinker keeps only the fields a caller needs (`label`, `probabilities`, `confirmed`, `suspicious`, etc.). **Teaching: API responses should be bounded. Don't store the full internal state in the API response — store only what the caller needs.**

5. **`store.mark_completed` after building the report:** the report dict is built from the graph result, then stored. If the graph returns a non-dict (bug), the `isinstance(result, dict)` check (line 380) catches it and marks the job as failed — no crash.

### The health monitor loop

```python
# gateway.py:171-188
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
```

Why this matters: three patterns:

1. **`while True: await asyncio.sleep(30)`:** the loop runs forever, probing every 30s. The `await asyncio.sleep(30)` is non-blocking — other coroutines (HTTP handlers, audit tasks) run during the sleep. This is the asyncio way to do periodic background work.

2. **`try/except` inside the loop:** if `_probe_services()` raises (e.g., all services unreachable, network error), the exception is caught and logged at DEBUG level. The loop continues — the next probe is in 30s. **Teaching: a background loop that crashes stops doing its job silently. Every `while True` loop needs a try/except that keeps the loop alive.**

3. **`_health_task.cancel()` on shutdown (line 192-193):** when the gateway shuts down, the health task is cancelled. Without this, the task would keep running during shutdown (trying to probe services that are being shut down), producing error logs. **Teaching: every `asyncio.create_task` needs a corresponding `cancel()` on shutdown. Otherwise the task outlives the gateway.**

### The service probe function

```python
# gateway.py:406-441
def _probe_urls() -> list[tuple[str, str]]:
    return [
        ("ml_api",        os.getenv("MODULE1_INFERENCE_URL", "http://localhost:8001") + "/health"),
        ("mcp_inference", f"http://localhost:{os.getenv('MCP_INFERENCE_PORT', '8010')}/health"),
        ("mcp_rag",       f"http://localhost:{os.getenv('MCP_RAG_PORT', '8011')}/health"),
        ("mcp_audit",     f"http://localhost:{os.getenv('MCP_AUDIT_PORT', '8012')}/health"),
        ("mcp_graph",     f"http://localhost:{os.getenv('MCP_GRAPH_INSPECTOR_PORT', '8013')}/health"),
        ("mcp_representation", f"http://localhost:{os.getenv('MCP_REPRESENTATION_PORT', '8014')}/health"),
    ]

async def _probe_services(timeout: float = 1.5) -> list[ServiceHealth]:
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
```

Why this matters: two patterns:

1. **Sequential probes:** the loop probes services one at a time. With 6 services × 1.5s timeout, worst-case is 9s. This is acceptable *in the background loop* (runs every 30s, not per-request). If it ran per-request, it would need `asyncio.gather` for concurrent probes. **Teaching: sequential is fine for background work (low frequency). Use `asyncio.gather` for per-request work (high frequency).**

2. **`ok=resp.status_code < 500`:** a 404 (service running but wrong endpoint) is treated as "ok" — the service is alive, just the health endpoint is misconfigured. Only 5xx (server error) and connection errors are "down." **Teaching: health checks should distinguish "service is down" (connection refused) from "endpoint is wrong" (404). Both are problems, but "down" means the gateway can't function; "wrong endpoint" means the monitoring is broken.**

### The `JobRecord` dataclass and `JobStatus` enum

```python
# job_store.py:51-60, 64-96
class JobStatus(str, enum.Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"

@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    submitted_at: str
    contract_code: str
    contract_address: str
    audit_timeout_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d.pop("contract_code", None)   # never expose contract_code in API
        return d
```

Why this matters: three patterns:

1. **`JobStatus(str, enum.Enum)`:** inherits from `str` so the value serializes to JSON as a plain string (`"queued"`, not `"JobStatus.queued"`). This matches the API contract — `JobResponse.status` is `str`, not `JobStatus`.

2. **`report` and `error` are mutually exclusive:** a completed job has `report` and `error=None`; a failed job has `error` and `report=None`. This is a convention, not enforced by the type system (both are `dict | None`). The convention is enforced by the state machine: `mark_completed` sets `report` and clears `error` (implicitly — it doesn't set `error`); `mark_failed` sets `error` and doesn't set `report`.

3. **`to_dict()` drops `contract_code`:** the API never returns the contract source in `GET /audit/{job_id}` — it can be MB-sized. Callers who need it can request it explicitly (via a separate endpoint, not yet built). `to_dict()` is the serialization boundary — it controls what the API exposes.

### The `AuditRequest` Pydantic model

```python
# models.py:45-100
class AuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_code: str = Field(..., min_length=1, max_length=MAX_CONTRACT_CHARS)
    contract_address: str | None = Field(default=None)
    audit_timeout_s: float = Field(default=300.0, gt=0, le=MAX_AUDIT_TIMEOUT_S)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("contract_code")
    @classmethod
    def _looks_like_solidity(cls, v: str) -> str:
        non_printable = sum(1 for c in v if not c.isprintable() and c not in "\n\r\t")
        if len(v) > 0 and non_printable / len(v) > 0.05:
            raise ValueError("contract_code contains >5% non-printable characters")
        return v
```

Why this matters: three patterns:

1. **`extra="forbid"`:** rejects unknown fields in the request body. If a client sends `{"contract_code": "...", "foo": "bar"}`, the request is rejected with 422. This prevents silent typos (`contract_adress` instead of `contract_address`) from being ignored. **Teaching: in APIs, reject unknown fields. Silent acceptance of typos leads to bugs that are hard to trace ("I set the timeout but it didn't work" → because the field was `timout`).**

2. **`max_length=MAX_CONTRACT_CHARS` (200,000):** hard cap on contract source size. Prevents DoS (a 1GB POST body would fill memory). Real-world contracts are typically <50KB; 200KB allows very large proxy implementations while blocking abuse.

3. **`_looks_like_solidity` validator:** rejects inputs with >5% non-printable characters (binary, compiled artifacts). Doesn't reject non-Solidity text (some legitimate inputs might not contain Solidity keywords). **Teaching: input validation should reject clearly-wrong inputs (binary) but not borderline ones (wrong language). The static-analysis nodes will report "not Solidity" themselves — the gateway just catches the "user pasted the wrong file" case.**

### The lifespan function — startup and shutdown

```python
# gateway.py:149-194
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"SENTINEL gateway v{GATEWAY_VERSION} starting up")

    # P10: Recover jobs left in RUNNING state after a crash.
    if hasattr(store, "recover_pending"):
        pending = store.recover_pending()
        for job in pending:
            store.mark_failed(job.job_id, "gateway restart (crash recovery)")

    # Initial service probe.
    if not skip_service_probes and not no_llm:
        services = await _probe_services()
        app.state.services = services

    # P10: Background health monitor.
    _health_task = None
    if not skip_service_probes:
        async def _health_loop():
            while True:
                await asyncio.sleep(30)
                services = await _probe_services()
                app.state.services = services

        _health_task = asyncio.create_task(_health_loop())

    yield  # gateway is running

    if _health_task:
        _health_task.cancel()
```

Why this matters: the lifespan function is the gateway's startup and shutdown logic. It runs *once* at startup (before any request) and *once* at shutdown (after all requests complete). Three things happen at startup:

1. **Crash recovery:** `recover_pending()` finds orphaned RUNNING jobs and marks them FAILED. This runs *before* the gateway accepts requests — no client sees a RUNNING job that's actually dead.

2. **Initial service probe:** `_probe_services()` runs once to populate `app.state.services` before the first `/health` request. Without this, the first `/health` call would have no cached services and would probe synchronously (up to 9s).

3. **Background health loop:** `_health_loop()` is started as an `asyncio.create_task`. It runs concurrently with request handlers. On shutdown (`yield` returns), the task is cancelled.

**Teaching: `hasattr(store, "recover_pending")` is a duck-type check.** The in-memory `JobStore` doesn't have `recover_pending` (it can't — in-memory jobs are lost on restart). `SqliteJobStore` does. The `hasattr` check lets the gateway work with both stores — the in-memory store for tests, the SQLite store for production. This is the benefit of the swappable interface: the gateway code doesn't know which store it's using; it checks for capabilities.

## Design Decision: SQLite vs Redis vs Postgres

> **How to read this section:** The table shows the options. The *elimination reasoning* shows how to think about the choice.

### The elimination process

**Redis — steel-man first:** "Redis is the industry standard for job queues. It's fast (in-memory), persistent (RDB snapshots), and supports multiple hosts. Why use SQLite when Redis is better?"

**Why it fails for SENTINEL (current scale):**
1. *Infrastructure:* Redis requires a Redis server process running. You need to install it, configure it, monitor it, and ensure it starts on boot. SQLite is a file — zero infrastructure.
2. *Network latency:* Redis is a separate process. Even on localhost, there's TCP overhead (~0.1ms per call). SQLite is in-process (no network).
3. *Client library:* Redis needs `redis-py` (or `aioredis`). SQLite uses Python's built-in `sqlite3` — no dependency.
4. *Eviction:* Redis has built-in eviction policies (LRU, LFU). SQLite needs manual eviction (`_evict_completed`). But manual eviction is 20 lines of code — not worth the infrastructure cost.

**Postgres — steel-man first:** "Postgres is the most robust relational database. ACID transactions, JSON columns, concurrent writes without locking. Why use SQLite when Postgres is production-grade?"

**Why it fails for SENTINEL (current scale):**
1. *Infrastructure:* Postgres requires a server process, configuration, user management, connection pooling. SQLite is a file.
2. *Overkill:* Postgres handles millions of rows, concurrent writes, complex queries. SENTINEL has 500 rows (max_completed=500) and 1-5 concurrent writes. Postgres's capabilities are unused.
3. *Connection management:* Postgres needs a connection pool (asyncpg, psycopg2). SQLite opens a connection per operation (microseconds).

**SQLite — why it survives:** it's a file. Zero infrastructure. Zero network. Zero dependencies. The write-lock limitation (single writer) triggers at ~100 concurrent writes — 20x above current load. When the trigger hits, the migration to Postgres is: (a) change `SqliteJobStore` to `PostgresJobStore`, (b) keep the same 7-method interface, (c) the gateway code doesn't change. The interface is the contract; the backend is swappable.

**The reasoning principle:** "Choose the simplest persistence that solves the *current* problem. Don't add infrastructure for problems you don't have yet. Design the interface so the backend is swappable — then migrate when the trigger hits, not before."

### When this decision would be wrong

**The reversal condition:** When the gateway runs on multiple hosts (e.g., 2 gateway processes behind a load balancer), SQLite's single-writer model breaks — both processes can't write to the same SQLite file simultaneously. At that point, migrate to Postgres (multi-process, row-level locking) or Redis (in-memory, atomic operations). The trigger: when you deploy a second gateway host. The migration is a new `PostgresJobStore` class implementing the same 7 methods.

## Technology Choice: SQLite

**The 5-question framework:**

1. **What category?** Embedded relational database (file-based, no server).
2. **What alternatives?** (a) SQLite (file, in-process), (b) Redis (in-memory, server), (c) Postgres (disk, server), (d) in-memory dict (no persistence).
3. **Why SQLite?** Zero infrastructure (file, no server), zero dependencies (built into Python), zero network (in-process), sufficient for 1-5 concurrent writes.
4. **When is Postgres better?** Multiple gateway hosts (multi-process writes), complex queries (JOINs, aggregations), >100 concurrent writes.
5. **Migration trigger:** Second gateway host deployed, or write contention detected (write latency >100ms consistently).

## Anti-Patterns

### ❌ In-memory store forever — "it's just a prototype"
**What it looks like:** `jobs: dict[str, JobRecord] = {}` — an in-memory dict. "We'll add persistence later. For now, the prototype works."
**Why someone would build this:** It's the fastest path to a working API. No file I/O, no schema, no serialization. The dict is the store.
**Why it's wrong:**
1. *Restart loses everything* — a crash, a deploy, a power failure erases all jobs. Clients polling `GET /audit/{job_id}` get 404.
2. *No debugging* — you can't inspect past jobs after a restart. "What happened to the audit I submitted 10 minutes ago?" → "It's gone."
3. *No crash recovery* — orphaned RUNNING jobs are invisible (they don't exist after restart).
**The right approach:** SQLite from day 1 for single-host. The interface is the same (7 methods). The cost is ~50 lines of code (schema + CRUD). The benefit is persistence, crash recovery, and debuggability.

### ❌ Per-request health probes — "probe on every /health call"
**What it looks like:** `services = await _probe_services()` inside the `/health` handler. "We want real-time health, not cached."
**Why someone would build this:** It sounds more accurate. "The cached result might be 30s stale — what if a service went down 5 seconds ago?"
**Why it's wrong:**
1. *9s latency per request* — 6 services × 1.5s timeout = 9s worst-case. Load balancers timeout at 5s and remove the gateway from rotation.
2. *Upstream load* — every `/health` call triggers 6 HTTP requests to upstream services. If 10 monitors probe `/health` every 5s, that's 60 upstream requests per 5s — DoS on your own services.
**The right approach:** Background health loop (every 30s) caches results in `app.state.services`. `/health` reads the cache. Latency: <1ms. Upstream load: 6 requests per 30s = 0.2 req/s per service.

## Mistakes & Fixes

### Mistake: In-memory `JobStore` lost all jobs on restart
**What happened:** The original gateway used an in-memory `OrderedDict` (`job_store.py`). The docstring literally said "NOT PRODUCTION-READY." Every gateway restart lost all jobs — queued, running, completed.
**Why it happened:** The in-memory store was the fastest path to a working API (WS6a prototype). Persistence was deferred.
**The fix:** `SqliteJobStore` (`sqlite_job_store.py`) — same 7-method interface, backed by SQLite. The swap was mechanical because the interface was clean: `create`, `mark_running`, `mark_completed`, `mark_failed`, `get`, `count_by_status`, `list_recent`. The gateway's `create_app(store: Any | None = None)` accepts any store — duck typing.
**The lesson:** Design the interface first, implement with the simplest backend (in-memory for prototyping), swap to a persistent backend when needed. The interface is the contract; the backend is an implementation detail. A clean 7-method interface makes the swap a few hours, not a rewrite.

### Mistake: `/health` probed 6 services on every request
**What happened:** The original `/health` endpoint called `_probe_services()` synchronously on every request. With 6 services × 1.5s timeout each, worst-case latency was 9s per `/health` call.
**Why it happened:** "We want real-time health." The per-request probe was the simplest implementation — no background task, no caching, no `app.state`.
**The fix:** Background `_health_loop()` probes every 30s and caches in `app.state.services`. `/health` reads the cache. Latency: <1ms.
**The lesson:** Expensive operations (network probes) should be cached, not repeated per-request. The staleness window (30s) is acceptable for a health check — a service that went down 5 seconds ago will be detected within 30s. For sub-second health detection, use a push model (services report their own health) instead of a pull model (gateway probes).

### Mistake: `/health` status ignored service health
**What happened:** The original `/health` status only considered job counts: `job_degraded = failed >= completed`. If the ML API was down (all audits failing), status was "degraded" (correct). But if the ML API was down and no audits had been submitted yet, status was "ok" (wrong — the gateway can't function).
**The fix:** `service_down = any(not s.ok for s in services)`. Status is "degraded" if job_degraded OR service_down.
**The lesson:** Health status should reflect *all* dimensions of gateway health, not just one. Job failure rate measures *past* problems; service health measures *current* problems. Both are needed: a gateway with 0 failed jobs but a dead ML API is not healthy.

### Mistake: No crash recovery — orphaned RUNNING jobs
**What happened:** If the gateway crashed while a job was RUNNING, the job stayed in RUNNING state forever. The client polled `GET /audit/{job_id}` and saw `status='running'` indefinitely.
**The fix:** `recover_pending()` on startup. `SELECT * FROM jobs WHERE status = 'running'` → `mark_failed(job_id, "gateway restart (crash recovery)")`.
**The lesson:** Any system with long-running background tasks needs crash recovery. The recovery should be: (a) find tasks in the "running" state, (b) mark them as "failed" (you can't resume them — the in-memory state is lost), (c) log the recovery so the human reviewer knows the job died in a crash, not in a code bug.

### Mistake: Type error — `store: JobStore | None` didn't accept `SqliteJobStore`
**What happened:** `create_app(store: JobStore | None = None)` had a type annotation of `JobStore`. When `SqliteJobStore` was passed, the type checker complained — `SqliteJobStore` doesn't inherit from `JobStore`.
**The fix:** Change to `store: Any | None = None`. Duck typing — any object with the right methods works. The type annotation is for documentation, not enforcement.
**The lesson:** When using duck typing (interface-based design), type annotations should be `Any` or a `Protocol`. Don't use a concrete class as the type annotation unless you're using inheritance. `SqliteJobStore` and `JobStore` share an interface but not inheritance — `Any` is the correct annotation.

## What Would Break If You Removed This?

**Remove SQLite (go back to in-memory):** restart loses all jobs. Client gets 404 on `GET /audit/{job_id}` after restart. No crash recovery — orphaned RUNNING jobs invisible.

**Remove the health monitor:** `/health` takes up to 9s per request. Load balancers timeout and remove the gateway from rotation. No audits accepted.

**Remove `recover_pending()`:** orphaned RUNNING jobs stay in RUNNING forever. Client polls indefinitely.

**Remove eviction:** database grows unbounded. `list_recent` query slows down as the table grows. Eventually the database file reaches GB size.

**Remove `_shrink_ml_result()`:** the full `ml_result` (including model internals, MLflow metadata) is stored in the report. The SQLite row grows to MB size. `GET /audit/{job_id}` response is huge.

**Remove the `asyncio.wait_for` timeout:** a hung audit runs forever, consuming a background task slot. Eventually all task slots are full and new audits can't start.

## At Scale

*Scale metric: concurrent audits (current: 1-5).*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| 1-5 concurrent (current) | SQLite, single-writer lock | — | — |
| 20 concurrent | SQLite still works | Write lock contention visible | Increase lock granularity |
| 100 concurrent | SQLite write latency rises | Write contention bottleneck | Migrate to Postgres |
| Multi-host | SQLite can't share | Multiple processes can't write to same file | Postgres + connection pool |

The SQLite write lock is the scale wall. At ~100 concurrent writes (each audit does ~4 writes: create + mark_running + mark_completed + evict), the lock serializes all writes — they queue. At 100 concurrent audits, write latency rises to ~10ms per write (lock contention). The trigger: when write latency consistently exceeds 100ms, migrate to Postgres (row-level locking, no single-writer bottleneck).

## Try It Yourself

> TRY IT: `cd agents && python -c "from src.api.sqlite_job_store import SqliteJobStore; import tempfile, os; td=tempfile.TemporaryDirectory(); s=SqliteJobStore(os.path.join(td.name,'test.db')); r=s.create('contract code','0xTEST',300.0); print(f'Created: {r.job_id[:8]} status={r.status.value}'); s.mark_running(r.job_id); s.mark_completed(r.job_id, {'verdict':'CONFIRMED'}); r2=s.get(r.job_id); print(f'Retrieved: status={r2.status.value} report={r2.report}'); print(f'Orphaned: {s.recover_pending()}')"`

> TRY IT: `cd agents && python -m pytest tests/test_p10_gateway.py -v` — runs all 13 P10 tests (SQLite CRUD, persistence, recovery, eviction, health status).

> TRY IT: `cd agents && python -c "from src.api.gateway import _probe_urls; [print(f'{name}: {url}') for name,url in _probe_urls()]"` — see the 6 upstream service URLs the gateway probes.

## Limitations & What's Missing

- **Single-host only.** SQLite's single-writer model means multiple gateway processes can't share a job queue. Multi-host deployment needs Postgres or Redis.

- **No job cancellation API.** There's no `POST /audit/{job_id}/cancel` endpoint. A long-running audit can't be cancelled — it runs until completion or timeout. The `asyncio.Task` object is stored in `app.state._tasks` but there's no API to cancel it.

- **No TTL on completed jobs.** Eviction happens when `max_completed` is exceeded (FIFO). There's no "delete jobs older than 7 days" policy. A job that completed 6 months ago is still in the database (if it's within the 500 most recent).

- **Sequential health probes.** `_probe_services()` probes 6 services sequentially. With `asyncio.gather`, it could probe all 6 concurrently (total time = max(individual_times) instead of sum). The sequential approach is fine for the background loop (30s interval) but would be slow for per-request probing.

- **No webhook/Slack alerting on service down.** The health monitor logs a warning when services are down, but doesn't send an alert (Slack webhook, email, PagerDuty). The operator must monitor logs.

- **No graph-state resume.** `recover_pending()` marks orphaned jobs as FAILED — it doesn't resume them from the LangGraph checkpoint. The checkpoint exists (SqliteSaver writes to `data/checkpoints.db`), but the gateway doesn't resume from it. This is P10+ work.

## Transferable Patterns

1. **Mechanical interface swap — design clean interface first, swap implementation later** — `JobStore` (7 methods) → `SqliteJobStore` (same 7 methods, different backend).
   - *Interview story:* "SENTINEL's gateway started with an in-memory job store — a dict with 7 methods. When we needed persistence, we wrote `SqliteJobStore` implementing the same 7 methods. The gateway code didn't change — only the store class name. The swap took 2 hours, not 2 days, because the interface was clean from the start. If we'd coupled the gateway to the in-memory store's internals (e.g., directly accessing `store._jobs`), the swap would have been a rewrite."
   - *When this pattern is WRONG:* when the interface is wrong. If `JobStore` exposed `self._jobs: OrderedDict` and the gateway iterated it directly, the SQLite swap would require changing every call site. The interface must be designed for swapping from day 1 — methods, not internals.

2. **Background health monitoring — cache expensive probes** — `_health_loop()` every 30s, `/health` reads cache.
   - *Interview story:* "Our `/health` endpoint probed 6 upstream services on every request — up to 9s latency. Load balancers timed out and removed us from rotation. We moved the probes to a background loop: every 30s, probe all services and cache the results in `app.state.services`. `/health` reads the cache in <1ms. The staleness window (30s) is acceptable — a service that went down 5 seconds ago is detected within 30s."
   - *When this pattern is WRONG:* when sub-second detection is required. If a service going down must be detected within 1 second, a 30s polling interval is too slow. In that case, use a push model (services report their own health via heartbeats) or a shorter interval (1-5s) with concurrent probes (`asyncio.gather`).

3. **Crash recovery — find orphans, mark them failed** — `recover_pending()` on startup.
   - *Interview story:* "When the SENTINEL gateway crashes, any job that was RUNNING is orphaned — the in-memory graph state is lost. On restart, `recover_pending()` finds all jobs with `status='running'` in SQLite and marks them as `failed` with the message 'gateway restart (crash recovery).' The client polling `GET /audit/{job_id}` sees `status='failed'` instead of polling forever. The human reviewer knows the job died in a crash, not in a code bug."
   - *When this pattern is WRONG:* when the system can resume the work. If the graph checkpoint (SqliteSaver) is available, the job can be resumed from the last completed node — not failed. Resuming is more complex (rebuild the graph, find the checkpoint, call `ainvoke(None, config={...})`) but avoids wasting the work done before the crash. Use `mark_failed` when resume is impossible; use `resume` when it's possible.

---

**Source files verified:**
- `agents/src/api/gateway.py:100-324, 328-402, 406-441, 444-470, 474-499` — create_app, lifespan, _run_job, _probe_services, _patch_no_llm, _derive_address, _shrink_ml_result
- `agents/src/api/sqlite_job_store.py:35-51, 54-76, 93-127, 129-175, 217-240, 242-252` — _SCHEMA, SqliteJobStore, create, mark_running/completed/failed, _evict_completed, recover_pending
- `agents/src/api/job_store.py:51-60, 64-96, 106-221` — JobStatus, JobRecord, JobStore (in-memory for comparison)
- `agents/src/api/models.py:30-41, 45-100, 104-175` — constants, AuditRequest, JobResponse, ServiceHealth, HealthResponse
- `agents/tests/test_p10_gateway.py` — 13 tests (SQLite CRUD, persistence, recovery, eviction, health)

**Verified against commit hash:** `c47898ea5`
