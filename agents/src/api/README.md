# API — Audit Gateway

FastAPI gateway (P10, 2026-06-26) that exposes SENTINEL's audit pipeline as an async
HTTP service. Accepts audit jobs, dispatches them to the LangGraph pipeline in the
background, and returns results via polling or webhook.

## Architecture

```
HTTP client
    │
    ▼
gateway.py  (FastAPI)
    │  POST /audit          → enqueue job → SqliteJobStore
    │  GET  /audit/{job_id} → poll status + result
    │  GET  /health         → service health (probes 6 downstream services)
    │
    ├── SqliteJobStore ──── data/jobs.db  (persists across restarts)
    │
    └── Background worker ──── LangGraph audit_graph.ainvoke(...)
                               writes result back to SqliteJobStore
```

## Files

| File | Purpose |
|------|---------|
| `gateway.py` | FastAPI app, route handlers, background audit dispatch |
| `job_store.py` | Abstract `JobStore` interface |
| `sqlite_job_store.py` | SQLite-backed implementation — crash-safe persistence |
| `models.py` | Pydantic request/response models |

## `gateway.py`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/audit` | Submit a new audit job. Returns `{job_id}` immediately. |
| `GET` | `/audit/{job_id}` | Poll job status and result. |
| `GET` | `/health` | Service liveness + downstream health check. |

### Background Health Monitor

A background task probes 6 downstream services every 30 seconds and caches results:

| Service | Port | What is checked |
|---------|------|----------------|
| ML inference | 8001 | `/health` HTTP 200 |
| MCP inference | 8010 | `/health` HTTP 200 |
| MCP RAG | 8011 | `/health` HTTP 200 |
| MCP audit | 8012 | `/health` HTTP 200 |
| MCP graph inspector | 8013 | `/health` HTTP 200 |
| MCP representation | 8014 | `/health` HTTP 200 |

`GET /health` response reflects cached probe results — `"degraded"` if any service is
down, `"healthy"` if all pass. This gives ops teams real signal rather than just job
queue depth.

## `sqlite_job_store.py` — Persistent Job Store

Replaces the old in-memory `JobStore`. Persists every job to `data/jobs.db` so the
gateway can be restarted without losing in-flight or completed jobs.

### Job Lifecycle

```
PENDING → RUNNING → DONE
                 ↘ FAILED
```

### Crash Recovery

`recover_pending()` is called at gateway startup. It finds all jobs still in `RUNNING`
state (i.e. the process crashed while they were executing) and marks them `FAILED` with
`reason="process_restart"`. This prevents jobs from being stuck in `RUNNING` forever.

### Usage

```python
from src.api.sqlite_job_store import SqliteJobStore

store = SqliteJobStore("data/jobs.db")
job_id = store.create(contract_code="...", contract_address="0x...")
store.set_running(job_id)
store.set_done(job_id, result={...})
job = store.get(job_id)     # {"status": "DONE", "result": {...}}
```

## `models.py` — Request/Response Models

```python
class AuditRequest(BaseModel):
    contract_code:    str
    contract_address: str = ""

class AuditResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    job_id:  str
    status:  str          # PENDING | RUNNING | DONE | FAILED
    result:  dict | None  # final_report when DONE
    error:   str | None   # reason when FAILED
```

## Starting the Gateway

```bash
cd agents
poetry run uvicorn src.api.gateway:app --host 0.0.0.0 --port 8080 --log-level info
```

## Running Tests

```bash
cd agents
poetry run pytest tests/test_gateway.py tests/test_p10_gateway.py -v
```

13 tests cover: job creation, status polling, crash recovery, health monitor degraded
state, and concurrent job submission.
