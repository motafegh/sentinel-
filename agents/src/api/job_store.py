"""
In-memory job store for the SENTINEL audit gateway (WS6a / C.1, 2026-06-22).

WHY A SEPARATE MODULE
=====================
The gateway's job lifecycle (create → queue → run → complete | fail) is
simple, but the dict itself needs to be thread-safe + size-bounded +
inspectable. Extracting it from `gateway.py` makes it testable in isolation
without spinning up FastAPI.

NOT PRODUCTION-READY
====================
This is a single-process dict. Restart = all jobs lost. For production:
  - swap for SQLite (single-host, no extra infra) OR
  - swap for Redis (multi-host, survives gateway restart, supports eviction).

Both swaps are mechanical: replace `JobStore._jobs` with a backend that
implements the same `get` / `put` / `update` / `list_recent` interface.
The class is kept deliberately small to make that swap easy.

CONCURRENCY
===========
We use a single `threading.Lock` for the dict, BUT the heavy work (running
the graph) happens in asyncio tasks OUTSIDE the lock. The lock only
protects the dict read/write — never the audit itself. This is safe because:
  1. The asyncio event loop runs in a single thread.
  2. `asyncio.create_task` schedules coroutines cooperatively.
  3. The lock is only held for dict operations (microseconds), never across
     an `await` on the graph.

EVICTION
========
To prevent memory bloat from runaway submissions, the store evicts the
oldest completed/failed jobs once `max_completed` is exceeded. The bound
is configurable but defaults to 100 — high enough for normal use, low
enough that a malicious client can't fill the dict.
"""

from __future__ import annotations

import enum
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


# ── Status enum ──────────────────────────────────────────────────────────
class JobStatus(str, enum.Enum):
    """Lifecycle status of an audit job.

    Inherits from `str` so the value serialises to JSON as a plain string
    (matching what the API contract documents).
    """
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"


# ── Record ──────────────────────────────────────────────────────────────
@dataclass
class JobRecord:
    """All the state for one audit job.

    `report` and `error` are mutually exclusive: a completed job has
    `report` and `error=None`; a failed job has `error` and `report=None`.

    `metadata` is opaque to the gateway — stored as-is, returned as-is
    on `GET /audit/{job_id}`.
    """
    job_id: str
    status: JobStatus
    submitted_at: str  # ISO-8601 UTC
    contract_code: str
    contract_address: str
    audit_timeout_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict matching the `JobResponse` Pydantic model.

        NOTE: the API never returns `contract_code` (it can be MB-sized);
        callers who need it can request it explicitly. We drop it here
        to keep `GET /audit/{job_id}` responses bounded.
        """
        d = asdict(self)
        d["status"] = self.status.value
        d.pop("contract_code", None)
        return d

    def to_full_dict(self) -> dict[str, Any]:
        """Like `to_dict` but includes `contract_code` — for debugging only."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ── Store ───────────────────────────────────────────────────────────────
class JobStore:
    """Thread-safe in-memory job record store.

    Threading model: see module docstring §CONCURRENCY. asyncio is enough
    in practice (the gateway's background tasks all run on the uvicorn
    event loop), but the lock keeps the class usable from non-async tests
    and from sync health-check code.
    """

    def __init__(self, max_completed: int = 100) -> None:
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._lock = threading.Lock()
        self.max_completed = max_completed

    # ── Mutators ──────────────────────────────────────────────────────
    def create(
        self,
        contract_code: str,
        contract_address: str,
        audit_timeout_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Create a new QUEUED job. Returns the record."""
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
            self._jobs[record.job_id] = record
        return record

    def mark_running(self, job_id: str) -> None:
        """Transition QUEUED → RUNNING. No-op if already past QUEUED."""
        with self._lock:
            r = self._jobs.get(job_id)
            if r is not None and r.status == JobStatus.QUEUED:
                r.status = JobStatus.RUNNING
                r.started_at = _utcnow_iso()

    def mark_completed(self, job_id: str, report: dict[str, Any]) -> None:
        """Transition RUNNING → COMPLETED and attach the report."""
        with self._lock:
            r = self._jobs.get(job_id)
            if r is not None and r.status == JobStatus.RUNNING:
                r.status = JobStatus.COMPLETED
                r.finished_at = _utcnow_iso()
                r.report = report
        # Evict after the lock is released — see _evict_completed.
        self._evict_completed()

    def mark_failed(self, job_id: str, error: str) -> None:
        """Transition QUEUED or RUNNING → FAILED and record the error."""
        with self._lock:
            r = self._jobs.get(job_id)
            if r is not None and r.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                r.status = JobStatus.FAILED
                r.finished_at = _utcnow_iso()
                r.error = error
        self._evict_completed()

    # ── Readers ───────────────────────────────────────────────────────
    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            return {
                s.value: sum(1 for r in self._jobs.values() if r.status == s)
                for s in JobStatus
            }

    def list_recent(self, n: int = 50) -> list[JobRecord]:
        """Return the N most recent jobs, newest first."""
        with self._lock:
            return list(reversed(self._jobs.values()))[:n]

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def __iter__(self) -> Iterator[JobRecord]:
        with self._lock:
            return iter(list(self._jobs.values()))

    # ── Internals ─────────────────────────────────────────────────────
    def _evict_completed(self) -> None:
        """Drop oldest COMPLETED/FAILED jobs when count exceeds `max_completed`.

        Eviction policy: keep at most `max_completed` terminal jobs (completed
        or failed). QUEUED/RUNNING jobs are NEVER evicted. If the queue is
        full of in-flight jobs, we don't evict them — only the terminal
        cache is bounded.

        Called OUTSIDE the per-method lock so we don't re-enter the lock.
        Acquires its own lock; safe because Python's GIL makes dict
        re-ordering atomic and `OrderedDict.move_to_end` is also atomic
        under the GIL. Still, we use the lock for hygiene.
        """
        with self._lock:
            terminal = [
                jid for jid, r in self._jobs.items()
                if r.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            ]
            excess = len(terminal) - self.max_completed
            if excess <= 0:
                return
            # Evict the oldest terminal jobs (FIFO by insertion order).
            for jid in terminal[:excess]:
                del self._jobs[jid]


# ── Helpers ─────────────────────────────────────────────────────────────
def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with a `Z` suffix (e.g. `2026-06-22T18:30:45.123Z`).

    We use the `Z` suffix (not `+00:00`) because:
      1. It's what the rest of the agents module uses (run_real_audit.py).
      2. It's the most common convention for HTTP API timestamps.

    Sub-second precision is preserved — useful for debugging race conditions
    on concurrent submissions.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
