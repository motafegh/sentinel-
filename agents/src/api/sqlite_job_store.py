"""
SQLite-backed job store for the SENTINEL audit gateway (P10, 2026-06-26).

Replaces the in-memory JobStore with persistent SQLite storage.
Survives gateway restarts — jobs are recovered on startup.

Interface matches JobStore: create, mark_running, mark_completed,
mark_failed, get, count_by_status, list_recent, __len__, __iter__.

Concurrency:
    SQLite with check_same_thread=False is safe for multi-threaded access
    when writes are serialized via a threading.Lock (same pattern as the
    in-memory store). The lock is only held for DB operations (microseconds),
    never across graph.ainvoke().

Schema:
    Single table `jobs` with columns matching JobRecord fields.
    `report` and `metadata` stored as JSON text.
    `contract_code` stored as TEXT (can be large but SQLite handles MB-sized
    rows fine for single-host use).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.api.job_store import JobRecord, JobStatus

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
    """Persistent SQLite-backed job store. Drop-in replacement for JobStore."""

    def __init__(
        self,
        db_path: str | Path = "data/jobs.db",
        max_completed: int = 500,
    ) -> None:
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

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            submitted_at=row["submitted_at"],
            contract_code=row["contract_code"],
            contract_address=row["contract_address"],
            audit_timeout_s=row["audit_timeout_s"],
            metadata=json.loads(row["metadata"] or "{}"),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            report=json.loads(row["report"]) if row["report"] else None,
            error=row["error"],
        )

    def create(
        self,
        contract_code: str,
        contract_address: str,
        audit_timeout_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> JobRecord:
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
                (
                    record.job_id,
                    record.status.value,
                    record.submitted_at,
                    record.contract_code,
                    record.contract_address,
                    record.audit_timeout_s,
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
            conn.close()
        return record

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """UPDATE jobs SET status = ?, started_at = ?
                   WHERE job_id = ? AND status = ?""",
                (JobStatus.RUNNING.value, _utcnow_iso(), job_id, JobStatus.QUEUED.value),
            )
            conn.commit()
            conn.close()

    def mark_completed(self, job_id: str, report: dict[str, Any]) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """UPDATE jobs SET status = ?, finished_at = ?, report = ?
                   WHERE job_id = ? AND status = ?""",
                (
                    JobStatus.COMPLETED.value,
                    _utcnow_iso(),
                    json.dumps(report, default=str),
                    job_id,
                    JobStatus.RUNNING.value,
                ),
            )
            conn.commit()
            conn.close()
        self._evict_completed()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """UPDATE jobs SET status = ?, finished_at = ?, error = ?
                   WHERE job_id = ? AND status IN (?, ?)""",
                (
                    JobStatus.FAILED.value,
                    _utcnow_iso(),
                    error,
                    job_id,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                ),
            )
            conn.commit()
            conn.close()
        self._evict_completed()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            conn.close()
        return self._row_to_record(row) if row else None

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
            ).fetchall()
            conn.close()
        counts = {s.value: 0 for s in JobStatus}
        for row in rows:
            counts[row["status"]] = row["cnt"]
        return counts

    def list_recent(self, n: int = 50) -> list[JobRecord]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (n,)
            ).fetchall()
            conn.close()
        return [self._row_to_record(row) for row in rows]

    def __len__(self) -> int:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT COUNT(*) as cnt FROM jobs").fetchone()
            conn.close()
        return row["cnt"] if row else 0

    def __iter__(self) -> Iterator[JobRecord]:
        return iter(self.list_recent(n=10000))

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
            conn.commit()
            conn.close()

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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
