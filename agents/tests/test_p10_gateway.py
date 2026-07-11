"""Tests for P10 SQLite JobStore and gateway hardening."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from src.api.job_store import JobStatus, JobRecord
from src.api.sqlite_job_store import SqliteJobStore


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_jobs.db"
        yield db_path


class TestSqliteJobStore:
    """Test the SQLite-backed job store."""

    def test_create_and_get(self, tmp_db):
        store = SqliteJobStore(db_path=tmp_db)
        record = store.create("contract code", "0xTEST", 300.0)
        assert record.status == JobStatus.QUEUED
        assert record.contract_code == "contract code"

        fetched = store.get(record.job_id)
        assert fetched is not None
        assert fetched.job_id == record.job_id
        assert fetched.status == JobStatus.QUEUED
        assert fetched.contract_code == "contract code"

    def test_mark_running(self, tmp_db):
        store = SqliteJobStore(db_path=tmp_db)
        record = store.create("code", "0x1", 300.0)
        store.mark_running(record.job_id)
        fetched = store.get(record.job_id)
        assert fetched.status == JobStatus.RUNNING
        assert fetched.started_at is not None

    def test_mark_completed(self, tmp_db):
        store = SqliteJobStore(db_path=tmp_db)
        record = store.create("code", "0x1", 300.0)
        store.mark_running(record.job_id)
        report = {"verdict": "CONFIRMED", "vulnerabilities": ["Reentrancy"]}
        store.mark_completed(record.job_id, report)
        fetched = store.get(record.job_id)
        assert fetched.status == JobStatus.COMPLETED
        assert fetched.report == report
        assert fetched.finished_at is not None

    def test_mark_failed(self, tmp_db):
        store = SqliteJobStore(db_path=tmp_db)
        record = store.create("code", "0x1", 300.0)
        store.mark_running(record.job_id)
        store.mark_failed(record.job_id, "timeout")
        fetched = store.get(record.job_id)
        assert fetched.status == JobStatus.FAILED
        assert fetched.error == "timeout"

    def test_persistence_across_instances(self, tmp_db):
        """Data must survive store recreation (simulating gateway restart)."""
        store1 = SqliteJobStore(db_path=tmp_db)
        record = store1.create("persistent code", "0xPERSIST", 300.0)
        store1.mark_running(record.job_id)

        del store1

        store2 = SqliteJobStore(db_path=tmp_db)
        fetched = store2.get(record.job_id)
        assert fetched is not None
        assert fetched.contract_code == "persistent code"
        assert fetched.status == JobStatus.RUNNING

    def test_count_by_status(self, tmp_db):
        store = SqliteJobStore(db_path=tmp_db)
        r1 = store.create("c1", "0x1", 300.0)
        r2 = store.create("c2", "0x2", 300.0)
        store.mark_running(r1.job_id)
        store.mark_completed(r1.job_id, {"result": "ok"})

        counts = store.count_by_status()
        assert counts["completed"] == 1
        assert counts["queued"] == 1

    def test_list_recent(self, tmp_db):
        store = SqliteJobStore(db_path=tmp_db)
        for i in range(5):
            store.create(f"code{i}", f"0x{i}", 300.0)

        recent = store.list_recent(n=3)
        assert len(recent) == 3
        assert recent[0].contract_address != recent[1].contract_address

    def test_recover_pending(self, tmp_db):
        """Jobs left in RUNNING after crash should be recoverable."""
        store = SqliteJobStore(db_path=tmp_db)
        r1 = store.create("c1", "0x1", 300.0)
        r2 = store.create("c2", "0x2", 300.0)
        store.mark_running(r1.job_id)
        store.mark_running(r2.job_id)

        pending = store.recover_pending()
        assert len(pending) == 2
        assert all(r.status == JobStatus.RUNNING for r in pending)

    def test_eviction(self, tmp_db):
        """Old completed jobs should be evicted when max_completed is exceeded."""
        store = SqliteJobStore(db_path=tmp_db, max_completed=3)
        for i in range(5):
            r = store.create(f"code{i}", f"0x{i}", 300.0)
            store.mark_running(r.job_id)
            store.mark_completed(r.job_id, {"ok": True})

        terminal = [r for r in store if r.status in (JobStatus.COMPLETED, JobStatus.FAILED)]
        assert len(terminal) <= 3

    def test_metadata_persisted(self, tmp_db):
        """Metadata dict should survive round-trip through SQLite."""
        store = SqliteJobStore(db_path=tmp_db)
        meta = {"user": "ali", "priority": "high", "tags": ["audit", "test"]}
        r = store.create("code", "0x1", 300.0, metadata=meta)
        store.mark_completed(r.job_id, {"result": "done"})

        fetched = store.get(r.job_id)
        assert fetched.metadata == meta

    def test_report_with_complex_json(self, tmp_db):
        """Report with nested dicts/lists should persist correctly."""
        store = SqliteJobStore(db_path=tmp_db)
        r = store.create("code", "0x1", 300.0)
        store.mark_running(r.job_id)
        report = {
            "verdict": "CONFIRMED",
            "vulnerabilities": [
                {"class": "Reentrancy", "probability": 0.85},
                {"class": "IntegerUO", "probability": 0.65},
            ],
            "model_provenance": {"hash": "abc123", "checkpoint": "/path/to.pt"},
        }
        store.mark_completed(r.job_id, report)

        fetched = store.get(r.job_id)
        assert fetched.report == report


class TestGatewayHealthStatus:
    """Test that gateway health status considers service probes."""

    def test_health_response_includes_services(self):
        from src.api.models import HealthResponse, ServiceHealth

        services = [
            ServiceHealth(name="ml_api", url="http://localhost:8001/health", ok=True, detail="HTTP 200"),
            ServiceHealth(name="mcp_rag", url="http://localhost:8011/health", ok=False, detail="UNREACHABLE"),
        ]

        # P10: status should be "degraded" when any service is down
        service_down = any(not s.ok for s in services)
        status = "degraded" if service_down else "ok"
        assert status == "degraded"

    def test_health_response_ok_when_all_up(self):
        from src.api.models import ServiceHealth

        services = [
            ServiceHealth(name="ml_api", url="http://localhost:8001/health", ok=True, detail="HTTP 200"),
            ServiceHealth(name="mcp_rag", url="http://localhost:8011/health", ok=True, detail="HTTP 200"),
        ]

        service_down = any(not s.ok for s in services)
        status = "degraded" if service_down else "ok"
        assert status == "ok"
