"""Tests for R0.2 atomic report/hotspot persistence with structured status."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.persistence.report_writer import (
    HOTSPOT_PERSISTENCE_TOOL_KEY,
    PERSISTENCE_TOOL_KEY,
    persist_hotspot,
    persist_report,
)


class TestPersistReport:
    def test_writes_job_scoped_report(self, tmp_path):
        jid = str(uuid.uuid4())
        state = {"job_id": jid}
        report = {"overall_label": "safe", "top_vulnerability": None}
        status = persist_report(state, report, tmp_path)
        assert status[PERSISTENCE_TOOL_KEY]["ran"] is True
        written = tmp_path / jid / "report.json"
        assert written.exists()
        persisted = json.loads(written.read_text())
        assert persisted["overall_label"] == report["overall_label"]
        assert persisted["top_vulnerability"] == report["top_vulnerability"]
        assert persisted["tool_status"][PERSISTENCE_TOOL_KEY]["ran"] is True

    def test_missing_job_id_returns_structured_skip(self, tmp_path):
        state = {"job_id": ""}
        status = persist_report(state, {"x": 1}, tmp_path)
        assert status[PERSISTENCE_TOOL_KEY]["ran"] is False
        assert status[PERSISTENCE_TOOL_KEY]["reason"] == "missing_or_invalid_job_id"
        assert list(tmp_path.glob("**/*.json")) == []

    def test_invalid_job_id_returns_structured_skip(self, tmp_path):
        state = {"job_id": "../../etc/passwd"}
        status = persist_report(state, {"x": 1}, tmp_path)
        assert status[PERSISTENCE_TOOL_KEY]["ran"] is False
        assert "UUID" in status[PERSISTENCE_TOOL_KEY]["detail"]

    def test_no_job_id_key_in_state(self, tmp_path):
        state = {}
        status = persist_report(state, {"x": 1}, tmp_path)
        assert status[PERSISTENCE_TOOL_KEY]["ran"] is False
        assert status[PERSISTENCE_TOOL_KEY]["reason"] == "missing_or_invalid_job_id"

    def test_atomic_write_replaces_existing(self, tmp_path):
        jid = str(uuid.uuid4())
        state = {"job_id": jid}
        persist_report(state, {"version": 1}, tmp_path)
        persist_report(state, {"version": 2}, tmp_path)
        written = tmp_path / jid / "report.json"
        assert json.loads(written.read_text())["version"] == 2

    def test_concurrent_same_address_different_jobs(self, tmp_path):
        jid1 = str(uuid.uuid4())
        jid2 = str(uuid.uuid4())
        persist_report({"job_id": jid1}, {"addr": "0xABC", "v": 1}, tmp_path)
        persist_report({"job_id": jid2}, {"addr": "0xABC", "v": 2}, tmp_path)
        assert (tmp_path / jid1 / "report.json").exists()
        assert (tmp_path / jid2 / "report.json").exists()
        assert json.loads((tmp_path / jid1 / "report.json").read_text())["v"] == 1
        assert json.loads((tmp_path / jid2 / "report.json").read_text())["v"] == 2

    def test_concurrent_same_job_uses_unique_temp_files(self, tmp_path):
        jid = str(uuid.uuid4())
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(
                pool.map(
                    lambda version: persist_report({"job_id": jid}, {"version": version}, tmp_path),
                    range(12),
                )
            )
        assert all(status[PERSISTENCE_TOOL_KEY]["ran"] for status in statuses)
        assert json.loads((tmp_path / jid / "report.json").read_text())["version"] in range(12)
        assert not list((tmp_path / jid).glob("*.tmp"))


class TestPersistHotspot:
    def test_writes_job_scoped_hotspot(self, tmp_path):
        jid = str(uuid.uuid4())
        state = {"job_id": jid}
        html = "<!DOCTYPE html><html><body>hotspot</body></html>"
        status = persist_hotspot(state, html, tmp_path)
        assert status[HOTSPOT_PERSISTENCE_TOOL_KEY]["ran"] is True
        written = tmp_path / jid / "hotspot.html"
        assert written.exists()
        assert "hotspot" in written.read_text()

    def test_missing_job_id_no_file(self, tmp_path):
        state = {"job_id": ""}
        status = persist_hotspot(state, "<html/>", tmp_path)
        assert status[HOTSPOT_PERSISTENCE_TOOL_KEY]["ran"] is False
        assert list(tmp_path.glob("**/*.html")) == []

    def test_report_directory_symlink_is_rejected(self, tmp_path):
        jid = str(uuid.uuid4())
        outside = tmp_path / "outside"
        outside.mkdir()
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / jid).symlink_to(outside, target_is_directory=True)
        status = persist_hotspot({"job_id": jid}, "<html/>", reports)
        assert status[HOTSPOT_PERSISTENCE_TOOL_KEY]["ran"] is False
        assert status[HOTSPOT_PERSISTENCE_TOOL_KEY]["reason"] == "write_failure"
        assert not (outside / "hotspot.html").exists()
