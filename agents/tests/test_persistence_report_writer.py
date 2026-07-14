"""Tests for R0.2 atomic report/hotspot persistence with structured status."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.persistence.report_writer import (
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
        assert json.loads(written.read_text()) == report

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


class TestPersistHotspot:
    def test_writes_job_scoped_hotspot(self, tmp_path):
        jid = str(uuid.uuid4())
        state = {"job_id": jid}
        html = "<!DOCTYPE html><html><body>hotspot</body></html>"
        status = persist_hotspot(state, html, tmp_path)
        assert status[PERSISTENCE_TOOL_KEY]["ran"] is True
        written = tmp_path / jid / "hotspot.html"
        assert written.exists()
        assert "hotspot" in written.read_text()

    def test_missing_job_id_no_file(self, tmp_path):
        state = {"job_id": ""}
        status = persist_hotspot(state, "<html/>", tmp_path)
        assert status[PERSISTENCE_TOOL_KEY]["ran"] is False
        assert list(tmp_path.glob("**/*.html")) == []
