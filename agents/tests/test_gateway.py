"""
Tests for the WS6a Phase C.1 FastAPI gateway (2026-06-22).

Coverage:
  - Models: AuditRequest validation (size, binary rejection, default timeout).
  - JobStore: state machine, eviction, thread safety, deterministic address.
  - Gateway: full E2E with a stub graph (success, fail, timeout).
  - Gateway: 404, 422, list, health, concurrent submissions.
  - Gateway: exception in graph is captured, not propagated.
  - Gateway: --no-llm flag pass-through.

A `StubGraph` helper (built per-test) keeps these tests fast and free of
real-service dependencies — the only slow op is the stub's `asyncio.sleep`,
which is microseconds.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# Make agents/ importable + make sure the agents venv is used.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.api.gateway import _derive_address, create_app
from src.api.job_store import JobRecord, JobStatus, JobStore
from src.api.models import (
    MAX_CONTRACT_CHARS,
    AuditRequest,
    ErrorResponse,
    HealthResponse,
    JobResponse,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════

VALID_SOLIDITY = (
    "// SPDX-License-Identifier: MIT\n"
    "pragma solidity ^0.8.0;\n"
    "contract Vault {\n"
    "    mapping(address => uint256) public balances;\n"
    "    function deposit() public payable { balances[msg.sender] += msg.value; }\n"
    "    function withdraw(uint256 amount) public {\n"
    "        require(balances[msg.sender] >= amount);\n"
    "        (bool ok, ) = msg.sender.call{value: amount}(\"\");\n"
    "        require(ok);\n"
    "        balances[msg.sender] -= amount;\n"
    "    }\n"
    "}\n"
)


class _StubGraph:
    """Stub LangGraph. Configurable per-instance via constructor.

    The real `langgraph.graph.StateGraph.ainvoke` returns a dict that
    matches the structure of the real graph output. Our stub returns a
    faithful subset — enough for the gateway to build a report.
    """

    def __init__(
        self,
        *,
        sleep_s: float = 0.01,
        raise_exc: Exception | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        self.sleep_s = sleep_s
        self.raise_exc = raise_exc
        self.result = result or {
            "final_report": {
                "overall_label": "confirmed_vulnerable",
                "overall_verdict": "VULNERABLE",
                "top_vulnerability": "Reentrancy",
            },
            "verdicts": {"Reentrancy": "CONFIRMED"},
            "ml_result": {
                "label": "suspicious",
                "probabilities": {"Reentrancy": 0.82},
                "truncated": False,
                "num_nodes": 42,
            },
            "static_findings": [{"tool": "slither", "detector": "reentrancy-eth"}],
            "rag_results": [],
            "audit_history": [],
            "routing_decisions": ["Reentrancy prob=0.82 >= 0.35 → static_analysis"],
            "consensus_verdict": {"Reentrancy": {"verdict": "CONFIRMED", "score": 0.82}},
            "debate_transcript": {},
            "confirmations": {"Reentrancy": ["ml:0.82", "slither:reentrancy-eth"]},
            "contradictions": {},
            "narrative": None,
            "error": None,
        }
        self.call_count = 0
        self.last_state: dict | None = None

    async def ainvoke(self, state: dict) -> dict:
        self.call_count += 1
        self.last_state = state
        if self.sleep_s > 0:
            await asyncio.sleep(self.sleep_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.result


def _make_client(
    graph: _StubGraph | None = None,
    *,
    store: JobStore | None = None,
    audit_timeout_s: float = 30.0,
) -> tuple[TestClient, JobStore, _StubGraph]:
    """Build a TestClient + JobStore + stub graph trio for one test.

    The TestClient is returned inside a `with` block, so the caller MUST
    use it as `with _make_client(...) as (client, store, graph):`. The
    context manager triggers the FastAPI lifespan, which is required for
    the background tasks to run.
    """
    g = graph or _StubGraph()
    s = store or JobStore()
    # We need to control audit_timeout_s per-request, but the gateway's
    # AuditRequest already takes it as a field. So we just leave it to
    # the request body.
    app = create_app(store=s, graph_factory=lambda: g, skip_service_probes=True)
    return TestClient(app), s, g


# ═══════════════════════════════════════════════════════════════════════════
# Model-level tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditRequestValidation:
    def test_valid_solidity_passes(self):
        req = AuditRequest(contract_code=VALID_SOLIDITY)
        assert req.contract_address is None
        assert req.audit_timeout_s == 300.0
        assert req.metadata == {}

    def test_empty_contract_code_rejected(self):
        with pytest.raises(Exception):  # ValidationError
            AuditRequest(contract_code="")

    def test_too_long_contract_code_rejected(self):
        with pytest.raises(Exception):
            AuditRequest(contract_code="x" * (MAX_CONTRACT_CHARS + 1))

    def test_max_size_boundary_accepted(self):
        # Exactly at the cap — accepted.
        req = AuditRequest(contract_code="x" * MAX_CONTRACT_CHARS)
        assert len(req.contract_code) == MAX_CONTRACT_CHARS

    def test_binary_input_rejected(self):
        # >5% non-printable characters triggers the validator.
        binary = "\x00\x01\x02\x03" * 100  # all non-printable
        with pytest.raises(Exception):
            AuditRequest(contract_code=binary)

    def test_audit_timeout_bounds(self):
        with pytest.raises(Exception):
            AuditRequest(contract_code=VALID_SOLIDITY, audit_timeout_s=0)
        with pytest.raises(Exception):
            AuditRequest(contract_code=VALID_SOLIDITY, audit_timeout_s=10_000)

    def test_extra_fields_rejected(self):
        # `extra="forbid"` is set on AuditRequest.
        import json
        with pytest.raises(Exception):
            AuditRequest.model_validate_json(
                json.dumps({"contract_code": VALID_SOLIDITY, "rogue_field": "evil"})
            )

    def test_audit_timeout_default(self):
        # The default timeout matches run_real_audit.py's default.
        req = AuditRequest(contract_code=VALID_SOLIDITY)
        assert req.audit_timeout_s == 300.0


# ═══════════════════════════════════════════════════════════════════════════
# JobStore tests (unit tests, no FastAPI)
# ═══════════════════════════════════════════════════════════════════════════

class TestJobStore:
    def test_create_returns_queued_record(self):
        store = JobStore()
        r = store.create(contract_code=VALID_SOLIDITY, contract_address="0xa",
                         audit_timeout_s=300.0)
        assert r.status == JobStatus.QUEUED
        assert r.started_at is None
        assert r.finished_at is None
        assert r.report is None
        assert r.error is None

    def test_full_lifecycle(self):
        store = JobStore()
        r = store.create(contract_code=VALID_SOLIDITY, contract_address="0xa",
                         audit_timeout_s=300.0)
        store.mark_running(r.job_id)
        assert store.get(r.job_id).status == JobStatus.RUNNING
        assert store.get(r.job_id).started_at is not None
        store.mark_completed(r.job_id, {"final_report": {"verdict": "VULN"}})
        rec = store.get(r.job_id)
        assert rec.status == JobStatus.COMPLETED
        assert rec.finished_at is not None
        assert rec.report == {"final_report": {"verdict": "VULN"}}

    def test_mark_completed_requires_running(self):
        """Can't mark COMPLETED from QUEUED — must transition through RUNNING."""
        store = JobStore()
        r = store.create(contract_code=VALID_SOLIDITY, contract_address="0xa",
                         audit_timeout_s=300.0)
        store.mark_completed(r.job_id, {})  # should be no-op
        assert store.get(r.job_id).status == JobStatus.QUEUED  # unchanged

    def test_mark_failed_from_queued(self):
        store = JobStore()
        r = store.create(contract_code=VALID_SOLIDITY, contract_address="0xa",
                         audit_timeout_s=300.0)
        store.mark_failed(r.job_id, "boom")
        rec = store.get(r.job_id)
        assert rec.status == JobStatus.FAILED
        assert rec.error == "boom"
        assert rec.finished_at is not None

    def test_mark_failed_from_running(self):
        store = JobStore()
        r = store.create(contract_code=VALID_SOLIDITY, contract_address="0xa",
                         audit_timeout_s=300.0)
        store.mark_running(r.job_id)
        store.mark_failed(r.job_id, "graph exploded")
        assert store.get(r.job_id).status == JobStatus.FAILED

    def test_get_unknown_returns_none(self):
        store = JobStore()
        assert store.get("nonexistent") is None

    def test_count_by_status(self):
        store = JobStore()
        for i in range(3):
            r = store.create(contract_code="c", contract_address="0x", audit_timeout_s=300)
            store.mark_running(r.job_id)
            store.mark_completed(r.job_id, {})
        r = store.create(contract_code="c", contract_address="0x", audit_timeout_s=300)
        store.mark_failed(r.job_id, "err")
        counts = store.count_by_status()
        assert counts == {"queued": 0, "running": 0, "completed": 3, "failed": 1}

    def test_eviction(self):
        store = JobStore(max_completed=3)
        ids = []
        for i in range(6):
            r = store.create(contract_code="c", contract_address=f"0x{i}", audit_timeout_s=300)
            store.mark_running(r.job_id)
            store.mark_completed(r.job_id, {})
            ids.append(r.job_id)
        assert len(store) == 3
        # The 3 oldest (ids[0..2]) should be evicted, ids[3..5] remain.
        for old in ids[:3]:
            assert store.get(old) is None
        for new in ids[3:]:
            assert store.get(new) is not None

    def test_eviction_preserves_in_flight(self):
        """QUEUED/RUNNING jobs must NEVER be evicted, only terminal ones."""
        store = JobStore(max_completed=2)
        # 5 in-flight + 3 terminal → 5 + (3-2)=6 total, all 5 in-flight kept.
        in_flight = []
        for i in range(5):
            r = store.create(contract_code="c", contract_address=f"0x{i}", audit_timeout_s=300)
            in_flight.append(r.job_id)
        for i in range(3):
            r = store.create(contract_code="c", contract_address=f"t{i}", audit_timeout_s=300)
            store.mark_running(r.job_id)
            store.mark_completed(r.job_id, {})
        # All 5 in-flight must still be present.
        for jid in in_flight:
            assert store.get(jid) is not None
        # And at most 2 terminal.
        terminal_count = sum(
            1 for r in store if r.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        )
        assert terminal_count == 2

    def test_thread_safety(self):
        """Concurrent create() calls from multiple threads must all succeed."""
        store = JobStore()
        ids = []
        errors = []
        def worker():
            try:
                r = store.create(contract_code="c", contract_address="0x", audit_timeout_s=300)
                ids.append(r.job_id)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(ids) == 50
        assert len(set(ids)) == 50  # all unique

    def test_to_dict_drops_contract_code(self):
        store = JobStore()
        r = store.create(contract_code="huge-source", contract_address="0xa",
                         audit_timeout_s=300)
        d = r.to_dict()
        assert "contract_code" not in d
        assert d["contract_address"] == "0xa"
        assert d["status"] == "queued"

    def test_to_full_dict_keeps_contract_code(self):
        store = JobStore()
        r = store.create(contract_code="huge-source", contract_address="0xa",
                         audit_timeout_s=300)
        d = r.to_full_dict()
        assert d["contract_code"] == "huge-source"

    def test_list_recent_newest_first(self):
        store = JobStore()
        ids = []
        for i in range(5):
            r = store.create(contract_code="c", contract_address=f"0x{i}", audit_timeout_s=300)
            ids.append(r.job_id)
        recent = store.list_recent(n=3)
        assert [r.job_id for r in recent] == list(reversed(ids[-3:]))


# ═══════════════════════════════════════════════════════════════════════════
# Address derivation
# ═══════════════════════════════════════════════════════════════════════════

class TestAddressDerivation:
    def test_deterministic(self):
        a1 = _derive_address(VALID_SOLIDITY)
        a2 = _derive_address(VALID_SOLIDITY)
        assert a1 == a2

    def test_different_inputs_different_addresses(self):
        a1 = _derive_address("contract A {}")
        a2 = _derive_address("contract B {}")
        assert a1 != a2

    def test_format_is_0x_prefixed_40hex(self):
        addr = _derive_address(VALID_SOLIDITY)
        assert addr.startswith("0x")
        assert len(addr) == 42
        int(addr[2:], 16)  # raises if not valid hex


# ═══════════════════════════════════════════════════════════════════════════
# Gateway HTTP tests (use TestClient as context manager for lifespan)
# ═══════════════════════════════════════════════════════════════════════════

class TestGatewayE2E:
    def test_submit_returns_202_and_job_id(self):
        with _make_client()[0] as client:
            resp = client.post("/audit", json={
                "contract_code": VALID_SOLIDITY,
                "contract_address": "0xtest",
            })
            assert resp.status_code == 202
            data = resp.json()
            assert "job_id" in data
            assert data["status"] == "queued"
            assert data["contract_address"] == "0xtest"
            assert "submitted_at" in data

    def test_full_lifecycle_success(self):
        with _make_client()[0] as client:
            # Submit
            resp = client.post("/audit", json={"contract_code": VALID_SOLIDITY})
            job_id = resp.json()["job_id"]
            # Poll until done
            for _ in range(50):
                resp = client.get(f"/audit/{job_id}")
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert data["status"] == "completed", f"job didn't complete: {data}"
            assert data["report"] is not None
            assert data["report"]["final_report"]["overall_verdict"] == "VULNERABLE"
            assert data["report"]["verdicts"]["Reentrancy"] == "CONFIRMED"
            assert data["started_at"] is not None
            assert data["finished_at"] is not None

    def test_lifecycle_failure(self):
        graph = _StubGraph(raise_exc=RuntimeError("graph kaboom"))
        with _make_client(graph=graph)[0] as client:
            resp = client.post("/audit", json={"contract_code": VALID_SOLIDITY})
            job_id = resp.json()["job_id"]
            for _ in range(50):
                resp = client.get(f"/audit/{job_id}")
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert data["status"] == "failed"
            assert "graph kaboom" in data["error"]
            assert data["report"] is None

    def test_lifecycle_timeout(self):
        # Stub sleeps longer than the audit_timeout_s we send.
        graph = _StubGraph(sleep_s=0.5)
        with _make_client(graph=graph)[0] as client:
            resp = client.post("/audit", json={
                "contract_code": VALID_SOLIDITY,
                "audit_timeout_s": 0.1,  # 100ms — less than the 500ms sleep
            })
            job_id = resp.json()["job_id"]
            for _ in range(50):
                resp = client.get(f"/audit/{job_id}")
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert data["status"] == "failed"
            assert "timed out" in data["error"].lower()

    def test_get_unknown_returns_404(self):
        with _make_client()[0] as client:
            resp = client.get("/audit/00000000-0000-0000-0000-000000000000")
            assert resp.status_code == 404
            assert "not found" in resp.json()["detail"].lower()

    def test_validation_empty_contract_code(self):
        with _make_client()[0] as client:
            resp = client.post("/audit", json={"contract_code": ""})
            assert resp.status_code == 422

    def test_validation_too_long_contract_code(self):
        with _make_client()[0] as client:
            resp = client.post("/audit", json={"contract_code": "x" * 200_001})
            assert resp.status_code == 422

    def test_validation_binary_input(self):
        with _make_client()[0] as client:
            resp = client.post("/audit", json={"contract_code": "\x00\x01" * 100})
            assert resp.status_code == 422

    def test_validation_missing_contract_code(self):
        with _make_client()[0] as client:
            resp = client.post("/audit", json={})
            assert resp.status_code == 422

    def test_validation_extra_field_rejected(self):
        with _make_client()[0] as client:
            resp = client.post("/audit", json={
                "contract_code": VALID_SOLIDITY,
                "rogue_field": "evil",
            })
            assert resp.status_code == 422

    def test_address_derived_when_omitted(self):
        graph = _StubGraph()
        with _make_client(graph=graph)[0] as client:
            resp = client.post("/audit", json={"contract_code": VALID_SOLIDITY})
            job_id = resp.json()["job_id"]
            for _ in range(50):
                resp = client.get(f"/audit/{job_id}")
                if resp.json()["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            # The graph saw the address that was derived server-side.
            assert graph.last_state["contract_address"].startswith("0x")
            assert len(graph.last_state["contract_address"]) == 42

    def test_concurrent_submissions_get_distinct_ids(self):
        with _make_client()[0] as client:
            ids = set()
            for _ in range(10):
                resp = client.post("/audit", json={"contract_code": VALID_SOLIDITY})
                ids.add(resp.json()["job_id"])
            assert len(ids) == 10

    def test_list_endpoint(self):
        with _make_client()[0] as client:
            # Submit 3 jobs
            for _ in range(3):
                client.post("/audit", json={"contract_code": VALID_SOLIDITY})
            resp = client.get("/audit")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) >= 3

    def test_health_endpoint(self):
        with _make_client()[0] as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert "gateway" in data
            assert "jobs" in data
            assert data["jobs"] == {"queued": 0, "running": 0, "completed": 0, "failed": 0}

    def test_health_after_jobs(self):
        with _make_client()[0] as client:
            client.post("/audit", json={"contract_code": VALID_SOLIDITY})
            # Don't wait for completion — just check counts after submission.
            time.sleep(0.1)
            resp = client.get("/health")
            data = resp.json()
            # At least one job should be queued or completed.
            total = sum(data["jobs"].values())
            assert total >= 1

    def test_root_endpoint(self):
        with _make_client()[0] as client:
            resp = client.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert data["service"] == "sentinel-audit-gateway"
            assert "0.1.0" in data["version"]

    def test_no_llm_query_param(self):
        """The `?no_llm=true` query param is accepted and recorded in metadata."""
        graph = _StubGraph()
        with _make_client(graph=graph)[0] as client:
            resp = client.post(
                "/audit?no_llm=true",
                json={"contract_code": VALID_SOLIDITY},
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]
            for _ in range(50):
                resp = client.get(f"/audit/{job_id}")
                if resp.json()["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            data = resp.json()
            assert data["metadata"].get("_effective_no_llm") is True

    def test_listening_on_test_mode(self):
        """Sanity: the app serves a valid OpenAPI schema."""
        with _make_client()[0] as client:
            resp = client.get("/openapi.json")
            assert resp.status_code == 200
            schema = resp.json()
            assert "/audit" in schema["paths"]
            assert "/audit/{job_id}" in schema["paths"]
            assert "/health" in schema["paths"]

    def test_graph_exception_does_not_kill_gateway(self):
        """A graph crash on one job must NOT crash the server."""
        graph1 = _StubGraph(raise_exc=RuntimeError("first job failed"))
        client, store, _ = _make_client(graph=graph1)
        with client:
            # First job: will fail.
            resp = client.post("/audit", json={"contract_code": VALID_SOLIDITY})
            job_id1 = resp.json()["job_id"]
            for _ in range(50):
                resp = client.get(f"/audit/{job_id1}")
                if resp.json()["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert resp.json()["status"] == "failed"
            # Second job: should still work (gateway is alive).
            # Replace the graph with a working one for the second job.
            # (The graph factory returns the same StubGraph instance, so
            # we need to reset its exception state.)
            graph1.raise_exc = None
            graph1.result = {
                "final_report": {"overall_label": "safe", "overall_verdict": "SAFE"},
                "verdicts": {}, "ml_result": {}, "static_findings": [],
                "rag_results": [], "audit_history": [], "routing_decisions": [],
                "consensus_verdict": {}, "debate_transcript": {},
                "confirmations": {}, "contradictions": {},
                "narrative": None, "error": None,
            }
            resp = client.post("/audit", json={"contract_code": VALID_SOLIDITY})
            job_id2 = resp.json()["job_id"]
            assert job_id1 != job_id2
            for _ in range(50):
                resp = client.get(f"/audit/{job_id2}")
                if resp.json()["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert resp.json()["status"] == "completed"


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: __init__ exports
# ═══════════════════════════════════════════════════════════════════════════

class TestPackageExports:
    def test_lazy_imports(self):
        # The __init__.py uses PEP 562 __getattr__ to defer submodule imports.
        from src.api import (
            AuditRequest,
            ErrorResponse,
            HealthResponse,
            JobResponse,
            ServiceHealth,
            JobRecord,
            JobStore,
            JobStatus,
            create_app,
        )
        assert AuditRequest is not None
        assert JobStore is not None
        assert JobStatus.QUEUED.value == "queued"
        assert callable(create_app)
