"""Tests for R0.3 authenticated services and signer isolation."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.gateway import create_app, GATEWAY_DEFAULT_HOST
from src.api.job_store import JobStore


class _StubGraph:
    async def ainvoke(self, state):
        return {"final_report": {"overall_label": "safe"}}


class TestGatewayLoopback:
    def test_default_host_is_loopback(self):
        assert GATEWAY_DEFAULT_HOST == "127.0.0.1" or os.getenv("GATEWAY_HOST")


class TestGatewayAuth:
    def _make_app(self, auth_enabled=True, token=None):
        g = _StubGraph()
        s = JobStore()
        return create_app(store=s, graph_factory=lambda: g, skip_service_probes=True, auth_enabled=auth_enabled)

    def test_unauthenticated_post_rejected_401(self):
        app = self._make_app(auth_enabled=True)
        with TestClient(app) as client:
            r = client.post("/audit", json={"contract_code": "contract C {}"})
            assert r.status_code == 401
            assert r.headers.get("www-authenticate") == "Bearer"

    def test_missing_auth_header_rejected(self):
        app = self._make_app(auth_enabled=True)
        with TestClient(app) as client:
            r = client.post("/audit", json={"contract_code": "contract C {}"})
            assert r.status_code == 401

    def test_malformed_auth_header_rejected(self):
        app = self._make_app(auth_enabled=True)
        with TestClient(app) as client:
            r = client.post(
                "/audit",
                json={"contract_code": "contract C {}"},
                headers={"Authorization": "Basic xyz"},
            )
            assert r.status_code == 401

    def test_invalid_token_rejected(self):
        os.environ["SENTINEL_GATEWAY_TOKEN"] = "secret"
        try:
            app = self._make_app(auth_enabled=True)
            with TestClient(app) as client:
                r = client.post(
                    "/audit",
                    json={"contract_code": "contract C {}"},
                    headers={"Authorization": "Bearer wrong-token"},
                )
                assert r.status_code == 401
        finally:
            del os.environ["SENTINEL_GATEWAY_TOKEN"]

    def test_valid_token_accepted(self):
        os.environ["SENTINEL_GATEWAY_TOKEN"] = "secret"
        try:
            app = self._make_app(auth_enabled=True)
            with TestClient(app) as client:
                r = client.post(
                    "/audit",
                    json={"contract_code": "contract C {}"},
                    headers={"Authorization": "Bearer secret"},
                )
                assert r.status_code == 202
        finally:
            del os.environ["SENTINEL_GATEWAY_TOKEN"]

    def test_auth_disabled_passes(self):
        app = self._make_app(auth_enabled=False)
        with TestClient(app) as client:
            r = client.post("/audit", json={"contract_code": "contract C {}"})
            assert r.status_code == 202

    def test_health_does_not_require_auth(self):
        app = self._make_app(auth_enabled=True)
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200


class TestSignerIsolation:
    def test_config_has_no_operator_key_env(self):
        config = Path(__file__).resolve().parents[1] / "src/mcp/servers/audit/_config.py"
        source = config.read_text()
        assert 'os.getenv("SENTINEL_OPERATOR_KEY"' not in source

    def test_submit_has_no_from_key(self):
        submit = Path(__file__).resolve().parents[1] / "src/mcp/servers/audit/_submit.py"
        source = submit.read_text()
        assert "from_key(_OPERATOR_KEY)" not in source

    def test_handlers_does_not_advertise_submit_audit(self):
        handlers = Path(__file__).resolve().parents[1] / "src/mcp/servers/audit/_handlers.py"
        source = handlers.read_text()
        assert 'name="submit_audit"' not in source

    def test_mcp_servers_default_to_loopback(self):
        servers_dir = Path(__file__).resolve().parents[1] / "src/mcp/servers"
        for py_file in servers_dir.rglob("*.py"):
            source = py_file.read_text()
            if 'uvicorn.run' in source and 'host=' in source:
                assert '"0.0.0.0"' not in source, f"{py_file.name} still binds to 0.0.0.0"
