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

    def test_empty_jwt_secret_rejects_jwt(self):
        """R0.6: JWT signed with empty HMAC key must be rejected."""
        import json as _json
        import jwt as _pyjwt
        import base64

        # Create JWT with empty secret (HMAC-SHA256 of "")
        header = base64.urlsafe_b64encode(_json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=").decode()
        payload = _json.dumps({
            "sub": "attacker", "scope": "write", "tenant_id": "default",
            "iat": 1000000, "exp": 9999999999,
        })
        payload_enc = base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
        import hmac, hashlib
        sig = base64.urlsafe_b64encode(
            hmac.new(b"", f"{header}.{payload_enc}".encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        token = f"{header}.{payload_enc}.{sig}"

        old_secret = os.environ.pop("SENTINEL_JWT_SECRET", None)
        old_token = os.environ.pop("SENTINEL_GATEWAY_TOKEN", None)
        try:
            app = self._make_app(auth_enabled=True)
            with TestClient(app) as client:
                r = client.post(
                    "/audit",
                    json={"contract_code": "contract C {}"},
                    headers={"Authorization": f"Bearer {token}"},
                )
                # Must be rejected: no JWT secret configured, and no static token
                assert r.status_code == 401
        finally:
            if old_secret is not None:
                os.environ["SENTINEL_JWT_SECRET"] = old_secret
            if old_token is not None:
                os.environ["SENTINEL_GATEWAY_TOKEN"] = old_token

    def test_cross_tenant_access_rejected(self):
        """R0.6: tenant B cannot GET tenant A's job."""
        from src.security.auth import create_token

        old_secret = os.environ.pop("SENTINEL_JWT_SECRET", None)
        os.environ["SENTINEL_JWT_SECRET"] = "r0-6-test-secret"
        try:
            token_a = create_token("tenant-a", scope="write", tenant_id="tenant-a")
            token_b = create_token("tenant-b", scope="read", tenant_id="tenant-b")

            app = self._make_app(auth_enabled=True)
            with TestClient(app) as client:
                r = client.post(
                    "/audit",
                    json={"contract_code": "contract C {}"},
                    headers={"Authorization": f"Bearer {token_a}"},
                )
                assert r.status_code == 202
                job_id = r.json()["job_id"]

                r2 = client.get(
                    f"/audit/{job_id}",
                    headers={"Authorization": f"Bearer {token_b}"},
                )
                assert r2.status_code == 404, f"tenant-b should not see tenant-a's job"

                # Tenant A should still be able to get their own job
                r3 = client.get(
                    f"/audit/{job_id}",
                    headers={"Authorization": f"Bearer {token_a}"},
                )
                assert r3.status_code == 200
        finally:
            os.environ["SENTINEL_JWT_SECRET"] = old_secret or ""

    def test_cross_tenant_list_is_filtered(self):
        """R0.6: tenant B's list does not leak tenant A's jobs."""
        from src.security.auth import create_token

        old_secret = os.environ.pop("SENTINEL_JWT_SECRET", None)
        os.environ["SENTINEL_JWT_SECRET"] = "r0-6-test-list-secret"
        try:
            token_a = create_token("tenant-a", scope="write", tenant_id="tenant-a")
            token_b = create_token("tenant-b", scope="read", tenant_id="tenant-b")

            app = self._make_app(auth_enabled=True)
            with TestClient(app) as client:
                client.post(
                    "/audit",
                    json={"contract_code": "contract C {}"},
                    headers={"Authorization": f"Bearer {token_a}"},
                )
                r = client.get(
                    "/audit",
                    headers={"Authorization": f"Bearer {token_b}"},
                )
                assert r.status_code == 200
                assert len(r.json()) == 0, f"tenant-b list should be empty, got {len(r.json())}"
        finally:
            os.environ["SENTINEL_JWT_SECRET"] = old_secret or ""

    def test_empty_jwt_secret_rejects_jwt(self):
        """R0.6: JWT signed with empty HMAC key must be rejected."""
        import json as _json
        import jwt as _pyjwt
        import base64

        # Create JWT with empty secret (HMAC-SHA256 of "")
        header = base64.urlsafe_b64encode(_json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=").decode()
        payload = _json.dumps({
            "sub": "attacker", "scope": "write", "tenant_id": "default",
            "iat": 1000000, "exp": 9999999999,
        })
        payload_enc = base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
        import hmac, hashlib
        sig = base64.urlsafe_b64encode(
            hmac.new(b"", f"{header}.{payload_enc}".encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        token = f"{header}.{payload_enc}.{sig}"

        old_secret = os.environ.pop("SENTINEL_JWT_SECRET", None)
        old_token = os.environ.pop("SENTINEL_GATEWAY_TOKEN", None)
        try:
            app = self._make_app(auth_enabled=True)
            with TestClient(app) as client:
                r = client.post(
                    "/audit",
                    json={"contract_code": "contract C {}"},
                    headers={"Authorization": f"Bearer {token}"},
                )
                # Must be rejected: no JWT secret configured, and no static token
                assert r.status_code == 401
        finally:
            if old_secret is not None:
                os.environ["SENTINEL_JWT_SECRET"] = old_secret
            if old_token is not None:
                os.environ["SENTINEL_GATEWAY_TOKEN"] = old_token


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
