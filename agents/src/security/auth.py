"""JWT-based authentication with scopes and tenant isolation for the SENTINEL gateway.

R0.3:
  - JWT tokens with scopes (read/write/admin) and tenant_id
  - Static bearer-token fallback for backward compatibility
  - Scope-enforced route access
  - Token generation helper
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Callable, Literal

from fastapi import Depends, HTTPException, Request

Scope = Literal["read", "write", "admin"]
_SCOPE_HIERARCHY: dict[str, int] = {"read": 1, "write": 2, "admin": 3}


def _jwt_secret() -> str | None:
    raw = os.getenv("SENTINEL_JWT_SECRET")
    if raw and raw.strip():
        return raw
    return None


def _jwt_enabled() -> bool:
    return _jwt_secret() is not None


def _static_token() -> str | None:
    return os.getenv("SENTINEL_GATEWAY_TOKEN") or None


def create_token(
    principal: str,
    scope: Scope = "read",
    tenant_id: str = "default",
    *,
    secret: str | None = None,
    expiry_s: int = 86400,
) -> str:
    import jwt as pyjwt

    actual_secret = secret or _jwt_secret()
    if not actual_secret:
        raise ValueError("SENTINEL_JWT_SECRET is not configured; cannot issue tokens")

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": principal,
        "scope": scope,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": now + expiry_s,
    }
    return pyjwt.encode(payload, actual_secret, algorithm="HS256")


def decode_token(token: str, *, secret: str | None = None) -> dict[str, Any] | None:
    import jwt as pyjwt

    actual_secret = secret or _jwt_secret()
    if not actual_secret:
        return None

    try:
        return pyjwt.decode(
            token,
            actual_secret,
            algorithms=["HS256"],
            options={"require": ["sub", "scope", "tenant_id", "iat", "exp"]},
        )
    except Exception:
        return None


def _scope_satisfies(required: str, provided: str | None) -> bool:
    if provided is None:
        return False
    return _SCOPE_HIERARCHY.get(provided, 0) >= _SCOPE_HIERARCHY.get(required, 0)


def _validate_token(token: str, *, secret: str | None = None) -> dict[str, Any]:
    """Validate a bearer token (JWT or static) and return claims.

    Raises HTTPException on invalid/missing tokens.
    """
    claims = decode_token(token, secret=secret)
    if claims is not None:
        return claims

    expected = secret or _static_token()
    if expected is not None and hmac.compare_digest(token, expected):
        return {
            "sub": "legacy-client",
            "scope": "admin",
            "tenant_id": "default",
        }

    raise HTTPException(
        status_code=401,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_bearer(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_header[len("Bearer "):]


def require_scope(scope: Scope, enabled: bool = True) -> Callable[[Request], dict[str, Any]]:
    """FastAPI dependency factory.

    Usage: ``_auth: dict = Depends(require_scope('write'))``
    """
    async def _auth_dependency(request: Request) -> dict[str, Any]:
        if not enabled:
            return {
                "authenticated": True,
                "principal": "anonymous",
                "scope": "admin",
                "tenant_id": "default",
            }

        token = _extract_bearer(request)
        claims = _validate_token(token)

        if not _scope_satisfies(scope, claims.get("scope")):
            raise HTTPException(
                status_code=403,
                detail=f"scope '{claims.get('scope')}' "
                       f"does not satisfy required scope '{scope}'",
            )

        return {
            "authenticated": True,
            "principal": claims.get("sub", "unknown"),
            "scope": claims.get("scope", "read"),
            "tenant_id": claims.get("tenant_id", "default"),
        }

    return _auth_dependency


__all__ = [
    "Scope",
    "create_token",
    "decode_token",
    "_jwt_enabled",
    "require_scope",
]
