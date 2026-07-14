"""Bearer token authentication for the SENTINEL gateway.

R0.3: public mutation routes (POST /audit) require a valid bearer token.
Missing or invalid tokens receive HTTP 401 with a WWW-Authenticate challenge.
"""

import hashlib
import hmac
import os
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


def _get_expected_token() -> str | None:
    """Return the expected bearer token from env, or None if not configured."""
    return os.getenv("SENTINEL_GATEWAY_TOKEN") or None


class BearerAuth:
    """FastAPI dependency that validates bearer tokens.

    When *enabled* is True (default for production), requests without a valid
    ``Authorization: Bearer <token>`` header receive HTTP 401. When *enabled*
    is False (test/development), all requests pass.
    """

    def __init__(self, enabled: bool = True, token: str | None = None) -> None:
        self.enabled = enabled
        self._token = token

    async def __call__(self, request: Request) -> dict[str, Any]:
        if not self.enabled:
            return {"authenticated": False, "reason": "auth_disabled"}

        expected = self._token or _get_expected_token()

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or malformed Authorization header. Expected: Bearer <token>",
                headers={"WWW-Authenticate": "Bearer"},
            )

        provided = auth_header[len("Bearer "):]
        if expected is None or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=401,
                detail="Invalid or unconfigured bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {"authenticated": True, "principal": "gateway-client"}


__all__ = ["BearerAuth"]
