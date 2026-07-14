"""Canonical execution-status wire producer for the ML service boundary."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def digest_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_live_result(
    payload: Mapping[str, Any],
    *,
    dependency: str,
    input_payload: Any,
    duration_ms: float,
) -> dict[str, Any]:
    """Attach the version-1 canonical live status to a successful response."""

    result = dict(payload)
    result["execution_status"] = {
        "schema_version": "1",
        "status": "SUCCEEDED",
        "attempted": True,
        "ran": True,
        "reason_code": "completed",
        "detail": "",
        "dependency": dependency,
        "provenance": "live",
        "input_digest": digest_payload(input_payload),
        "output_digest": digest_payload(result),
        "duration_ms": max(float(duration_ms), 0.0),
        "attempt": 1,
    }
    return result


__all__ = ["bind_live_result", "digest_payload"]
