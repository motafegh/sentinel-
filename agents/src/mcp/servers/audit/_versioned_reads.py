# agents/src/mcp/servers/audit/_versioned_reads.py
"""Version-aware read helpers for AuditRegistry V1/V2/V3.

The live MCP query names are intentionally protocol-neutral. This module makes
that true by reading all historical registry versions and returning explicit
`protocol_version` metadata instead of silently treating the V1 scalar storage
as the whole registry.

V2/V3 class-score values are returned as raw field elements. We do not convert
them into probabilities here: the tracked proxy is a fixed-point computation
and signed field decoding must follow the exact artifact/settings semantics.
Inventing a scalar verdict at this persistence boundary would be misleading.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ._decode import _decode_audit_result


def _timestamp_iso(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        if timestamp > 0
        else "never"
    )


def _bytes32_hex(value: Any) -> str:
    if isinstance(value, str):
        if value.startswith("0x") and len(value) == 66:
            return value.lower()
        raw = bytes.fromhex(value[2:] if value.startswith("0x") else value)
    else:
        raw = bytes(value)
    if len(raw) != 32:
        raise ValueError(f"expected bytes32 value, got {len(raw)} bytes")
    return "0x" + raw.hex()


def decode_v1(result: tuple, contract_address: str) -> dict[str, Any]:
    decoded = _decode_audit_result(result, contract_address)
    decoded["protocol_version"] = "v1"
    return decoded


def decode_v2(result: tuple, contract_address: str) -> dict[str, Any]:
    if len(result) != 6:
        raise ValueError(f"AuditResultV2 tuple must contain 6 fields, got {len(result)}")
    scores, proof_hash, model_hash, timestamp, agent, verified = result
    score_felts = [int(value) for value in scores]
    if len(score_felts) != 10:
        raise ValueError(f"AuditResultV2 must contain 10 class scores, got {len(score_felts)}")
    timestamp = int(timestamp)
    return {
        "protocol_version": "v2",
        "contract_address": contract_address,
        "class_score_felts": score_felts,
        "proof_hash": _bytes32_hex(proof_hash),
        "model_hash": _bytes32_hex(model_hash),
        "timestamp": timestamp,
        "timestamp_iso": _timestamp_iso(timestamp),
        "agent": agent,
        "verified": bool(verified),
        "proof_scope": "legacy_proxy_only_unbound",
    }


def decode_v3(result: tuple, contract_address: str) -> dict[str, Any]:
    if len(result) != 15:
        raise ValueError(f"AuditResultV3 tuple must contain 15 fields, got {len(result)}")
    (
        scores,
        proof_hash,
        request_digest,
        public_signals_hash,
        contract_code_hash,
        teacher_model_hash,
        proxy_bundle_hash,
        data_version_hash,
        class_schema_hash,
        round_id,
        timestamp,
        agent,
        policy_signer,
        verifier,
        verified,
    ) = result
    score_felts = [int(value) for value in scores]
    if len(score_felts) != 10:
        raise ValueError(f"AuditResultV3 must contain 10 class scores, got {len(score_felts)}")
    timestamp = int(timestamp)
    return {
        "protocol_version": "v3",
        "submission_protocol": "context_attested_v3",
        "proof_scope": "legacy_proxy_only_unbound",
        "contract_address": contract_address,
        "class_score_felts": score_felts,
        "proof_hash": _bytes32_hex(proof_hash),
        "request_digest": _bytes32_hex(request_digest),
        "public_signals_hash": _bytes32_hex(public_signals_hash),
        "contract_code_hash": _bytes32_hex(contract_code_hash),
        "teacher_model_hash": _bytes32_hex(teacher_model_hash),
        "proxy_bundle_hash": _bytes32_hex(proxy_bundle_hash),
        "data_version_hash": _bytes32_hex(data_version_hash),
        "class_schema_hash": _bytes32_hex(class_schema_hash),
        "round_id": int(round_id),
        "timestamp": timestamp,
        "timestamp_iso": _timestamp_iso(timestamp),
        "agent": agent,
        "policy_signer": policy_signer,
        "verifier": verifier,
        "verified": bool(verified),
    }


async def count_by_protocol(registry: Any, address: str) -> dict[str, int]:
    """Return exact counts for each persisted registry protocol."""
    v3 = int(await registry.functions.getAuditCountV3(address).call())
    v2 = int(await registry.functions.getAuditCountV2(address).call())
    v1 = int(await registry.functions.getAuditCount(address).call())
    if min(v1, v2, v3) < 0:
        raise ValueError("audit counts must be non-negative")
    return {"v3": v3, "v2": v2, "v1": v1}


async def latest_across_protocols(registry: Any, address: str) -> dict[str, Any]:
    """Return the newest persisted audit across V1/V2/V3 by block timestamp."""
    counts = await count_by_protocol(registry, address)
    candidates: list[dict[str, Any]] = []

    if counts["v3"]:
        candidates.append(
            decode_v3(await registry.functions.getLatestAuditV3(address).call(), address)
        )
    if counts["v2"]:
        candidates.append(
            decode_v2(await registry.functions.getLatestAuditV2(address).call(), address)
        )
    if counts["v1"]:
        candidates.append(
            decode_v1(await registry.functions.getLatestAudit(address).call(), address)
        )

    latest = max(candidates, key=lambda row: int(row["timestamp"])) if candidates else None
    return {
        "contract_address": address,
        "exists": latest is not None,
        "total_count": sum(counts.values()),
        "counts_by_protocol": counts,
        "latest": latest,
    }


async def history_across_protocols(
    registry: Any,
    address: str,
    *,
    limit: int,
) -> dict[str, Any]:
    """Return merged V1/V2/V3 history sorted newest first."""
    if limit < 1:
        raise ValueError("history limit must be at least 1")

    v3_raw = await registry.functions.getAuditHistoryV3(address).call()
    v2_raw = await registry.functions.getAuditHistoryV2(address).call()
    v1_raw = await registry.functions.getAuditHistory(address).call()

    records = [decode_v3(row, address) for row in v3_raw]
    records.extend(decode_v2(row, address) for row in v2_raw)
    records.extend(decode_v1(row, address) for row in v1_raw)
    records.sort(key=lambda row: int(row["timestamp"]), reverse=True)

    counts = {"v3": len(v3_raw), "v2": len(v2_raw), "v1": len(v1_raw)}
    return {
        "contract_address": address,
        "total_count": len(records),
        "returned": min(len(records), limit),
        "counts_by_protocol": counts,
        "records": records[:limit],
    }


__all__ = [
    "count_by_protocol",
    "decode_v1",
    "decode_v2",
    "decode_v3",
    "history_across_protocols",
    "latest_across_protocols",
]
