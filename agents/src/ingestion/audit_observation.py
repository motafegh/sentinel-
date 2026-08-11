"""Versioned, policy-free on-chain audit observations.

Event decoding is deliberately separated from feedback eligibility and RAG
mutation. Seeing an `AuditSubmittedV3` event proves that the registry emitted a
V3 observation; it does not by itself define whether that observation should be
used as training/RAG feedback.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

PROTOCOL_V1 = "v1"
PROTOCOL_V3 = "v3"

AUDIT_EVENT_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "contractAddress", "type": "address"},
            {"indexed": False, "name": "proofHash", "type": "bytes32"},
            {"indexed": True, "name": "agent", "type": "address"},
            {"indexed": False, "name": "scoreFieldElement", "type": "uint256"},
        ],
        "name": "AuditSubmitted",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "contractAddress", "type": "address"},
            {"indexed": True, "name": "requestDigest", "type": "bytes32"},
            {"indexed": True, "name": "proofHash", "type": "bytes32"},
            {"indexed": False, "name": "agent", "type": "address"},
            {"indexed": False, "name": "roundId", "type": "uint256"},
        ],
        "name": "AuditSubmittedV3",
        "type": "event",
    },
]

_HEX32_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"event missing required field: {key}")
    return mapping[key]


def _hex(value: Any, *, bytes_len: int | None = None) -> str:
    if isinstance(value, str):
        out = value if value.startswith("0x") else "0x" + value
    elif hasattr(value, "hex"):
        raw = value.hex()
        out = raw if str(raw).startswith("0x") else "0x" + str(raw)
    else:
        out = "0x" + bytes(value).hex()
    out = out.lower()
    if bytes_len is not None and len(out) != 2 + bytes_len * 2:
        raise ValueError(f"expected {bytes_len}-byte hex value, got {out!r}")
    return out


def _address(value: Any, key: str) -> str:
    out = str(value)
    if not _ADDRESS_RE.fullmatch(out):
        raise ValueError(f"{key} is not a 20-byte Ethereum address: {out!r}")
    return out


def _base_event(event: Mapping[str, Any]) -> dict[str, Any]:
    block_number = int(_required(event, "blockNumber"))
    log_index = int(event.get("logIndex", 0))
    transaction_index = int(event.get("transactionIndex", 0))
    if block_number < 0 or log_index < 0 or transaction_index < 0:
        raise ValueError("block/log/transaction indices must be non-negative")
    tx_hash = _hex(_required(event, "transactionHash"), bytes_len=32)
    return {
        "block_number": block_number,
        "transaction_index": transaction_index,
        "log_index": log_index,
        "tx_hash": tx_hash,
        "event_identity": f"{tx_hash}:{log_index}",
    }


def format_v1_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Decode historical scalar `AuditSubmitted` without changing its meaning."""
    args = _required(event, "args")
    if not isinstance(args, Mapping):
        raise ValueError("event args must be a mapping")
    score = int(_required(args, "scoreFieldElement"))
    if score < 0:
        raise ValueError("scoreFieldElement must be non-negative")
    return {
        **_base_event(event),
        "protocol_version": PROTOCOL_V1,
        "event_name": "AuditSubmitted",
        "contract_address": _address(_required(args, "contractAddress"), "contractAddress"),
        "proof_hash": _hex(_required(args, "proofHash"), bytes_len=32),
        "agent": _address(_required(args, "agent"), "agent"),
        "score_field_element": score,
    }


def format_v3_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Decode compact `AuditSubmittedV3` identity without feedback policy."""
    args = _required(event, "args")
    if not isinstance(args, Mapping):
        raise ValueError("event args must be a mapping")
    request_digest = _hex(_required(args, "requestDigest"), bytes_len=32)
    if not _HEX32_RE.fullmatch(request_digest):
        raise ValueError("requestDigest must be bytes32")
    round_id = int(_required(args, "roundId"))
    if round_id < 0:
        raise ValueError("roundId must be non-negative")
    return {
        **_base_event(event),
        "protocol_version": PROTOCOL_V3,
        "event_name": "AuditSubmittedV3",
        "submission_protocol": "context_attested_v3",
        "proof_scope": "legacy_proxy_only_unbound",
        "contract_address": _address(_required(args, "contractAddress"), "contractAddress"),
        "request_digest": request_digest,
        "proof_hash": _hex(_required(args, "proofHash"), bytes_len=32),
        "agent": _address(_required(args, "agent"), "agent"),
        "round_id": round_id,
    }


def observation_sort_key(observation: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(observation["block_number"]),
        int(observation.get("transaction_index", 0)),
        int(observation.get("log_index", 0)),
    )


__all__ = [
    "AUDIT_EVENT_ABI",
    "PROTOCOL_V1",
    "PROTOCOL_V3",
    "format_v1_event",
    "format_v3_event",
    "observation_sort_key",
]
