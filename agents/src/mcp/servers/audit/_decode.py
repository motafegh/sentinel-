# agents/src/mcp/servers/audit/_decode.py
"""
AuditResult tuple decoding + mock data generators.

Pure functions over the on-chain AuditResult tuple and the EZKL scale
factor. No state, no I/O. Separated from _handlers so the math + the
mock fixtures can change independently of the MCP protocol surface.

AuditResult struct (Solidity):
    scoreFieldElement  uint256    BN254 field element encoding the score
    proofHash          bytes32    keccak256 of the ZK proof bytes
    timestamp          uint256    Unix timestamp of submission
    agent              address    Submitter's wallet address
    verified           bool       True if ZK proof passed on-chain verify
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ._config import EZKL_SCALE_FACTOR


def _decode_audit_result(
    result: tuple,
    contract_address: str,
) -> dict[str, Any]:
    """
    Convert a raw AuditResult tuple from the contract to a clean dict.

    AuditResult tuple layout (indices):
        0  scoreFieldElement  uint256  BN254 field element
        1  proofHash          bytes32  keccak256(proof bytes)
        2  timestamp          uint256  Unix epoch seconds
        3  agent              address  Submitter's wallet
        4  verified           bool     On-chain ZK proof passed

    Score decoding:
        score = scoreFieldElement / EZKL_SCALE_FACTOR (= 2^13 = 8192)
        This is the same factor used in run_proof.py and extract_calldata.py.
        Example: 4497 / 8192 = 0.5490 → "vulnerable"
    """
    score_field_element: int  = int(result[0])
    proof_hash_bytes:    bytes = result[1]
    timestamp:           int  = int(result[2])
    agent:               str  = result[3]
    verified:            bool = bool(result[4])

    # Decode score from field element
    score: float = score_field_element / EZKL_SCALE_FACTOR
    label: str   = "vulnerable" if score >= 0.50 else "safe"

    # Convert timestamp to ISO string for readability
    timestamp_iso: str = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        if timestamp > 0 else "never"
    )

    # Convert bytes32 proof hash to hex string
    proof_hash_hex: str = "0x" + proof_hash_bytes.hex() if proof_hash_bytes else "0x" + "0" * 64

    return {
        "contract_address":    contract_address,
        "score":               round(score, 4),
        "score_field_element": score_field_element,
        "label":               label,
        "threshold":           0.50,       # binary phase threshold
        "proof_hash":          proof_hash_hex,
        "timestamp":           timestamp,
        "timestamp_iso":       timestamp_iso,
        "agent":               agent,
        "verified":            verified,
    }


def _mock_audit_result(contract_address: str) -> dict[str, Any]:
    """
    Realistic fake audit result for development and CI.

    Mirrors _decode_audit_result() output shape exactly — swapping
    mock → real requires zero changes to callers.
    """
    return {
        "contract_address":    contract_address,
        "score":               0.7314,
        "score_field_element": 5993,       # 5993 / 8192 ≈ 0.7314
        "label":               "vulnerable",
        "threshold":           0.50,
        "proof_hash":          "0x" + "ab" * 32,
        "timestamp":           1713200000,
        "timestamp_iso":       "2026-04-15T12:00:00+00:00",
        "agent":               "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
        "verified":            True,
    }


def _mock_history(contract_address: str, limit: int) -> list[dict[str, Any]]:
    """Realistic fake audit history — two entries to exercise pagination."""
    if limit == 0:
        return []
    records = [
        {
            **_mock_audit_result(contract_address),
            "timestamp":     1713200000,
            "timestamp_iso": "2026-04-15T12:00:00+00:00",
            "score":         0.7314,
            "label":         "vulnerable",
        },
    ]
    if limit >= 2:
        records.append({
            **_mock_audit_result(contract_address),
            "timestamp":     1712900000,
            "timestamp_iso": "2026-04-12T03:20:00+00:00",
            "score":         0.4102,
            "score_field_element": 3362,
            "label":         "safe",
        })
    return records[:limit]