"""V3 context-attested submission truth for AGENTS boundaries.

This module is intentionally separate from ``submission.py``. The latter is the
R0/V2 compatibility view; overloading it would make proof scope, policy
attestation, signing, transaction state, and finality look like one boolean.

V3 truth keeps those dimensions independent. It contains no private key,
transaction construction, RPC broadcast, or model decision policy.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

SUBMISSION_V3_SCHEMA_VERSION = "3"
V3_SUBMISSION_PROTOCOL = "context_attested_v3"
V3_NEURAL_PROOF_SCOPE = "legacy_proxy_only_unbound"

PROOF_STATES = frozenset({"not_requested", "verified", "rejected", "unavailable", "failed"})
CONTEXT_STATES = frozenset({"not_requested", "attested", "rejected", "unavailable", "failed"})
SIGNER_STATES = frozenset({"not_requested", "eligible", "signed", "rejected", "unavailable", "failed"})
TX_STATES = frozenset(
    {
        "not_requested",
        "prepared",
        "signed",
        "broadcast",
        "pending",
        "confirmed",
        "reverted",
        "dropped",
        "replaced",
        "failed",
        "unavailable",
    }
)
FINALITY_STATES = frozenset(
    {"not_requested", "ineligible", "pending", "confirmed", "reverted", "failed", "unavailable"}
)

_HASH_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{64}$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def _state(source: Mapping[str, Any], key: str, allowed: frozenset[str]) -> str:
    value = str(source.get(key) or "not_requested")
    if value not in allowed:
        raise ValueError(f"invalid {key}: {value!r}; allowed={sorted(allowed)}")
    return value


def _hash_or_none(source: Mapping[str, Any], key: str) -> str | None:
    value = source.get(key)
    if value in (None, ""):
        return None
    value = str(value)
    if not _HASH_RE.fullmatch(value):
        raise ValueError(f"{key} must be a 32-byte hex identity")
    raw = value[2:] if value.startswith("0x") else value
    return "0x" + raw.lower()


def _address_or_none(source: Mapping[str, Any], key: str) -> str | None:
    value = source.get(key)
    if value in (None, ""):
        return None
    value = str(value)
    if not _ADDRESS_RE.fullmatch(value):
        raise ValueError(f"{key} must be a 20-byte 0x-prefixed address")
    return value


def normalize_v3_submission(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return lossless, fail-closed V3 submission/finality truth.

    Positive finality is never inferred from a request digest, a proof, or a
    policy decision alone. ``verified_audit_finality`` is true only when the
    caller explicitly supplies it *and* the independent proof/context/signer/
    transaction/finality states are mutually consistent.
    """

    source = dict(value or {})
    protocol = str(source.get("submission_protocol") or V3_SUBMISSION_PROTOCOL)
    proof_scope = str(source.get("proof_scope") or V3_NEURAL_PROOF_SCOPE)
    if protocol != V3_SUBMISSION_PROTOCOL:
        raise ValueError(f"unsupported V3 submission_protocol: {protocol!r}")
    if proof_scope != V3_NEURAL_PROOF_SCOPE:
        raise ValueError(
            "V3 context attestation does not upgrade the neural proof scope; "
            f"expected {V3_NEURAL_PROOF_SCOPE!r}, got {proof_scope!r}"
        )

    proof_state = _state(source, "proof_state", PROOF_STATES)
    context_state = _state(source, "context_state", CONTEXT_STATES)
    signer_state = _state(source, "signer_state", SIGNER_STATES)
    transaction_state = _state(source, "transaction_state", TX_STATES)
    finality_state = _state(source, "finality_state", FINALITY_STATES)

    requested_finality = source.get("verified_audit_finality") is True
    positive_states = {
        "proof_state": proof_state == "verified",
        "context_state": context_state == "attested",
        "signer_state": signer_state == "signed",
        "transaction_state": transaction_state == "confirmed",
        "finality_state": finality_state == "confirmed",
    }
    if requested_finality and not all(positive_states.values()):
        missing = sorted(key for key, ok in positive_states.items() if not ok)
        raise ValueError(
            "verified_audit_finality=true is inconsistent with independent states: "
            + ", ".join(missing)
        )

    request_digest = _hash_or_none(source, "request_digest")
    if requested_finality and request_digest is None:
        raise ValueError("confirmed V3 finality requires request_digest")

    chain_id_raw = source.get("chain_id")
    chain_id = None if chain_id_raw in (None, "") else int(chain_id_raw)
    if chain_id is not None and chain_id <= 0:
        raise ValueError("chain_id must be positive")

    round_id_raw = source.get("round_id")
    round_id = None if round_id_raw in (None, "") else int(round_id_raw)
    if round_id is not None and round_id < 0:
        raise ValueError("round_id must be non-negative")

    result = {
        "schema_version": SUBMISSION_V3_SCHEMA_VERSION,
        "submission_protocol": protocol,
        "proof_scope": proof_scope,
        "proof_state": proof_state,
        "context_state": context_state,
        "signer_state": signer_state,
        "transaction_state": transaction_state,
        "finality_state": finality_state,
        "verified_audit_finality": requested_finality,
        "request_digest": request_digest,
        "chain_id": chain_id,
        "registry_address": _address_or_none(source, "registry_address"),
        "contract_address": _address_or_none(source, "contract_address"),
        "agent": _address_or_none(source, "agent"),
        "policy_signer": _address_or_none(source, "policy_signer"),
        "verifier": _address_or_none(source, "verifier"),
        "round_id": round_id,
        "teacher_model_hash": _hash_or_none(source, "teacher_model_hash"),
        "proxy_bundle_hash": _hash_or_none(source, "proxy_bundle_hash"),
        "data_version_hash": _hash_or_none(source, "data_version_hash"),
        "class_schema_hash": _hash_or_none(source, "class_schema_hash"),
        "proof_hash": _hash_or_none(source, "proof_hash"),
        "public_signals_hash": _hash_or_none(source, "public_signals_hash"),
        "policy_reason": source.get("policy_reason"),
        "failure_reason": source.get("failure_reason"),
    }

    # A positive finality claim must also identify the chain/registry/target.
    if requested_finality:
        required = ("chain_id", "registry_address", "contract_address", "agent", "proof_hash")
        missing = [key for key in required if result[key] is None]
        if missing:
            raise ValueError(
                "confirmed V3 finality missing required identity fields: " + ", ".join(missing)
            )

    return result


__all__ = [
    "CONTEXT_STATES",
    "FINALITY_STATES",
    "PROOF_STATES",
    "SIGNER_STATES",
    "SUBMISSION_V3_SCHEMA_VERSION",
    "TX_STATES",
    "V3_NEURAL_PROOF_SCOPE",
    "V3_SUBMISSION_PROTOCOL",
    "normalize_v3_submission",
]
