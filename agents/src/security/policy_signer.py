"""SENTINEL policy-signer boundary.

This module deliberately contains **no private key, transaction construction,
ABI contract call, RPC broadcast, or receipt handling**. It defines the policy
contract between the analysis/MCP process and a separately isolated signing
service.

Two facts must remain distinct:

1. The current EZKL V2 proof scope is ``legacy_proxy_only_unbound`` and is
   rejected for direct finality.
2. The V3 *submission protocol* can bind that exact proxy proof to audit context
   with an EIP-712 policy attestation. The signature does not upgrade what the
   ZK circuit proves; it supplies a separately authenticated provenance/context
   statement checked by ``AuditRegistry.submitAuditV3``.

The functions below may construct/validate the EIP-712 digest that a dedicated
signing service is allowed to sign. They never sign it here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from eth_abi import encode
from web3 import Web3


class PolicyDecision(Enum):
    REJECTED = "policy_rejected"
    ACCEPTED = "policy_accepted"


REJECT_REASON_UNBOUND = "proof_scope_not_identity_bound"
REJECT_REASON_NO_SCOPE = "no_proof_scope"
REJECT_REASON_INVALID_V3 = "invalid_context_attested_v3_request"

LEGACY_PROOF_SCOPE = "legacy_proxy_only_unbound"
V3_SUBMISSION_PROTOCOL = "context_attested_v3"
V3_TOTAL_PUBLIC_SIGNALS = 138
V3_INPUT_OFFSET = 128
V3_NUM_CLASSES = 10

_EIP712_DOMAIN_TYPE = (
    "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
)
_EIP712_NAME = "SENTINEL Audit Registry"
_EIP712_VERSION = "3"
_AUDIT_REQUEST_V3_TYPE = (
    "SentinelAuditV3(address agent,address contractAddress,bytes32 contractCodeHash,"
    "uint256 roundId,bytes32 teacherModelHash,bytes32 proxyBundleHash,"
    "bytes32 dataVersionHash,bytes32 classSchemaHash,bytes32 proofHash,"
    "bytes32 publicSignalsHash,bytes32 classScoreFeltsHash,uint256 deadline)"
)

_HASH_RE = re.compile(r"^(?:0x)?([0-9a-fA-F]{64})$")


@dataclass
class PolicyResult:
    decision: PolicyDecision
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "details": self.details,
        }


def _bytes32(value: str, *, field_name: str, allow_zero: bool = False) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a 32-byte hex string")
    match = _HASH_RE.fullmatch(value)
    if not match:
        raise ValueError(f"{field_name} must be exactly 32 bytes of hex")
    raw = bytes.fromhex(match.group(1))
    if not allow_zero and raw == b"\x00" * 32:
        raise ValueError(f"{field_name} must not be zero")
    return raw


def _address(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not Web3.is_address(value):
        raise ValueError(f"{field_name} must be a valid Ethereum address")
    address = Web3.to_checksum_address(value)
    if int(address, 16) == 0:
        raise ValueError(f"{field_name} must not be the zero address")
    return address


def _hex32(value: bytes) -> str:
    if len(value) != 32:
        raise ValueError(f"expected 32 bytes, got {len(value)}")
    return "0x" + value.hex()


def _keccak_abi(types: list[str], values: list[Any]) -> bytes:
    return bytes(Web3.keccak(encode(types, values)))


@dataclass(frozen=True)
class AuditRequestV3:
    """Unsigned, fully bound request eligible for the isolated V3 signer."""

    agent: str
    contract_address: str
    contract_code_hash: str
    chain_id: int
    registry_address: str
    round_id: int
    teacher_model_hash: str
    proxy_bundle_hash: str
    data_version_hash: str
    class_schema_hash: str
    proof_hash: str
    public_signals_hash: str
    class_score_felts_hash: str
    deadline: int
    digest: str
    proof_scope: str = LEGACY_PROOF_SCOPE
    submission_protocol: str = V3_SUBMISSION_PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "contract_address": self.contract_address,
            "contract_code_hash": self.contract_code_hash,
            "chain_id": self.chain_id,
            "registry_address": self.registry_address,
            "round_id": self.round_id,
            "teacher_model_hash": self.teacher_model_hash,
            "proxy_bundle_hash": self.proxy_bundle_hash,
            "data_version_hash": self.data_version_hash,
            "class_schema_hash": self.class_schema_hash,
            "proof_hash": self.proof_hash,
            "public_signals_hash": self.public_signals_hash,
            "class_score_felts_hash": self.class_score_felts_hash,
            "deadline": self.deadline,
            "digest": self.digest,
            "proof_scope": self.proof_scope,
            "submission_protocol": self.submission_protocol,
        }


def _domain_separator(*, chain_id: int, registry_address: str) -> bytes:
    if not isinstance(chain_id, int) or chain_id <= 0:
        raise ValueError("chain_id must be a positive integer")
    registry = _address(registry_address, field_name="registry_address")
    return _keccak_abi(
        ["bytes32", "bytes32", "bytes32", "uint256", "address"],
        [
            Web3.keccak(text=_EIP712_DOMAIN_TYPE),
            Web3.keccak(text=_EIP712_NAME),
            Web3.keccak(text=_EIP712_VERSION),
            chain_id,
            registry,
        ],
    )


def compute_v3_digest(
    *,
    agent: str,
    contract_address: str,
    contract_code_hash: str,
    chain_id: int,
    registry_address: str,
    round_id: int,
    teacher_model_hash: str,
    proxy_bundle_hash: str,
    data_version_hash: str,
    class_schema_hash: str,
    proof_hash: str,
    public_signals_hash: str,
    class_score_felts_hash: str,
    deadline: int,
) -> str:
    """Compute the exact EIP-712 digest used by ``AuditRegistry`` V3."""
    if not isinstance(round_id, int) or round_id < 0:
        raise ValueError("round_id must be a non-negative integer")
    if not isinstance(deadline, int) or deadline <= 0:
        raise ValueError("deadline must be a positive unix timestamp")

    agent_address = _address(agent, field_name="agent")
    target_address = _address(contract_address, field_name="contract_address")
    registry = _address(registry_address, field_name="registry_address")

    code_hash = _bytes32(contract_code_hash, field_name="contract_code_hash")
    teacher_hash = _bytes32(teacher_model_hash, field_name="teacher_model_hash")
    bundle_hash = _bytes32(proxy_bundle_hash, field_name="proxy_bundle_hash")
    data_hash = _bytes32(data_version_hash, field_name="data_version_hash")
    schema_hash = _bytes32(class_schema_hash, field_name="class_schema_hash")
    proof_hash_raw = _bytes32(proof_hash, field_name="proof_hash", allow_zero=True)
    signals_hash = _bytes32(public_signals_hash, field_name="public_signals_hash")
    score_hash = _bytes32(class_score_felts_hash, field_name="class_score_felts_hash")

    struct_hash = _keccak_abi(
        [
            "bytes32",
            "address",
            "address",
            "bytes32",
            "uint256",
            "bytes32",
            "bytes32",
            "bytes32",
            "bytes32",
            "bytes32",
            "bytes32",
            "bytes32",
            "uint256",
        ],
        [
            Web3.keccak(text=_AUDIT_REQUEST_V3_TYPE),
            agent_address,
            target_address,
            code_hash,
            round_id,
            teacher_hash,
            bundle_hash,
            data_hash,
            schema_hash,
            proof_hash_raw,
            signals_hash,
            score_hash,
            deadline,
        ],
    )
    domain = _domain_separator(chain_id=chain_id, registry_address=registry)
    return _hex32(bytes(Web3.keccak(b"\x19\x01" + domain + struct_hash)))


def build_v3_request(
    *,
    agent: str,
    contract_address: str,
    contract_code_hash: str,
    chain_id: int,
    registry_address: str,
    round_id: int,
    teacher_model_hash: str,
    proxy_bundle_hash: str,
    data_version_hash: str,
    class_schema_hash: str,
    proof_bytes: bytes,
    public_signals: Sequence[int],
    class_score_felts: Sequence[int],
    deadline: int,
) -> AuditRequestV3:
    """Build a fail-closed unsigned V3 signing request from exact artifacts."""
    if not isinstance(proof_bytes, (bytes, bytearray)) or len(proof_bytes) == 0:
        raise ValueError("proof_bytes must be non-empty")
    if len(public_signals) != V3_TOTAL_PUBLIC_SIGNALS:
        raise ValueError(
            f"public_signals must contain exactly {V3_TOTAL_PUBLIC_SIGNALS} values"
        )
    if len(class_score_felts) != V3_NUM_CLASSES:
        raise ValueError(
            f"class_score_felts must contain exactly {V3_NUM_CLASSES} values"
        )

    signals = [int(value) for value in public_signals]
    scores = [int(value) for value in class_score_felts]
    if any(value < 0 or value >= 2**256 for value in signals):
        raise ValueError("public_signals values must fit uint256")
    if any(value < 0 or value >= 2**256 for value in scores):
        raise ValueError("class_score_felts values must fit uint256")
    if signals[V3_INPUT_OFFSET:] != scores:
        raise ValueError("class_score_felts do not match public proof outputs")

    proof_hash = _hex32(bytes(Web3.keccak(bytes(proof_bytes))))
    signals_hash = _hex32(
        _keccak_abi(["uint256[]"], [signals])
    )
    scores_hash = _hex32(
        _keccak_abi(["uint256[10]"], [scores])
    )

    digest = compute_v3_digest(
        agent=agent,
        contract_address=contract_address,
        contract_code_hash=contract_code_hash,
        chain_id=chain_id,
        registry_address=registry_address,
        round_id=round_id,
        teacher_model_hash=teacher_model_hash,
        proxy_bundle_hash=proxy_bundle_hash,
        data_version_hash=data_version_hash,
        class_schema_hash=class_schema_hash,
        proof_hash=proof_hash,
        public_signals_hash=signals_hash,
        class_score_felts_hash=scores_hash,
        deadline=deadline,
    )

    return AuditRequestV3(
        agent=_address(agent, field_name="agent"),
        contract_address=_address(contract_address, field_name="contract_address"),
        contract_code_hash=_hex32(_bytes32(contract_code_hash, field_name="contract_code_hash")),
        chain_id=chain_id,
        registry_address=_address(registry_address, field_name="registry_address"),
        round_id=round_id,
        teacher_model_hash=_hex32(_bytes32(teacher_model_hash, field_name="teacher_model_hash")),
        proxy_bundle_hash=_hex32(_bytes32(proxy_bundle_hash, field_name="proxy_bundle_hash")),
        data_version_hash=_hex32(_bytes32(data_version_hash, field_name="data_version_hash")),
        class_schema_hash=_hex32(_bytes32(class_schema_hash, field_name="class_schema_hash")),
        proof_hash=proof_hash,
        public_signals_hash=signals_hash,
        class_score_felts_hash=scores_hash,
        deadline=deadline,
        digest=digest,
    )


def evaluate_v3_request(
    request: AuditRequestV3,
    *,
    now_timestamp: int | None = None,
) -> PolicyResult:
    """Validate an unsigned V3 request before handing it to a signer.

    Acceptance here means **eligible for the isolated signer to consider**. It
    is not a signature, transaction, broadcast, receipt, or proof-finality
    claim.
    """
    try:
        expected_digest = compute_v3_digest(
            agent=request.agent,
            contract_address=request.contract_address,
            contract_code_hash=request.contract_code_hash,
            chain_id=request.chain_id,
            registry_address=request.registry_address,
            round_id=request.round_id,
            teacher_model_hash=request.teacher_model_hash,
            proxy_bundle_hash=request.proxy_bundle_hash,
            data_version_hash=request.data_version_hash,
            class_schema_hash=request.class_schema_hash,
            proof_hash=request.proof_hash,
            public_signals_hash=request.public_signals_hash,
            class_score_felts_hash=request.class_score_felts_hash,
            deadline=request.deadline,
        )
    except (TypeError, ValueError) as exc:
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_INVALID_V3,
            details={"error": str(exc)},
        )

    if request.proof_scope != LEGACY_PROOF_SCOPE:
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_INVALID_V3,
            details={"error": f"unexpected proof_scope:{request.proof_scope}"},
        )
    if request.submission_protocol != V3_SUBMISSION_PROTOCOL:
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_INVALID_V3,
            details={
                "error": f"unexpected submission_protocol:{request.submission_protocol}"
            },
        )
    if request.digest.lower() != expected_digest.lower():
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_INVALID_V3,
            details={"error": "request_digest_mismatch"},
        )

    now_ts = int(time.time()) if now_timestamp is None else int(now_timestamp)
    if request.deadline < now_ts:
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_INVALID_V3,
            details={"error": "request_expired", "deadline": request.deadline},
        )

    return PolicyResult(
        decision=PolicyDecision.ACCEPTED,
        reason=None,
        details={
            "submission_protocol": request.submission_protocol,
            "proof_scope": request.proof_scope,
            "digest": request.digest,
            "note": (
                "Eligible for isolated policy signing. The underlying ZK proof "
                "remains proxy-only; context binding is supplied by this V3 "
                "attestation and enforced by AuditRegistry."
            ),
        },
    )


def evaluate_submission(
    *,
    proof_scope: str,
    contract_address: str,
    chain_id: int,
    round_id: int,
    model_hash: str,
    **kwargs: Any,
) -> PolicyResult:
    """Evaluate the legacy proof-only submission path.

    No proof-scope string can make this path eligible. V3 acceptance requires a
    fully constructed ``AuditRequestV3`` and ``evaluate_v3_request`` instead.
    """
    if not proof_scope or proof_scope == "none":
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_NO_SCOPE,
            details={
                "proof_scope": proof_scope,
                "contract_address": contract_address,
                "chain_id": chain_id,
                "round_id": round_id,
            },
        )

    if proof_scope == LEGACY_PROOF_SCOPE:
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_UNBOUND,
            details={
                "proof_scope": proof_scope,
                "contract_address": contract_address,
                "chain_id": chain_id,
                "round_id": round_id,
                "model_hash": model_hash,
                "note": (
                    "The EZKL V2 proof does not bind audit context. Use the "
                    "context-attested V3 request path; do not relabel the proof."
                ),
            },
        )

    return PolicyResult(
        decision=PolicyDecision.REJECTED,
        reason=f"proof_scope_not_accepted:{proof_scope}",
        details={
            "proof_scope": proof_scope,
            "contract_address": contract_address,
            "chain_id": chain_id,
            "round_id": round_id,
            "model_hash": model_hash,
            "note": (
                "Caller-provided proof scope is not an authorization mechanism. "
                "Only a validated context-attested V3 request can cross the "
                "signing boundary."
            ),
        },
    )


__all__ = [
    "AuditRequestV3",
    "LEGACY_PROOF_SCOPE",
    "PolicyDecision",
    "PolicyResult",
    "REJECT_REASON_INVALID_V3",
    "REJECT_REASON_NO_SCOPE",
    "REJECT_REASON_UNBOUND",
    "V3_INPUT_OFFSET",
    "V3_NUM_CLASSES",
    "V3_SUBMISSION_PROTOCOL",
    "V3_TOTAL_PUBLIC_SIGNALS",
    "build_v3_request",
    "compute_v3_digest",
    "evaluate_submission",
    "evaluate_v3_request",
]
