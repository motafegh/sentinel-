"""R0-F3: Policy-signer boundary — rejects V2/unbound proofs.

This is the **separation boundary** between the analysis/MCP process and
on-chain submission. The analysis process constructs a typed unsigned
submission request. The policy signer validates it and either:
  - rejects (returns policy_rejected with reason), or
  - accepts (returns prepared for signing — R4 key management).

No raw private key, no transaction construction, no ABI encoding, and
no RPC access live in this module. That belongs to the key-management
service in R4.

R0-R4 separation:
  R0: this module — typed rejection of V2/unbound proofs
  R4: key management, actual signing, broadcast, receipt monitoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyDecision(Enum):
    REJECTED = "policy_rejected"
    ACCEPTED = "policy_accepted"


REJECT_REASON_UNBOUND = "proof_scope_not_identity_bound"
REJECT_REASON_NO_SCOPE = "no_proof_scope"


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


def evaluate_submission(
    *,
    proof_scope: str,
    contract_address: str,
    chain_id: int,
    round_id: int,
    model_hash: str,
    **kwargs: Any,
) -> PolicyResult:
    """Evaluate a submission request against policy rules.

    Current policy (R0-F3):
      - Only proofs with proof_scope == 'typed_identity_bound_v3' are eligible.
      - V2 proofs (legacy_proxy_only_unbound) are categorically rejected.
      - Missing proof_scope is rejected.
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

    if proof_scope == "legacy_proxy_only_unbound":
        return PolicyResult(
            decision=PolicyDecision.REJECTED,
            reason=REJECT_REASON_UNBOUND,
            details={
                "proof_scope": proof_scope,
                "contract_address": contract_address,
                "chain_id": chain_id,
                "round_id": round_id,
                "model_hash": model_hash,
                "note": "V2 proofs do not bind contract/chain/round/model identity. "
                        "Full typed identity binding requires R3 V3 protocol work.",
            },
        )

    # Future: typed_identity_bound_v3 passes through
    if proof_scope == "typed_identity_bound_v3":
        return PolicyResult(
            decision=PolicyDecision.ACCEPTED,
            details={
                "proof_scope": proof_scope,
                "contract_address": contract_address,
                "chain_id": chain_id,
                "round_id": round_id,
                "model_hash": model_hash,
                "note": "V3 identity-bound proof — accepted for submission (R4 signing)",
            },
        )

    return PolicyResult(
        decision=PolicyDecision.REJECTED,
        reason=f"unknown_proof_scope:{proof_scope}",
        details={"proof_scope": proof_scope},
    )


__all__ = [
    "PolicyDecision",
    "PolicyResult",
    "evaluate_submission",
    "REJECT_REASON_UNBOUND",
    "REJECT_REASON_NO_SCOPE",
]
