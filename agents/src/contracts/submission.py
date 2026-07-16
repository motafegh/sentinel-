"""Canonical submission-truth contract shared across AGENTS boundaries."""

from __future__ import annotations

from typing import Any, Mapping

SUBMISSION_SCHEMA_VERSION = "1"


def normalize_submission(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the lossless R0 containment view of submission/finality truth."""
    source = dict(value or {})
    proof_scope = str(source.get("proof_scope") or "none")
    status = str(source.get("status") or "not_requested")
    policy_decision = str(
        source.get("policy_decision") or source.get("decision") or "not_requested"
    )
    policy_reason = source.get("policy_reason")
    if policy_reason is None and policy_decision.startswith("policy_"):
        policy_reason = source.get("reason")
    eligible = source.get("verified_audit_eligible") is True
    ineligible_reason = source.get("finality_ineligible_reason")
    if not eligible and not ineligible_reason:
        ineligible_reason = (
            "proof_scope_not_identity_bound"
            if proof_scope == "legacy_proxy_only_unbound"
            else "no_proof_scope"
        )
    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "proof_scope": proof_scope,
        "status": status,
        "policy_decision": policy_decision,
        "policy_reason": policy_reason,
        "verified_audit_eligible": eligible,
        "finality_ineligible_reason": None if eligible else str(ineligible_reason),
    }


__all__ = ["SUBMISSION_SCHEMA_VERSION", "normalize_submission"]
