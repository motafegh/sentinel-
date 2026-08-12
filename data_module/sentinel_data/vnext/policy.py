"""Pure DATA vNext policy interpretation.

No filesystem writes occur here.  The module translates one frozen Phase-3
ledger row plus the accepted Phase-5 policy and Phase-6 contract role into the
canonical vNext semantic state used by the builder and validator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CLASS_NAMES: tuple[str, ...] = (
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
)

TRAINING_ROLES = {"TRAIN_STRONG", "TRAIN_WEAK"}
POSITIVE_EVAL_ROLES = {"MODEL_SELECTION", "INTERNAL_AUDIT"}


@dataclass(frozen=True)
class SemanticDecision:
    outcome_state: str
    target_value: int | None
    training_signal: str
    training_strength: str
    source_policy_loss_eligible: bool
    outcome_metric_eligible: bool
    policy_decision_id: str
    reason_code: str


def validate_policy_surface(policy: dict[str, Any]) -> None:
    """Fail closed if a policy other than the accepted R4 policy is supplied."""
    if policy.get("policy_version") != "data-vnext-policy-v1":
        raise ValueError("DATA vNext requires policy_version=data-vnext-policy-v1")
    if policy.get("status") != "ACCEPTED_G5":
        raise ValueError("DATA vNext requires accepted policy status ACCEPTED_G5")
    vocab = policy.get("class_vocabulary") or {}
    if tuple(vocab.get("classes") or ()) != CLASS_NAMES:
        raise ValueError("DATA vNext class vocabulary/order mismatch")
    if vocab.get("feature_schema_version") != "v9":
        raise ValueError("DATA vNext requires graph feature schema v9")


def _enabled(policy: dict[str, Any], class_name: str) -> bool:
    cfg = policy["class_supervision"][class_name]
    return cfg["status"] == "ENABLED"


def semantic_decision(row: dict[str, Any], policy: dict[str, Any], role: str) -> SemanticDecision:
    """Return the canonical semantic/training decision for one ledger row.

    The role affects only final metric/use eligibility.  It never upgrades weak
    or unknown evidence into canonical truth.
    """
    validate_policy_surface(policy)

    class_name = str(row["class_name"])
    if class_name not in CLASS_NAMES:
        raise ValueError(f"unknown class: {class_name}")

    historical_target = row.get("historical_target")
    source = str(row.get("primary_source") or "")
    historical_positive = historical_target == 1

    # Disabled classes remain present positionally but have no supervised target.
    if not _enabled(policy, class_name):
        return SemanticDecision(
            outcome_state="NOT_REVIEWED" if historical_positive else "UNKNOWN",
            target_value=None,
            training_signal="NONE",
            training_strength="NONE",
            source_policy_loss_eligible=False,
            outcome_metric_eligible=False,
            policy_decision_id="R4-D-002",
            reason_code="SUPERVISION_DISABLED_PENDING_EVIDENCE",
        )

    # SolidiFI's only historical positive is the injected class and is accepted
    # as strong positive evidence.  Non-target zeros are never negatives.
    if source == "solidifi" and historical_positive:
        return SemanticDecision(
            outcome_state="CONFIRMED_POSITIVE",
            target_value=1,
            training_signal="POSITIVE",
            training_strength="STRONG",
            source_policy_loss_eligible=True,
            outcome_metric_eligible=role in POSITIVE_EVAL_ROLES,
            policy_decision_id="R4-D-002",
            reason_code="SOLIDIFI_INJECTED_CLASS_STRONG_POSITIVE",
        )

    # Approved SmartBugs direct categories are strong positives, except
    # Timestamp: the committed ledger cannot distinguish time_manipulation from
    # the superseded bad_randomness->Timestamp mapping, so it fails closed.
    if source == "smartbugs_curated" and historical_positive:
        if class_name == "Timestamp":
            return SemanticDecision(
                outcome_state="NOT_REVIEWED",
                target_value=None,
                training_signal="NONE",
                training_strength="NONE",
                source_policy_loss_eligible=False,
                outcome_metric_eligible=False,
                policy_decision_id="R4-D-002",
                reason_code="SMARTBUGS_TIMESTAMP_NATIVE_CATEGORY_AMBIGUOUS",
            )
        approved = set(policy["sources"]["smartbugs_curated"]["approved_mappings"].values())
        if class_name in approved:
            return SemanticDecision(
                outcome_state="CONFIRMED_POSITIVE",
                target_value=1,
                training_signal="POSITIVE",
                training_strength="STRONG",
                source_policy_loss_eligible=True,
                outcome_metric_eligible=role in POSITIVE_EVAL_ROLES,
                policy_decision_id="R4-D-002",
                reason_code="SMARTBUGS_APPROVED_DIRECT_STRONG_POSITIVE",
            )

    # The only retained DIVE supervised signal is weak Front Running -> TOD.
    if (
        source == "dive"
        and historical_positive
        and class_name == "TransactionOrderDependence"
    ):
        cfg = policy["sources"]["dive"]["mapped_category_policy"]["Front Running"]
        if cfg.get("training_strength") != "WEAK" or cfg.get("target_value") != 1:
            raise ValueError("DIVE TOD policy no longer matches frozen Phase-5 decision")
        return SemanticDecision(
            outcome_state="NOT_REVIEWED",
            target_value=1,
            training_signal="POSITIVE",
            training_strength="WEAK",
            source_policy_loss_eligible=True,
            outcome_metric_eligible=False,
            policy_decision_id="R4-D-002",
            reason_code="DIVE_TOD_WEAK_POSITIVE_ONLY",
        )

    # All other positives are masked/not-reviewed.  Historical zeros and
    # non-assertions remain unknown and never become target=0.
    return SemanticDecision(
        outcome_state="NOT_REVIEWED" if historical_positive else "UNKNOWN",
        target_value=None,
        training_signal="NONE",
        training_strength="NONE",
        source_policy_loss_eligible=False,
        outcome_metric_eligible=False,
        policy_decision_id="R4-D-002",
        reason_code="MASKED_OR_UNLABELED_NO_AUTHORIZED_TARGET",
    )


def effective_loss_mask(decision: SemanticDecision, role: str) -> bool:
    """Combine source-policy eligibility with the frozen Phase-6 role."""
    if not decision.source_policy_loss_eligible:
        return False
    if decision.training_strength == "STRONG":
        return role == "TRAIN_STRONG"
    if decision.training_strength == "WEAK":
        return role == "TRAIN_WEAK"
    return False


def source_claim_state(row: dict[str, Any]) -> str:
    """Map Phase-3 reconstructed source state into the vNext source-claim enum."""
    value = str(row.get("source_native_state") or "NOT_RECONSTRUCTED")
    return {
        "EXPLICIT_POSITIVE": "POSITIVE",
        "EXPLICIT_NEGATIVE": "EXPLICIT_ZERO",
        "UNKNOWN": "UNKNOWN",
        "ABSENT": "NO_ASSERTION",
        "UNSUPPORTED": "UNSUPPORTED",
        "DROPPED_CATEGORY": "DROPPED_CATEGORY",
        "MAPPED_NONVULNERABLE": "OUT_OF_TAXONOMY",
        "UNAVAILABLE": "UNAVAILABLE",
        "MIXED": "UNKNOWN",
        "NOT_RECONSTRUCTED": "UNKNOWN",
    }.get(value, "UNKNOWN")


def crosswalk_action(row: dict[str, Any]) -> tuple[str, str | None]:
    """Map historical crosswalk state into a vNext semantic action.

    SmartBugs Timestamp is deliberately no-target because native category
    identity was lost in the committed ledger.
    """
    source = str(row.get("primary_source") or "")
    class_name = str(row.get("class_name") or "")
    historical_target = row.get("historical_target")

    if source == "smartbugs_curated" and class_name == "Timestamp" and historical_target == 1:
        return "LOSSY_NO_CANONICAL_TARGET", None

    native = str(row.get("crosswalk_action") or "UNKNOWN")
    if native == "DIRECT":
        return "DIRECT", class_name
    if native == "LOSSY_MAP":
        return "SEMANTIC_COMPRESSION", class_name
    if native == "DROP":
        return "DROPPED_CATEGORY", None
    if native == "MAP_NONVULNERABLE":
        return "OUT_OF_TAXONOMY_NO_CANONICAL_TARGET", None
    if native == "UNSUPPORTED":
        return "UNSUPPORTED", None
    return "NO_ASSERTION", None


def role_eligibility_for_row(role: str, decision: SemanticDecision) -> list[str]:
    """Return row-level role metadata without inventing a role for EXCLUDED."""
    allowed = {
        "TRAIN_STRONG",
        "TRAIN_WEAK",
        "TRAIN_UNLABELED",
        "MODEL_SELECTION",
        "INTERNAL_AUDIT",
        "CASE_STUDY",
    }
    roles: list[str] = [role] if role in allowed else []
    if not decision.outcome_metric_eligible:
        roles.append("EXCLUDE_OUTCOME_METRICS")
    return roles
