"""Feedback eligibility boundary for on-chain audit observations.

Only the historical V1 scalar policy is encoded here, unchanged. V3 has no
measured feedback-promotion policy yet, so every V3 observation is returned as
explicitly not evaluated / policy unavailable. It must not enter RAG through
the V1 threshold by accident.
"""

from __future__ import annotations

from typing import Any, Mapping

from .audit_observation import PROTOCOL_V1, PROTOCOL_V3

# Historical compatibility policy preserved exactly from feedback_loop.py.
# This is NOT a V3 threshold and must not be reused for V3 class-score felts.
LEGACY_V1_SCORE_THRESHOLD = 5734
LEGACY_V1_POLICY_VERSION = "legacy-v1-scalar-5734"
V3_POLICY_VERSION = None


def evaluate_feedback_eligibility(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit eligibility decision without mutating RAG state."""
    protocol = str(observation.get("protocol_version") or "")

    if protocol == PROTOCOL_V3:
        return {
            "eligible": False,
            "status": "not_evaluated",
            "reason_code": "v3_feedback_policy_unavailable",
            "reason": (
                "V3 feedback promotion requires a measured/versioned policy from "
                "the promoted R4 model/data/calibration lineage; legacy V1 scalar "
                "thresholds are not applicable"
            ),
            "policy_version": V3_POLICY_VERSION,
            "protocol_version": PROTOCOL_V3,
            "attempted": False,
        }

    if protocol == PROTOCOL_V1:
        if "score_field_element" not in observation:
            raise ValueError("V1 observation missing score_field_element")
        score = int(observation["score_field_element"])
        if score < 0:
            raise ValueError("V1 score_field_element must be non-negative")
        eligible = score >= LEGACY_V1_SCORE_THRESHOLD
        return {
            "eligible": eligible,
            "status": "eligible" if eligible else "ineligible",
            "reason_code": (
                "legacy_v1_scalar_threshold_met"
                if eligible
                else "legacy_v1_scalar_threshold_not_met"
            ),
            "reason": (
                "Historical V1 scalar compatibility policy only. "
                f"score_field_element={score}, threshold={LEGACY_V1_SCORE_THRESHOLD}."
            ),
            "policy_version": LEGACY_V1_POLICY_VERSION,
            "protocol_version": PROTOCOL_V1,
            "attempted": True,
        }

    raise ValueError(f"unsupported feedback observation protocol: {protocol!r}")


__all__ = [
    "LEGACY_V1_POLICY_VERSION",
    "LEGACY_V1_SCORE_THRESHOLD",
    "V3_POLICY_VERSION",
    "evaluate_feedback_eligibility",
]
