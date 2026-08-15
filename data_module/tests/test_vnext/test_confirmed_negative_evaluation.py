from __future__ import annotations

from sentinel_data.vnext.confirmed_negative_evaluation import (
    build_review_queue,
    minimum_zero_false_positive_sample_size,
    validate_adjudications,
)
from sentinel_data.vnext.policy import CLASS_NAMES


def _row(contract_id: str, group_id: str, *, role: str = "TRAIN_UNLABELED") -> dict:
    row = {
        "contract_id": contract_id,
        "group_id": group_id,
        "source": "dive",
        "role": role,
    }
    for index in range(len(CLASS_NAMES)):
        row[f"target_{index}"] = None
        row[f"source_loss_eligible_{index}"] = False
        row[f"effective_loss_mask_{index}"] = False
        row[f"outcome_metric_mask_{index}"] = False
        row[f"outcome_state_{index}"] = "UNKNOWN"
    return row


def _queue():
    return build_review_queue(
        [
            _row("c1", "g1"),
            _row("c2", "g1"),
            _row("c3", "g2"),
        ],
        dataset_version="sentinel-r4-vnext-v2",
        partition_version="r4-vnext-roles-v2",
        publication_manifest_sha256="a" * 64,
        enabled_class_names=("Reentrancy",),
        per_class=2,
    )


def _valid_adjudication(candidate, *, primary="reviewer-a", second="reviewer-b"):
    return {
        "candidate_id": candidate["candidate_id"],
        "contract_id": candidate["contract_id"],
        "group_id": candidate["group_id"],
        "class_index": candidate["class_index"],
        "class_name": candidate["class_name"],
        "decision": "CONFIRMED_NEGATIVE",
        "negative_scope": "CLASS_SPECIFIC_ONLY",
        "primary_review": {
            "reviewer_id": primary,
            "reviewed_at": "2026-08-15T20:00:00Z",
            "rationale": "Complete class-specific code review found no reentrancy-capable external interaction path.",
            "code_scope_complete": True,
            "all_file_graph_components_reviewed": True,
            "contradictory_positive_evidence_found": False,
            "evidence": [
                {
                    "type": "MANUAL_CLASS_SPECIFIC_REVIEW",
                    "reference": "review/c1/reentrancy",
                    "summary": "All externally callable paths and state-changing call sites were reviewed.",
                    "independent_of_training_label": True,
                }
            ],
        },
        "independent_verification": {
            "status": "AGREES",
            "reviewer_id": second,
            "reviewed_at": "2026-08-15T20:30:00Z",
            "rationale": "Independent inspection agrees that the class-specific negative claim is supported.",
            "evidence": [
                {
                    "type": "TOOL_ANALYSIS",
                    "reference": "verify/c1/reentrancy",
                    "summary": "Independent static analysis found no conflicting positive path.",
                    "independent_of_training_label": True,
                }
            ],
        },
    }


def test_zero_false_positive_planning_bound_is_59_for_5_percent_at_95_percent():
    assert minimum_zero_false_positive_sample_size(
        max_false_positive_rate=0.05,
        confidence=0.95,
    ) == 59


def test_queue_is_deterministic_and_never_claims_negative_truth():
    first = _queue()
    second = _queue()
    assert first == second
    assert first["queued_cells"] == 2
    assert len({row["group_id"] for row in first["candidates"]}) == 2
    assert all(row["negative_truth_claim"] is False for row in first["candidates"])
    assert all(row["current_target_value"] is None for row in first["candidates"])


def test_same_reviewer_cannot_self_verify_confirmed_negative():
    queue = _queue()
    candidate = queue["candidates"][0]
    report = validate_adjudications(
        queue,
        [_valid_adjudication(candidate, primary="same", second="same")],
    )
    assert report["status"] == "FAIL"
    assert report["confirmed_negative_cells"] == 0
    assert report["training_target_authorized"] is False


def test_valid_dual_review_is_evaluation_only():
    queue = _queue()
    candidate = queue["candidates"][0]
    report = validate_adjudications(queue, [_valid_adjudication(candidate)])
    assert report["status"] == "PASS"
    assert report["confirmed_negative_cells"] == 1
    accepted = report["accepted_cells"][0]
    assert accepted["target_value"] == 0
    assert accepted["outcome_state"] == "CONFIRMED_NEGATIVE"
    assert accepted["usage_authority"] == "EVALUATION_ONLY_NOT_TRAINING_AUTHORITY"
    assert report["training_target_authorized"] is False
    assert report["threshold_fit_authorized"] is False
