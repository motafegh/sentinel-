"""Evidence-honest confirmed-negative evaluation tooling for R4 Phase 8.

This module does not infer negative truth from missing labels, source silence, or
unlabeled status. It only:

* builds deterministic class-balanced review queues from currently unlabeled
  leakage groups;
* validates explicit human/independent adjudications;
* emits confirmed-negative cells for evaluation-only use.

A confirmed-negative evaluation cell is not automatically a training target.
Promoting any accepted negative into optimizer supervision requires a separate
versioned policy/role decision.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from sentinel_data.vnext.policy import CLASS_NAMES

QUEUE_SCHEMA = "sentinel-r4-confirmed-negative-review-queue-v1"
ADJUDICATION_SCHEMA = "sentinel-r4-confirmed-negative-adjudication-v1"
ACCEPTED_SCHEMA = "sentinel-r4-confirmed-negative-evaluation-v1"
DEFAULT_QUEUE_SALT = "r4-phase8-confirmed-negative-pilot-v1"
DEFAULT_ELIGIBLE_ROLES = frozenset({"TRAIN_UNLABELED"})

DECISIONS = frozenset({"CONFIRMED_NEGATIVE", "NOT_CONFIRMED", "EXCLUDE"})
PRIMARY_EVIDENCE_TYPES = frozenset(
    {
        "MANUAL_CLASS_SPECIFIC_REVIEW",
        "FORMAL_STATIC_ARGUMENT",
        "TRUSTED_EXPLICIT_NEGATIVE_SOURCE",
    }
)
CORROBORATING_EVIDENCE_TYPES = frozenset(
    {
        "DYNAMIC_PROPERTY_TEST",
        "TOOL_ANALYSIS",
        "DOCUMENTATION",
    }
)
ALL_EVIDENCE_TYPES = PRIMARY_EVIDENCE_TYPES | CORROBORATING_EVIDENCE_TYPES


class ConfirmedNegativeEvidenceError(ValueError):
    """Raised when a review/adjudication attempts to fail open."""


def _rank(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_id(
    publication_manifest_sha256: str,
    group_id: str,
    contract_id: str,
    class_index: int,
) -> str:
    return "r4neg-" + _rank(
        QUEUE_SCHEMA,
        publication_manifest_sha256,
        group_id,
        contract_id,
        class_index,
    )[:32]


def minimum_zero_false_positive_sample_size(
    *,
    max_false_positive_rate: float,
    confidence: float,
) -> int:
    """Minimum negative examples needed for a zero-FP one-sided binomial bound.

    If zero false positives are observed in ``n`` independent negative examples,
    ``(1-max_false_positive_rate)**n <= 1-confidence`` is sufficient to bound
    the true false-positive rate below ``max_false_positive_rate`` at the
    requested confidence level.

    This helper is a planning bound only. Group dependence, selection bias,
    threshold fitting, and multiple-class testing still require explicit
    evaluation design.
    """

    if not 0.0 < max_false_positive_rate < 1.0:
        raise ValueError("max_false_positive_rate must be between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return math.ceil(
        math.log(1.0 - confidence) / math.log(1.0 - max_false_positive_rate)
    )


def build_review_queue(
    ml_rows: Iterable[dict[str, Any]],
    *,
    dataset_version: str,
    partition_version: str,
    publication_manifest_sha256: str,
    enabled_class_names: Iterable[str],
    per_class: int = 25,
    seed_salt: str = DEFAULT_QUEUE_SALT,
    eligible_roles: Iterable[str] = DEFAULT_ELIGIBLE_ROLES,
) -> dict[str, Any]:
    """Build a deterministic pilot queue without asserting negative truth.

    Candidate cells come only from currently unlabeled groups and must have a
    null target with no source-policy/effective loss eligibility. One leakage
    group may appear at most once across the entire queue, including across
    different vulnerability classes.

    Queue membership is *review reservation*, not a label. If a later PU
    objective consumes unlabeled examples, queue/reserved groups must remain
    outside optimizer use until the evaluation decision is closed.
    """

    if per_class < 1:
        raise ValueError("per_class must be >= 1")
    enabled = tuple(str(name) for name in enabled_class_names)
    invalid_classes = sorted(set(enabled) - set(CLASS_NAMES))
    if invalid_classes:
        raise ValueError(f"unknown enabled classes: {invalid_classes}")
    roles = frozenset(str(role) for role in eligible_roles)
    if not roles:
        raise ValueError("eligible_roles must not be empty")

    rows = [dict(row) for row in ml_rows if str(row.get("role")) in roles]
    rows.sort(key=lambda row: (str(row["group_id"]), str(row["contract_id"])))

    candidates: list[dict[str, Any]] = []
    available_by_class: dict[str, int] = {}
    selected_groups: set[str] = set()

    for class_name in enabled:
        class_index = CLASS_NAMES.index(class_name)
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_id = str(row["group_id"])
            if group_id in selected_groups:
                continue
            if row.get(f"target_{class_index}") is not None:
                continue
            if bool(row.get(f"source_loss_eligible_{class_index}")):
                continue
            if bool(row.get(f"effective_loss_mask_{class_index}")):
                continue
            if bool(row.get(f"outcome_metric_mask_{class_index}")):
                continue
            outcome = str(row.get(f"outcome_state_{class_index}") or "")
            if outcome not in {"UNKNOWN", "NOT_REVIEWED"}:
                continue
            by_group[group_id].append(row)

        representative_rows: list[dict[str, Any]] = []
        for group_id, group_rows in sorted(by_group.items()):
            representative = min(
                group_rows,
                key=lambda row: _rank(
                    seed_salt,
                    class_name,
                    group_id,
                    row["contract_id"],
                ),
            )
            representative_rows.append(representative)
        representative_rows.sort(
            key=lambda row: _rank(
                seed_salt,
                class_name,
                row["group_id"],
                row["contract_id"],
            )
        )

        available_by_class[class_name] = len(representative_rows)
        if len(representative_rows) < per_class:
            raise ValueError(
                "confirmed-negative queue cannot satisfy globally distinct group "
                f"reservations for {class_name}: requested={per_class} "
                f"available_unreserved={len(representative_rows)}"
            )

        for ordinal, row in enumerate(representative_rows[:per_class], start=1):
            group_id = str(row["group_id"])
            contract_id = str(row["contract_id"])
            if group_id in selected_groups:  # defensive fail-closed assertion
                raise AssertionError(f"queue group reused across classes: {group_id}")
            selected_groups.add(group_id)
            candidates.append(
                {
                    "schema": QUEUE_SCHEMA,
                    "candidate_id": _candidate_id(
                        publication_manifest_sha256,
                        group_id,
                        contract_id,
                        class_index,
                    ),
                    "dataset_version": dataset_version,
                    "partition_version": partition_version,
                    "publication_manifest_sha256": publication_manifest_sha256,
                    "contract_id": contract_id,
                    "group_id": group_id,
                    "source": str(row.get("source") or ""),
                    "role_at_queue_creation": str(row["role"]),
                    "class_index": class_index,
                    "class_name": class_name,
                    "current_target_value": None,
                    "current_outcome_state": str(
                        row.get(f"outcome_state_{class_index}") or ""
                    ),
                    "candidate_status": "PENDING_REVIEW",
                    "queue_ordinal_within_class": ordinal,
                    "negative_truth_claim": False,
                    "reservation_policy": (
                        "REVIEW_PENDING_DO_NOT_ADD_GROUP_TO_FUTURE_OPTIMIZER"
                    ),
                    "reason": (
                        "UNKNOWN_CLASS_CELL_IN_CURRENTLY_UNLABELED_GROUP;"
                        "SOURCE_ABSENCE_IS_NOT_NEGATIVE_EVIDENCE"
                    ),
                }
            )

    if len(selected_groups) != len(candidates):
        raise AssertionError("confirmed-negative queue lost global group uniqueness")

    by_class = Counter(str(row["class_name"]) for row in candidates)
    return {
        "schema": QUEUE_SCHEMA,
        "status": "PILOT_REVIEW_QUEUE_NOT_NEGATIVE_TRUTH",
        "dataset_version": dataset_version,
        "partition_version": partition_version,
        "publication_manifest_sha256": publication_manifest_sha256,
        "seed_salt": seed_salt,
        "eligible_roles": sorted(roles),
        "requested_per_enabled_class": per_class,
        "enabled_classes": list(enabled),
        "available_unreserved_groups_by_class": available_by_class,
        "queued_cells_by_class": {
            name: int(by_class.get(name, 0)) for name in enabled
        },
        "queued_cells": len(candidates),
        "reserved_group_ids": sorted(selected_groups),
        "group_uniqueness_scope": "GLOBAL_ACROSS_ENABLED_CLASSES",
        "future_optimizer_rule": (
            "Any queue or accepted-evaluation group must remain outside a future "
            "PU/unlabeled optimizer population until a new role policy explicitly "
            "reconciles the reservation."
        ),
        "negative_truth_claim": False,
        "candidates": candidates,
    }


def _require_text(value: Any, field: str, *, minimum: int = 1) -> str:
    text = str(value or "").strip()
    if len(text) < minimum:
        raise ConfirmedNegativeEvidenceError(
            f"{field} must contain at least {minimum} non-whitespace characters"
        )
    return text


def _validate_review_block(
    block: Any,
    *,
    name: str,
    require_scope: bool,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(block, dict):
        raise ConfirmedNegativeEvidenceError(f"{name} must be an object")
    reviewer = _require_text(block.get("reviewer_id"), f"{name}.reviewer_id")
    _require_text(block.get("reviewed_at"), f"{name}.reviewed_at")
    _require_text(block.get("rationale"), f"{name}.rationale", minimum=20)

    if require_scope:
        if block.get("code_scope_complete") is not True:
            raise ConfirmedNegativeEvidenceError(
                f"{name}.code_scope_complete must be true"
            )
        if block.get("all_file_graph_components_reviewed") is not True:
            raise ConfirmedNegativeEvidenceError(
                f"{name}.all_file_graph_components_reviewed must be true"
            )
        if block.get("contradictory_positive_evidence_found") is not False:
            raise ConfirmedNegativeEvidenceError(
                f"{name}.contradictory_positive_evidence_found must be false"
            )

    evidence = block.get("evidence") or []
    if not isinstance(evidence, list):
        raise ConfirmedNegativeEvidenceError(f"{name}.evidence must be a list")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(evidence):
        if not isinstance(raw, dict):
            raise ConfirmedNegativeEvidenceError(
                f"{name}.evidence[{index}] must be an object"
            )
        kind = _require_text(raw.get("type"), f"{name}.evidence[{index}].type")
        if kind not in ALL_EVIDENCE_TYPES:
            raise ConfirmedNegativeEvidenceError(
                f"unsupported evidence type {kind!r}"
            )
        reference = _require_text(
            raw.get("reference"),
            f"{name}.evidence[{index}].reference",
        )
        summary = _require_text(
            raw.get("summary"),
            f"{name}.evidence[{index}].summary",
            minimum=10,
        )
        if raw.get("independent_of_training_label") is not True:
            raise ConfirmedNegativeEvidenceError(
                f"{name}.evidence[{index}] must be independent_of_training_label=true"
            )
        normalized.append(
            {
                "type": kind,
                "reference": reference,
                "summary": summary,
                "independent_of_training_label": True,
            }
        )
    return reviewer, normalized


def validate_adjudications(
    queue: dict[str, Any],
    adjudications: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate explicit adjudications and emit evaluation-only negatives.

    Manual negative confirmation requires a complete primary review, at least
    one direct class-specific evidence type, and a distinct independent reviewer
    that explicitly agrees. A failed/ambiguous review never becomes target 0.
    """

    if queue.get("schema") != QUEUE_SCHEMA:
        raise ConfirmedNegativeEvidenceError("review queue schema mismatch")
    queue_by_id = {
        str(row["candidate_id"]): row for row in (queue.get("candidates") or [])
    }
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    decisions = Counter()
    errors: list[dict[str, str]] = []

    for raw in adjudications:
        row = dict(raw)
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            errors.append({"candidate_id": "", "error": "missing candidate_id"})
            continue
        if candidate_id in seen:
            errors.append(
                {"candidate_id": candidate_id, "error": "duplicate adjudication"}
            )
            continue
        seen.add(candidate_id)
        candidate = queue_by_id.get(candidate_id)
        if candidate is None:
            errors.append(
                {"candidate_id": candidate_id, "error": "candidate not in queue"}
            )
            continue

        decision = str(row.get("decision") or "")
        if decision not in DECISIONS:
            errors.append(
                {
                    "candidate_id": candidate_id,
                    "error": f"invalid decision {decision!r}",
                }
            )
            continue
        decisions[decision] += 1

        for field in ("contract_id", "group_id", "class_index", "class_name"):
            if field in row and row[field] != candidate[field]:
                errors.append(
                    {
                        "candidate_id": candidate_id,
                        "error": f"{field} does not match queue",
                    }
                )
                decision = "INVALID"
                break
        if decision == "INVALID":
            continue

        if decision != "CONFIRMED_NEGATIVE":
            if not str(row.get("rationale") or "").strip():
                errors.append(
                    {
                        "candidate_id": candidate_id,
                        "error": f"{decision} requires rationale",
                    }
                )
            continue

        try:
            if row.get("negative_scope") != "CLASS_SPECIFIC_ONLY":
                raise ConfirmedNegativeEvidenceError(
                    "negative_scope must be CLASS_SPECIFIC_ONLY"
                )
            primary_reviewer, primary_evidence = _validate_review_block(
                row.get("primary_review"),
                name="primary_review",
                require_scope=True,
            )
            if not any(
                item["type"] in PRIMARY_EVIDENCE_TYPES
                for item in primary_evidence
            ):
                raise ConfirmedNegativeEvidenceError(
                    "confirmed negative requires at least one primary class-specific evidence type"
                )

            verification = row.get("independent_verification")
            if not isinstance(verification, dict):
                raise ConfirmedNegativeEvidenceError(
                    "independent_verification must be an object"
                )
            if verification.get("status") != "AGREES":
                raise ConfirmedNegativeEvidenceError(
                    "independent_verification.status must be AGREES"
                )
            verification_reviewer, verification_evidence = _validate_review_block(
                verification,
                name="independent_verification",
                require_scope=False,
            )
            if not verification_evidence:
                raise ConfirmedNegativeEvidenceError(
                    "independent verification requires at least one evidence record"
                )
            if verification_reviewer == primary_reviewer:
                raise ConfirmedNegativeEvidenceError(
                    "independent verification reviewer must differ from primary reviewer"
                )

            accepted.append(
                {
                    "schema": ACCEPTED_SCHEMA,
                    "candidate_id": candidate_id,
                    "dataset_version": candidate["dataset_version"],
                    "partition_version": candidate["partition_version"],
                    "publication_manifest_sha256": candidate[
                        "publication_manifest_sha256"
                    ],
                    "contract_id": candidate["contract_id"],
                    "group_id": candidate["group_id"],
                    "class_index": candidate["class_index"],
                    "class_name": candidate["class_name"],
                    "outcome_state": "CONFIRMED_NEGATIVE",
                    "target_value": 0,
                    "usage_authority": "EVALUATION_ONLY_NOT_TRAINING_AUTHORITY",
                    "primary_reviewer_id": primary_reviewer,
                    "independent_reviewer_id": verification_reviewer,
                    "primary_evidence": primary_evidence,
                    "verification_evidence": verification_evidence,
                    "limitations": [
                        "Class-specific absence only; does not imply globally safe contract.",
                        "Not optimizer authority without a separate versioned policy decision.",
                        "Evaluation group must remain leakage-disjoint from future optimizer roles.",
                    ],
                }
            )
        except ConfirmedNegativeEvidenceError as exc:
            errors.append({"candidate_id": candidate_id, "error": str(exc)})

    accepted.sort(
        key=lambda row: (
            int(row["class_index"]),
            str(row["group_id"]),
            str(row["contract_id"]),
        )
    )
    accepted_groups = sorted({str(row["group_id"]) for row in accepted})
    accepted_by_class = Counter(str(row["class_name"]) for row in accepted)
    return {
        "schema": ACCEPTED_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "queue_schema": QUEUE_SCHEMA,
        "adjudication_schema": ADJUDICATION_SCHEMA,
        "dataset_version": queue.get("dataset_version"),
        "partition_version": queue.get("partition_version"),
        "publication_manifest_sha256": queue.get(
            "publication_manifest_sha256"
        ),
        "adjudications_seen": len(seen),
        "decision_counts": dict(sorted(decisions.items())),
        "confirmed_negative_cells": len(accepted),
        "confirmed_negative_cells_by_class": dict(
            sorted(accepted_by_class.items())
        ),
        "reserved_group_ids": accepted_groups,
        "training_target_authorized": False,
        "threshold_fit_authorized": False,
        "calibration_fit_authorized": False,
        "errors": errors,
        "accepted_cells": accepted,
    }


__all__ = [
    "ACCEPTED_SCHEMA",
    "ADJUDICATION_SCHEMA",
    "ALL_EVIDENCE_TYPES",
    "ConfirmedNegativeEvidenceError",
    "DEFAULT_QUEUE_SALT",
    "PRIMARY_EVIDENCE_TYPES",
    "QUEUE_SCHEMA",
    "build_review_queue",
    "minimum_zero_false_positive_sample_size",
    "validate_adjudications",
]
