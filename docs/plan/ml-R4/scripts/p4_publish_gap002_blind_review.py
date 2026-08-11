#!/usr/bin/env python3
"""Publish the locked source-only blind semantic review for R4-GAP-002.

This script does not run static tools or models.  It materializes the primary
semantic verdicts already made against the checksum-verified blind Solidity
bundle.  Contract identities and source hashes are read from review_tasks.jsonl
so the review stays bound to the frozen sample.

The reviewer is explicitly recorded as an AI primary semantic reviewer.  These
verdicts do not create CONFIRMED_NEGATIVE outcomes and are not acceptance-grade
labels.  Their purpose is the Phase-4 source/stratum role decision.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

GAP_ID = "R4-GAP-002"
SAMPLE_VERSION = "r4-gap-002-sample-v1"
EXPECTED_SAMPLE_SHA256 = "2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9"
TASKS = Path("docs/plan/ml-R4/review_bundles/r4_gap002_blind_review_bundle_v1_extracted/review_tasks.jsonl")
SAMPLE = Path("docs/plan/ml-R4/manifests/p4_gap002_initial_sample.jsonl")
OUT = Path("docs/plan/ml-R4/reviews/R4-GAP-002/p4_gap002_blind_semantic_review_v1.jsonl")
REPORT = Path("docs/plan/ml-R4/findings/06_gap002_blind_semantic_review_report.json")

# Ordinals were reviewed against the frozen class rubric using source only.
# Any ordinal not listed as SUPPORT/UNCLEAR/CONFLICT is DOES_NOT_SUPPORT.
VERDICTS = {
    "DenialOfService": {
        "support": [],
        "unclear": [],
        "conflict": [],
    },
    "IntegerUO": {
        "support": [9, 12, 13],
        "unclear": [14],
        "conflict": [],
    },
    "Timestamp": {
        "support": [11, 15, 16, 18],
        "unclear": [13],
        "conflict": [],
    },
    "TransactionOrderDependence": {
        "support": [1, 2, 3, 4, 8, 10, 12, 13, 14, 15, 16, 17],
        "unclear": [],
        "conflict": [11, 18, 19],
    },
    "UnusedReturn": {
        "support": [3, 4, 7, 8, 10, 11, 13, 15, 19],
        "unclear": [],
        "conflict": [],
    },
}

ROLE = {
    "DenialOfService": "MASK_OR_EXCLUDE",
    "IntegerUO": "MASK_OR_EXCLUDE",
    "Timestamp": "MASK_OR_EXCLUDE",
    "TransactionOrderDependence": "TRAIN_WEAK",
    "UnusedReturn": "MASK_OR_EXCLUDE",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reason_code(class_name: str, ordinal: int, state: str) -> str:
    if class_name == "DenialOfService":
        if ordinal in {9, 16}:
            return "FIXED_OR_TRUSTED_RECIPIENT_FAILURE_NOT_ATTACKER_CONTROLLED"
        if ordinal == 20:
            return "CALLER_LOCAL_REVERT_NOT_PERSISTENT_GLOBAL_DOS"
        if ordinal in {3, 4, 14}:
            return "EXTERNAL_DEPENDENCY_OR_LOOP_WITHOUT_ATTACKER_PERSISTENT_BLOCKER"
        return "NO_CANONICAL_ATTACKER_CONTROLLED_PERSISTENT_DOS"

    if class_name == "IntegerUO":
        if state == "SUPPORTS_POSITIVE":
            return {
                9: "PRE08_ATTACKER_CONTROLLED_RAW_MULTIPLICATION",
                12: "PRE08_REVERSED_GUARDS_ALLOW_UNDERFLOW",
                13: "PRE08_ATTACKER_CONTROLLED_RAW_MULTIPLICATION",
            }[ordinal]
        if state == "UNCLEAR_INSUFFICIENT":
            return "RAW_VALUE_CRITICAL_ARITHMETIC_BUT_REACHABILITY_BOUNDS_UNCLEAR"
        if ordinal in {4, 6}:
            return "INTENTIONAL_GUARDED_UNCHECKED_OPTIMIZATION"
        if ordinal in {1, 2, 3, 10, 15, 17, 19}:
            return "SOL08_CHECKED_ARITHMETIC_NO_UNSAFE_PATH"
        if ordinal in {7, 11, 18, 20}:
            return "PROTECTED_OR_GUARDED_PRE08_ARITHMETIC"
        return "NO_SECURITY_RELEVANT_WRAPAROUND_PATH"

    if class_name == "Timestamp":
        if state == "SUPPORTS_POSITIVE":
            return {
                11: "TIMESTAMP_DIRECTLY_CHANGES_MINT_CAP",
                15: "SHORT_AUCTION_OUTCOME_TIMESTAMP_BOUNDARY",
                16: "CROWDSALE_OPENING_TIMESTAMP_BOUNDARY",
                18: "SHORT_SNIPING_WINDOW_PUNITIVE_CLASSIFICATION",
            }[ordinal]
        if state == "UNCLEAR_INSUFFICIENT":
            return "SECURITY_TIMELOCK_MATERIALITY_DEPENDS_ON_CONFIGURATION"
        if ordinal in {4, 5, 8, 9, 10, 14, 17}:
            return "ROUTER_DEADLINE_ONLY"
        if ordinal in {6, 20}:
            return "LONG_DURATION_WINDOW_TIMESTAMP_SKEW_IMMATERIAL"
        return "TIMESTAMP_NOT_MATERIALLY_SECURITY_SENSITIVE"

    if class_name == "TransactionOrderDependence":
        if state == "SUPPORTS_POSITIVE":
            if ordinal == 3:
                return "ERC721_APPROVAL_REVOCATION_ORDERING_RACE"
            if ordinal == 12:
                return "PUBLIC_UNIQUE_ASSET_MARKET_ORDERING_RACE"
            return "ERC20_ALLOWANCE_OVERWRITE_ORDERING_RACE"
        if state == "CLASS_BOUNDARY_CONFLICT":
            return {
                11: "ROOT_CAUSE_ACCESS_STATE_OVERWRITE_NOT_ORDERING",
                18: "INTENDED_LAST_DEPOSITOR_GAME_ORDERING",
                19: "INTENDED_LAST_PLAYER_GAME_ORDERING",
            }[ordinal]
        return "NO_ADVERSARIAL_MATERIAL_TRANSACTION_ORDER_RACE"

    if class_name == "UnusedReturn":
        if state == "SUPPORTS_POSITIVE":
            if ordinal == 3:
                return "ERC20_TRANSFER_BOOL_IGNORED"
            if ordinal == 10:
                return "TOKEN_TRANSFER_BOOL_IGNORED"
            if ordinal == 4:
                return "LOW_LEVEL_CALL_RESULT_DISCARDED"
            return "LOW_LEVEL_CALL_SUCCESS_ASSIGNED_BUT_NEVER_CHECKED"
        if ordinal in {1, 9}:
            return "LOW_LEVEL_CALL_RESULT_CHECKED"
        if ordinal in {2, 6, 17}:
            return "NATIVE_TRANSFER_REVERTS_NO_BOOL_TO_IGNORE"
        if ordinal in {12, 14, 16, 20}:
            return "EXTERNAL_CALL_RESULT_EXPLICITLY_CHECKED_OR_VOID"
        return "NO_MEANINGFUL_IGNORED_EXTERNAL_RETURN"

    raise RuntimeError(f"unknown class: {class_name}")


def state_for(class_name: str, ordinal: int) -> str:
    spec = VERDICTS[class_name]
    memberships = sum(ordinal in spec[k] for k in ("support", "unclear", "conflict"))
    if memberships > 1:
        raise RuntimeError(f"ordinal assigned to multiple verdicts: {class_name} {ordinal}")
    if ordinal in spec["support"]:
        return "SUPPORTS_POSITIVE"
    if ordinal in spec["unclear"]:
        return "UNCLEAR_INSUFFICIENT"
    if ordinal in spec["conflict"]:
        return "CLASS_BOUNDARY_CONFLICT"
    return "DOES_NOT_SUPPORT_POSITIVE"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    if sha256_file(SAMPLE) != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError("frozen sample SHA-256 mismatch")

    tasks = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
    if len(tasks) != 100:
        raise RuntimeError(f"expected 100 blind tasks, got {len(tasks)}")
    if any(t.get("gap_id") != GAP_ID or t.get("sample_version") != SAMPLE_VERSION for t in tasks):
        raise RuntimeError("unexpected task identity")
    if any(t.get("review_state") is not None for t in tasks):
        raise RuntimeError("input task bundle is not blind")

    by_class: dict[str, list[dict]] = {}
    rows: list[dict] = []
    for task in tasks:
        cls = task["class_name"]
        ordinal = int(task["stratum_ordinal"])
        if cls not in VERDICTS or not 1 <= ordinal <= 20:
            raise RuntimeError(f"unexpected review task: {cls} {ordinal}")
        state = state_for(cls, ordinal)
        row = {
            "schema": "r4-gap-semantic-review-row-v1",
            "gap_id": GAP_ID,
            "sample_version": SAMPLE_VERSION,
            "batch_id": task["batch_id"],
            "class_index": int(task["class_index"]),
            "class_name": cls,
            "stratum_ordinal": ordinal,
            "contract_id": task["contract_id"],
            "review_group_id": task["review_group_id"],
            "source_file": task["source_file"],
            "source_file_sha256": task["source_file_sha256"],
            "reviewer_kind": "AI_PRIMARY_SEMANTIC_REVIEW",
            "reviewer_model": "GPT-5.6 Sol",
            "review_mode": "SOURCE_ONLY_BLIND",
            "review_state": state,
            "reason_code": reason_code(cls, ordinal, state),
            "creates_confirmed_negative": False,
            "model_tool_evidence_revealed": False,
        }
        rows.append(row)
        by_class.setdefault(cls, []).append(row)

    if len({(r["class_name"], r["stratum_ordinal"]) for r in rows}) != 100:
        raise RuntimeError("review ordinal identity is not unique")
    if len({r["contract_id"] for r in rows}) != 100:
        raise RuntimeError("contract identity reuse detected")
    if len({r["review_group_id"] for r in rows}) != 100:
        raise RuntimeError("review-group reuse detected")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    strata = {}
    for cls, class_rows in by_class.items():
        counts = Counter(r["review_state"] for r in class_rows)
        k = counts["SUPPORTS_POSITIVE"]
        lo, hi = wilson(k, len(class_rows))
        strata[cls] = {
            "n": len(class_rows),
            "supports_positive": k,
            "does_not_support_positive": counts["DOES_NOT_SUPPORT_POSITIVE"],
            "unclear_insufficient": counts["UNCLEAR_INSUFFICIENT"],
            "class_boundary_conflict": counts["CLASS_BOUNDARY_CONFLICT"],
            "observed_support_rate": k / len(class_rows),
            "wilson_95_descriptive_interval": [lo, hi],
            "role_recommendation": ROLE[cls],
        }

    report = {
        "schema": "r4-gap-blind-semantic-review-report-v1",
        "gap_id": GAP_ID,
        "sample_version": SAMPLE_VERSION,
        "sample_sha256": EXPECTED_SAMPLE_SHA256,
        "review_rows_sha256": sha256_file(OUT),
        "reviewer_kind": "AI_PRIMARY_SEMANTIC_REVIEW",
        "review_mode": "SOURCE_ONLY_BLIND",
        "task_count": len(rows),
        "strata": dict(sorted(strata.items(), key=lambda kv: by_class[kv[0]][0]["class_index"])),
        "limitations": [
            "Single AI semantic reviewer; not inter-rater or acceptance-grade evidence.",
            "Intervals are descriptive Wilson intervals for the deterministic group-aware screening sample, not promotion thresholds.",
            "DOES_NOT_SUPPORT_POSITIVE does not create a confirmed negative outcome.",
            "No model prediction, tool vote, merger outcome, or non-target historical label was used in the blind pass.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
