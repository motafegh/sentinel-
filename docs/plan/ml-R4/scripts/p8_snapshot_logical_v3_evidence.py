#!/usr/bin/env python3
"""Create a Git-safe, coherence-validated snapshot of logical-V3 evidence.

The V3 logical build and research reports remain under Git-ignored DATA roots.
This helper validates that all decision-critical reports describe the same V3
publication, physical binding, and hardened source commit before copying
anything into durable docs. It then sanitizes local paths, summarizes the large
selector report, and binds every snapshot file by SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
SOURCE_ROOT = DATA_ROOT / "r4-v3-logical-build"
PUBLICATION_ROOT = DATA_ROOT / "exports/sentinel-r4-vnext-v3"
OUTPUT_ROOT = REPO_ROOT / "docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3"

DATASET_VERSION = "sentinel-r4-vnext-v3"
GROUPING_VERSION = "r4-leakage-groups-v3"
PARTITION_VERSION = "r4-vnext-roles-v3"
BOUND_STATUS = "LOGICAL_V3_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED"
GPU_STATUS = "LOGICAL_V3_BOUNDED_RESEARCH_COMPLETE"
QUEUE_SCHEMA = "sentinel-r4-confirmed-negative-review-queue-v1"
QUEUE_STATUS = "PILOT_REVIEW_QUEUE_NOT_NEGATIVE_TRUTH"
QUEUE_EXPECTED_PER_CLASS = 25
QUEUE_CLASS_INDEX = {
    "CallToUnknown": 0,
    "DenialOfService": 1,
    "ExternalBug": 2,
    "IntegerUO": 4,
    "MishandledException": 5,
    "Reentrancy": 6,
    "Timestamp": 7,
    "TransactionOrderDependence": 8,
}
QUEUE_ENABLED_CLASSES = tuple(QUEUE_CLASS_INDEX)
QUEUE_ALLOWED_OUTCOME_STATES = frozenset({"UNKNOWN", "NOT_REVIEWED"})
QUEUE_EXPECTED_CELLS = QUEUE_EXPECTED_PER_CLASS * len(QUEUE_ENABLED_CLASSES)

SMALL_REPORTS = (
    "logical_v3_summary.json",
    "logical_v3_acceptance.json",
    "grouping_breadth_audit_v1.json",
    "representation_sensitivity_v1.json",
    "confirmed_negative_review_queue_v1.json",
    "selector_gpu_compare_v1.json",
)
PUBLICATION_REPORTS = {
    "logical_v3_manifest.json": "manifest.json",
    "logical_v3_partition_manifest.json": "partition_manifest.json",
    "logical_v3_representation_binding_report.json": "representation_binding_report.json",
}
LARGE_REPORT = "bounded_window_selector_v1.json"
LARGE_SUMMARY = "bounded_window_selector_v1.summary.json"
COHERENCE_REPORT = "snapshot_coherence_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        root = str(REPO_ROOT)
        if value == root:
            return "<REPO_ROOT>"
        if value.startswith(root + "/"):
            return "<REPO_ROOT>/" + value[len(root) + 1 :]
    return value


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing logical-v3 evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"logical-v3 evidence is not a JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_sanitize(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _all_true(mapping: Any) -> bool:
    return isinstance(mapping, dict) and bool(mapping) and all(
        value is True for value in mapping.values()
    )


def _queue_candidate_id(
    publication_manifest_sha256: str,
    group_id: str,
    contract_id: str,
    class_index: int,
) -> str:
    payload = "\0".join(
        str(part)
        for part in (
            QUEUE_SCHEMA,
            publication_manifest_sha256,
            group_id,
            contract_id,
            class_index,
        )
    )
    return "r4neg-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def validate_queue_coherence(
    queue: dict[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, bool]:
    """Validate the exact V3 8x25 pilot queue contract without inferring truth."""

    raw_candidates = queue.get("candidates")
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    candidate_groups = [
        str(row.get("group_id") or "") for row in candidates if isinstance(row, dict)
    ]
    candidate_ids = [
        str(row.get("candidate_id") or "") for row in candidates if isinstance(row, dict)
    ]
    reserved = queue.get("reserved_group_ids")
    reserved_groups = reserved if isinstance(reserved, list) else []
    enabled = queue.get("enabled_classes")
    enabled_classes = enabled if isinstance(enabled, list) else []
    declared_by_class = queue.get("queued_cells_by_class")
    declared_by_class = declared_by_class if isinstance(declared_by_class, dict) else {}

    class_counts: Counter[str] = Counter()
    ordinals: defaultdict[str, list[int]] = defaultdict(list)
    structural_rows_ok = True
    outcome_states_ok = True
    candidate_ids_ok = True

    for row in candidates:
        if not isinstance(row, dict):
            structural_rows_ok = False
            outcome_states_ok = False
            candidate_ids_ok = False
            continue
        class_name = str(row.get("class_name") or "")
        class_counts[class_name] += 1
        try:
            class_index = int(row.get("class_index"))
            ordinal = int(row.get("queue_ordinal_within_class"))
        except (TypeError, ValueError):
            structural_rows_ok = False
            candidate_ids_ok = False
            continue
        ordinals[class_name].append(ordinal)
        expected_index = QUEUE_CLASS_INDEX.get(class_name, -1)
        structural_rows_ok = structural_rows_ok and (
            row.get("schema") == QUEUE_SCHEMA
            and row.get("dataset_version") == DATASET_VERSION
            and row.get("partition_version") == PARTITION_VERSION
            and row.get("publication_manifest_sha256") == manifest_sha256
            and row.get("candidate_status") == "PENDING_REVIEW"
            and row.get("current_target_value") is None
            and row.get("negative_truth_claim") is False
            and row.get("role_at_queue_creation") == "TRAIN_UNLABELED"
            and class_index == expected_index
            and 1 <= ordinal <= QUEUE_EXPECTED_PER_CLASS
            and bool(str(row.get("contract_id") or ""))
            and bool(str(row.get("group_id") or ""))
        )
        outcome_states_ok = outcome_states_ok and (
            row.get("current_outcome_state") in QUEUE_ALLOWED_OUTCOME_STATES
        )
        expected_candidate_id = _queue_candidate_id(
            manifest_sha256,
            str(row.get("group_id") or ""),
            str(row.get("contract_id") or ""),
            class_index,
        )
        candidate_ids_ok = candidate_ids_ok and (
            row.get("candidate_id") == expected_candidate_id
        )

    expected_by_class = {
        name: QUEUE_EXPECTED_PER_CLASS for name in QUEUE_ENABLED_CLASSES
    }
    ordinals_exact = all(
        sorted(ordinals.get(name, []))
        == list(range(1, QUEUE_EXPECTED_PER_CLASS + 1))
        for name in QUEUE_ENABLED_CLASSES
    )

    return {
        "queue_schema_status_valid": queue.get("schema") == QUEUE_SCHEMA
        and queue.get("status") == QUEUE_STATUS,
        "queue_expected_size": len(candidates) == QUEUE_EXPECTED_CELLS
        and queue.get("queued_cells") == QUEUE_EXPECTED_CELLS,
        "queue_enabled_classes_exact": enabled_classes == list(QUEUE_ENABLED_CLASSES)
        and queue.get("requested_per_enabled_class") == QUEUE_EXPECTED_PER_CLASS
        and queue.get("eligible_roles") == ["TRAIN_UNLABELED"],
        "queue_class_balance_exact": dict(class_counts) == expected_by_class
        and declared_by_class == expected_by_class
        and ordinals_exact,
        "queue_groups_globally_unique": len(candidate_groups)
        == len(set(candidate_groups))
        == QUEUE_EXPECTED_CELLS,
        "queue_reserved_groups_match_candidates": len(reserved_groups)
        == QUEUE_EXPECTED_CELLS
        and len(set(reserved_groups)) == QUEUE_EXPECTED_CELLS
        and set(reserved_groups) == set(candidate_groups),
        "queue_candidate_ids_valid_unique": candidate_ids_ok
        and len(candidate_ids) == QUEUE_EXPECTED_CELLS
        and len(set(candidate_ids)) == QUEUE_EXPECTED_CELLS
        and all(candidate_ids),
        "queue_candidates_pending_unknown": structural_rows_ok,
        "queue_candidate_outcomes_allowed": outcome_states_ok,
    }


def validate_snapshot_coherence(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    partition: dict[str, Any],
    binding: dict[str, Any],
    binding_sha256: str,
    summary: dict[str, Any],
    acceptance: dict[str, Any],
    grouping_audit: dict[str, Any],
    sensitivity: dict[str, Any],
    sensitivity_sha256: str,
    queue: dict[str, Any],
    selector: dict[str, Any],
    selector_sha256: str,
    gpu: dict[str, Any],
    current_source_commit: str,
) -> dict[str, Any]:
    """Fail closed unless every decision-critical report binds to one V3 lineage."""

    rep_meta = manifest.get("representation_binding_report") or {}
    rep_digest = str(rep_meta.get("binding_digest_sha256") or "")
    sensitivity_lineage = sensitivity.get("lineage") or {}
    selector_lineage = selector.get("lineage") or {}
    acceptance_lineage = acceptance.get("lineage") or {}
    gpu_scope = gpu.get("runtime_scope") or {}
    report_source_commits = {
        "acceptance": acceptance_lineage.get("source_commit"),
        "sensitivity": sensitivity_lineage.get("source_commit"),
        "selector": selector_lineage.get("source_commit"),
        "queue": queue.get("source_commit"),
        "gpu": gpu.get("source_commit"),
    }

    checks = {
        "manifest_dataset_v3": manifest.get("dataset_version") == DATASET_VERSION,
        "manifest_grouping_v3": manifest.get("grouping_version") == GROUPING_VERSION,
        "manifest_partition_v3": manifest.get("partition_version") == PARTITION_VERSION,
        "manifest_physically_bound": manifest.get("status") == BOUND_STATUS,
        "manifest_confirmed_negatives_zero": manifest.get("confirmed_negative_rows") == 0,
        "manifest_address_authority_false": manifest.get("address_literal_grouping_authority") is False,
        "binding_manifest_sha_matches_file": rep_meta.get("sha256") == binding_sha256,
        "binding_passed": binding.get("passed") is True,
        "binding_dataset_v3": binding.get("dataset_version") == DATASET_VERSION,
        "binding_grouping_v3": binding.get("logical_grouping_version") == GROUPING_VERSION,
        "binding_partition_v3": binding.get("logical_partition_version") == PARTITION_VERSION,
        "binding_digest_matches_manifest": bool(rep_digest)
        and binding.get("binding_digest_sha256") == rep_digest,
        "partition_version_v3": partition.get("partition_version") == PARTITION_VERSION,
        "partition_grouping_v3": partition.get("grouping_version") == GROUPING_VERSION,
        "partition_confirmed_negatives_zero": partition.get("confirmed_negative_rows") == 0,
        "partition_role_counts_match_manifest": partition.get("role_contract_counts")
        == manifest.get("role_contract_counts"),
        "summary_versions_match": (summary.get("versions") or {}).get("dataset") == DATASET_VERSION
        and (summary.get("versions") or {}).get("grouping") == GROUPING_VERSION
        and (summary.get("versions") or {}).get("partition") == PARTITION_VERSION,
        "summary_binding_matches": (summary.get("physical_binding") or {}).get(
            "binding_digest_sha256"
        )
        == rep_digest,
        "summary_binding_passed": (summary.get("physical_binding") or {}).get("passed") is True,
        "summary_semantics_preserved": (summary.get("semantic_invariants") or {}).get(
            "target_counts_unchanged_from_v2"
        )
        is True
        and (summary.get("semantic_invariants") or {}).get(
            "training_strength_counts_unchanged_from_v2"
        )
        is True
        and (summary.get("semantic_invariants") or {}).get("confirmed_negative_rows") == 0,
        "acceptance_pass": acceptance.get("status") == "PASS"
        and _all_true(acceptance.get("checks")),
        "acceptance_manifest_matches": acceptance_lineage.get("publication_manifest_sha256")
        == manifest_sha256,
        "acceptance_binding_matches": acceptance_lineage.get(
            "representation_binding_digest_sha256"
        )
        == rep_digest,
        "acceptance_versions_match": acceptance_lineage.get("dataset_version") == DATASET_VERSION
        and acceptance_lineage.get("grouping_version") == GROUPING_VERSION
        and acceptance_lineage.get("partition_version") == PARTITION_VERSION,
        "acceptance_training_unauthorized": acceptance.get("training_authorized") is False,
        "grouping_audit_v3_policy_pass": (grouping_audit.get("v3_policy_check") or {}).get("passed")
        is True,
        "grouping_audit_address_edges_zero": (grouping_audit.get("v3_policy_check") or {}).get(
            "address_edge_count"
        )
        == 0,
        "sensitivity_manifest_matches": sensitivity_lineage.get("publication_manifest_sha256")
        == manifest_sha256,
        "sensitivity_binding_matches": sensitivity_lineage.get(
            "representation_binding_digest_sha256"
        )
        == rep_digest,
        "sensitivity_versions_match": sensitivity_lineage.get("dataset_version") == DATASET_VERSION
        and sensitivity_lineage.get("grouping_version") == GROUPING_VERSION
        and sensitivity_lineage.get("partition_version") == PARTITION_VERSION,
        "sensitivity_training_unauthorized": sensitivity.get("full_training_authorized") is False,
        "queue_manifest_matches": queue.get("publication_manifest_sha256") == manifest_sha256,
        "queue_versions_match": queue.get("dataset_version") == DATASET_VERSION
        and queue.get("partition_version") == PARTITION_VERSION,
        "queue_not_negative_truth": queue.get("negative_truth_claim") is False,
        "queue_global_group_uniqueness_declared": queue.get("group_uniqueness_scope")
        == "GLOBAL_ACROSS_ENABLED_CLASSES",
        **validate_queue_coherence(queue, manifest_sha256=manifest_sha256),
        "selector_manifest_matches": selector_lineage.get("publication_manifest_sha256")
        == manifest_sha256,
        "selector_binding_matches": selector_lineage.get("representation_binding_digest_sha256")
        == rep_digest,
        "selector_versions_match": selector_lineage.get("dataset_version") == DATASET_VERSION
        and selector_lineage.get("grouping_version") == GROUPING_VERSION
        and selector_lineage.get("partition_version") == PARTITION_VERSION,
        "selector_experiment_only": selector.get("experiment_only") is True
        and selector.get("promotion_authorized") is False
        and selector.get("changes_bound_representations") is False,
        "selector_complete_no_regressions": selector.get("failures_total") == 0
        and selector.get("guarded_target_coverage_regressed_records") == 0,
        "gpu_status_complete": gpu.get("status") == GPU_STATUS,
        "gpu_manifest_matches": gpu.get("publication_manifest_sha256") == manifest_sha256,
        "gpu_binding_matches": gpu.get("representation_binding_digest_sha256") == rep_digest,
        "gpu_versions_match": gpu.get("dataset_version") == DATASET_VERSION
        and gpu.get("grouping_version") == GROUPING_VERSION
        and gpu.get("partition_version") == PARTITION_VERSION,
        "gpu_sensitivity_hash_matches": gpu.get("sensitivity_report_sha256")
        == sensitivity_sha256,
        "gpu_identical_initialization": gpu.get("identical_initialization_verified") is True,
        "gpu_worst_case_probes_complete": int(gpu_scope.get("worst_case_probes_required", -1)) > 0
        and gpu_scope.get("worst_case_probes_completed")
        == gpu_scope.get("worst_case_probes_required"),
        "gpu_no_checkpoint_or_run12": gpu_scope.get("checkpoint_written") is False
        and gpu_scope.get("run12_weights_loaded") is False,
        "gpu_training_and_promotion_unauthorized": gpu.get("full_training_authorized") is False
        and gpu.get("selector_promotion_authorized") is False,
        "research_source_commit_present": bool(current_source_commit)
        and all(bool(value) for value in report_source_commits.values()),
        "research_source_commit_consistent": all(
            value == current_source_commit for value in report_source_commits.values()
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema": "sentinel-r4-logical-v3-snapshot-coherence-v1",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "lineage": {
            "dataset_version": DATASET_VERSION,
            "grouping_version": GROUPING_VERSION,
            "partition_version": PARTITION_VERSION,
            "publication_manifest_sha256": manifest_sha256,
            "representation_binding_digest_sha256": rep_digest,
            "representation_binding_report_sha256": binding_sha256,
            "representation_sensitivity_sha256": sensitivity_sha256,
            "bounded_selector_source_report_sha256": selector_sha256,
            "source_commit": current_source_commit,
            "report_source_commits": report_source_commits,
        },
        "decision_boundary": (
            "PASS means the snapshot inputs are mutually coherent and bound to one V3 "
            "publication, physical representation lineage, and hardened source commit. "
            "It does not create negative truth, promote the selector, or authorize full training."
        ),
    }
    if failures:
        raise ValueError(
            "logical-v3 evidence snapshot coherence failed: " + ", ".join(failures)
        )
    return report


def main() -> int:
    # Load and validate every source before creating the durable output directory.
    summary = _load(SOURCE_ROOT / "logical_v3_summary.json")
    acceptance = _load(SOURCE_ROOT / "logical_v3_acceptance.json")
    grouping_audit = _load(SOURCE_ROOT / "grouping_breadth_audit_v1.json")
    sensitivity_path = SOURCE_ROOT / "representation_sensitivity_v1.json"
    sensitivity = _load(sensitivity_path)
    queue = _load(SOURCE_ROOT / "confirmed_negative_review_queue_v1.json")
    gpu = _load(SOURCE_ROOT / "selector_gpu_compare_v1.json")
    selector_path = SOURCE_ROOT / LARGE_REPORT
    selector = _load(selector_path)

    manifest_path = PUBLICATION_ROOT / "manifest.json"
    partition_path = PUBLICATION_ROOT / "partition_manifest.json"
    binding_path = PUBLICATION_ROOT / "representation_binding_report.json"
    manifest = _load(manifest_path)
    partition = _load(partition_path)
    binding = _load(binding_path)

    coherence = validate_snapshot_coherence(
        manifest=manifest,
        manifest_sha256=_sha256(manifest_path),
        partition=partition,
        binding=binding,
        binding_sha256=_sha256(binding_path),
        summary=summary,
        acceptance=acceptance,
        grouping_audit=grouping_audit,
        sensitivity=sensitivity,
        sensitivity_sha256=_sha256(sensitivity_path),
        queue=queue,
        selector=selector,
        selector_sha256=_sha256(selector_path),
        gpu=gpu,
        current_source_commit=_source_commit(),
    )

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"logical-v3 snapshot already contains files: {OUTPUT_ROOT}; "
            "do not overwrite durable evidence"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    source_reports = {
        "logical_v3_summary.json": summary,
        "logical_v3_acceptance.json": acceptance,
        "grouping_breadth_audit_v1.json": grouping_audit,
        "representation_sensitivity_v1.json": sensitivity,
        "confirmed_negative_review_queue_v1.json": queue,
        "selector_gpu_compare_v1.json": gpu,
    }
    for name, value in source_reports.items():
        _write(OUTPUT_ROOT / name, value)

    publication_reports = {
        "logical_v3_manifest.json": manifest,
        "logical_v3_partition_manifest.json": partition,
        "logical_v3_representation_binding_report.json": binding,
    }
    for name, value in publication_reports.items():
        _write(OUTPUT_ROOT / name, value)

    large = json.loads(json.dumps(selector))
    records = large.pop("records", [])
    large["source_report_sha256"] = _sha256(selector_path)
    large["source_report_bytes"] = selector_path.stat().st_size
    large["records_omitted_from_git_snapshot"] = len(records)
    large["snapshot_scope"] = (
        "Decision-level top-level summary only; per-contract records remain "
        "local and are bound by source_report_sha256."
    )
    _write(OUTPUT_ROOT / LARGE_SUMMARY, large)
    _write(OUTPUT_ROOT / COHERENCE_REPORT, coherence)

    hashes = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(OUTPUT_ROOT.glob("*.json"))
    ]
    (OUTPUT_ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(hashes) + "\n", encoding="utf-8"
    )

    print(f"snapshot={OUTPUT_ROOT.relative_to(REPO_ROOT)}")
    print("coherence=PASS")
    for path in sorted(OUTPUT_ROOT.iterdir()):
        print(f"{path.name}\t{path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "validate_queue_coherence",
    "validate_snapshot_coherence",
]
