#!/usr/bin/env python3
"""Read-only acceptance profiler for the locally rebuilt Phase-8 repaired lineage.

Run after preprocessing, source claims, grouping, evidence ledger,
representations, publication, and physical binding.  The script compares actual
local outputs to the 2026-08-14 historical audit without turning expected
recoveries into facts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
PREPROCESSED_ROOT = DATA_ROOT / "sentinel-preprocessed-r4-v2"
REPRESENTATIONS_ROOT = DATA_ROOT / "representations-r4-v2"
BUILD_ROOT = DATA_ROOT / "r4-v2-build"
PUBLICATION_ROOT = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
ACTIVE_SOURCES = ("dive", "smartbugs_curated", "solidifi")

HISTORICAL = {
    "raw_manifest_records": {
        "dive": 22330,
        "smartbugs_curated": 143,
        "solidifi": 350,
        "total": 22823,
    },
    "vnext_contracts": 22493,
    "represented_contracts": 21657,
    "strong_semantic_cells": 403,
    "weak_semantic_cells": 604,
    "effective_loss_cells": 852,
    "outcome_metric_cells": 118,
    "address_only_positive_drops": 65,
    "solc_049_valid_wrapper_drop": 1,
    "recoverable_direct_smartbugs_timestamp_cells": 5,
    "potential_additional_strong_cells_before_refreeze": 71,
    "normalized_code_groups": 120,
    "normalized_code_group_records": 288,
    "groups_split_across_historical_group_ids": 10,
    "represented_over_four_windows": 18491,
    "optimizer_cells_over_four_windows": 612,
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_dropped(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_summary(source: str) -> dict[str, Any]:
    directory = PREPROCESSED_ROOT / source
    manifest = _read_json(directory / "repaired_preprocessing_manifest.json")
    dropped = _read_dropped(directory / "dropped.csv")
    rep_manifest = _read_json(
        REPRESENTATIONS_ROOT / source / "repaired_representation_manifest.json"
    )
    return {
        "preprocessing_manifest_present": manifest is not None,
        "records_prepared": (manifest or {}).get("records_prepared"),
        "records_dropped": (manifest or {}).get("records_dropped"),
        "manifest_records_total": (manifest or {}).get("manifest_records_total"),
        "records_requested": (manifest or {}).get("records_requested"),
        "complete_source_build": (manifest or {}).get("complete_source_build"),
        "raw_manifest_verification_passed": (manifest or {}).get(
            "raw_manifest_verification_passed"
        ),
        "artifacts_written": (manifest or {}).get("artifacts_written"),
        "exact_normalized_duplicates_aggregated": (manifest or {}).get(
            "exact_normalized_duplicates_aggregated"
        ),
        "normalized_code_groups": len(
            (manifest or {}).get("normalized_code_groups_requiring_group_atomic_roles")
            or {}
        ),
        "address_candidate_groups": len(
            (manifest or {}).get("address_family_candidates_not_auto_merged") or {}
        ),
        "drop_reasons": dict(sorted(Counter(row.get("reason") for row in dropped).items())),
        "address_duplicate_drop_rows": sum(
            "address" in str(row.get("reason") or "").lower() for row in dropped
        ),
        "representation_manifest_present": rep_manifest is not None,
        "representations_written": (rep_manifest or {}).get("representations_written"),
        "representations_failed": (rep_manifest or {}).get("representations_failed"),
        "complete_source_build_verified_for_representation": (rep_manifest or {}).get(
            "complete_source_build_verified"
        ),
        "representation_preprocessing_manifest_sha256": (rep_manifest or {}).get(
            "preprocessing_manifest_sha256"
        ),
        "preprocessing_manifest_sha256": _sha256(
            directory / "repaired_preprocessing_manifest.json"
        ),
    }


def main() -> int:
    claims = _read_jsonl(BUILD_ROOT / "source_claims.jsonl")
    grouping = _read_json(BUILD_ROOT / "grouping.json") or {}
    ledger = _read_json(BUILD_ROOT / "evidence_ledger_v2_manifest.json") or {}
    publication = _read_json(PUBLICATION_ROOT / "manifest.json") or {}
    binding = _read_json(PUBLICATION_ROOT / "representation_binding_report.json") or {}
    partition = _read_json(PUBLICATION_ROOT / "partition_manifest.json") or {}
    publication_manifest_path = PUBLICATION_ROOT / "manifest.json"
    binding_path = PUBLICATION_ROOT / "representation_binding_report.json"

    source_summaries = {source: _source_summary(source) for source in ACTIVE_SOURCES}
    claim_strengths = Counter(str(row.get("training_strength")) for row in claims)
    smartbugs_time = [
        row
        for row in claims
        if row.get("source") == "smartbugs_curated"
        and row.get("native_category") == "time_manipulation"
        and row.get("mapped_class_name") == "Timestamp"
        and row.get("training_strength") == "STRONG"
        and row.get("target_value") == 1
    ]
    smartbugs_bad_randomness_targeted = [
        row
        for row in claims
        if row.get("source") == "smartbugs_curated"
        and row.get("native_category") == "bad_randomness"
        and row.get("target_value") is not None
    ]
    address_deletion = sum(
        int(summary["address_duplicate_drop_rows"] or 0)
        for summary in source_summaries.values()
    )
    expected_raw_counts = HISTORICAL["raw_manifest_records"]
    complete_sources = all(
        summary.get("complete_source_build") is True
        and summary.get("raw_manifest_verification_passed") is True
        and summary.get("records_requested") == expected_raw_counts[source]
        and summary.get("manifest_records_total") == expected_raw_counts[source]
        and (summary.get("records_prepared") or 0)
        + (summary.get("records_dropped") or 0)
        == expected_raw_counts[source]
        for source, summary in source_summaries.items()
    )
    representation_source_bindings = all(
        summary.get("complete_source_build_verified_for_representation") is True
        and summary.get("representation_preprocessing_manifest_sha256")
        == summary.get("preprocessing_manifest_sha256")
        for summary in source_summaries.values()
    )
    publication_artifacts = publication.get("artifacts") or {}
    ledger_artifacts = ledger.get("artifacts") or {}
    ledger_sha = (ledger_artifacts.get("ledger") or {}).get("sha256")
    ledger_manifest_sha = _sha256(BUILD_ROOT / "evidence_ledger_v2_manifest.json")
    ledger_bound = bool(
        ledger_sha
        and ledger_manifest_sha
        and (publication_artifacts.get("evidence_ledger") or {}).get("sha256")
        == ledger_sha
        and (publication_artifacts.get("evidence_ledger_manifest") or {}).get(
            "sha256"
        )
        == ledger_manifest_sha
    )
    binding_meta = publication.get("representation_binding_report") or {}
    binding_bound = (
        binding.get("passed") is True
        and binding_meta.get("sha256") == _sha256(binding_path)
        and binding_meta.get("binding_digest_sha256")
        == binding.get("binding_digest_sha256")
    )
    role_coverage = partition.get("strong_group_coverage_by_role_and_class") or {}
    enabled_role_coverage = all(
        role_coverage.get(role)
        and all(int(count) >= 1 for count in role_coverage[role].values())
        for role in ("TRAIN_STRONG", "MODEL_SELECTION", "INTERNAL_AUDIT")
    )

    checks = {
        "all_preprocessing_manifests_present": all(
            summary["preprocessing_manifest_present"]
            for summary in source_summaries.values()
        ),
        "all_representation_manifests_present": all(
            summary["representation_manifest_present"]
            for summary in source_summaries.values()
        ),
        "all_preprocessing_sources_complete_and_reconciled": complete_sources,
        "representations_bound_to_complete_preprocessing": representation_source_bindings,
        "no_address_based_deletion": address_deletion == 0,
        "source_claim_index_present": bool(claims),
        "no_source_claim_target_zero": all(row.get("target_value") != 0 for row in claims),
        "smartbugs_bad_randomness_not_targeted": not smartbugs_bad_randomness_targeted,
        "grouping_present": bool(grouping.get("artifact_to_group")),
        "evidence_ledger_present": ledger.get("ledger_version") == "evidence-ledger-r4-v2",
        "evidence_ledger_no_confirmed_negatives": ledger.get("confirmed_negative_rows") == 0,
        "publication_is_repaired_v2": publication.get("dataset_version") == "sentinel-r4-vnext-v2",
        "publication_no_confirmed_negatives": publication.get("confirmed_negative_rows") == 0,
        "physical_binding_passed": binding.get("passed") is True,
        "physical_binding_hash_bound_to_publication": binding_bound,
        "evidence_ledger_hash_bound_to_publication": ledger_bound,
        "enabled_classes_have_strong_coverage_in_all_evaluation_roles": enabled_role_coverage,
        "representation_failures_zero": all(
            summary.get("representations_failed") == 0
            for summary in source_summaries.values()
        ),
        "ledger_publication_contract_count_match": isinstance(
            ledger.get("contracts"), int
        )
        and ledger.get("contracts")
        == (publication.get("population") or {}).get("contracts"),
        "ledger_publication_target_counts_match": isinstance(
            ledger.get("target_counts"), dict
        )
        and ledger.get("target_counts") == publication.get("target_counts"),
        "ledger_publication_strength_counts_match": isinstance(
            ledger.get("training_strength_counts"), dict
        )
        and ledger.get("training_strength_counts")
        == publication.get("training_strength_counts"),
    }
    repository_data_acceptance_passed = all(checks.values())

    repaired_contracts = (publication.get("population") or {}).get("contracts")
    represented = binding.get("checked_contracts")
    strong_cells = (publication.get("training_strength_counts") or {}).get("STRONG")
    weak_cells = (publication.get("training_strength_counts") or {}).get("WEAK")
    result = {
        "schema": "sentinel-r4-phase8-repaired-lineage-audit-v1",
        "repository_data_acceptance_passed": repository_data_acceptance_passed,
        "training_authorized": False,
        "publication_manifest_sha256": _sha256(publication_manifest_path),
        "representation_binding_digest_sha256": binding.get(
            "binding_digest_sha256"
        ),
        "representation_binding_report_sha256": _sha256(binding_path),
        "checks": checks,
        "historical_2026_08_14_baseline": HISTORICAL,
        "actual_repaired_observations": {
            "sources": source_summaries,
            "source_claim_rows": len(claims),
            "source_claim_strength_counts": dict(sorted(claim_strengths.items())),
            "direct_smartbugs_time_manipulation_strong_claims": len(smartbugs_time),
            "smartbugs_bad_randomness_targeted_claims": len(
                smartbugs_bad_randomness_targeted
            ),
            "contract_identities": repaired_contracts,
            "represented_required_contracts": represented,
            "strong_semantic_cells": strong_cells,
            "weak_semantic_cells": weak_cells,
            "effective_loss_cells": publication.get("effective_loss_cells"),
            "outcome_metric_cells": publication.get("outcome_metric_cells"),
            "group_count": len(grouping.get("groups") or []),
  "role_contract_counts": publication.get("role_contract_counts") or {},
            "cross_source_exact_identities": len(
                grouping.get("cross_source_exact_identities") or {}
            ),
            "binding_digest_sha256": binding.get("binding_digest_sha256"),
            "token_coverage": binding.get("token_coverage"),
            "graph_population": binding.get("graph_population"),
        },
        "deltas_vs_historical": {
            "contract_identities": (
                repaired_contracts - HISTORICAL["vnext_contracts"]
                if isinstance(repaired_contracts, int)
                else None
            ),
            "represented_required_contracts": (
                represented - HISTORICAL["represented_contracts"]
                if isinstance(represented, int)
                else None
            ),
            "strong_semantic_cells": (
                strong_cells - HISTORICAL["strong_semantic_cells"]
                if isinstance(strong_cells, int)
                else None
            ),
            "weak_semantic_cells": (
                weak_cells - HISTORICAL["weak_semantic_cells"]
                if isinstance(weak_cells, int)
                else None
            ),
        },
        "hypotheses_to_compare_not_pass_criteria": {
            "address_only_positive_records_previously_lost": 65,
            "valid_solc_0_4_9_record_previously_lost": 1,
            "direct_smartbugs_timestamp_cells_expected_from_bound_provenance": 5,
            "maximum_identified_strong_cell_gain_before_representation_and_refreeze": 71,
            "note": "Actual repaired counts may differ after exact-content aggregation, normalization/compile outcomes, cross-source identity aggregation, final leakage grouping, representation success, and role re-freezing."
        },
        "next_required_gate": (
            "bounded repaired-data GPU micro-smoke and explicit launch reauthorization"
            if repository_data_acceptance_passed
            else "resolve failed physical DATA acceptance checks"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if repository_data_acceptance_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
