#!/usr/bin/env python3
"""Audit V9 -> V10 V2.6 with full-population structural evidence.

V4 preserves every population, token, version, runtime, call-edge, and binding
check from the V2 transition audit.  It then validates the passed 355-identity
V2.6 evidence chain and independently re-proves its two admissible classes
against the actual full candidate:

* exact node-index-invariant labelled graph equivalence; and
* deterministic persistent-storage WRITE corrections backed by the stable
  three-run semantic evidence.

The bounded evidence selects the admissible class; it never waives a check on
the full candidate.  This diagnostic cannot grant physical acceptance or
authorize training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import p8_audit_v10_transition as audit_v2
import p8_probe_v10_v25_full_population as full_population
from p8_probe_v10_structural_drift import _load_graph, _sha256, compare_graphs


AUDIT_SCHEMA = "sentinel-r4-v9-to-v10-transition-audit-v4"
PROBE_SCHEMA = "sentinel-r4-v10-v26-full-population-probe-v2"
INDEX_DECISION = "PROVEN_EXACT_NODE_INDEX_INVARIANT_EQUIVALENCE"
WRITE_DECISION = "PROVEN_DUPLICATE_SAFE_STORAGE_WRITE_CORRECTION"
EXPECTED_IDENTITIES = 355
EXPECTED_INDEX_IDENTITIES = 6
EXPECTED_WRITE_IDENTITIES = 349
EXPECTED_REPEAT_GENERATIONS = 3
_INTERNAL_MAX_RECORDS = 1_000_000


def _validate_full_population_evidence(
    probe: dict[str, Any],
    evidence_reports: list[dict[str, Any]],
) -> tuple[set[str], set[str], dict[str, list[dict[str, Any]]]]:
    """Return exact index/write sets and stable semantic target groups."""

    if probe.get("schema") != PROBE_SCHEMA:
        raise ValueError("unexpected V2.6 full-population probe schema")
    if probe.get("passed") is not True:
        raise ValueError("V2.6 full-population probe did not pass")
    if probe.get("zero_unexplained_drift") is not True:
        raise ValueError("V2.6 full-population probe reports unexplained drift")
    if list(probe.get("blocking_identities") or []):
        raise ValueError("V2.6 full-population probe contains blocking identities")
    if probe.get("physical_acceptance") is not False:
        raise ValueError("V2.6 evidence unexpectedly claims physical acceptance")
    if probe.get("training_authorized") is not False:
        raise ValueError("V2.6 evidence unexpectedly authorizes training")
    if int(probe.get("unexpected_identities", -1)) != EXPECTED_IDENTITIES:
        raise ValueError("V2.6 evidence identity census mismatch")
    if int(probe.get("repeat_generations", -1)) != EXPECTED_REPEAT_GENERATIONS:
        raise ValueError("V2.6 repeat-generation census mismatch")
    if int(probe.get("semantic_evidence_repeats", -1)) != EXPECTED_REPEAT_GENERATIONS:
        raise ValueError("V2.6 semantic-evidence repeat census mismatch")
    if probe.get("semantic_evidence_stable") is not True:
        raise ValueError("V2.6 semantic evidence is not stable")

    expected_counts = {
        INDEX_DECISION: EXPECTED_INDEX_IDENTITIES,
        WRITE_DECISION: EXPECTED_WRITE_IDENTITIES,
    }
    if dict(probe.get("decision_counts") or {}) != expected_counts:
        raise ValueError("V2.6 evidence decision census mismatch")

    rows = list(probe.get("contracts") or [])
    if len(rows) != EXPECTED_IDENTITIES:
        raise ValueError("V2.6 evidence contract record count mismatch")
    index_identities: set[str] = set()
    write_identities: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        logical = str(row.get("contract") or "")
        if not logical or logical in seen:
            raise ValueError("V2.6 evidence contains empty or duplicate identity")
        seen.add(logical)
        if row.get("passed") is not True:
            raise ValueError(f"V2.6 evidence row is not passed: {logical}")
        decision = row.get("decision")
        if decision == INDEX_DECISION:
            index_identities.add(logical)
        elif decision == WRITE_DECISION:
            write_identities.add(logical)
        else:
            raise ValueError(f"V2.6 evidence has unapproved decision: {logical}")

    if len(index_identities) != EXPECTED_INDEX_IDENTITIES:
        raise ValueError("V2.6 index-equivalence census mismatch")
    if len(write_identities) != EXPECTED_WRITE_IDENTITIES:
        raise ValueError("V2.6 storage-WRITE census mismatch")
    if index_identities & write_identities:
        raise ValueError("V2.6 evidence classes overlap")

    if len(evidence_reports) != EXPECTED_REPEAT_GENERATIONS:
        raise ValueError("exactly three semantic-evidence reports are required")
    targets, stable, errors = full_population._validated_evidence(
        evidence_reports,
        audit_sha256=str(probe.get("source_audit_sha256") or ""),
        binding_digest=str(probe.get("candidate_binding_digest_sha256") or ""),
    )
    if errors:
        raise ValueError(f"invalid V2.6 semantic evidence: {sorted(set(errors))}")
    if not stable:
        raise ValueError("V2.6 semantic evidence projection differs across repeats")
    if set(targets) != write_identities:
        raise ValueError("semantic target population does not match WRITE decisions")
    return index_identities, write_identities, targets


def _base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        accepted_v9_root=args.accepted_v9_root,
        candidate_root=args.candidate_root,
        reference_v10_root=args.reference_v10_root,
        preprocessed_root=args.preprocessed_root,
        output=args.output,
        progress_every=args.progress_every,
        max_errors=_INTERNAL_MAX_RECORDS,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    probe = json.loads(args.full_population_probe.read_text(encoding="utf-8"))
    evidence_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.semantic_evidence
    ]
    index_identities, write_identities, targets = (
        _validate_full_population_evidence(probe, evidence_reports)
    )

    base = audit_v2.build_report(_base_args(args))
    evidence_sha256 = [_sha256(path) for path in args.semantic_evidence]
    if base.get("passed") is not True:
        result = dict(base)
        result.update(
            {
                "schema": AUDIT_SCHEMA,
                "passed": False,
                "status": "FAIL_BASE_TRANSITION_MECHANICS",
                "raw_v2_status": base.get("status"),
                "full_population_probe_sha256": _sha256(args.full_population_probe),
                "semantic_evidence_sha256": evidence_sha256,
                "structural_reconciliation_passed": False,
                "structural_evidence_records": [],
                "structural_evidence_failures": [
                    "base V2 transition mechanics did not pass"
                ],
                "physical_acceptance": False,
                "training_authorized": False,
            }
        )
        return result

    raw_records = list(base.get("structural_drift_contracts") or [])
    raw_non_parse_only = {
        str(row["contract"])
        for row in raw_records
        if row.get("v9_parse_only") is False
    }
    historical_parse_only = {
        str(row["contract"])
        for row in raw_records
        if row.get("v9_parse_only") is True
    }
    approved = index_identities | write_identities
    evidence_failures: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []

    for logical in sorted(approved):
        try:
            reference = _load_graph(
                args.reference_v10_root, logical, require_primary_runtime=False
            )
            candidate = _load_graph(
                args.candidate_root, logical, require_primary_runtime=True
            )
            target_rows = targets.get(logical, [])
            canonical_reference = full_population._canonicalize(
                reference.graph, logical, target_rows
            )
            canonical_candidate = full_population._canonicalize(
                candidate.graph, logical, target_rows
            )
            comparison = compare_graphs(
                canonical_reference,
                canonical_candidate,
                max_search_states=args.max_search_states,
            )
            passed = comparison["exact_node_index_invariant_equivalent"] is True
            record = {
                "contract": logical,
                "evidence_class": (
                    "deterministic_persistent_storage_write_correction"
                    if logical in write_identities
                    else "node_order_index_equivalence"
                ),
                "passed": passed,
                "canonicalized_write_groups": len(target_rows),
                "canonicalized_write_occurrences": sum(
                    int(row.get("candidate_multiplicity", 0)) for row in target_rows
                ),
                "comparison": comparison,
            }
            evidence_records.append(record)
            if not passed:
                evidence_failures.append(
                    {
                        "contract": logical,
                        "detail": "full candidate failed its V2.6 evidence class",
                    }
                )
        except Exception as exc:
            evidence_records.append(
                {
                    "contract": logical,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            evidence_failures.append(
                {
                    "contract": logical,
                    "detail": f"reconciliation raised {type(exc).__name__}: {exc}",
                }
            )

    missing_from_raw = sorted(approved - raw_non_parse_only)
    unapproved_raw = sorted(raw_non_parse_only - approved)
    for logical in missing_from_raw:
        evidence_failures.append(
            {
                "contract": logical,
                "detail": "evidenced identity is absent from actual raw drift census",
            }
        )
    for logical in unapproved_raw:
        evidence_failures.append(
            {
                "contract": logical,
                "detail": "actual non-parse-only drift lacks V2.6 evidence",
            }
        )

    structural_passed = not evidence_failures
    passed = bool(base["passed"] and structural_passed)
    totals = dict(base.get("totals") or {})
    totals.update(
        {
            "graphs_with_raw_non_parse_only_structural_drift": len(raw_non_parse_only),
            "graphs_with_historical_v9_parse_only_structural_drift": len(
                historical_parse_only
            ),
            "graphs_with_proven_v26_index_equivalence": sum(
                row.get("passed") is True
                and row.get("evidence_class") == "node_order_index_equivalence"
                for row in evidence_records
            ),
            "graphs_with_proven_v26_storage_write_correction": sum(
                row.get("passed") is True
                and row.get("evidence_class")
                == "deterministic_persistent_storage_write_correction"
                for row in evidence_records
            ),
            "graphs_with_unexplained_non_parse_only_structural_drift": len(
                evidence_failures
            ),
        }
    )

    result = dict(base)
    result.update(
        {
            "schema": AUDIT_SCHEMA,
            "passed": passed,
            "status": (
                "PASS_TRANSITION_EVIDENCE_RECONCILED_PENDING_PHYSICAL_DECISION"
                if passed
                else "PASS_BASE_MECHANICS_WITH_STRUCTURAL_EVIDENCE_BLOCKER"
            ),
            "raw_v2_status": base.get("status"),
            "full_population_probe_schema": probe.get("schema"),
            "full_population_probe_sha256": _sha256(args.full_population_probe),
            "bounded_source_audit_sha256": probe.get("source_audit_sha256"),
            "bounded_source_candidate_binding_digest_sha256": probe.get(
                "candidate_binding_digest_sha256"
            ),
            "semantic_evidence_sha256": evidence_sha256,
            "bounded_index_equivalence_identities": len(index_identities),
            "bounded_storage_write_identities": len(write_identities),
            "historical_v9_parse_only_structural_drift_identities": len(
                historical_parse_only
            ),
            "raw_non_parse_only_structural_drift_identities": len(
                raw_non_parse_only
            ),
            "evidenced_identities_absent_from_raw_drift": missing_from_raw,
            "unapproved_raw_non_parse_only_structural_drift_identities": unapproved_raw,
            "structural_reconciliation_passed": structural_passed,
            "structural_evidence_records": evidence_records[: args.max_errors],
            "structural_evidence_records_truncated": max(
                0, len(evidence_records) - args.max_errors
            ),
            "structural_evidence_failures": evidence_failures[: args.max_errors],
            "structural_evidence_failures_truncated": max(
                0, len(evidence_failures) - args.max_errors
            ),
            "totals": dict(sorted(totals.items())),
            "physical_acceptance_blockers": (
                ["full candidate has structural drift not reconciled by V2.6 evidence"]
                if evidence_failures
                else []
            ),
            "physical_acceptance": False,
            "training_authorized": False,
            "remaining_stop_lines": [
                "explicit_review_of_complete_v4_transition_report",
                "explicit_physical_acceptance_decision_record",
                "training_authorization_remains_separate",
            ],
            "limitations": [
                "V4 reuses all V2 population, token-byte, version, call-IR, edge-count, runtime, and candidate-binding checks.",
                "The full-population evidence selects only the admissible class; all 355 identities are independently re-proven against this full candidate and its own binding digest.",
                "Historical accepted-V9 parse-only identities remain a separate expected repair class.",
                "Any additional non-parse-only structural difference remains blocking.",
                "This report never grants physical acceptance and never authorizes training.",
            ],
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted-v9-root",
        type=Path,
        default=Path("data_module/data/representations-r4-v2"),
    )
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-v10-root", type=Path, required=True)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data_module/data/sentinel-preprocessed-r4-v2"),
    )
    parser.add_argument("--full-population-probe", type=Path, required=True)
    parser.add_argument(
        "--semantic-evidence", action="append", type=Path, default=[]
    )
    parser.add_argument("--max-search-states", type=int, default=200_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-errors", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_errors < 1:
        raise ValueError("--max-errors must be >= 1")
    if args.max_search_states < 1:
        raise ValueError("--max-search-states must be >= 1")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
