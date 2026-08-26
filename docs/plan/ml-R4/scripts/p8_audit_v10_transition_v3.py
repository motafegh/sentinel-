#!/usr/bin/env python3
"""Audit the complete V9 -> V10 transition with bounded V2.5 evidence binding.

This V3 audit preserves the complete mechanical checks performed by
``p8_audit_v10_transition.py`` and adds a fail-closed reconciliation layer for
structural drift through unchanged edge type 10.

Only two non-parse-only structural evidence classes are admissible:

* exact node-index-invariant labelled graph equivalence, for identities already
  proven by the bounded V2.5 reproducibility report; and
* deterministic ``CFG_NODE_WRITE`` corrections, for identities/nodes backed by
  exact Slither-0.10 persistent-storage lvalue evidence and the same bounded
  V2.5 reproducibility report.

Every other non-parse-only difference remains blocking. Historical accepted-V9
parse-only identities remain a separate expected-repair class exactly as in the
V2 audit. This script never grants physical acceptance or authorizes training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import p8_audit_v10_transition as audit_v2
import p8_probe_v10_v25_reproducibility as v25
from p8_probe_v10_structural_drift import (
    PRIMARY_SLITHER_VERSION,
    _load_graph,
    _sha256,
    compare_graphs,
)
from sentinel_data.preprocessing.r4_versions import (
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)


BOUNDED_SCHEMA = "sentinel-r4-v10-v25-reproducibility-probe-v1"
AUDIT_SCHEMA = "sentinel-r4-v9-to-v10-transition-audit-v3"
INDEX_DECISION = "V25_NODE_ORDER_INDEX_EQUIVALENCE_REPRODUCED"
WRITE_DECISION = "V25_DETERMINISTIC_STORAGE_WRITE_CORRECTION_PROVEN"
EXPECTED_UNEXPECTED_IDENTITIES = 20
EXPECTED_INDEX_IDENTITIES = 8
EXPECTED_WRITE_IDENTITIES = 12
EXPECTED_REPEAT_GENERATIONS = 3
_INTERNAL_MAX_RECORDS = 1_000_000


def _validate_bounded_evidence(
    report: dict[str, Any],
    *,
    semantic_evidence_path: Path,
) -> tuple[set[str], set[str]]:
    """Validate and return the exact bounded index/write identity sets."""

    if report.get("schema") != BOUNDED_SCHEMA:
        raise ValueError("unexpected bounded V2.5 reproducibility schema")
    if report.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError("bounded V2.5 extractor identity mismatch")
    if report.get("slither_analyzer") != PRIMARY_SLITHER_VERSION:
        raise ValueError("bounded V2.5 evidence is not bound to exact Slither 0.10.0")
    if report.get("bounded_v25_reproducibility_passed") is not True:
        raise ValueError("bounded V2.5 reproducibility did not pass")
    if report.get("zero_unexplained_drift") is not True:
        raise ValueError("bounded V2.5 evidence still reports unexplained drift")
    if report.get("physical_acceptance") is not False:
        raise ValueError("bounded V2.5 evidence unexpectedly claims physical acceptance")
    if report.get("training_authorized") is not False:
        raise ValueError("bounded V2.5 evidence unexpectedly authorizes training")
    if list(report.get("blocking_identities") or []):
        raise ValueError("bounded V2.5 evidence still contains blocking identities")
    if int(report.get("unexpected_identities", -1)) != EXPECTED_UNEXPECTED_IDENTITIES:
        raise ValueError("bounded V2.5 unexpected-identity count mismatch")
    if int(report.get("index_equivalence_identities", -1)) != EXPECTED_INDEX_IDENTITIES:
        raise ValueError("bounded V2.5 index-equivalence count mismatch")
    if int(report.get("semantic_correction_identities", -1)) != EXPECTED_WRITE_IDENTITIES:
        raise ValueError("bounded V2.5 semantic-correction count mismatch")
    if int(report.get("repeat_generations", -1)) != EXPECTED_REPEAT_GENERATIONS:
        raise ValueError("bounded V2.5 repeat-generation count mismatch")

    semantic_sha = _sha256(semantic_evidence_path)
    if report.get("semantic_evidence_sha256") != semantic_sha:
        raise ValueError("bounded V2.5 report is not bound to the supplied semantic evidence")

    rows = list(report.get("contracts") or [])
    if len(rows) != EXPECTED_UNEXPECTED_IDENTITIES:
        raise ValueError("bounded V2.5 contract record count mismatch")

    index_identities: set[str] = set()
    write_identities: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        logical = str(row.get("contract") or "")
        if not logical or logical in seen:
            raise ValueError("bounded V2.5 contains empty or duplicate contract identity")
        seen.add(logical)
        if row.get("passed") is not True:
            raise ValueError(f"bounded V2.5 row is not passed: {logical}")
        decision = row.get("decision")
        if decision == INDEX_DECISION:
            index_identities.add(logical)
        elif decision == WRITE_DECISION:
            write_identities.add(logical)
        else:
            raise ValueError(
                f"bounded V2.5 row has an unapproved decision for {logical}: {decision!r}"
            )

    if len(index_identities) != EXPECTED_INDEX_IDENTITIES:
        raise ValueError("bounded V2.5 index decision census mismatch")
    if len(write_identities) != EXPECTED_WRITE_IDENTITIES:
        raise ValueError("bounded V2.5 WRITE decision census mismatch")
    if index_identities & write_identities:
        raise ValueError("bounded V2.5 evidence classes overlap")
    if len(index_identities | write_identities) != EXPECTED_UNEXPECTED_IDENTITIES:
        raise ValueError("bounded V2.5 evidence does not cover exactly 20 identities")

    return index_identities, write_identities


def _reconcile_index_equivalence(reference_graph: Any, candidate_graph: Any) -> dict[str, Any]:
    comparison = compare_graphs(reference_graph, candidate_graph)
    passed = comparison["exact_node_index_invariant_equivalent"] is True
    return {
        "evidence_class": "node_order_index_equivalence",
        "passed": passed,
        "comparison": comparison,
        "failures": [] if passed else [comparison["isomorphism_reason"]],
    }


def _reconcile_storage_write(
    reference_graph: Any,
    candidate_graph: Any,
    *,
    logical: str,
    targets: set[tuple[str, tuple[int, ...]]],
) -> dict[str, Any]:
    writes_passed, write_failures = v25._all_targets_are_write(
        candidate_graph,
        logical,
        targets,
    )
    canonical_reference = v25._canonicalize_expected_writes(
        reference_graph,
        logical,
        targets,
    )
    canonical_candidate = v25._canonicalize_expected_writes(
        candidate_graph,
        logical,
        targets,
    )
    comparison = compare_graphs(canonical_reference, canonical_candidate)
    equivalent = comparison["exact_node_index_invariant_equivalent"] is True
    passed = writes_passed and equivalent
    failures: list[Any] = []
    if write_failures:
        failures.append({"semantic_write_failures": write_failures})
    if not equivalent:
        failures.append(
            {
                "canonical_comparison_failure": comparison["isomorphism_reason"],
                "classification": comparison["classification"],
            }
        )
    return {
        "evidence_class": "deterministic_storage_write_correction",
        "passed": passed,
        "target_nodes": len(targets),
        "semantic_write_failures": write_failures,
        "canonical_comparison": comparison,
        "failures": failures,
    }


def _base_args(args: argparse.Namespace) -> SimpleNamespace:
    # Force complete structural recording internally. V3 applies its own output
    # truncation only after every drift identity has been reconciled.
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
    semantic_report = json.loads(args.semantic_evidence.read_text(encoding="utf-8"))
    semantic_targets = v25._semantic_targets(semantic_report)

    bounded = json.loads(args.bounded_v25_report.read_text(encoding="utf-8"))
    index_identities, write_identities = _validate_bounded_evidence(
        bounded,
        semantic_evidence_path=args.semantic_evidence,
    )
    if set(semantic_targets) != write_identities:
        raise ValueError(
            "semantic-evidence contract set does not exactly match bounded WRITE decisions"
        )

    base = audit_v2.build_report(_base_args(args))
    if base.get("passed") is not True:
        # Mechanics/population/version/call reconciliation must pass before
        # structural evidence can be considered.
        result = dict(base)
        result.update(
            {
                "schema": AUDIT_SCHEMA,
                "passed": False,
                "status": "FAIL_BASE_TRANSITION_MECHANICS",
                "raw_v2_status": base.get("status"),
                "bounded_v25_report_sha256": _sha256(args.bounded_v25_report),
                "semantic_evidence_sha256": _sha256(args.semantic_evidence),
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

    raw_structural_records = list(base.get("structural_drift_contracts") or [])
    raw_non_parse_only = {
        str(row["contract"])
        for row in raw_structural_records
        if row.get("v9_parse_only") is False
    }
    historical_parse_only = {
        str(row["contract"])
        for row in raw_structural_records
        if row.get("v9_parse_only") is True
    }

    evidence_records: list[dict[str, Any]] = []
    evidence_failures: list[dict[str, Any]] = []

    # Re-prove every bounded identity against the actual full candidate. The
    # bounded report is authority for which class may be used, never a waiver.
    for logical in sorted(index_identities | write_identities):
        try:
            reference = _load_graph(
                args.reference_v10_root,
                logical,
                require_primary_runtime=False,
            )
            candidate = _load_graph(
                args.candidate_root,
                logical,
                require_primary_runtime=True,
            )
            if logical in index_identities:
                reconciliation = _reconcile_index_equivalence(
                    reference.graph,
                    candidate.graph,
                )
            else:
                reconciliation = _reconcile_storage_write(
                    reference.graph,
                    candidate.graph,
                    logical=logical,
                    targets=semantic_targets[logical],
                )
            record = {
                "contract": logical,
                "bounded_decision": (
                    INDEX_DECISION if logical in index_identities else WRITE_DECISION
                ),
                **reconciliation,
            }
            evidence_records.append(record)
            if not reconciliation["passed"]:
                evidence_failures.append(
                    {
                        "contract": logical,
                        "detail": "full candidate failed its bounded evidence class",
                        "evidence_class": reconciliation["evidence_class"],
                    }
                )
        except Exception as exc:
            evidence_records.append(
                {
                    "contract": logical,
                    "bounded_decision": (
                        INDEX_DECISION if logical in index_identities else WRITE_DECISION
                    ),
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            evidence_failures.append(
                {
                    "contract": logical,
                    "detail": f"evidence reconciliation raised {type(exc).__name__}: {exc}",
                }
            )

    approved_non_parse_only = index_identities | write_identities
    unapproved_raw_drift = sorted(raw_non_parse_only - approved_non_parse_only)
    for logical in unapproved_raw_drift:
        evidence_failures.append(
            {
                "contract": logical,
                "detail": "non-parse-only structural drift is absent from bounded V2.5 evidence",
            }
        )

    structural_reconciliation_passed = not evidence_failures
    passed = bool(base["passed"] and structural_reconciliation_passed)

    totals = dict(base.get("totals") or {})
    totals.update(
        {
            "graphs_with_raw_non_parse_only_structural_drift": len(raw_non_parse_only),
            "graphs_with_historical_v9_parse_only_structural_drift": len(
                historical_parse_only
            ),
            "graphs_with_proven_v25_index_equivalence": sum(
                1
                for row in evidence_records
                if row.get("evidence_class") == "node_order_index_equivalence"
                and row.get("passed") is True
            ),
            "graphs_with_proven_v25_storage_write_correction": sum(
                1
                for row in evidence_records
                if row.get("evidence_class")
                == "deterministic_storage_write_correction"
                and row.get("passed") is True
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
            "bounded_v25_report_sha256": _sha256(args.bounded_v25_report),
            "bounded_v25_schema": bounded.get("schema"),
            "bounded_v25_repeat_generations": bounded.get("repeat_generations"),
            "semantic_evidence_sha256": _sha256(args.semantic_evidence),
            "semantic_evidence_source_reports": semantic_report.get("source_reports"),
            "bounded_index_equivalence_identities": len(index_identities),
            "bounded_storage_write_identities": len(write_identities),
            "historical_v9_parse_only_structural_drift_identities": len(
                historical_parse_only
            ),
            "raw_non_parse_only_structural_drift_identities": len(
                raw_non_parse_only
            ),
            "unapproved_raw_non_parse_only_structural_drift_identities": unapproved_raw_drift,
            "structural_reconciliation_passed": structural_reconciliation_passed,
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
                [
                    "full candidate contains non-parse-only structural drift that is not re-proven by the exact bounded V2.5 evidence classes"
                ]
                if evidence_failures
                else []
            ),
            "physical_acceptance": False,
            "training_authorized": False,
            "remaining_stop_lines": [
                "explicit_review_of_complete_v3_transition_report",
                "explicit_physical_acceptance_decision_record",
                "training_authorization_remains_separate",
            ],
            "limitations": [
                "V3 reuses all V2 population, token-byte, version, call-IR, edge-count, and binding checks before applying structural evidence reconciliation.",
                "Historical accepted-V9 parse-only identities remain an expected repair class; they are not silently counted as ordinary structural equivalence.",
                "The 8 node-order identities are accepted only after exact labelled directed-multigraph isomorphism through unchanged edge type 10 is re-proven against the actual full candidate.",
                "The 12 WRITE identities are accepted only after every evidenced target is CFG_NODE_WRITE and canonicalized reference/candidate graphs remain exactly node-index-invariant equivalent through unchanged edge type 10.",
                "Any other non-parse-only structural difference remains blocking.",
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
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("data_module/data/representations-r4-v3-candidate"),
    )
    parser.add_argument("--reference-v10-root", type=Path, required=True)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data_module/data/sentinel-preprocessed-r4-v2"),
    )
    parser.add_argument("--bounded-v25-report", type=Path, required=True)
    parser.add_argument("--semantic-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-errors", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_errors < 1:
        raise ValueError("--max-errors must be >= 1")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
