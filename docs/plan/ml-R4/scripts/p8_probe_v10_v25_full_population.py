#!/usr/bin/env python3
"""Validate full-population V10 V2.5 structural evidence and three repeats.

Only duplicate-safe source groups with stable expression-level persistent-write
proof are canonicalized.  Every graph must then satisfy exact labelled directed
multigraph isomorphism through unchanged edge type 10.  Missing or unstable
evidence remains blocking, and this diagnostic never grants acceptance or
training authority.
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from sentinel_data.representation.graph_schema import NODE_TYPES

from p8_collect_v10_v25_full_population_write_evidence import SCHEMA as EVIDENCE_SCHEMA
from p8_generate_v10_v25_structural_repeat import SCHEMA as REPEAT_SCHEMA
from p8_probe_v10_structural_drift import (
    PRIMARY_SLITHER_VERSION,
    _identity_key,
    _load_graph,
    _sha256,
    _unexpected_identities,
    compare_graphs,
)


SCHEMA = "sentinel-r4-v10-v26-full-population-probe-v2"
WRITE_TYPE = "CFG_NODE_WRITE"
WRITE_TYPE_ID = int(NODE_TYPES[WRITE_TYPE])
MAX_NODE_TYPE_ID = float(max(NODE_TYPES.values()))


def _evidence_projection(report: dict[str, Any]) -> Any:
    return {
        "unexpected_identities": report.get("unexpected_identities"),
        "contracts_with_write_drift": report.get("contracts_with_write_drift"),
        "target_groups": report.get("target_groups"),
        "duplicate_target_groups": report.get("duplicate_target_groups"),
        "storage_mutation_groups_proven": report.get(
            "storage_mutation_groups_proven"
        ),
        "unresolved_write_groups": report.get("unresolved_write_groups"),
        "non_write_or_population_drift": report.get(
            "non_write_or_population_drift"
        ),
        "contracts": report.get("contracts"),
    }


def _validated_evidence(
    reports: list[dict[str, Any]],
    *,
    audit_sha256: str,
    binding_digest: str,
) -> tuple[dict[str, list[dict[str, Any]]], bool, list[str]]:
    errors: list[str] = []
    for index, report in enumerate(reports, start=1):
        if report.get("schema") != EVIDENCE_SCHEMA:
            errors.append(f"evidence_{index}_schema")
        if report.get("source_audit_sha256") != audit_sha256:
            errors.append(f"evidence_{index}_audit_binding")
        if report.get("candidate_binding_digest_sha256") != binding_digest:
            errors.append(f"evidence_{index}_candidate_binding")
        if report.get("slither_analyzer") != PRIMARY_SLITHER_VERSION:
            errors.append(f"evidence_{index}_runtime")

    stable = all(
        _evidence_projection(report) == _evidence_projection(reports[0])
        for report in reports[1:]
    )
    if not stable:
        errors.append("semantic_evidence_not_stable_across_three_repeats")

    targets: dict[str, list[dict[str, Any]]] = {}
    for contract in reports[0].get("contracts") or []:
        logical = str(contract.get("contract") or "")
        rows = []
        for group in contract.get("target_groups") or []:
            if group.get("write_proven") is True:
                rows.append(group)
        if rows:
            targets[logical] = rows
    if reports[0].get("unresolved_write_groups"):
        errors.append("unresolved_storage_mutation_groups")
    if reports[0].get("non_write_or_population_drift"):
        errors.append("non_write_or_population_drift")
    return targets, stable, errors


def _canonicalize(graph: Any, logical: str, targets: list[dict[str, Any]]) -> Any:
    result = copy.deepcopy(graph)
    metadata = list(getattr(result, "node_metadata", None) or [])
    groups: dict[tuple[str, tuple[int, ...], str], list[int]] = {}
    for index, row in enumerate(metadata):
        groups.setdefault(_identity_key(row), []).append(index)

    for target in targets:
        key = (
            str(target.get("name") or ""),
            tuple(int(line) for line in target.get("source_lines") or []),
            str(target.get("coarse_type") or ""),
        )
        indices = groups.get(key) or []
        expected = int(target.get("candidate_multiplicity", -1))
        reference_expected = int(target.get("reference_multiplicity", -1))
        if len(indices) not in {expected, reference_expected}:
            raise ValueError(
                f"{logical} target {key!r} resolves to {len(indices)} nodes; "
                f"expected {reference_expected} or {expected}"
            )
        for index in indices:
            metadata[index] = dict(metadata[index])
            metadata[index]["type"] = WRITE_TYPE
            result.x[index, 0] = WRITE_TYPE_ID / MAX_NODE_TYPE_ID
    result.node_metadata = metadata
    return result


def _validate_repeat_reports(
    reports: list[dict[str, Any]],
    *,
    identities: list[str],
    audit_sha256: str,
    binding_digest: str,
) -> list[str]:
    errors: list[str] = []
    expected = set(identities)
    for index, report in enumerate(reports, start=1):
        if report.get("schema") != REPEAT_SCHEMA:
            errors.append(f"repeat_{index}_schema")
        if report.get("passed") is not True:
            errors.append(f"repeat_{index}_failed")
        if report.get("source_audit_sha256") != audit_sha256:
            errors.append(f"repeat_{index}_audit_binding")
        if report.get("candidate_binding_digest_sha256") != binding_digest:
            errors.append(f"repeat_{index}_candidate_binding")
        if (report.get("runtime") or {}).get("slither_analyzer") != PRIMARY_SLITHER_VERSION:
            errors.append(f"repeat_{index}_runtime")
        observed = {str(row.get("contract") or "") for row in report.get("records") or []}
        if observed != expected:
            errors.append(f"repeat_{index}_population")
    return errors


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.repeat_root) != 3 or len(args.repeat_report) != 3:
        raise ValueError("exactly three repeat roots and reports are required")
    if len(args.semantic_evidence) != 3:
        raise ValueError("exactly three semantic-evidence reports are required")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    audit_sha256 = _sha256(args.audit)
    binding_digest = str(audit.get("candidate_binding_digest_sha256") or "")
    identities = _unexpected_identities(audit)
    repeat_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.repeat_report]
    evidence_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.semantic_evidence]

    errors = _validate_repeat_reports(
        repeat_reports,
        identities=identities,
        audit_sha256=audit_sha256,
        binding_digest=binding_digest,
    )
    targets, evidence_stable, evidence_errors = _validated_evidence(
        evidence_reports,
        audit_sha256=audit_sha256,
        binding_digest=binding_digest,
    )
    errors.extend(evidence_errors)

    contracts: list[dict[str, Any]] = []
    blockers: list[str] = []
    decision_counts: dict[str, int] = {}
    for logical in identities:
        reference = _load_graph(args.reference_root, logical, require_primary_runtime=False)
        candidate = _load_graph(args.candidate_root, logical, require_primary_runtime=True)
        repeats = [
            _load_graph(root, logical, require_primary_runtime=True)
            for root in args.repeat_root
        ]
        target_rows = targets.get(logical, [])
        graphs = {
            "reference": _canonicalize(reference.graph, logical, target_rows),
            "candidate": _canonicalize(candidate.graph, logical, target_rows),
            **{
                f"repeat_{index}": _canonicalize(artifact.graph, logical, target_rows)
                for index, artifact in enumerate(repeats, start=1)
            },
        }

        comparisons: dict[str, Any] = {}
        names = list(graphs)
        for left_index, right_index in combinations(range(len(names)), 2):
            left_name, right_name = names[left_index], names[right_index]
            comparisons[f"{left_name}__vs__{right_name}"] = compare_graphs(
                graphs[left_name],
                graphs[right_name],
                max_search_states=args.max_search_states,
            )
        passed = all(
            row["exact_node_index_invariant_equivalent"] is True
            for row in comparisons.values()
        )
        if passed and target_rows:
            decision = "PROVEN_DUPLICATE_SAFE_STORAGE_WRITE_CORRECTION"
        elif passed:
            decision = "PROVEN_EXACT_NODE_INDEX_INVARIANT_EQUIVALENCE"
        else:
            decision = "UNRESOLVED_STRUCTURAL_OR_SEMANTIC_DRIFT"
            blockers.append(logical)
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        contracts.append(
            {
                "contract": logical,
                "decision": decision,
                "passed": passed,
                "canonicalized_write_groups": len(target_rows),
                "canonicalized_write_occurrences": sum(
                    int(row["candidate_multiplicity"]) for row in target_rows
                ),
                "comparisons": comparisons,
            }
        )

    if blockers:
        errors.append("unresolved_graph_drift")
    passed = not errors and not blockers
    return {
        "schema": SCHEMA,
        "passed": passed,
        "status": (
            "FULL_POPULATION_STRUCTURAL_EVIDENCE_PASS"
            if passed
            else "FULL_POPULATION_STRUCTURAL_EVIDENCE_BLOCKED"
        ),
        "source_audit_sha256": audit_sha256,
        "candidate_binding_digest_sha256": binding_digest,
        "unexpected_identities": len(identities),
        "repeat_generations": 3,
        "semantic_evidence_repeats": 3,
        "semantic_evidence_stable": evidence_stable,
        "proven_write_contracts": len(targets),
        "decision_counts": dict(sorted(decision_counts.items())),
        "blocking_identities": blockers,
        "errors": sorted(set(errors)),
        "zero_unexplained_drift": passed,
        "contracts": contracts,
        "physical_acceptance": False,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--repeat-root", action="append", type=Path, default=[])
    parser.add_argument("--repeat-report", action="append", type=Path, default=[])
    parser.add_argument("--semantic-evidence", action="append", type=Path, default=[])
    parser.add_argument("--max-search-states", type=int, default=200_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_search_states < 1:
        raise ValueError("max search states must be >= 1")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
