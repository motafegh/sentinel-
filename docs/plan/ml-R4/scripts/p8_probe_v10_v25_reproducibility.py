#!/usr/bin/env python3
"""Verify bounded V10 v2.5 reproducibility after CFG WRITE semantic resolution.

This diagnostic consumes three fresh V2.5 regenerations of the 20 identities
already reported as unexpected structural drift. It combines two previously
established evidence classes without weakening the graph comparison:

1. eight identities whose frozen-reference drift was proven to be node-index
   permutation only; and
2. twelve identities whose drifting CFG statements were independently proven
   to mutate persistent storage through expression-level lvalues rooted either
   directly in a ``StateVariable`` or in a
   ``LocalVariable(location='storage', is_storage=True)``.

For the second class, the tool canonicalizes only those explicitly evidenced
node identities to ``CFG_NODE_WRITE`` in an in-memory copy of each graph, then
requires exact labelled directed-multigraph isomorphism through unchanged edge
type 10. Any additional feature, metadata, topology, missing-node, runtime, or
extractor-version difference remains blocking.

This tool does not modify artifacts, physical acceptance, or training authority.
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.representation.graph_schema import NODE_TYPES

from p8_probe_v10_structural_drift import (
    PRIMARY_SLITHER_VERSION,
    _load_graph,
    _sha256,
    _unexpected_identities,
    compare_graphs,
)


EXPECTED_SEMANTIC_CONTRACTS = 12
EXPECTED_UNEXPECTED_IDENTITIES = 20
WRITE_TYPE_NAME = "CFG_NODE_WRITE"
WRITE_TYPE_ID = int(NODE_TYPES[WRITE_TYPE_NAME])
MAX_NODE_TYPE_ID = float(max(NODE_TYPES.values()))


def _node_key(metadata: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return (
        str(metadata.get("name") or ""),
        tuple(int(line) for line in (metadata.get("source_lines") or [])),
    )


def _is_persistent_write_root(root: Any) -> bool:
    """Match the V2.5 classifier's positive persistent-storage evidence rule."""

    if not isinstance(root, dict):
        return False
    if root.get("class") == "StateVariable":
        return True
    return (
        root.get("location") == "storage"
        and root.get("is_storage") is True
    )


def _semantic_targets(report: dict[str, Any]) -> dict[str, set[tuple[str, tuple[int, ...]]]]:
    if report.get("schema") != "sentinel-r4-v10-cfg-write-evidence-v1":
        raise ValueError("unexpected CFG write semantic-evidence schema")
    if report.get("slither_analyzer") != PRIMARY_SLITHER_VERSION:
        raise ValueError("semantic evidence was not produced under exact Slither 0.10.0")
    if report.get("all_requested_nodes_found") is not True:
        raise ValueError("semantic evidence contains missing requested nodes")

    contracts = list(report.get("contracts") or [])
    if len(contracts) != EXPECTED_SEMANTIC_CONTRACTS:
        raise ValueError(
            f"expected {EXPECTED_SEMANTIC_CONTRACTS} semantic-evidence contracts, "
            f"found {len(contracts)}"
        )

    targets: dict[str, set[tuple[str, tuple[int, ...]]]] = {}
    for contract in contracts:
        logical = str(contract.get("contract") or "")
        if not logical:
            raise ValueError("semantic evidence contains an empty contract identity")
        rows = list(contract.get("nodes") or [])
        expected_count = int(contract.get("requested_nodes", -1))
        if len(rows) != expected_count:
            raise ValueError(f"semantic-evidence node count mismatch for {logical}")
        node_keys = {
            (
                str(row.get("name") or ""),
                tuple(int(line) for line in (row.get("source_lines") or [])),
            )
            for row in rows
        }
        if len(node_keys) != len(rows):
            raise ValueError(f"duplicate semantic-evidence node identity for {logical}")

        # Re-prove the evidence direction from the report itself instead of
        # trusting only its contract membership. This intentionally mirrors the
        # V2.5 classifier's positive persistent-write rule.
        for row in rows:
            expression_writes = list(row.get("expression_writes") or [])
            persistent_roots = [
                expression.get("root_variable")
                for expression in expression_writes
                if _is_persistent_write_root(expression.get("root_variable"))
            ]
            if not persistent_roots:
                raise ValueError(
                    f"semantic evidence does not prove a persistent-storage write for "
                    f"{logical} node {row.get('name')!r}"
                )

        targets[logical] = node_keys
    return targets


def _require_v25_sidecar(logical: str, sidecar: dict[str, Any]) -> None:
    if sidecar.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError(
            f"{logical} extractor mismatch: expected "
            f"{V10_REPRESENTATION_EXTRACTOR_VERSION!r}, found "
            f"{sidecar.get('extractor_version')!r}"
        )


def _canonicalize_expected_writes(
    graph: Any,
    logical: str,
    targets: set[tuple[str, tuple[int, ...]]],
) -> Any:
    result = copy.deepcopy(graph)
    metadata = list(getattr(result, "node_metadata", None) or [])
    by_key: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for index, row in enumerate(metadata):
        by_key.setdefault(_node_key(row), []).append(index)

    for key in sorted(targets):
        indices = by_key.get(key) or []
        if len(indices) != 1:
            raise ValueError(
                f"{logical} expected semantic node {key!r} resolves to "
                f"{len(indices)} graph nodes"
            )
        index = indices[0]
        metadata[index] = dict(metadata[index])
        metadata[index]["type"] = WRITE_TYPE_NAME
        result.x[index, 0] = WRITE_TYPE_ID / MAX_NODE_TYPE_ID

    result.node_metadata = metadata
    return result


def _all_targets_are_write(
    graph: Any,
    logical: str,
    targets: set[tuple[str, tuple[int, ...]]],
) -> tuple[bool, list[dict[str, Any]]]:
    metadata = list(getattr(graph, "node_metadata", None) or [])
    observed: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    for row in metadata:
        observed.setdefault(_node_key(row), []).append(str(row.get("type") or ""))

    failures: list[dict[str, Any]] = []
    for key in sorted(targets):
        types = observed.get(key) or []
        if types != [WRITE_TYPE_NAME]:
            failures.append(
                {
                    "name": key[0],
                    "source_lines": list(key[1]),
                    "observed_types": types,
                }
            )
    return not failures, failures


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.repeat_root) != 3:
        raise ValueError("exactly three --repeat-root values are required")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    identities = _unexpected_identities(audit)
    if len(identities) != EXPECTED_UNEXPECTED_IDENTITIES:
        raise ValueError(
            f"expected {EXPECTED_UNEXPECTED_IDENTITIES} unexpected identities, "
            f"found {len(identities)}"
        )

    semantic_report = json.loads(args.semantic_evidence.read_text(encoding="utf-8"))
    semantic_targets = _semantic_targets(semantic_report)
    unknown_semantic = sorted(set(semantic_targets) - set(identities))
    if unknown_semantic:
        raise ValueError(
            f"semantic evidence contains identities absent from the transition audit: "
            f"{unknown_semantic}"
        )

    index_only = sorted(set(identities) - set(semantic_targets))
    if len(index_only) != 8:
        raise ValueError(
            f"expected 8 permutation-only identities after semantic split, found "
            f"{len(index_only)}"
        )

    contracts: list[dict[str, Any]] = []
    blockers: list[str] = []

    for logical in identities:
        reference = _load_graph(
            args.reference_root,
            logical,
            require_primary_runtime=False,
        )
        repeats = [
            _load_graph(root, logical, require_primary_runtime=True)
            for root in args.repeat_root
        ]
        for artifact in repeats:
            _require_v25_sidecar(logical, artifact.sidecar)

        repeat_comparisons: dict[str, Any] = {}
        repeat_pair_pass = True
        for left_index, right_index in combinations(range(3), 2):
            key = f"repeat_{left_index + 1}__vs__repeat_{right_index + 1}"
            comparison = compare_graphs(
                repeats[left_index].graph,
                repeats[right_index].graph,
                max_search_states=args.max_search_states,
            )
            repeat_comparisons[key] = comparison
            if comparison["exact_node_index_invariant_equivalent"] is not True:
                repeat_pair_pass = False

        semantic_failures: dict[str, list[dict[str, Any]]] = {}
        if logical in semantic_targets:
            targets = semantic_targets[logical]
            for index, artifact in enumerate(repeats, start=1):
                passed, failures = _all_targets_are_write(
                    artifact.graph, logical, targets
                )
                if not passed:
                    semantic_failures[f"repeat_{index}"] = failures

            canonical_reference = _canonicalize_expected_writes(
                reference.graph, logical, targets
            )
            canonical_repeats = [
                _canonicalize_expected_writes(artifact.graph, logical, targets)
                for artifact in repeats
            ]
            reference_comparisons = {
                f"reference_canonical__vs__repeat_{index}": compare_graphs(
                    canonical_reference,
                    graph,
                    max_search_states=args.max_search_states,
                )
                for index, graph in enumerate(canonical_repeats, start=1)
            }
            reference_pass = all(
                row["exact_node_index_invariant_equivalent"] is True
                for row in reference_comparisons.values()
            )
            decision = (
                "V25_DETERMINISTIC_STORAGE_WRITE_CORRECTION_PROVEN"
                if repeat_pair_pass and not semantic_failures and reference_pass
                else "BLOCKED_V25_STORAGE_WRITE_REPRODUCIBILITY"
            )
        else:
            reference_comparisons = {
                f"reference__vs__repeat_{index}": compare_graphs(
                    reference.graph,
                    artifact.graph,
                    max_search_states=args.max_search_states,
                )
                for index, artifact in enumerate(repeats, start=1)
            }
            reference_pass = all(
                row["exact_node_index_invariant_equivalent"] is True
                for row in reference_comparisons.values()
            )
            decision = (
                "V25_NODE_ORDER_INDEX_EQUIVALENCE_REPRODUCED"
                if repeat_pair_pass and reference_pass
                else "BLOCKED_V25_INDEX_EQUIVALENCE_REPRODUCIBILITY"
            )

        passed = decision in {
            "V25_DETERMINISTIC_STORAGE_WRITE_CORRECTION_PROVEN",
            "V25_NODE_ORDER_INDEX_EQUIVALENCE_REPRODUCED",
        }
        if not passed:
            blockers.append(logical)

        contracts.append(
            {
                "contract": logical,
                "evidence_class": (
                    "deterministic_storage_write_correction"
                    if logical in semantic_targets
                    else "node_order_index_equivalence"
                ),
                "decision": decision,
                "passed": passed,
                "semantic_write_failures": semantic_failures,
                "repeat_comparisons": repeat_comparisons,
                "reference_comparisons": reference_comparisons,
                "artifacts": {
                    f"repeat_{index}": {
                        "graph_path": str(artifact.graph_path),
                        "graph_sha256": _sha256(artifact.graph_path),
                        "sidecar_path": str(artifact.sidecar_path),
                        "sidecar_sha256": _sha256(artifact.sidecar_path),
                        "extractor_version": artifact.sidecar.get("extractor_version"),
                        "slither_runtime": artifact.sidecar.get("slither_runtime"),
                    }
                    for index, artifact in enumerate(repeats, start=1)
                },
            }
        )

    return {
        "schema": "sentinel-r4-v10-v25-reproducibility-probe-v1",
        "source_audit_sha256": _sha256(args.audit),
        "semantic_evidence_sha256": _sha256(args.semantic_evidence),
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "slither_analyzer": PRIMARY_SLITHER_VERSION,
        "unexpected_identities": len(identities),
        "semantic_correction_identities": len(semantic_targets),
        "index_equivalence_identities": len(index_only),
        "repeat_generations": len(args.repeat_root),
        "contracts": contracts,
        "blocking_identities": blockers,
        "zero_unexplained_drift": not blockers,
        "bounded_v25_reproducibility_passed": not blockers,
        "physical_acceptance": False,
        "training_authorized": False,
        "limitations": [
            "This is a bounded 20-identity V2.5 reproducibility decision, not full-population physical acceptance.",
            "Only node identities independently proven as persistent-storage writes are canonicalized to CFG_NODE_WRITE.",
            "After that explicit correction, exact labelled directed-multigraph isomorphism through edge type 10 is still required.",
            "Any additional feature, metadata, topology, runtime, extractor, or missing-node difference remains blocking.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "docs/plan/ml-R4/reviews/R4-GAP-008/v10_transition_audit_v2.json"
        ),
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--semantic-evidence", type=Path, required=True)
    parser.add_argument("--repeat-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-search-states", type=int, default=200_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_search_states < 1:
        raise ValueError("--max-search-states must be >= 1")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["bounded_v25_reproducibility_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
