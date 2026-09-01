#!/usr/bin/env python3
"""Probe the R4 V10 structural blocker with exact node-index-invariant comparison.

This tool is diagnostic only. It compares the frozen Slither-0.10 V10 structural
reference, the current V2.4 candidate, and one or more repeated bounded
regenerations for the identities already reported as unexpected drift.

It never changes physical acceptance or training authorization. A graph is called
semantically equivalent only when an exact labelled directed-multigraph
isomorphism is found through unchanged edge type 10. Search-limit exhaustion is
reported as inconclusive, never as equivalent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import torch

MAX_UNCHANGED_EDGE_TYPE = 10
DEFAULT_MAX_SEARCH_STATES = 200_000
PRIMARY_SLITHER_VERSION = "0.10.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_blob(metadata: Any) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _node_labels(graph: Any) -> list[tuple[str, tuple[float, ...]]]:
    metadata = list(getattr(graph, "node_metadata", None) or [])
    if len(metadata) != int(graph.x.shape[0]):
        raise ValueError("node_metadata must be index-aligned with graph.x")
    return [
        (
            _metadata_blob(metadata[index]),
            tuple(float(value) for value in graph.x[index].tolist()),
        )
        for index in range(int(graph.x.shape[0]))
    ]


def _identity_key(metadata: dict[str, Any]) -> tuple[str, tuple[int, ...], str]:
    node_type = str(metadata.get("type") or "")
    coarse_type = "CFG_NODE" if node_type.startswith("CFG_NODE_") else node_type
    return (
        str(metadata.get("name") or ""),
        tuple(int(line) for line in (metadata.get("source_lines") or [])),
        coarse_type,
    )


def _filtered_edges(
    graph: Any, max_edge_type: int = MAX_UNCHANGED_EDGE_TYPE
) -> Counter[tuple[int, int, int]]:
    counter: Counter[tuple[int, int, int]] = Counter()
    edge_attr = graph.edge_attr.detach().cpu()
    edge_index = graph.edge_index.detach().cpu()
    for position, edge_type in enumerate(edge_attr.tolist()):
        edge_type = int(edge_type)
        if edge_type <= max_edge_type:
            counter[
                (
                    int(edge_index[0, position]),
                    int(edge_index[1, position]),
                    edge_type,
                )
            ] += 1
    return counter


def _edge_topology_equal_through(
    left: Any, right: Any, max_edge_type: int = MAX_UNCHANGED_EDGE_TYPE
) -> bool:
    return _filtered_edges(left, max_edge_type) == _filtered_edges(
        right, max_edge_type
    )


def _node_structural_signatures(
    labels: list[tuple[str, tuple[float, ...]]],
    edges: Counter[tuple[int, int, int]],
) -> list[tuple[Any, ...]]:
    outgoing: dict[int, Counter[tuple[int, tuple[str, tuple[float, ...]]]]] = defaultdict(Counter)
    incoming: dict[int, Counter[tuple[int, tuple[str, tuple[float, ...]]]]] = defaultdict(Counter)
    for (src, dst, edge_type), count in edges.items():
        outgoing[src][(edge_type, labels[dst])] += count
        incoming[dst][(edge_type, labels[src])] += count

    def _freeze(counter: Counter[Any]) -> tuple[tuple[Any, int], ...]:
        return tuple(sorted(counter.items(), key=lambda item: repr(item[0])))

    return [
        (labels[index], _freeze(outgoing[index]), _freeze(incoming[index]))
        for index in range(len(labels))
    ]


class IsomorphismResult(NamedTuple):
    equivalent: bool | None
    mapping: dict[int, int] | None
    search_states: int
    reason: str


def exact_semantic_isomorphism(
    left: Any,
    right: Any,
    *,
    max_edge_type: int = MAX_UNCHANGED_EDGE_TYPE,
    max_search_states: int = DEFAULT_MAX_SEARCH_STATES,
) -> IsomorphismResult:
    """Prove exact labelled multigraph equivalence while allowing node reindexing."""

    left_count = int(left.x.shape[0])
    right_count = int(right.x.shape[0])
    if left_count != right_count:
        return IsomorphismResult(False, None, 0, "node_count_differs")

    left_labels = _node_labels(left)
    right_labels = _node_labels(right)
    if Counter(left_labels) != Counter(right_labels):
        return IsomorphismResult(False, None, 0, "node_semantic_labels_differ")

    left_edges = _filtered_edges(left, max_edge_type)
    right_edges = _filtered_edges(right, max_edge_type)
    if sum(left_edges.values()) != sum(right_edges.values()):
        return IsomorphismResult(False, None, 0, "unchanged_edge_count_differs")

    left_edge_types = Counter()
    right_edge_types = Counter()
    for (_, _, edge_type), count in left_edges.items():
        left_edge_types[edge_type] += count
    for (_, _, edge_type), count in right_edges.items():
        right_edge_types[edge_type] += count
    if left_edge_types != right_edge_types:
        return IsomorphismResult(False, None, 0, "unchanged_edge_type_counts_differ")

    left_signatures = _node_structural_signatures(left_labels, left_edges)
    right_signatures = _node_structural_signatures(right_labels, right_edges)
    if Counter(left_signatures) != Counter(right_signatures):
        return IsomorphismResult(False, None, 0, "node_neighbourhood_signatures_differ")

    right_by_signature: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, signature in enumerate(right_signatures):
        right_by_signature[signature].append(index)

    candidates = {
        index: tuple(right_by_signature[left_signatures[index]])
        for index in range(left_count)
    }

    left_between: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    right_between: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for (src, dst, edge_type), count in left_edges.items():
        left_between[(src, dst)][edge_type] += count
    for (src, dst, edge_type), count in right_edges.items():
        right_between[(src, dst)][edge_type] += count

    mapping: dict[int, int] = {}
    used_right: set[int] = set()
    search_states = 0

    # Pre-map singleton invariant classes.
    for left_index in sorted(candidates):
        options = candidates[left_index]
        if len(options) == 1:
            right_index = options[0]
            if right_index in used_right:
                return IsomorphismResult(False, None, 0, "singleton_mapping_collision")
            mapping[left_index] = right_index
            used_right.add(right_index)

    def compatible(left_index: int, right_index: int) -> bool:
        if left_between[(left_index, left_index)] != right_between[
            (right_index, right_index)
        ]:
            return False
        for mapped_left, mapped_right in mapping.items():
            if left_between[(left_index, mapped_left)] != right_between[
                (right_index, mapped_right)
            ]:
                return False
            if left_between[(mapped_left, left_index)] != right_between[
                (mapped_right, right_index)
            ]:
                return False
        return True

    # Validate the complete pre-mapped subgraph in O(E), rather than calling
    # ``compatible`` for every singleton against every other singleton. Large
    # contracts can have more than a thousand nodes that become uniquely
    # identifiable after the neighbourhood signature pass; the old O(N^2)
    # loop made those already-deterministic mappings appear to hang.
    mapped_left_edges = Counter(
        (mapping[src], mapping[dst], edge_type, count)
        for (src, dst, edge_type), count in left_edges.items()
        if src in mapping and dst in mapping
    )
    mapped_right_edges = Counter(
        (src, dst, edge_type, count)
        for (src, dst, edge_type), count in right_edges.items()
        if src in used_right and dst in used_right
    )
    if mapped_left_edges != mapped_right_edges:
        return IsomorphismResult(False, None, 0, "singleton_edges_differ")

    ambiguous = [
        index
        for index in range(left_count)
        if index not in mapping
    ]
    ambiguous.sort(key=lambda index: (len(candidates[index]), repr(left_signatures[index]), index))

    limit_exhausted = False

    def search(position: int) -> bool:
        nonlocal search_states, limit_exhausted
        if position == len(ambiguous):
            remapped = Counter(
                (mapping[src], mapping[dst], edge_type, count)
                for (src, dst, edge_type), count in left_edges.items()
            )
            expected = Counter(
                (src, dst, edge_type, count)
                for (src, dst, edge_type), count in right_edges.items()
            )
            return remapped == expected

        if search_states >= max_search_states:
            limit_exhausted = True
            return False

        left_index = ambiguous[position]
        for right_index in candidates[left_index]:
            if right_index in used_right:
                continue
            search_states += 1
            if not compatible(left_index, right_index):
                continue
            mapping[left_index] = right_index
            used_right.add(right_index)
            if search(position + 1):
                return True
            used_right.remove(right_index)
            del mapping[left_index]
            if limit_exhausted:
                return False
        return False

    if search(0):
        return IsomorphismResult(True, dict(mapping), search_states, "exact_isomorphism")
    if limit_exhausted:
        return IsomorphismResult(
            None, None, search_states, "search_limit_exhausted_fail_closed"
        )
    return IsomorphismResult(False, None, search_states, "no_exact_isomorphism")


def _unique_identity_diffs(left: Any, right: Any) -> list[dict[str, Any]]:
    left_meta = list(getattr(left, "node_metadata", None) or [])
    right_meta = list(getattr(right, "node_metadata", None) or [])
    left_groups: dict[tuple[str, tuple[int, ...], str], list[int]] = defaultdict(list)
    right_groups: dict[tuple[str, tuple[int, ...], str], list[int]] = defaultdict(list)
    for index, metadata in enumerate(left_meta):
        left_groups[_identity_key(metadata)].append(index)
    for index, metadata in enumerate(right_meta):
        right_groups[_identity_key(metadata)].append(index)

    diffs: list[dict[str, Any]] = []
    for key in sorted(set(left_groups) & set(right_groups), key=repr):
        if len(left_groups[key]) != 1 or len(right_groups[key]) != 1:
            continue
        left_index = left_groups[key][0]
        right_index = right_groups[key][0]
        left_type = str(left_meta[left_index].get("type") or "")
        right_type = str(right_meta[right_index].get("type") or "")
        left_features = tuple(float(v) for v in left.x[left_index].tolist())
        right_features = tuple(float(v) for v in right.x[right_index].tolist())
        if left_type != right_type or left_features != right_features:
            diffs.append(
                {
                    "identity": {
                        "name": key[0],
                        "source_lines": list(key[1]),
                        "coarse_type": key[2],
                    },
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_type": left_type,
                    "right_type": right_type,
                    "left_features": list(left_features),
                    "right_features": list(right_features),
                }
            )
    return diffs


def compare_graphs(
    left: Any,
    right: Any,
    *,
    max_search_states: int = DEFAULT_MAX_SEARCH_STATES,
) -> dict[str, Any]:
    raw_features_equal = torch.equal(left.x, right.x)
    raw_metadata_equal = getattr(left, "node_metadata", None) == getattr(
        right, "node_metadata", None
    )
    raw_topology_equal = _edge_topology_equal_through(left, right)
    # Index identity is itself a complete isomorphism proof. Avoid invoking the
    # permutation search when labels and unchanged topology already match
    # byte-for-byte; this is especially important for repeated inherited CFG
    # nodes with identical labels.
    iso = (
        IsomorphismResult(
            True,
            {index: index for index in range(int(left.x.shape[0]))},
            0,
            "raw_index_identity",
        )
        if raw_features_equal and raw_metadata_equal and raw_topology_equal
        else exact_semantic_isomorphism(
            left, right, max_search_states=max_search_states
        )
    )

    if iso.equivalent is True and (
        not raw_features_equal or not raw_metadata_equal or not raw_topology_equal
    ):
        classification = "NODE_ORDER_INDEX_NONDETERMINISM_PROVEN"
    elif iso.equivalent is True:
        classification = "RAW_STRUCTURE_EQUAL"
    elif (
        int(left.x.shape[0]) == int(right.x.shape[0])
        and raw_topology_equal
        and (not raw_features_equal or not raw_metadata_equal)
    ):
        classification = "FEATURE_OR_METADATA_CLASSIFICATION_DRIFT"
    elif iso.equivalent is None:
        classification = "INCONCLUSIVE_FAIL_CLOSED"
    else:
        classification = "SEMANTIC_STRUCTURE_DRIFT"

    return {
        "classification": classification,
        "raw_node_features_equal": raw_features_equal,
        "raw_node_metadata_equal": raw_metadata_equal,
        "raw_unchanged_edge_topology_equal": raw_topology_equal,
        "exact_node_index_invariant_equivalent": iso.equivalent,
        "isomorphism_reason": iso.reason,
        "isomorphism_search_states": iso.search_states,
        "node_count_left": int(left.x.shape[0]),
        "node_count_right": int(right.x.shape[0]),
        "unchanged_edge_count_left": sum(_filtered_edges(left).values()),
        "unchanged_edge_count_right": sum(_filtered_edges(right).values()),
        "unique_identity_semantic_diffs": _unique_identity_diffs(left, right),
    }


class _LoadedArtifact(NamedTuple):
    graph_path: Path
    graph: Any
    sidecar_path: Path
    sidecar: dict[str, Any]


def _load_graph(
    root: Path,
    logical: str,
    *,
    require_primary_runtime: bool,
) -> _LoadedArtifact:
    source, contract_id = logical.split("/", 1)
    graph_path = root / source / f"{contract_id}.pt"
    sidecar_path = root / source / f"{contract_id}.rep.json"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if require_primary_runtime:
        runtime = sidecar.get("slither_runtime")
        if not isinstance(runtime, dict):
            raise ValueError(
                f"{logical} is missing the bound Slither runtime in {sidecar_path}"
            )
        if runtime.get("slither_analyzer") != PRIMARY_SLITHER_VERSION:
            raise ValueError(
                f"{logical} was not generated under exact Slither "
                f"{PRIMARY_SLITHER_VERSION}: {runtime.get('slither_analyzer')!r}"
            )
        if runtime.get("runtime_role") != "primary":
            raise ValueError(
                f"{logical} does not use the primary Slither runtime: "
                f"{runtime.get('runtime_role')!r}"
            )
        if runtime.get("required_for_physical_acceptance") != PRIMARY_SLITHER_VERSION:
            raise ValueError(
                f"{logical} has inconsistent physical-acceptance runtime binding"
            )

    return _LoadedArtifact(
        graph_path=graph_path,
        graph=torch.load(graph_path, map_location="cpu", weights_only=False),
        sidecar_path=sidecar_path,
        sidecar=sidecar,
    )


def _unexpected_identities(audit: dict[str, Any]) -> list[str]:
    identities = [
        str(record["contract"])
        for record in (audit.get("structural_drift_contracts") or [])
        if record.get("v9_parse_only") is False
    ]
    expected = int(
        (audit.get("totals") or {}).get(
            "graphs_with_unexpected_structural_drift_from_reference_through_edge_10",
            -1,
        )
    )
    if expected < 0:
        raise ValueError("audit is missing unexpected structural-drift count")
    if len(identities) != expected:
        raise ValueError(
            f"audit exposes {len(identities)} unexpected identities but totals require {expected}"
        )
    return identities


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    identities = _unexpected_identities(audit)
    roots = [
        ("reference", args.reference_root, False),
        ("candidate", args.candidate_root, True),
        *[
            (f"repeat_{index}", root, True)
            for index, root in enumerate(args.repeat_root, start=1)
        ],
    ]

    contracts: list[dict[str, Any]] = []
    blockers: list[str] = []
    for logical in identities:
        loaded: dict[str, _LoadedArtifact] = {
            name: _load_graph(
                root,
                logical,
                require_primary_runtime=require_primary_runtime,
            )
            for name, root, require_primary_runtime in roots
        }
        comparisons: dict[str, Any] = {}
        for left_index in range(len(roots)):
            for right_index in range(left_index + 1, len(roots)):
                left_name = roots[left_index][0]
                right_name = roots[right_index][0]
                comparisons[f"{left_name}__vs__{right_name}"] = compare_graphs(
                    loaded[left_name].graph,
                    loaded[right_name].graph,
                    max_search_states=args.max_search_states,
                )

        reference_candidate = comparisons["reference__vs__candidate"]
        repeat_vs_reference = [
            comparisons[f"reference__vs__repeat_{index}"]
            for index in range(1, len(args.repeat_root) + 1)
        ]
        repeat_vs_candidate = [
            comparisons[f"candidate__vs__repeat_{index}"]
            for index in range(1, len(args.repeat_root) + 1)
        ]

        if reference_candidate["exact_node_index_invariant_equivalent"] is True:
            decision = "NODE_ORDER_INDEX_NONDETERMINISM_PROVEN"
        elif any(
            row["exact_node_index_invariant_equivalent"] is None
            for row in [reference_candidate, *repeat_vs_reference, *repeat_vs_candidate]
        ):
            decision = "INCONCLUSIVE_FAIL_CLOSED"
        elif args.repeat_root:
            ref_matches = [
                row["exact_node_index_invariant_equivalent"] is True
                for row in repeat_vs_reference
            ]
            cand_matches = [
                row["exact_node_index_invariant_equivalent"] is True
                for row in repeat_vs_candidate
            ]
            if any(ref_matches) and any(cand_matches):
                semantic_kind = (
                    "FEATURE_CLASSIFICATION"
                    if reference_candidate["classification"]
                    == "FEATURE_OR_METADATA_CLASSIFICATION_DRIFT"
                    else "STRUCTURAL"
                )
                decision = f"SLITHER_{semantic_kind}_NONDETERMINISM_PROVEN"
            elif all(ref_matches):
                decision = "CANDIDATE_ONE_OFF_DRIFT_REPEAT_MATCHES_REFERENCE"
            elif all(cand_matches):
                decision = "REPRODUCIBLE_CANDIDATE_DRIFT_VS_REFERENCE"
            else:
                decision = "UNRESOLVED_MULTIPLE_REPEAT_STATES"
        else:
            decision = "REPEATED_REGENERATION_REQUIRED"

        if decision not in {
            "NODE_ORDER_INDEX_NONDETERMINISM_PROVEN",
        }:
            blockers.append(logical)

        contracts.append(
            {
                "contract": logical,
                "decision": decision,
                "artifacts": {
                    name: {
                        "graph_path": str(artifact.graph_path),
                        "graph_sha256": _sha256(artifact.graph_path),
                        "sidecar_path": str(artifact.sidecar_path),
                        "sidecar_sha256": _sha256(artifact.sidecar_path),
                        "slither_runtime": artifact.sidecar.get("slither_runtime"),
                    }
                    for name, artifact in loaded.items()
                },
                "comparisons": comparisons,
            }
        )

    return {
        "schema": "sentinel-r4-v10-structural-drift-probe-v1",
        "source_audit_sha256": _sha256(args.audit),
        "unexpected_identities": len(identities),
        "repeat_generations": len(args.repeat_root),
        "contracts": contracts,
        "unresolved_or_semantic_drift_identities": blockers,
        "zero_unexplained_drift": not blockers,
        "physical_acceptance": False,
        "training_authorized": False,
        "limitations": [
            "This is a strict structural diagnostic, not a physical-acceptance decision.",
            "Exact node-index-invariant equivalence is accepted only after a complete labelled directed-multigraph isomorphism through edge type 10.",
            "Search-limit exhaustion is inconclusive and remains a blocker.",
            "Feature/metadata or reproducible structural differences require explicit source/evidence review before any acceptance rule changes.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/plan/ml-R4/reviews/R4-GAP-008/v10_transition_audit_v2.json"),
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--repeat-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-search-states", type=int, default=DEFAULT_MAX_SEARCH_STATES
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_search_states < 1:
        raise ValueError("--max-search-states must be >= 1")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["zero_unexplained_drift"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
