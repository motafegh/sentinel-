#!/usr/bin/env python3
"""Collect duplicate-safe source evidence for every full-population WRITE drift.

Targets are derived mechanically from the transition audit plus the bound
reference/candidate graphs.  A source statement is parsed once even when the
graph contains inherited duplicate occurrences; the report preserves the exact
graph multiplicity on both sides.  Expression-level storage roots are the only
positive semantic proof.  Mutable SlithIR lvalues are retained as diagnostics,
not silently promoted to proof.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from p8_probe_v10_cfg_write_evidence import (
    _canonical_records,
    _expression_record,
    _ir_lvalue_record,
    _load_meta,
    _load_slither,
    _merge_node_record,
    _record_key,
    _select_targets,
    _stable_state_variables_written,
    _variable_record,
)
from p8_probe_v10_structural_drift import (
    PRIMARY_SLITHER_VERSION,
    _identity_key,
    _sha256,
    _unexpected_identities,
)
from sentinel_data.representation.v10_cfg_determinism import (
    _call_mutates_persistent_storage,
)


SCHEMA = "sentinel-r4-v10-v26-full-write-evidence-v2"
WRITE_TYPE = "CFG_NODE_WRITE"


def _persistent_root(root: Any) -> bool:
    if not isinstance(root, dict):
        return False
    if root.get("class") == "StateVariable":
        return True
    # This mirrors the V2.5 guard's positive LocalVariable contract.  Slither
    # 0.10 reports ``location='default'`` for some storage-reference parameters
    # and inherited locals while its resolved ``is_storage`` property is true.
    # Requiring the display location as well would reject the exact evidence
    # consumed by the implementation.
    return root.get("is_storage") is True


def _storage_mutating_call_record(node: Any) -> dict[str, Any] | None:
    """Return stable AST evidence for a proven storage collection mutator."""

    if not _call_mutates_persistent_storage(node):
        return None
    expression = getattr(node, "expression", None)
    called = getattr(expression, "called", None)
    receiver = getattr(called, "expression", None)
    return {
        "method": str(getattr(called, "member_name", "") or ""),
        "receiver": _expression_record(receiver),
        "persistent_storage_proven": True,
    }


def _graph_groups(graph: Any) -> dict[tuple[str, tuple[int, ...], str], list[int]]:
    groups: dict[tuple[str, tuple[int, ...], str], list[int]] = defaultdict(list)
    metadata = list(getattr(graph, "node_metadata", None) or [])
    if len(metadata) != int(graph.x.shape[0]):
        raise ValueError("node metadata is not index aligned")
    for index, row in enumerate(metadata):
        groups[_identity_key(row)].append(index)
    return groups


def _derive_targets(
    *, reference_root: Path, candidate_root: Path, identities: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    targets: dict[str, list[dict[str, Any]]] = {}
    unexplained: list[dict[str, Any]] = []
    for logical in identities:
        source, contract_id = logical.split("/", 1)
        reference = torch.load(
            reference_root / source / f"{contract_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        candidate = torch.load(
            candidate_root / source / f"{contract_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        reference_groups = _graph_groups(reference)
        candidate_groups = _graph_groups(candidate)
        if set(reference_groups) != set(candidate_groups):
            unexplained.append(
                {"contract": logical, "reason": "coarse_identity_population_differs"}
            )
            continue

        rows: list[dict[str, Any]] = []
        for key in sorted(reference_groups, key=repr):
            reference_indices = reference_groups[key]
            candidate_indices = candidate_groups[key]
            reference_types = Counter(
                str(reference.node_metadata[index].get("type") or "")
                for index in reference_indices
            )
            candidate_types = Counter(
                str(candidate.node_metadata[index].get("type") or "")
                for index in candidate_indices
            )
            if reference_types == candidate_types:
                continue
            row = {
                "name": key[0],
                "source_lines": list(key[1]),
                "coarse_type": key[2],
                "reference_multiplicity": len(reference_indices),
                "candidate_multiplicity": len(candidate_indices),
                "reference_types": dict(sorted(reference_types.items())),
                "candidate_types": dict(sorted(candidate_types.items())),
            }
            if (
                len(reference_indices) != len(candidate_indices)
                or key[2] != "CFG_NODE"
                or WRITE_TYPE not in set(reference_types) | set(candidate_types)
            ):
                unexplained.append(
                    {"contract": logical, "reason": "non_write_group_drift", **row}
                )
                continue
            rows.append(row)
        if rows:
            targets[logical] = rows
    return targets, unexplained


def _source_record(node: Any, function: Any) -> dict[str, Any]:
    expression_writes = _canonical_records(
        [
            _expression_record(expression)
            for expression in (
                getattr(node, "variables_written_as_expression", None) or []
            )
        ]
    )
    storage_mutating_call = _storage_mutating_call_record(node)
    return {
        "name": str(node),
        "source_lines": [
            int(line)
            for line in (
                getattr(getattr(node, "source_mapping", None), "lines", None) or []
            )
        ],
        "function": str(getattr(function, "canonical_name", "")),
        "node_type": str(getattr(node, "type", "")),
        "variable_declaration": _variable_record(
            getattr(node, "variable_declaration", None)
        ),
        "expression_writes": expression_writes,
        "expression_persistent_write_proven": any(
            _persistent_root(expression.get("root_variable"))
            for expression in expression_writes
        ),
        "storage_mutating_call": storage_mutating_call,
        "storage_mutating_call_proven": storage_mutating_call is not None,
        "state_variables_written": _stable_state_variables_written(expression_writes),
        "state_variables_read": sorted(
            str(getattr(variable, "canonical_name", variable))
            for variable in (getattr(node, "state_variables_read", None) or [])
        ),
        "ir_lvalues": _canonical_records(
            [
                record
                for operation in (getattr(node, "irs", None) or [])
                if (record := _ir_lvalue_record(operation)) is not None
            ]
        ),
    }


def _collect_contract(
    payload: tuple[str, str, tuple[tuple[str, tuple[int, ...]], ...]]
) -> dict[str, Any]:
    logical, preprocessed_root_text, wanted = payload
    preprocessed_root = Path(preprocessed_root_text)
    source, contract_id = logical.split("/", 1)
    source_path = preprocessed_root / source / f"{contract_id}.sol"
    meta_path = preprocessed_root / source / f"{contract_id}.meta.json"
    meta = _load_meta(meta_path)
    selected = set(_select_targets(source_path, meta))
    slither = _load_slither(source_path, meta)
    wanted_set = set(wanted)
    observed: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []

    for contract in slither.contracts:
        if contract.name not in selected:
            continue
        functions = list(contract.functions)
        for parent in getattr(contract, "inheritance", None) or []:
            functions.extend(getattr(parent, "functions", None) or [])
        seen_nodes: set[int] = set()
        for function in functions:
            for node in getattr(function, "nodes", None) or []:
                if id(node) in seen_nodes:
                    continue
                seen_nodes.add(id(node))
                lines = tuple(
                    int(line)
                    for line in (
                        getattr(getattr(node, "source_mapping", None), "lines", None)
                        or []
                    )
                )
                key = (str(node), lines)
                if key not in wanted_set:
                    continue
                record = _source_record(node, function)
                previous = observed.get(key)
                if previous is None:
                    observed[key] = record
                else:
                    try:
                        observed[key] = _merge_node_record(previous, record)
                        observed[key]["expression_persistent_write_proven"] = any(
                            _persistent_root(expression.get("root_variable"))
                            for expression in observed[key]["expression_writes"]
                        )
                        call_records = [
                            record
                            for record in (
                                previous.get("storage_mutating_call"),
                                record.get("storage_mutating_call"),
                            )
                            if record is not None
                        ]
                        if len({_record_key(row) for row in call_records}) > 1:
                            raise ValueError(
                                "conflicting duplicate-node storage-mutating call evidence"
                            )
                        call_record = call_records[0] if call_records else None
                        observed[key]["storage_mutating_call"] = call_record
                        observed[key]["storage_mutating_call_proven"] = (
                            call_record is not None
                        )
                    except ValueError as exc:
                        conflicts.append({"name": key[0], "source_lines": list(key[1]), "error": str(exc)})

    missing = sorted(wanted_set - set(observed), key=repr)
    return {
        "contract": logical,
        "nodes": [observed[key] for key in sorted(observed, key=repr)],
        "missing_nodes": [
            {"name": name, "source_lines": list(lines)} for name, lines in missing
        ],
        "conflicts": conflicts,
    }


def _run_workers(payloads: list[tuple[Any, ...]], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return list(map(_collect_contract, payloads))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return list(executor.map(_collect_contract, payloads, chunksize=1))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    slither_version = importlib.metadata.version("slither-analyzer")
    if slither_version != PRIMARY_SLITHER_VERSION:
        raise RuntimeError(
            f"full write evidence requires Slither {PRIMARY_SLITHER_VERSION}; "
            f"found {slither_version}"
        )
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    identities = _unexpected_identities(audit)
    targets, unexplained = _derive_targets(
        reference_root=args.reference_root,
        candidate_root=args.candidate_root,
        identities=identities,
    )
    payloads = []
    target_rows_by_contract: dict[str, list[dict[str, Any]]] = {}
    for logical, rows in sorted(targets.items()):
        target_rows_by_contract[logical] = rows
        wanted = tuple(
            (str(row["name"]), tuple(int(line) for line in row["source_lines"]))
            for row in rows
        )
        payloads.append((logical, str(args.preprocessed_root), wanted))
    collected = _run_workers(payloads, args.workers)

    contracts: list[dict[str, Any]] = []
    positive = 0
    unresolved_nodes: list[dict[str, Any]] = []
    for source_record in collected:
        logical = source_record["contract"]
        nodes = {
            (row["name"], tuple(row["source_lines"])): row
            for row in source_record["nodes"]
        }
        groups = []
        for target in target_rows_by_contract[logical]:
            key = (target["name"], tuple(target["source_lines"]))
            evidence = nodes.get(key)
            proven = bool(
                evidence
                and (
                    evidence.get("expression_persistent_write_proven") is True
                    or evidence.get("storage_mutating_call_proven") is True
                )
            )
            positive += int(proven)
            group = {**target, "semantic_evidence": evidence, "write_proven": proven}
            groups.append(group)
            if not proven:
                unresolved_nodes.append(
                    {
                        "contract": logical,
                        "name": target["name"],
                        "source_lines": target["source_lines"],
                        "reason": (
                            "missing_source_node"
                            if evidence is None
                            else "no_stable_persistent_storage_mutation_evidence"
                        ),
                    }
                )
        contracts.append(
            {
                "contract": logical,
                "target_groups": groups,
                "missing_nodes": source_record["missing_nodes"],
                "conflicts": source_record["conflicts"],
            }
        )

    target_group_count = sum(len(rows) for rows in targets.values())
    return {
        "schema": SCHEMA,
        "source_audit_sha256": _sha256(args.audit),
        "candidate_binding_digest_sha256": audit.get(
            "candidate_binding_digest_sha256"
        ),
        "slither_analyzer": slither_version,
        "unexpected_identities": len(identities),
        "contracts_with_write_drift": len(targets),
        "target_groups": target_group_count,
        "duplicate_target_groups": sum(
            row["candidate_multiplicity"] > 1
            for rows in targets.values()
            for row in rows
        ),
        "storage_mutation_groups_proven": positive,
        "unresolved_write_groups": unresolved_nodes,
        "non_write_or_population_drift": unexplained,
        "all_target_groups_resolved": not unresolved_nodes and not unexplained,
        "contracts": contracts,
        "workers": args.workers,
        "physical_acceptance": False,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data_module/data/sentinel-preprocessed-r4-v2"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_target_groups_resolved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
