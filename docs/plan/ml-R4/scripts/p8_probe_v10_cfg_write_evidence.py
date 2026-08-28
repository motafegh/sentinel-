#!/usr/bin/env python3
"""Inspect stable expression-level write evidence for the V10 CFG drift set.

This protected-local diagnostic reads the existing structural-repeat report,
re-parses only contracts that still contain feature/metadata drift under exact
Slither 0.10.0, and reports the expression lvalue roots that existed before
SlithIR reference propagation. It does not write representations, change
acceptance, or authorize training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from sentinel_data.representation.graph_extractor import (
    GraphExtractionConfig,
    _build_solc_args,
)
from sentinel_data.representation.r4_orchestrator import (
    _load_meta,
    _resolve_solc_binary,
    _select_targets,
)


PRIMARY_SLITHER_VERSION = "0.10.0"


def _root_variable(expression: Any) -> Any | None:
    from slither.core.expressions.identifier import Identifier
    from slither.core.expressions.index_access import IndexAccess
    from slither.core.expressions.member_access import MemberAccess

    current = expression
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, Identifier):
            return getattr(current, "value", None)
        if isinstance(current, MemberAccess):
            current = getattr(current, "expression", None)
            continue
        if isinstance(current, IndexAccess):
            current = getattr(current, "expression_left", None)
            continue
        return None
    return None


def _variable_record(variable: Any | None) -> dict[str, Any] | None:
    if variable is None:
        return None
    try:
        is_storage = bool(variable.is_storage)
    except Exception:
        is_storage = None
    return {
        "class": type(variable).__name__,
        "name": str(getattr(variable, "name", "") or ""),
        "location": getattr(variable, "location", None),
        "is_storage": is_storage,
    }


def _expression_record(expression: Any) -> dict[str, Any]:
    return {
        "class": type(expression).__name__,
        "text": str(expression),
        "root_variable": _variable_record(_root_variable(expression)),
    }


def _record_key(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _canonical_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a byte-stable, duplicate-free ordering for diagnostic records."""

    unique = {_record_key(record): record for record in records}
    return [unique[key] for key in sorted(unique)]


def _stable_state_variables_written(
    expression_writes: list[dict[str, Any]],
) -> list[str]:
    """Report only direct state-variable roots, avoiding unstable Slither IR aliases."""

    return sorted(
        {
            str(root.get("name") or "")
            for expression in expression_writes
            if isinstance((root := expression.get("root_variable")), dict)
            and root.get("class") == "StateVariable"
            and root.get("name")
        }
    )


def _merge_node_record(
    existing: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Merge duplicate Slither views of one semantic node deterministically."""

    identity_fields = (
        "name",
        "source_lines",
        "function",
        "node_type",
        "variable_declaration",
    )
    for field in identity_fields:
        if existing.get(field) != candidate.get(field):
            raise ValueError(
                f"conflicting duplicate-node field {field}: "
                f"{existing.get(field)!r} != {candidate.get(field)!r}"
            )

    merged = dict(existing)
    for field in ("expression_writes", "ir_lvalues"):
        merged[field] = _canonical_records(
            list(existing.get(field) or []) + list(candidate.get(field) or [])
        )
    for field in ("state_variables_written", "state_variables_read"):
        merged[field] = sorted(
            {
                str(value)
                for value in (
                    list(existing.get(field) or [])
                    + list(candidate.get(field) or [])
                )
            }
        )
    return merged


def _ir_lvalue_record(operation: Any) -> dict[str, Any] | None:
    lvalue = getattr(operation, "lvalue", None)
    if lvalue is None:
        return None
    origin = getattr(lvalue, "points_to_origin", None)
    return {
        "operation": type(operation).__name__,
        "lvalue_class": type(lvalue).__name__,
        "lvalue": str(lvalue),
        "points_to_origin": _variable_record(origin),
    }


def _requested_nodes(report: dict[str, Any]) -> dict[str, set[tuple[str, tuple[int, ...]]]]:
    requested: dict[str, set[tuple[str, tuple[int, ...]]]] = {}
    for contract in report.get("contracts", []):
        comparison = (contract.get("comparisons") or {}).get(
            "reference__vs__candidate", {}
        )
        if comparison.get("exact_node_index_invariant_equivalent") is True:
            continue
        rows = set()
        for diff in comparison.get("unique_identity_semantic_diffs", []):
            identity = diff.get("identity") or {}
            rows.add(
                (
                    str(identity.get("name") or ""),
                    tuple(int(line) for line in identity.get("source_lines") or []),
                )
            )
        if rows:
            requested[str(contract["contract"])] = rows
    return requested


def _load_slither(source_path: Path, meta: dict[str, Any]) -> Any:
    from slither import Slither

    solc_binary = _resolve_solc_binary(str(meta.get("solc_version", "")))
    config = GraphExtractionConfig(
        solc_binary=solc_binary,
        solc_version=str(meta.get("solc_version", "")),
        allow_paths=str(source_path.parent),
    )
    kwargs: dict[str, Any] = {
        "detectors_to_run": [],
        "skip_analyze": False,
        "solc_args": _build_solc_args(config),
    }
    if solc_binary is not None:
        kwargs["solc"] = str(solc_binary)
    return Slither(str(source_path), **kwargs)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    slither_version = importlib.metadata.version("slither-analyzer")
    if slither_version != PRIMARY_SLITHER_VERSION:
        raise RuntimeError(
            "CFG write evidence requires exact slither-analyzer "
            f"{PRIMARY_SLITHER_VERSION}; found {slither_version}"
        )

    probe = json.loads(args.repeat_report.read_text(encoding="utf-8"))
    requested = _requested_nodes(probe)
    contracts: list[dict[str, Any]] = []

    for logical, wanted in sorted(requested.items()):
        source, contract_id = logical.split("/", 1)
        meta_path = args.preprocessed_root / source / f"{contract_id}.meta.json"
        source_path = args.preprocessed_root / source / f"{contract_id}.sol"
        if not meta_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"missing protected source for {logical}")

        meta = _load_meta(meta_path)
        targets = set(_select_targets(source_path, meta))
        slither = _load_slither(source_path, meta)
        observed_by_key: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}

        for contract in slither.contracts:
            if contract.name not in targets:
                continue
            seen_nodes: set[int] = set()
            functions = list(contract.functions)
            for parent in getattr(contract, "inheritance", None) or []:
                functions.extend(getattr(parent, "functions", None) or [])
            for function in functions:
                for node in getattr(function, "nodes", None) or []:
                    if id(node) in seen_nodes:
                        continue
                    seen_nodes.add(id(node))
                    mapping = getattr(node, "source_mapping", None)
                    lines = tuple(
                        int(line)
                        for line in (
                            getattr(mapping, "lines", None) or []
                        )
                    )
                    key = (str(node), lines)
                    if key not in wanted:
                        continue

                    expression_writes = _canonical_records(
                        [
                            _expression_record(expression)
                            for expression in (
                                getattr(
                                    node,
                                    "variables_written_as_expression",
                                    None,
                                )
                                or []
                            )
                        ]
                    )
                    record = {
                        "name": str(node),
                        "source_lines": list(lines),
                        "function": str(getattr(function, "canonical_name", "")),
                        "node_type": str(getattr(node, "type", "")),
                        "variable_declaration": _variable_record(
                            getattr(node, "variable_declaration", None)
                        ),
                        "expression_writes": expression_writes,
                        # Slither 0.10 may expose the same storage write either
                        # through state_variables_written or only through an
                        # earlier expression-level storage lvalue depending on
                        # internal reference propagation order. The latter is
                        # the governing evidence seam, so report only stable
                        # direct StateVariable expression roots here.
                        "state_variables_written": _stable_state_variables_written(
                            expression_writes
                        ),
                        "state_variables_read": sorted(
                            str(getattr(variable, "canonical_name", variable))
                            for variable in (
                                getattr(node, "state_variables_read", None) or []
                            )
                        ),
                        "ir_lvalues": _canonical_records(
                            [
                                ir_record
                                for operation in (
                                    getattr(node, "irs", None) or []
                                )
                                if (ir_record := _ir_lvalue_record(operation))
                                is not None
                            ]
                        ),
                    }
                    existing = observed_by_key.get(key)
                    observed_by_key[key] = (
                        record
                        if existing is None
                        else _merge_node_record(existing, record)
                    )

        observed = list(observed_by_key.values())
        observed_keys = set(observed_by_key)
        missing = sorted(wanted - observed_keys)
        contracts.append(
            {
                "contract": logical,
                "requested_nodes": len(wanted),
                "observed_nodes": len(observed),
                "missing_nodes": [
                    {"name": name, "source_lines": list(lines)}
                    for name, lines in missing
                ],
                "nodes": sorted(
                    observed,
                    key=lambda row: (
                        tuple(row["source_lines"]),
                        row["name"],
                    ),
                ),
            }
        )

    return {
        "schema": "sentinel-r4-v10-cfg-write-evidence-v1",
        "slither_analyzer": slither_version,
        "contracts_requested": len(requested),
        "contracts": contracts,
        "all_requested_nodes_found": all(
            not row["missing_nodes"] for row in contracts
        ),
        "physical_acceptance": False,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat-report",
        type=Path,
        default=Path(
            "docs/plan/ml-R4/reviews/R4-GAP-008/"
            "v10_structural_drift_repeat_probe_v1.json"
        ),
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data_module/data/sentinel-preprocessed-r4-v2"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["all_requested_nodes_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
