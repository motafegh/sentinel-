#!/usr/bin/env python3
"""Expand semantic WRITE evidence only for deterministic V2.5 blocker diffs.

This protected-local diagnostic consumes the bounded V2.5 reproducibility
report. It is deliberately narrower than a general corpus scan:

* only contracts still marked ``BLOCKED_V25_STORAGE_WRITE_REPRODUCIBILITY``
  are considered;
* all three V2.5 repeats must already be exactly node-index-invariant
  equivalent to one another;
* previously requested semantic WRITE nodes must have zero repeat failures;
* every remaining frozen-reference difference must be a lower CFG class moving
  deterministically to ``CFG_NODE_WRITE`` in all three repeats.

Only after those fail-closed prerequisites hold does the script re-parse the
protected source under exact Slither 0.10.0 and collect the same expression-level
lvalue/storage evidence used by ``p8_probe_v10_cfg_write_evidence.py``.

It does not modify representations, change physical acceptance, or authorize
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from p8_probe_v10_cfg_write_evidence import build_report as build_cfg_write_report


SOURCE_SCHEMA = "sentinel-r4-v10-v25-reproducibility-probe-v1"
BLOCKER_DECISION = "BLOCKED_V25_STORAGE_WRITE_REPRODUCIBILITY"
WRITE_TYPE = "CFG_NODE_WRITE"
OUTPUT_SCHEMA = "sentinel-r4-v10-cfg-write-evidence-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diff_signature(diff: dict[str, Any]) -> tuple[Any, ...]:
    identity = diff.get("identity") or {}
    return (
        str(identity.get("name") or ""),
        tuple(int(line) for line in (identity.get("source_lines") or [])),
        str(diff.get("left_type") or ""),
        str(diff.get("right_type") or ""),
        tuple(float(value) for value in (diff.get("left_features") or [])),
        tuple(float(value) for value in (diff.get("right_features") or [])),
    )


def _derive_adapter_report(
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a structural-probe-shaped adapter for only justified new WRITE diffs."""

    if report.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"unexpected V2.5 reproducibility schema: {report.get('schema')!r}")

    declared_blockers = [str(value) for value in (report.get("blocking_identities") or [])]
    if not declared_blockers:
        raise ValueError("V2.5 report has no blocking identities to investigate")

    rows_by_contract = {
        str(row.get("contract") or ""): row
        for row in (report.get("contracts") or [])
        if row.get("contract")
    }
    if sorted(rows_by_contract) != sorted(
        str(row.get("contract") or "")
        for row in (report.get("contracts") or [])
        if row.get("contract")
    ):
        raise ValueError("V2.5 report contains duplicate contract identities")

    adapter_contracts: list[dict[str, Any]] = []
    derived_blockers: list[str] = []
    total_requested_nodes = 0

    for logical in declared_blockers:
        row = rows_by_contract.get(logical)
        if row is None:
            raise ValueError(f"blocking identity is absent from contract rows: {logical}")
        if row.get("decision") != BLOCKER_DECISION:
            raise ValueError(
                f"{logical} blocker has unexpected decision {row.get('decision')!r}"
            )
        if row.get("semantic_write_failures"):
            raise ValueError(
                f"{logical} still has failures in the previously evidenced WRITE set"
            )

        repeat_comparisons = row.get("repeat_comparisons") or {}
        if len(repeat_comparisons) != 3:
            raise ValueError(
                f"{logical} must expose exactly three pairwise repeat comparisons"
            )
        for name, comparison in repeat_comparisons.items():
            if comparison.get("exact_node_index_invariant_equivalent") is not True:
                raise ValueError(
                    f"{logical} is not repeat-deterministic: {name} is not equivalent"
                )

        reference_comparisons = row.get("reference_comparisons") or {}
        if len(reference_comparisons) != 3:
            raise ValueError(
                f"{logical} must expose exactly three reference comparisons"
            )

        representative: list[dict[str, Any]] | None = None
        expected_signatures: set[tuple[Any, ...]] | None = None
        for name, comparison in sorted(reference_comparisons.items()):
            diffs = list(comparison.get("unique_identity_semantic_diffs") or [])
            if not diffs:
                raise ValueError(f"{logical} {name} has no semantic diffs to investigate")
            signatures = {_diff_signature(diff) for diff in diffs}
            if len(signatures) != len(diffs):
                raise ValueError(f"{logical} {name} contains duplicate semantic diffs")
            if expected_signatures is None:
                expected_signatures = signatures
                representative = diffs
            elif signatures != expected_signatures:
                raise ValueError(
                    f"{logical} reference differences are not identical across V2.5 repeats"
                )

        assert representative is not None
        for diff in representative:
            left_type = str(diff.get("left_type") or "")
            right_type = str(diff.get("right_type") or "")
            if right_type != WRITE_TYPE or left_type == WRITE_TYPE:
                raise ValueError(
                    f"{logical} contains a remaining diff outside lower-class -> WRITE: "
                    f"{left_type!r} -> {right_type!r}"
                )
            identity = diff.get("identity") or {}
            if not identity.get("name") or not identity.get("source_lines"):
                raise ValueError(f"{logical} contains an incomplete semantic node identity")

        total_requested_nodes += len(representative)
        derived_blockers.append(logical)
        adapter_contracts.append(
            {
                "contract": logical,
                "comparisons": {
                    "reference__vs__candidate": {
                        "exact_node_index_invariant_equivalent": False,
                        "unique_identity_semantic_diffs": representative,
                    }
                },
            }
        )

    if sorted(derived_blockers) != sorted(declared_blockers):
        raise ValueError("derived blocker set does not match declared blocker set")

    adapter = {
        "schema": "sentinel-r4-v10-v25-blocker-write-adapter-v1",
        "contracts": adapter_contracts,
    }
    provenance = {
        "blocking_identities": derived_blockers,
        "requested_nodes": total_requested_nodes,
    }
    return adapter, provenance


def _storage_write_proof(report: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    for contract in report.get("contracts") or []:
        logical = str(contract.get("contract") or "")
        for node in contract.get("nodes") or []:
            persistent = []
            for expression in node.get("expression_writes") or []:
                root = expression.get("root_variable")
                if (
                    isinstance(root, dict)
                    and root.get("location") == "storage"
                    and root.get("is_storage") is True
                ):
                    persistent.append(root)
            if not persistent:
                failures.append(
                    {
                        "contract": logical,
                        "name": str(node.get("name") or ""),
                        "source_lines": list(node.get("source_lines") or []),
                    }
                )
    return not failures, failures


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.reproducibility_report.read_text(encoding="utf-8"))
    adapter, provenance = _derive_adapter_report(source)

    with tempfile.TemporaryDirectory(prefix="sentinel-r4-v25-write-evidence-") as tmp:
        adapter_path = Path(tmp) / "adapter.json"
        adapter_path.write_text(
            json.dumps(adapter, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        semantic = build_cfg_write_report(
            SimpleNamespace(
                repeat_report=adapter_path,
                preprocessed_root=args.preprocessed_root,
            )
        )

    if semantic.get("schema") != OUTPUT_SCHEMA:
        raise ValueError("underlying CFG write probe returned an unexpected schema")

    storage_proven, storage_failures = _storage_write_proof(semantic)
    semantic.update(
        {
            "evidence_scope": "v25_remaining_blocker_expansion",
            "source_reproducibility_report": str(args.reproducibility_report),
            "source_reproducibility_report_sha256": _sha256(args.reproducibility_report),
            "derived_blocking_identities": provenance["blocking_identities"],
            "derived_requested_nodes": provenance["requested_nodes"],
            "all_requested_nodes_storage_write_proven": storage_proven,
            "storage_write_proof_failures": storage_failures,
            "physical_acceptance": False,
            "training_authorized": False,
        }
    )
    return semantic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reproducibility-report",
        type=Path,
        default=Path(
            "docs/plan/ml-R4/reviews/R4-GAP-008/"
            "v10_v25_reproducibility_probe_v1.json"
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
    passed = (
        report.get("all_requested_nodes_found") is True
        and report.get("all_requested_nodes_storage_write_proven") is True
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
