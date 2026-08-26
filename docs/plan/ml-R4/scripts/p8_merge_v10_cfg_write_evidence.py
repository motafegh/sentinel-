#!/usr/bin/env python3
"""Merge compatible V10 CFG WRITE semantic-evidence reports deterministically.

The bounded V2.5 investigation can discover additional storage-write nodes only
after the deterministic extractor exposes them. This utility combines the
original semantic-evidence report with one or more fail-closed expansion reports
without editing JSON by hand.

Merge rules are intentionally strict:

* every input must use the same CFG-write evidence schema and exact Slither
  0.10.0 runtime;
* every input must report all requested nodes found;
* contract/node identities are keyed by (contract, node name, source lines);
* duplicate node identities must have byte-equivalent JSON evidence;
* missing-node evidence is never merged away;
* the output remains diagnostic only and never grants physical acceptance or
  training authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "sentinel-r4-v10-cfg-write-evidence-v1"
PRIMARY_SLITHER_VERSION = "0.10.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_key(row: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return (
        str(row.get("name") or ""),
        tuple(int(line) for line in (row.get("source_lines") or [])),
    )


def _validate(report: dict[str, Any], path: Path) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError(f"{path} has unexpected schema {report.get('schema')!r}")
    if report.get("slither_analyzer") != PRIMARY_SLITHER_VERSION:
        raise ValueError(
            f"{path} was not produced under exact Slither {PRIMARY_SLITHER_VERSION}"
        )
    if report.get("all_requested_nodes_found") is not True:
        raise ValueError(f"{path} contains missing requested nodes")

    contracts = list(report.get("contracts") or [])
    declared_count = int(report.get("contracts_requested", -1))
    if declared_count != len(contracts):
        raise ValueError(
            f"{path} contract count mismatch: declared={declared_count}, observed={len(contracts)}"
        )

    seen_contracts: set[str] = set()
    for contract in contracts:
        logical = str(contract.get("contract") or "")
        if not logical or logical in seen_contracts:
            raise ValueError(f"{path} contains empty/duplicate contract identity {logical!r}")
        seen_contracts.add(logical)
        if contract.get("missing_nodes"):
            raise ValueError(f"{path} contains missing nodes for {logical}")
        rows = list(contract.get("nodes") or [])
        requested = int(contract.get("requested_nodes", -1))
        observed = int(contract.get("observed_nodes", -1))
        if requested != len(rows) or observed != len(rows):
            raise ValueError(f"{path} node-count mismatch for {logical}")
        keys = [_node_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{path} contains duplicate node evidence for {logical}")


def merge_reports(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two evidence reports are required")

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate(report, path)
        loaded.append((path, report))

    contracts: dict[str, dict[tuple[str, tuple[int, ...]], dict[str, Any]]] = {}
    for path, report in loaded:
        for contract in report["contracts"]:
            logical = str(contract["contract"])
            target = contracts.setdefault(logical, {})
            for row in contract["nodes"]:
                key = _node_key(row)
                existing = target.get(key)
                if existing is None:
                    target[key] = row
                    continue
                if existing != row:
                    raise ValueError(
                        f"conflicting evidence for {logical} node {key!r} between inputs"
                    )

    merged_contracts: list[dict[str, Any]] = []
    for logical in sorted(contracts):
        nodes = sorted(
            contracts[logical].values(),
            key=lambda row: (tuple(row.get("source_lines") or []), str(row.get("name") or "")),
        )
        merged_contracts.append(
            {
                "contract": logical,
                "requested_nodes": len(nodes),
                "observed_nodes": len(nodes),
                "missing_nodes": [],
                "nodes": nodes,
            }
        )

    return {
        "schema": SCHEMA,
        "evidence_scope": "merged_v25_semantic_write_evidence",
        "slither_analyzer": PRIMARY_SLITHER_VERSION,
        "contracts_requested": len(merged_contracts),
        "contracts": merged_contracts,
        "all_requested_nodes_found": True,
        "source_reports": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "evidence_scope": report.get("evidence_scope"),
                "contracts_requested": report.get("contracts_requested"),
            }
            for path, report in loaded
        ],
        "physical_acceptance": False,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = merge_reports(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
