#!/usr/bin/env python3
"""Generate or bind the protected-local R4-D-010 v10 graph candidate.

Regression mode emits only explicitly named contracts into a disposable root.
Full mode requires a fresh canonical candidate root, regenerates every source,
then runs the diagnostic population binder.  Neither mode accepts data or
authorizes training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from sentinel_data.preprocessing.r4_versions import (
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.graph_schema_versions import get_graph_schema
from sentinel_data.representation.r4_orchestrator import (
    _extract_one,
    represent_repaired_source,
)
from sentinel_data.vnext.r4_v10_binding import bind_v10_candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_candidate_root(root: Path) -> None:
    if root.name != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError(f"output root must be named {V10_REPRESENTATION_ROOT_NAME!r}")


def _locate_contract(preprocessed_root: Path, contract_id: str) -> tuple[str, Path, Path]:
    if len(contract_id) != 64 or any(ch not in "0123456789abcdef" for ch in contract_id):
        raise ValueError(f"invalid contract ID {contract_id!r}")
    matches = sorted(preprocessed_root.glob(f"*/{contract_id}.meta.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one preprocessed identity for {contract_id}; found {len(matches)}"
        )
    meta_path = matches[0]
    source = meta_path.parent.name
    source_path = meta_path.with_name(f"{contract_id}.sol")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    return source, source_path, meta_path


def _regression(args: argparse.Namespace) -> dict[str, Any]:
    if not args.contract_id:
        raise ValueError("regression mode requires at least one --contract-id")
    _require_candidate_root(args.output_root)
    schema = get_graph_schema(V10_GRAPH_SCHEMA_VERSION)
    expectations: dict[str, Any] = {}
    if args.expectations is not None:
        expectation_payload = json.loads(args.expectations.read_text(encoding="utf-8"))
        expectations = dict(expectation_payload.get("contracts") or {})
        if set(expectations) != set(args.contract_id):
            raise ValueError("regression expectations must exactly cover requested contracts")
    records: list[dict[str, Any]] = []
    for contract_id in args.contract_id:
        source, source_path, meta_path = _locate_contract(
            args.preprocessed_root, contract_id
        )
        output_dir = args.output_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        destinations = [
            output_dir / f"{contract_id}{suffix}"
            for suffix in (".pt", ".tokens.pt", ".rep.json")
        ]
        if any(path.exists() for path in destinations):
            raise FileExistsError(f"regression output already exists for {source}/{contract_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        provenance = _extract_one(
            source,
            source_path,
            meta,
            output_dir,
            graph_schema_version=V10_GRAPH_SCHEMA_VERSION,
            extractor_version=V10_REPRESENTATION_EXTRACTOR_VERSION,
            accepted_tokens_dir=args.accepted_v9_root / source,
        )
        graph_path, token_path, sidecar_path = destinations
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        edge_counts = {
            name: int((graph.edge_attr == edge_id).sum())
            for name, edge_id in schema.edge_types.items()
            if edge_id >= 11
        }
        accepted_token = args.accepted_v9_root / source / f"{contract_id}.tokens.pt"
        expected_counts = expectations.get(contract_id)
        expectation_match = (
            None
            if expected_counts is None
            else edge_counts == expected_counts.get("call_edge_counts")
        )
        records.append(
            {
                "contract_id": contract_id,
                "source": source,
                "graph_sha256": _sha256(graph_path),
                "tokens_sha256": _sha256(token_path),
                "sidecar_sha256": _sha256(sidecar_path),
                "accepted_v9_token_sha256": _sha256(accepted_token),
                "token_bytes_identical": _sha256(token_path) == _sha256(accepted_token),
                "call_edge_counts": edge_counts,
                "expected_call_edge_counts": (
                    None if expected_counts is None else expected_counts.get("call_edge_counts")
                ),
                "source_review_expectation_match": expectation_match,
                "unclassified_call_ir": list(
                    getattr(graph, "unclassified_call_ir", []) or []
                ),
                "call_mapping_errors": list(
                    getattr(graph, "call_mapping_errors", []) or []
                ),
                "classified_call_ir_counts": dict(
                    getattr(graph, "classified_call_ir_counts", {}) or {}
                ),
                "emitted_call_edge_counts": dict(
                    getattr(graph, "emitted_call_edge_counts", {}) or {}
                ),
                "graph_extraction_mode": provenance["graph_extraction_mode"],
            }
        )
    passed = all(
        row["token_bytes_identical"]
        and not row["unclassified_call_ir"]
        and not row["call_mapping_errors"]
        and row["classified_call_ir_counts"] == row["emitted_call_edge_counts"]
        and row["source_review_expectation_match"] is not False
        for row in records
    )
    return {
        "schema": "sentinel-r4-v10-bounded-regression-v1",
        "passed": passed,
        "status": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "physical_acceptance": False,
        "training_authorized": False,
        "expectations_sha256": (
            None if args.expectations is None else _sha256(args.expectations)
        ),
        "records": records,
    }


def _full(args: argparse.Namespace) -> dict[str, Any]:
    if args.contract_id:
        raise ValueError("full mode does not accept --contract-id")
    _require_candidate_root(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"full candidate root is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_results: list[dict[str, Any]] = []
    for preprocessed_dir in sorted(path for path in args.preprocessed_root.iterdir() if path.is_dir()):
        source = preprocessed_dir.name
        accepted_tokens_dir = args.accepted_v9_root / source
        if not accepted_tokens_dir.is_dir():
            raise FileNotFoundError(accepted_tokens_dir)
        result = represent_repaired_source(
            source,
            preprocessed_dir,
            args.output_root / source,
            n_workers=args.workers,
            graph_schema_version=V10_GRAPH_SCHEMA_VERSION,
            extractor_version=V10_REPRESENTATION_EXTRACTOR_VERSION,
            accepted_tokens_dir=accepted_tokens_dir,
        )
        source_results.append(
            {
                "source": result.source,
                "contracts_seen": result.contracts_seen,
                "representations_written": result.representations_written,
                "representations_failed": result.representations_failed,
                "duration_s": result.duration_s,
            }
        )
    generation_passed = all(row["representations_failed"] == 0 for row in source_results)
    binding = None
    if generation_passed:
        binding = bind_v10_candidate(
            candidate_root=args.output_root,
            accepted_v9_root=args.accepted_v9_root,
            report_path=args.output_root / "v10_candidate_binding_report.json",
        )
    return {
        "schema": "sentinel-r4-v10-full-generation-v1",
        "passed": generation_passed and bool(binding and binding["passed"]),
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "source_results": source_results,
        "binding_digest_sha256": None if binding is None else binding["binding_digest_sha256"],
        "physical_acceptance": False,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("regression", "full", "bind"), required=True)
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data_module/data/sentinel-preprocessed-r4-v2"))
    parser.add_argument("--accepted-v9-root", type=Path, default=Path("data_module/data/representations-r4-v2"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-id", action="append", default=[])
    parser.add_argument("--expectations", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.mode == "regression":
        report = _regression(args)
    elif args.mode == "full":
        report = _full(args)
    else:
        if args.contract_id or args.expectations is not None:
            raise ValueError("bind mode does not accept contract expectations")
        report = bind_v10_candidate(
            candidate_root=args.output_root,
            accepted_v9_root=args.accepted_v9_root,
            report_path=args.report,
        )
    if args.report is not None and args.mode != "bind":
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
