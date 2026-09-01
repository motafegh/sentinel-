#!/usr/bin/env python3
"""Regenerate exactly the primary-runtime identities named by a V10 audit.

This is a protected-local diagnostic generator.  It derives its complete
population from the transition audit, uses fresh spawned worker processes under
the exact primary Slither runtime, and validates every emitted graph, token, and
sidecar.  It never changes the candidate, acceptance, or training authority.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from sentinel_data.preprocessing.r4_versions import (
    V10_GRAPH_SCHEMA_VERSION,
    V10_PRIMARY_SLITHER_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
    V10_SLITHER_RUNTIME_EXCEPTIONS,
)
from sentinel_data.representation.graph_schema_versions import get_graph_schema
from sentinel_data.representation.r4_orchestrator import _represent_worker
from sentinel_data.vnext.r4_binding import _validate_graph, _validate_tokens

from p8_generate_v10_v25_primary_attempt import (
    ARTIFACT_SUFFIXES,
    _inventory_attempt_artifacts,
)
from p8_probe_v10_structural_drift import (
    _sha256,
    _unexpected_identities,
)
from p8_stage_v10_v25_primary_attempt import _validate_primary_sidecar


SCHEMA = "sentinel-r4-v10-v26-structural-repeat-v1"


def _require_runtime() -> dict[str, str]:
    slither = importlib.metadata.version("slither-analyzer")
    crytic = importlib.metadata.version("crytic-compile")
    if slither != V10_PRIMARY_SLITHER_VERSION:
        raise RuntimeError(
            "structural repeat requires exact slither-analyzer "
            f"{V10_PRIMARY_SLITHER_VERSION}; found {slither}"
        )
    return {
        "slither_analyzer": slither,
        "crytic_compile": crytic,
        "runtime_role": "primary",
    }


def _run_workers(
    worker_args: list[tuple[str, ...]], workers: int
) -> list[tuple[bool, dict[str, Any] | None, dict[str, str] | None]]:
    if workers == 1:
        return list(map(_represent_worker, worker_args))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return list(executor.map(_represent_worker, worker_args, chunksize=1))


def _validate_outputs(
    *,
    output_root: Path,
    accepted_v9_root: Path,
    identities: list[str],
) -> list[dict[str, Any]]:
    expected = {tuple(logical.split("/", 1)) for logical in identities}
    artifacts = _inventory_attempt_artifacts(output_root)
    if set(artifacts) != expected:
        missing = sorted(expected - set(artifacts))
        extra = sorted(set(artifacts) - expected)
        raise ValueError(
            f"repeat population mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    schema = get_graph_schema(V10_GRAPH_SCHEMA_VERSION)
    records: list[dict[str, Any]] = []
    for source, contract_id in sorted(expected):
        logical = f"{source}/{contract_id}"
        paths = artifacts[(source, contract_id)]
        if set(paths) != set(ARTIFACT_SUFFIXES):
            raise ValueError(f"incomplete repeat artifacts for {logical}")
        sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
        _validate_primary_sidecar(logical, sidecar)
        if sidecar.get("sha256") != contract_id or sidecar.get("source") != source:
            raise ValueError(f"repeat sidecar identity mismatch for {logical}")

        graph = torch.load(paths["graph"], map_location="cpu", weights_only=False)
        _validate_graph(torch, graph, sidecar, num_edge_types=schema.num_edge_types)
        if getattr(graph, "graph_schema_version", None) != V10_GRAPH_SCHEMA_VERSION:
            raise ValueError(f"repeat graph schema mismatch for {logical}")
        if (
            getattr(graph, "representation_extractor_version", None)
            != V10_REPRESENTATION_EXTRACTOR_VERSION
        ):
            raise ValueError(f"repeat extractor mismatch for {logical}")
        if list(getattr(graph, "unclassified_call_ir", []) or []):
            raise ValueError(f"repeat contains unclassified call IR for {logical}")
        if list(getattr(graph, "call_mapping_errors", []) or []):
            raise ValueError(f"repeat contains call mapping errors for {logical}")
        if getattr(graph, "classified_call_ir_counts", None) != getattr(
            graph, "emitted_call_edge_counts", None
        ):
            raise ValueError(f"repeat call counts differ for {logical}")

        tokens = torch.load(paths["tokens"], map_location="cpu", weights_only=True)
        _validate_tokens(torch, tokens, sidecar)
        accepted_token = accepted_v9_root / source / f"{contract_id}.tokens.pt"
        if not accepted_token.is_file():
            raise FileNotFoundError(accepted_token)
        if _sha256(paths["tokens"]) != _sha256(accepted_token):
            raise ValueError(f"repeat token bytes changed for {logical}")
        records.append(
            {
                "contract": logical,
                "graph_sha256": _sha256(paths["graph"]),
                "tokens_sha256": _sha256(paths["tokens"]),
                "sidecar_sha256": _sha256(paths["sidecar"]),
            }
        )
    return records


def build_repeat(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.output_root.name != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError(f"output root must be named {V10_REPRESENTATION_ROOT_NAME!r}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"repeat output root is not empty: {args.output_root}")

    runtime = _require_runtime()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    identities = _unexpected_identities(audit)
    if len(identities) != len(set(identities)):
        raise ValueError("audit contains duplicate unexpected identities")
    exception_ids = set(V10_SLITHER_RUNTIME_EXCEPTIONS)
    forbidden = [logical for logical in identities if logical.split("/", 1)[1] in exception_ids]
    if forbidden:
        raise ValueError(f"audit includes non-primary runtime identities: {forbidden[:5]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_args: list[tuple[str, ...]] = []
    for logical in identities:
        source, contract_id = logical.split("/", 1)
        meta_path = args.preprocessed_root / source / f"{contract_id}.meta.json"
        source_path = args.preprocessed_root / source / f"{contract_id}.sol"
        accepted_token = args.accepted_v9_root / source / f"{contract_id}.tokens.pt"
        for path in (meta_path, source_path, accepted_token):
            if not path.is_file():
                raise FileNotFoundError(path)
        output_dir = args.output_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        worker_args.append(
            (
                source,
                str(meta_path),
                str(meta_path.parent),
                str(output_dir),
                V10_GRAPH_SCHEMA_VERSION,
                V10_REPRESENTATION_EXTRACTOR_VERSION,
                str(args.accepted_v9_root / source),
            )
        )

    results = _run_workers(worker_args, args.workers)
    failures = [failure for passed, _, failure in results if not passed and failure]
    if failures:
        return {
            "schema": SCHEMA,
            "passed": False,
            "runtime": runtime,
            "source_audit_sha256": _sha256(args.audit),
            "candidate_binding_digest_sha256": audit.get(
                "candidate_binding_digest_sha256"
            ),
            "contracts_requested": len(identities),
            "contracts_generated": sum(passed for passed, _, _ in results),
            "failures": failures,
            "physical_acceptance": False,
            "training_authorized": False,
        }

    records = _validate_outputs(
        output_root=args.output_root,
        accepted_v9_root=args.accepted_v9_root,
        identities=identities,
    )
    return {
        "schema": SCHEMA,
        "passed": True,
        "runtime": runtime,
        "source_audit_sha256": _sha256(args.audit),
        "candidate_binding_digest_sha256": audit.get(
            "candidate_binding_digest_sha256"
        ),
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "contracts_requested": len(identities),
        "contracts_generated": len(records),
        "workers": args.workers,
        "records": records,
        "physical_acceptance": False,
        "training_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data_module/data/sentinel-preprocessed-r4-v2"),
    )
    parser.add_argument(
        "--accepted-v9-root",
        type=Path,
        default=Path("data_module/data/representations-r4-v2"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_repeat(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
