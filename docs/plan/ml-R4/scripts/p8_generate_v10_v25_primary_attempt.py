#!/usr/bin/env python3
"""Generate the primary-runtime portion of a fresh V10 V2.5 candidate attempt.

This is Stage A of the heterogeneous-runtime V2.5 build.  It enumerates the
complete accepted-V9 population, verifies the same identities exist in repaired
preprocessing, and generates every ordinary identity under exact primary
Slither 0.10.0.  Identities declared in ``V10_SLITHER_RUNTIME_EXCEPTIONS`` are
never extracted in this process; they are written only as structured deferred
failure records for later identity-bound generation.

The output is an attempt root, not a bound candidate.  Physical acceptance and
training authorization remain false.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    V10_GRAPH_SCHEMA_VERSION,
    V10_PRIMARY_SLITHER_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
    V10_SLITHER_RUNTIME_EXCEPTIONS,
)
from sentinel_data.representation.r4_orchestrator import _represent_worker


SCHEMA = "sentinel-r4-v10-v25-primary-attempt-v1"


def _inventory_sidecars(root: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for path in sorted(root.glob("*/*.rep.json")):
        key = (path.parent.name, path.name.removesuffix(".rep.json"))
        if key in result:
            raise ValueError(f"duplicate accepted identity: {key}")
        result[key] = path
    return result


def _inventory_preprocessed(root: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for path in sorted(root.glob("*/*.meta.json")):
        key = (path.parent.name, path.name.removesuffix(".meta.json"))
        if key in result:
            raise ValueError(f"duplicate preprocessed identity: {key}")
        source_path = path.with_name(f"{key[1]}.sol")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        result[key] = path
    return result


def _require_primary_runtime() -> dict[str, str]:
    slither = importlib.metadata.version("slither-analyzer")
    crytic = importlib.metadata.version("crytic-compile")
    if slither != V10_PRIMARY_SLITHER_VERSION:
        raise RuntimeError(
            "V2.5 primary attempt requires exact slither-analyzer "
            f"{V10_PRIMARY_SLITHER_VERSION}; found {slither}"
        )
    return {
        "slither_analyzer": slither,
        "crytic_compile": crytic,
        "runtime_role": "primary",
    }


def _resolve_exception_keys(
    accepted: dict[tuple[str, str], Path],
) -> set[tuple[str, str]]:
    by_id: dict[str, list[tuple[str, str]]] = {}
    for key in accepted:
        by_id.setdefault(key[1], []).append(key)
    result: set[tuple[str, str]] = set()
    for contract_id in V10_SLITHER_RUNTIME_EXCEPTIONS:
        matches = by_id.get(contract_id) or []
        if len(matches) != 1:
            raise ValueError(
                f"runtime exception {contract_id} resolves to {len(matches)} accepted identities"
            )
        result.add(matches[0])
    return result


def _deferred_failure(contract_id: str) -> dict[str, str]:
    required = V10_SLITHER_RUNTIME_EXCEPTIONS[contract_id]
    return {
        "meta_path": f"{contract_id}.meta.json",
        "error_type": "IdentityBoundRuntimeDeferred",
        "error": (
            "V10 primary attempt intentionally deferred this identity to its "
            f"required slither-analyzer {required} runtime"
        ),
    }


def _run_workers(
    worker_args: list[tuple[str, ...]],
    workers: int,
) -> list[tuple[bool, dict[str, Any] | None, dict[str, str] | None]]:
    if workers == 1:
        return list(map(_represent_worker, worker_args))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_represent_worker, worker_args, chunksize=1))


def build_primary_attempt(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.output_root.name != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError("output root has the wrong versioned basename")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"primary attempt root is not empty: {args.output_root}")

    runtime = _require_primary_runtime()
    accepted = _inventory_sidecars(args.accepted_v9_root)
    preprocessed = _inventory_preprocessed(args.preprocessed_root)
    if set(accepted) != set(preprocessed):
        missing = sorted(set(accepted) - set(preprocessed))
        extra = sorted(set(preprocessed) - set(accepted))
        raise ValueError(
            "accepted/preprocessed population mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    exception_keys = _resolve_exception_keys(accepted)
    ordinary_keys = set(accepted) - exception_keys
    args.output_root.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(accepted):
        by_source.setdefault(key[0], []).append(key)

    source_results: list[dict[str, Any]] = []
    unexpected_failures: list[dict[str, str]] = []
    total_written = 0
    total_deferred = 0
    started = time.monotonic()

    for source, source_keys in sorted(by_source.items()):
        output_dir = args.output_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        accepted_tokens_dir = args.accepted_v9_root / source
        ordinary_source_keys = [key for key in source_keys if key in ordinary_keys]
        deferred_source_keys = [key for key in source_keys if key in exception_keys]

        worker_args = [
            (
                source,
                str(preprocessed[key]),
                str(args.preprocessed_root / source),
                str(output_dir),
                V10_GRAPH_SCHEMA_VERSION,
                V10_REPRESENTATION_EXTRACTOR_VERSION,
                str(accepted_tokens_dir),
            )
            for key in ordinary_source_keys
        ]
        results = _run_workers(worker_args, args.workers)
        failures = [
            failure
            for passed, _, failure in results
            if not passed and failure is not None
        ]
        failures.extend(_deferred_failure(key[1]) for key in deferred_source_keys)
        failures.sort(key=lambda row: row["meta_path"])
        if failures:
            with (output_dir / "representation_failures.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for row in failures:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

        ordinary_failures = [
            row
            for row in failures
            if row.get("error_type") != "IdentityBoundRuntimeDeferred"
        ]
        unexpected_failures.extend(
            {"source": source, **row} for row in ordinary_failures
        )
        written = sum(passed for passed, _, _ in results)
        deferred = len(deferred_source_keys)
        total_written += written
        total_deferred += deferred
        mode_counts = Counter(
            str(provenance.get("graph_extraction_mode"))
            for passed, provenance, _ in results
            if passed and provenance is not None
        )

        manifest = {
            "schema": "sentinel-r4-v10-v25-primary-attempt-source-v1",
            "status": "PRIMARY_ATTEMPT_PENDING_STAGING",
            "source": source,
            "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
            "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
            "contracts_requested": len(source_keys),
            "ordinary_contracts_requested": len(ordinary_source_keys),
            "identity_bound_runtime_deferred": deferred,
            "representations_written": written,
            "representations_failed": len(failures),
            "unexpected_failures": len(ordinary_failures),
            "representation_workers": args.workers,
            "graph_extraction_mode_counts": dict(sorted(mode_counts.items())),
            "physical_acceptance": False,
            "training_authorized": False,
        }
        (output_dir / "repaired_representation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_results.append(manifest)

    observed = _inventory_sidecars(args.output_root)
    expected_written = len(ordinary_keys)
    population_ok = set(observed) == ordinary_keys
    passed = (
        not unexpected_failures
        and population_ok
        and total_written == expected_written
        and total_deferred == len(exception_keys)
    )

    return {
        "schema": SCHEMA,
        "passed": passed,
        "status": (
            "PRIMARY_ATTEMPT_PASS_EXCEPTION_FILL_REQUIRED"
            if passed
            else "PRIMARY_ATTEMPT_FAIL"
        ),
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "runtime": runtime,
        "accepted_v9_contracts": len(accepted),
        "preprocessed_contracts": len(preprocessed),
        "ordinary_contracts_expected": len(ordinary_keys),
        "representations_written": total_written,
        "runtime_exception_contracts": len(exception_keys),
        "runtime_exception_identities": [
            f"{source}/{contract_id}" for source, contract_id in sorted(exception_keys)
        ],
        "runtime_exceptions_deferred": total_deferred,
        "observed_attempt_contracts": len(observed),
        "attempt_population_matches_expected_primary": population_ok,
        "unexpected_failures_total": len(unexpected_failures),
        "unexpected_failures": unexpected_failures[:200],
        "source_results": source_results,
        "duration_s": time.monotonic() - started,
        "physical_acceptance": False,
        "training_authorized": False,
        "next_required_step": "validate and stage the primary attempt before identity-bound exception generation",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_primary_attempt(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
