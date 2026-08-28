#!/usr/bin/env python3
"""Stage a fresh V10 V2.5 candidate from the primary Slither-0.10 attempt.

The complete primary attempt is intentionally allowed to fail only for identities
listed in ``V10_SLITHER_RUNTIME_EXCEPTIONS``.  This tool validates that boundary
fail-closed, then transfers only the successful V2.5 graph/token/sidecar triples
into a fresh candidate root.  The declared runtime exception identities remain
missing for a later identity-bound regression under their required runtime.

This is a build-staging operation, not candidate binding, physical acceptance, or
training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    V10_GRAPH_SCHEMA_VERSION,
    V10_PRIMARY_SLITHER_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
    V10_SLITHER_RUNTIME_EXCEPTIONS,
)


SCHEMA = "sentinel-r4-v10-v25-primary-stage-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for sidecar in sorted(root.glob("*/*.rep.json")):
        contract_id = sidecar.name.removesuffix(".rep.json")
        key = (sidecar.parent.name, contract_id)
        if key in result:
            raise ValueError(f"duplicate representation identity: {key}")
        result[key] = sidecar
    return result


def _exception_keys(accepted: dict[tuple[str, str], Path]) -> set[tuple[str, str]]:
    by_id: dict[str, list[tuple[str, str]]] = {}
    for key in accepted:
        by_id.setdefault(key[1], []).append(key)
    keys: set[tuple[str, str]] = set()
    for contract_id in V10_SLITHER_RUNTIME_EXCEPTIONS:
        matches = by_id.get(contract_id) or []
        if len(matches) != 1:
            raise ValueError(
                f"runtime exception {contract_id} resolves to {len(matches)} accepted identities"
            )
        keys.add(matches[0])
    return keys


def _deferred_failure_keys(root: Path) -> set[tuple[str, str]]:
    failed: set[tuple[str, str]] = set()
    for path in sorted(root.glob("*/representation_failures.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            meta_path = str(row.get("meta_path") or "")
            if not meta_path.endswith(".meta.json"):
                raise ValueError(f"invalid primary-attempt failure row in {path}")
            contract_id = meta_path.removesuffix(".meta.json")
            if row.get("error_type") != "IdentityBoundRuntimeDeferred":
                raise ValueError(
                    f"primary-attempt failure is not an identity-bound runtime deferral: {path}"
                )
            required_runtime = V10_SLITHER_RUNTIME_EXCEPTIONS.get(contract_id)
            expected_detail = (
                None
                if required_runtime is None
                else f"required slither-analyzer {required_runtime} runtime"
            )
            if expected_detail is None or expected_detail not in str(row.get("error") or ""):
                raise ValueError(
                    f"primary-attempt deferral has the wrong required runtime: {path}"
                )
            key = (path.parent.name, contract_id)
            if key in failed:
                raise ValueError(f"duplicate primary-attempt deferral: {key}")
            failed.add(key)
    return failed


def _validate_primary_sidecar(
    logical: str,
    sidecar: dict[str, Any],
) -> None:
    if sidecar.get("schema_version") != V10_GRAPH_SCHEMA_VERSION:
        raise ValueError(f"{logical} schema mismatch")
    if sidecar.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError(f"{logical} extractor mismatch")
    if sidecar.get("token_lineage") != "accepted_v9_byte_copy":
        raise ValueError(f"{logical} token lineage mismatch")
    if str(sidecar.get("graph_extraction_mode") or "").startswith("slither_parse_only"):
        raise ValueError(f"{logical} remained parse-only")
    if bool(sidecar.get("graph_analysis_degraded")):
        raise ValueError(f"{logical} reports degraded analysis")
    if list(sidecar.get("unclassified_call_ir") or []):
        raise ValueError(f"{logical} contains unclassified call IR")
    if list(sidecar.get("call_mapping_errors") or []):
        raise ValueError(f"{logical} contains call mapping errors")
    if sidecar.get("classified_call_ir_counts") != sidecar.get(
        "emitted_call_edge_counts"
    ):
        raise ValueError(f"{logical} classified/emitted call counts differ")
    runtime = dict(sidecar.get("slither_runtime") or {})
    if runtime.get("slither_analyzer") != V10_PRIMARY_SLITHER_VERSION:
        raise ValueError(f"{logical} was not generated under primary Slither")
    if runtime.get("runtime_role") != "primary":
        raise ValueError(f"{logical} runtime role is not primary")
    if runtime.get("required_for_physical_acceptance") != V10_PRIMARY_SLITHER_VERSION:
        raise ValueError(f"{logical} primary runtime binding mismatch")


def _transfer(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def stage_primary_attempt(args: argparse.Namespace) -> dict[str, Any]:
    if args.primary_attempt_root.name != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError("primary attempt root has the wrong versioned basename")
    if args.output_root.name != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError("staged output root has the wrong versioned basename")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"staged output root is not empty: {args.output_root}")

    accepted = _inventory(args.accepted_v9_root)
    attempt = _inventory(args.primary_attempt_root)
    exception_keys = _exception_keys(accepted)
    expected_primary = set(accepted) - exception_keys

    if set(attempt) != expected_primary:
        missing = sorted(expected_primary - set(attempt))
        extra = sorted(set(attempt) - expected_primary)
        raise ValueError(
            "primary attempt population mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    failed_keys = _deferred_failure_keys(args.primary_attempt_root)
    if failed_keys != exception_keys:
        raise ValueError(
            "primary attempt failure set does not exactly match runtime exceptions: "
            f"observed={sorted(failed_keys)} expected={sorted(exception_keys)}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    transfers: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for source, contract_id in sorted(expected_primary):
        logical = f"{source}/{contract_id}"
        source_dir = args.primary_attempt_root / source
        accepted_dir = args.accepted_v9_root / source
        destination_dir = args.output_root / source
        destination_dir.mkdir(parents=True, exist_ok=True)

        sidecar_path = source_dir / f"{contract_id}.rep.json"
        graph_path = source_dir / f"{contract_id}.pt"
        token_path = source_dir / f"{contract_id}.tokens.pt"
        accepted_token = accepted_dir / f"{contract_id}.tokens.pt"
        for path in (sidecar_path, graph_path, token_path, accepted_token):
            if not path.is_file():
                raise FileNotFoundError(path)

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        _validate_primary_sidecar(logical, sidecar)
        if sidecar.get("sha256") != contract_id or sidecar.get("source") != source:
            raise ValueError(f"{logical} sidecar identity mismatch")
        token_sha = _sha256(token_path)
        accepted_token_sha = _sha256(accepted_token)
        if token_sha != accepted_token_sha:
            raise ValueError(f"{logical} token bytes changed")

        for path in (graph_path, token_path, sidecar_path):
            transfers[_transfer(path, destination_dir / path.name)] += 1

        records.append(
            {
                "contract": logical,
                "graph_sha256": _sha256(graph_path),
                "tokens_sha256": token_sha,
                "sidecar_sha256": _sha256(sidecar_path),
            }
        )

    staged = _inventory(args.output_root)
    if set(staged) != expected_primary:
        raise RuntimeError("staged primary population does not reconcile after transfer")

    source_manifest_hashes = {
        path.parent.name: _sha256(path)
        for path in sorted(
            args.primary_attempt_root.glob("*/repaired_representation_manifest.json")
        )
    }
    report = {
        "schema": SCHEMA,
        "passed": True,
        "status": "PRIMARY_STAGE_PASS_EXCEPTION_FILL_REQUIRED",
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "primary_slither_version": V10_PRIMARY_SLITHER_VERSION,
        "accepted_v9_contracts": len(accepted),
        "primary_attempt_contracts": len(attempt),
        "staged_primary_contracts": len(staged),
        "runtime_exception_contracts": len(exception_keys),
        "missing_runtime_exception_identities": [
            f"{source}/{contract_id}" for source, contract_id in sorted(exception_keys)
        ],
        "primary_attempt_root": str(args.primary_attempt_root),
        "primary_attempt_source_manifest_sha256": source_manifest_hashes,
        "transfer_file_counts": dict(sorted(transfers.items())),
        "records_digest_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "physical_acceptance": False,
        "training_authorized": False,
        "next_required_step": "generate exactly the declared runtime exception identities under their identity-bound runtimes, then bind",
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-attempt-root", type=Path, required=True)
    parser.add_argument(
        "--accepted-v9-root",
        type=Path,
        default=Path("data_module/data/representations-r4-v2"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = stage_primary_attempt(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
