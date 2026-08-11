#!/usr/bin/env python3
"""Run the complete local Phase-3 G3 preparation gate in fail-closed order.

This wrapper intentionally does not invoke DVC. From repository root:

    ml/.venv/bin/python docs/plan/ml-R4/scripts/p3_run_local_gate.py

Publication is manifest-last: all production outputs are first materialized and
validated inside a repository-local staging directory. Canonical output paths
are replaced only after semantic, strict schema-surface, and cryptographic
artifact-binding checks pass. An interrupted/failed preparation therefore
cannot publish a new VALIDATED manifest for a partial ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def root_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    resolved_root = ROOT.resolve()
    resolved = root_path(path).resolve()
    try:
        return str(resolved.relative_to(resolved_root))
    except ValueError as exc:
        raise ValueError(f"output path must remain inside repository: {path}") from exc


def generation_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = proc.stdout.strip().lower()
    if proc.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(
            "could not resolve exact Git generation commit; run the gate from a Git checkout"
        )
    return commit


def run_step(label: str, command: list[str]) -> int:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit={result.returncode})", file=sys.stderr)
    else:
        print(f"PASS: {label}")
    return result.returncode


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_manifest_for_binding(
    manifest_path: Path,
    *,
    ledger_path: Path,
    evidence_path: Path,
    strict_report: Path,
    commit: str,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "VALIDATED"
    manifest["generation_commit"] = commit
    manifest["ledger_parquet"] = {
        "path": repo_relative(ledger_path),
        "sha256": sha256_file(ledger_path),
    }
    manifest["evidence_jsonl"] = {
        "path": repo_relative(evidence_path),
        "sha256": sha256_file(root_path(evidence_path)),
    }
    manifest["validation_report"] = {
        "path": repo_relative(strict_report),
        "sha256": sha256_file(strict_report),
    }
    limitations = list(manifest.get("limitations") or [])
    marker = "Final Phase-3 validation identity is the strict schema-surface + semantic report."
    if marker not in limitations:
        limitations.append(marker)
    manifest["limitations"] = limitations
    write_json(manifest_path, manifest)
    return manifest


def publish_file(staged: Path, destination: Path) -> None:
    destination = root_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(destination)


def mark_manifest_failed(manifest_path: Path, reason: str) -> None:
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    manifest["status"] = "FAILED"
    limitations = list(manifest.get("limitations") or [])
    marker = f"Artifact publication/binding failure: {reason}"
    if marker not in limitations:
        limitations.append(marker)
    manifest["limitations"] = limitations
    write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--representations-root",
        type=Path,
        default=Path("data_module/data/representations"),
    )
    p.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("docs/plan/ml-R4/ledger/evidence_ledger_v1.parquet"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/plan/ml-R4/manifests/evidence_ledger_v1.materialized.json"),
    )
    p.add_argument(
        "--evidence",
        type=Path,
        default=Path("docs/plan/ml-R4/manifests/evidence_items_v1.jsonl"),
    )
    p.add_argument(
        "--semantic-report",
        type=Path,
        default=Path("docs/plan/ml-R4/findings/04_evidence_ledger_validation_report.json"),
    )
    p.add_argument(
        "--strict-report",
        type=Path,
        default=Path("docs/plan/ml-R4/findings/04_evidence_ledger_strict_validation_report.json"),
    )
    p.add_argument(
        "--binding-report",
        type=Path,
        default=Path("docs/plan/ml-R4/findings/04_evidence_ledger_artifact_binding_report.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable

    self_tests = [
        (
            "semantic validator self-tests",
            [python, str(SCRIPT_DIR / "test_p3_validate_evidence_ledger.py")],
        ),
        (
            "strict schema-surface validator self-tests",
            [python, str(SCRIPT_DIR / "test_p3_validate_evidence_ledger_strict.py")],
        ),
        (
            "artifact-binding validator self-tests",
            [python, str(SCRIPT_DIR / "test_p3_validate_artifact_bindings.py")],
        ),
    ]
    for label, command in self_tests:
        rc = run_step(label, command)
        if rc:
            return rc

    try:
        commit = generation_commit()
        # Validate all eventual publication paths before expensive materialization.
        for path in (
            args.output_parquet,
            args.manifest,
            args.evidence,
            args.semantic_report,
            args.strict_report,
            args.binding_report,
        ):
            repo_relative(path)
    except (ValueError, RuntimeError) as exc:
        print(f"GATE PRECONDITION ERROR: {exc}", file=sys.stderr)
        return 2

    staging_parent = ROOT / "docs/plan/ml-R4"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p3_gate_", dir=staging_parent) as tmp:
        stage = Path(tmp)
        staged_ledger = stage / "evidence_ledger_v1.parquet"
        staged_manifest = stage / "evidence_ledger_v1.materialized.json"
        staged_semantic = stage / "semantic_validation.json"
        staged_strict = stage / "strict_validation.json"
        staged_binding = stage / "artifact_binding.json"

        materialize = [
            python,
            str(SCRIPT_DIR / "p3_materialize_evidence_ledger.py"),
            "--representations-root",
            str(args.representations_root),
            "--evidence",
            str(args.evidence),
            "--output-parquet",
            str(staged_ledger),
            "--output-manifest",
            str(staged_manifest),
            "--output-report",
            str(staged_semantic),
        ]
        rc = run_step("protected population materialization (staged)", materialize)
        if rc:
            return rc

        strict_cmd = [
            python,
            str(SCRIPT_DIR / "p3_validate_evidence_ledger_strict.py"),
            "--ledger",
            str(staged_ledger),
            "--evidence",
            str(args.evidence),
            "--manifest",
            str(staged_manifest),
            "--report",
            str(staged_strict),
        ]
        rc = run_step("strict production-ledger validation (staged)", strict_cmd)
        if rc:
            return rc

        # First bind to the staged artifacts. This proves the complete candidate
        # package before any canonical file is replaced.
        prepare_manifest_for_binding(
            staged_manifest,
            ledger_path=staged_ledger,
            evidence_path=args.evidence,
            strict_report=staged_strict,
            commit=commit,
        )
        binding_cmd = [
            python,
            str(SCRIPT_DIR / "p3_validate_artifact_bindings.py"),
            "--manifest",
            str(staged_manifest),
            "--root",
            str(ROOT),
            "--report",
            str(staged_binding),
        ]
        rc = run_step("candidate artifact-binding validation (staged)", binding_cmd)
        if rc:
            return rc

        # Rewrite only path references for final publication. File identities do
        # not change during same-filesystem replace(). The manifest is published
        # last so no partial run can advertise a new VALIDATED package.
        manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
        manifest["ledger_parquet"]["path"] = repo_relative(args.output_parquet)
        manifest["validation_report"]["path"] = repo_relative(args.strict_report)
        write_json(staged_manifest, manifest)

        publish_file(staged_ledger, args.output_parquet)
        publish_file(staged_semantic, args.semantic_report)
        publish_file(staged_strict, args.strict_report)
        publish_file(staged_manifest, args.manifest)  # manifest last

    # Re-hash the canonical package after promotion. If anything changed during
    # publication, mark the manifest FAILED and keep G3 blocked.
    final_binding_cmd = [
        python,
        str(SCRIPT_DIR / "p3_validate_artifact_bindings.py"),
        "--manifest",
        str(args.manifest),
        "--root",
        str(ROOT),
        "--report",
        str(args.binding_report),
    ]
    rc = run_step("canonical artifact-binding validation", final_binding_cmd)
    if rc:
        mark_manifest_failed(root_path(args.manifest), "canonical artifact binding did not validate")
        return rc

    print("\n=== PHASE 3 LOCAL GATE PREPARATION PASS ===")
    print(f"ledger:          {args.output_parquet}")
    print(f"manifest:        {args.manifest}")
    print(f"strict report:   {args.strict_report}")
    print(f"binding report:  {args.binding_report}")
    print(f"generation SHA:  {commit}")
    print("G3 still requires repository review/registration before Phase 4 begins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
