#!/usr/bin/env python3
"""Run the complete local Phase-3 G3 preparation gate in fail-closed order.

This wrapper intentionally does not invoke DVC. From repository root:

    ml/.venv/bin/python docs/plan/ml-R4/scripts/p3_run_local_gate.py

It runs dataset-independent semantic + strict tests, materializes the protected
224,930-row ledger, then performs strict validation over the generated Parquet.
The materialized manifest is rewritten so the strict report—not the weaker
intermediate report—is the final validation identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_step(label: str, command: list[str]) -> int:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit={result.returncode})", file=sys.stderr)
    else:
        print(f"PASS: {label}")
    return result.returncode


def update_manifest(manifest_path: Path, strict_report: Path, passed: bool) -> None:
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "VALIDATED" if passed else "FAILED"
    if strict_report.exists():
        manifest["validation_report"] = {
            "path": str(strict_report.relative_to(ROOT)),
            "sha256": sha256_file(strict_report),
        }
    limitations = list(manifest.get("limitations") or [])
    marker = (
        "Final Phase-3 validation identity is the strict schema-surface + semantic report."
        if passed
        else "Strict Phase-3 validation failed; G3 must remain blocked until corrected and rerun."
    )
    if marker not in limitations:
        limitations.append(marker)
    manifest["limitations"] = limitations
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "--strict-report",
        type=Path,
        default=Path("docs/plan/ml-R4/findings/04_evidence_ledger_strict_validation_report.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable

    steps = [
        (
            "semantic validator self-tests",
            [python, str(SCRIPT_DIR / "test_p3_validate_evidence_ledger.py")],
        ),
        (
            "strict schema-surface validator self-tests",
            [python, str(SCRIPT_DIR / "test_p3_validate_evidence_ledger_strict.py")],
        ),
    ]
    for label, command in steps:
        rc = run_step(label, command)
        if rc:
            return rc

    materialize = [
        python,
        str(SCRIPT_DIR / "p3_materialize_evidence_ledger.py"),
        "--representations-root",
        str(args.representations_root),
        "--evidence",
        str(args.evidence),
        "--output-parquet",
        str(args.output_parquet),
        "--output-manifest",
        str(args.manifest),
    ]
    rc = run_step("protected population materialization", materialize)
    if rc:
        update_manifest(args.manifest, args.strict_report, False)
        return rc

    strict_cmd = [
        python,
        str(SCRIPT_DIR / "p3_validate_evidence_ledger_strict.py"),
        "--ledger",
        str(args.output_parquet),
        "--evidence",
        str(args.evidence),
        "--manifest",
        str(args.manifest),
        "--report",
        str(args.strict_report),
    ]
    rc = run_step("strict production-ledger validation", strict_cmd)
    passed = rc == 0
    update_manifest(args.manifest, args.strict_report, passed)
    if not passed:
        return rc

    print("\n=== PHASE 3 LOCAL GATE PREPARATION PASS ===")
    print(f"ledger:   {args.output_parquet}")
    print(f"manifest: {args.manifest}")
    print(f"report:   {args.strict_report}")
    print("G3 still requires repository review/registration before Phase 4 begins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
