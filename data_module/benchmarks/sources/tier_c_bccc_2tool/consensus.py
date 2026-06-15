"""Tier C builder — BCCC 2-tool consensus (extends proven ME methodology).

Methodology (proven on MishandledException = 658 contracts from 5,154):
  1. Compile probe — try solc 0.4.24, 0.4.25, 0.5.0, 0.5.17
     Apply tempfile.TemporaryDirectory() per-call (fixes 99%→84.5% aderyn success)
  2. Slither audit with class-specific detector list
  3. Aderyn audit
  4. 2-tool consensus = intersection of slither + aderyn findings
  5. Tier 3 deep audit — manual code review on STRONG-only matches
     (precision ~21% in ME per project_bccc_2tool_audit_2026-06-14.md)
  6. Output: high-confidence contracts per class

Per-class detector lists (from SWC Registry + Slither docs):

  MishandledException: unchecked-transfer, arbitrary-send-eth, void-cst, tautology
  Reentrancy:          reentrancy-eth, reentrancy-no-eth, reentrancy-benign, reentrancy-events
  IntegerUO:           divide-before-multiply, incorrect-shift, tautology, void-cst
  UnusedReturn:        unchecked-lowlevel, unchecked-send, unchecked-transfer
  Timestamp:           timestamp (slither 0.10+)
  TransactionOrder:    tx-origin (slither 0.10+)
  DenialOfService:     calls-loop, locked-ether
  CallToUnknown:       arbitrary-send-eth, controlled-delegatecall
  ExternalBug:         arbitrary-send-eth, delegatecall-loop, msg-value-loop
  GasException:        gas-limit, tautology (most are FP per Phase 4)

Expected yields (Phase 4 audit + ME precedent):
  IntegerUO 16,740    →  5-10% high-confidence (~800-1,600)
  Reentrancy 17,698   →  2-5% high-confidence (~350-880)
  UnusedReturn 3,229  →  5-10% high-confidence (~160-320)
  Timestamp 2,674     →  10-20% high-confidence (~270-540)
  MishandledException → 658 (DONE)
  Others              →  < 5% (likely not worth the time)
"""
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
BCCC_DIR = REPO_ROOT / "BCCC-SCsVul-2024" / "SourceCodes"

# Class-specific detector lists (verified in Slither 0.11.5 + Aderyn 0.6.8 docs)
CLASS_DETECTORS = {
    "MishandledException": {
        "slither": ["unchecked-transfer", "arbitrary-send-eth", "void-cst", "tautology"],
        "aderyn": ["unchecked-send", "unchecked-transfer", "arbitrary-send-eth"],
    },
    "Reentrancy": {
        "slither": ["reentrancy-eth", "reentrancy-no-eth", "reentrancy-benign", "reentrancy-events"],
        "aderyn": ["reentrancy", "reentrancy-eth", "reentrancy-no-eth"],
    },
    "IntegerUO": {
        "slither": ["divide-before-multiply", "incorrect-shift", "tautology", "void-cst"],
        "aderyn": ["divide-before-multiply", "incorrect-shift"],
    },
    "UnusedReturn": {
        "slither": ["unchecked-lowlevel", "unchecked-send", "unchecked-transfer"],
        "aderyn": ["unchecked-lowlevel", "unchecked-send", "unchecked-transfer"],
    },
    "Timestamp": {
        "slither": ["timestamp"],
        "aderyn": ["timestamp"],
    },
    "TransactionOrderDependence": {
        "slither": ["tx-origin"],
        "aderyn": ["tx-origin"],
    },
    "DenialOfService": {
        "slither": ["calls-loop", "locked-ether"],
        "aderyn": ["calls-loop"],
    },
    "CallToUnknown": {
        "slither": ["arbitrary-send-eth", "controlled-delegatecall"],
        "aderyn": ["arbitrary-send-eth", "controlled-delegatecall"],
    },
    "ExternalBug": {
        "slither": ["arbitrary-send-eth", "delegatecall-loop", "msg-value-loop"],
        "aderyn": ["arbitrary-send-eth", "delegatecall-loop", "msg-value-loop"],
    },
    "GasException": {
        "slither": ["gas-limit", "tautology"],
        "aderyn": ["tautology"],
    },
}

# Solc versions to try (BCCC is 92% pre-0.6; need old versions)
SOLC_VERSIONS = [
    "/home/motafeq/.solc-select/artifacts/solc-0.4.24/solc-0.4.24",
    "/home/motafeq/.solc-select/artifacts/solc-0.4.25/solc-0.4.25",
    "/home/motafeq/.solc-select/artifacts/solc-0.5.0/solc-0.5.0",
    "/home/motafeq/.solc-select/artifacts/solc-0.5.17/solc-0.5.17",
    "/home/motafeq/.solc-select/artifacts/solc-0.4.21/solc-0.4.21",
    "/home/motafeq/.solc-select/artifacts/solc-0.4.13/solc-0.4.13",
]
SLITHER = "/home/motafeq/.venv/bin/slither"
ADERYN = "/home/motafeq/.cargo/bin/aderyn"


def compile_with_solc(sol_path: Path) -> str | None:
    """Try to compile with our solc versions; return version on success."""
    for solc in SOLC_VERSIONS:
        if not Path(solc).exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = subprocess.run(
                    [solc, "--bin", str(sol_path)],
                    capture_output=True, timeout=30, cwd=tmp
                )
                if result.returncode == 0:
                    return Path(solc).parent.name
            except (subprocess.TimeoutExpired, Exception):
                continue
    return None


def slither_audit(sol_path: Path, detectors: list) -> set:
    """Returns set of detector names that fired."""
    findings = set()
    for det in detectors:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = subprocess.run(
                    [SLITHER, str(sol_path), "--detect", det, "--json", "-"],
                    capture_output=True, timeout=60, cwd=tmp
                )
                output = result.stdout.decode("utf-8", errors="replace")
                if result.returncode != 0 and f'"{det}"' in output:
                    findings.add(det)
                elif '"impact": "High"' in output or '"impact": "Medium"' in output:
                    # Any high/medium in the JSON means a detector fired
                    if f'"{det}"' in output:
                        findings.add(det)
            except (subprocess.TimeoutExpired, Exception):
                continue
    return findings


def aderyn_audit(sol_path: Path, detectors: list) -> set:
    """Returns set of detector names that fired."""
    findings = set()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "report.md"
        try:
            result = subprocess.run(
                [ADERYN, str(sol_path), "--output", str(report_path)],
                capture_output=True, timeout=60
            )
            if report_path.exists():
                report = report_path.read_text()
                for det in detectors:
                    if det in report:
                        findings.add(det)
        except (subprocess.TimeoutExpired, Exception):
            pass
    return findings


def audit_class(class_name: str, bccc_class_dir: Path) -> dict:
    """Run 2-tool consensus audit on a BCCC class folder."""
    detectors = CLASS_DETECTORS.get(class_name)
    if not detectors:
        return {"class": class_name, "error": "no detector config"}

    print(f"\n=== Auditing {class_name} ({bccc_class_dir}) ===")
    sol_files = sorted(bccc_class_dir.glob("*.sol"))
    print(f"  Total contracts: {len(sol_files)}")

    # TODO: implement the full audit
    # For now, this is a skeleton — the actual implementation would:
    # 1. Compile each contract (with solc retry)
    # 2. Run slither with class detectors
    # 3. Run aderyn with class detectors
    # 4. Compute intersection (2-tool consensus)
    # 5. For STRONG-only matches, run Tier 3 deep audit
    # 6. Save high-confidence contracts to results/

    return {
        "class": class_name,
        "total_contracts": len(sol_files),
        "compilable": 0,  # TODO
        "slither_findings": 0,  # TODO
        "aderyn_findings": 0,  # TODO
        "two_tool_consensus": 0,  # TODO
        "strong_only": 0,  # TODO
        "high_confidence": 0,  # TODO
        "status": "SKELETON — TODO: implement full audit pipeline (3-5 days effort)",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_name", help="Specific BCCC class to audit")
    parser.add_argument("--all", action="store_true", help="Audit all classes")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.class_name:
        classes = [args.class_name]
    else:
        classes = list(CLASS_DETECTORS.keys())

    for cls in classes:
        bccc_dir = BCCC_DIR / cls
        if not bccc_dir.exists():
            print(f"WARNING: BCCC class dir not found: {bccc_dir}")
            continue
        result = audit_class(cls, bccc_dir)
        out = results_dir / f"{cls.lower().replace(' ', '_')}_2tool.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"  Result written: {out}")


if __name__ == "__main__":
    main()
