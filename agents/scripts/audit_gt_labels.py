#!/usr/bin/env python3
"""
Audit all contracts in the manual_hand_written_contracts/ corpus for complete
GT labels. Runs Slither + Aderyn on each, extracts detector findings, maps
them to SENTINEL vulnerability classes, and flags any class detected by tools
that is NOT in the contract's `// expect:` header.

Output: a JSON report per contract + a summary of label gaps.

Usage:
    cd ~/projects/sentinel/agents
    PATH="$PWD/.venv/bin:$HOME/.cargo/bin:$PATH" .venv/bin/python scripts/audit_gt_labels.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Make agents/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestration.routing import CLASS_TO_DETECTORS

# Reverse map: detector name → classes
DETECTOR_TO_CLASSES: dict[str, list[str]] = {}
for cls, dets in CLASS_TO_DETECTORS.items():
    for det in dets:
        DETECTOR_TO_CLASSES.setdefault(det, []).append(cls)

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "manual_hand_written_contracts"


def parse_expect_header(sol_path: Path) -> list[str]:
    """Parse `// expect:` labels from the first 20 lines."""
    try:
        for line in sol_path.read_text().splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("// expect:"):
                payload = stripped[len("// expect:"):].strip()
                return [c.strip() for c in payload.split(",") if c.strip()]
    except OSError:
        pass
    return []


def run_slither(sol_path: Path) -> list[dict]:
    """Run Slither on a contract, return findings list."""
    try:
        proc = subprocess.run(
            ["slither", str(sol_path), "--json", "-"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            # Slither writes JSON to stdout even on non-zero exit (detectors fire)
            pass
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []
        findings = []
        for d in data.get("results", {}).get("detectors", []):
            findings.append({
                "check": d.get("check", ""),
                "impact": d.get("impact", ""),
                "confidence": d.get("confidence", ""),
                "description": d.get("description", "")[:200],
            })
        return findings
    except Exception as e:
        return [{"error": f"slither failed: {e}"}]


def run_aderyn(sol_path: Path) -> list[dict]:
    """Run Aderyn on a contract (needs a temp dir), return findings list."""
    try:
        with tempfile.TemporaryDirectory() as td:
            import shutil
            shutil.copy(sol_path, td)
            proc = subprocess.run(
                ["aderyn", "--stdout", td],
                capture_output=True, text=True, timeout=30,
            )
            # Aderyn outputs markdown — parse the issue lines
            findings = []
            for line in proc.stdout.splitlines():
                # Lines like: "| H-01 | reentrancy-state-change | ..."
                # or "| L-01 | unchecked-low-level-call | ..."
                m = re.match(r'\|\s*([HL])-\d+\s*\|\s*([a-z-]+)\s*\|', line)
                if m:
                    severity = "High" if m.group(1) == "H" else "Low"
                    detector = m.group(2)
                    findings.append({
                        "check": detector,
                        "impact": severity,
                        "confidence": "Medium",
                        "description": line.strip()[:200],
                    })
            return findings
    except Exception as e:
        return [{"error": f"aderyn failed: {e}"}]


def map_finding_to_classes(finding: dict) -> list[str]:
    """Map a Slither/Aderyn finding to SENTINEL classes via CLASS_TO_DETECTORS."""
    detector = finding.get("check", "").lower().strip()
    if not detector:
        return []
    # Direct match in CLASS_TO_DETECTORS
    classes = set()
    for det, cls_list in DETECTOR_TO_CLASSES.items():
        if det.lower() == detector:
            classes.update(cls_list)
    # Token-overlap match (same as _signals_for_class in nodes.py)
    det_tokens = {tok for tok in detector.split("-") if len(tok) > 3}
    for cls, dets in CLASS_TO_DETECTORS.items():
        cls_tokens = {tok for d in dets for tok in d.split("-") if len(tok) > 3}
        if det_tokens & cls_tokens:
            classes.add(cls)
    return sorted(classes)


def main():
    if not CORPUS_DIR.is_dir():
        print(f"ERROR: corpus dir not found: {CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path("eval/gt_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    sol_files = sorted(CORPUS_DIR.rglob("*.sol"))
    print(f"Auditing {len(sol_files)} contracts...")

    results = []
    label_gaps = []

    for i, sol_path in enumerate(sol_files, 1):
        stem = sol_path.stem
        rel_path = sol_path.relative_to(CORPUS_DIR)
        gt_labels = parse_expect_header(sol_path)
        gt_set = set(gt_labels)

        # Run tools
        slither_findings = run_slither(sol_path)
        aderyn_findings = run_aderyn(sol_path)

        # Map findings to classes
        detected_classes = set()
        for f in slither_findings + aderyn_findings:
            if "error" in f:
                continue
            for cls in map_finding_to_classes(f):
                detected_classes.add(cls)

        # Find gaps: classes detected by tools but NOT in GT labels
        # Exclude "NonVulnerable" (that's not a real class, it's the safe folder)
        gaps = detected_classes - gt_set
        gaps.discard("NonVulnerable")

        result = {
            "stem": stem,
            "path": str(rel_path),
            "gt_labels": gt_labels,
            "slither_findings": slither_findings,
            "aderyn_findings": aderyn_findings,
            "detected_classes": sorted(detected_classes),
            "label_gaps": sorted(gaps),
        }
        results.append(result)

        if gaps:
            label_gaps.append({
                "stem": stem,
                "path": str(rel_path),
                "gt_labels": gt_labels,
                "gaps": sorted(gaps),
                "detected_classes": sorted(detected_classes),
            })

        if i % 10 == 0 or i == len(sol_files):
            print(f"  [{i}/{len(sol_files)}] {len(label_gaps)} gaps found so far")

    # Write per-contract results
    (output_dir / "gt_audit_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Write gap summary
    (output_dir / "gt_label_gaps.json").write_text(json.dumps(label_gaps, indent=2, default=str))

    # Console summary
    print()
    print("=" * 72)
    print(f"GT LABEL AUDIT COMPLETE — {len(sol_files)} contracts")
    print(f"Label gaps found: {len(label_gaps)} contracts with missing classes")
    print("=" * 72)
    for gap in label_gaps:
        print(f"  {gap['stem']}")
        print(f"    GT: {gap['gt_labels']}")
        print(f"    Gaps: {gap['gaps']}")
        print(f"    All detected: {gap['detected_classes']}")
        print()
    print(f"Full results: {output_dir}/gt_audit_results.json")
    print(f"Gap summary:  {output_dir}/gt_label_gaps.json")


if __name__ == "__main__":
    main()
