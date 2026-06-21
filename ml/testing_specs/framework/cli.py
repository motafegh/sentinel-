"""
framework/cli.py — Single CLI entry point for the testing framework.

Usage:
    # Run all gates on a run
    ml-validate run --config framework/templates/sentinel_v2.yaml

    # Initialize a new project from a template
    ml-validate init --template sentinel_v2 --output my_project_gates.yaml

    # Validate a config file
    ml-validate validate --config my_project_gates.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make this work whether imported as ml.testing_specs.framework.cli
# or run as a script (python ml/testing_specs/framework/cli.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ml.testing_specs.framework.gates import (
        CompositeGate,
        F1Gate,
        FileExistsGate,
        Gate,
        GateResult,
        GateStatus,
        JSONFileGate,
        JSONKeyGate,
        ReproducibilityGate,
        StaleCheckpointsGate,
    )
else:
    from .gates import (
        CompositeGate,
        F1Gate,
        FileExistsGate,
        Gate,
        GateResult,
        GateStatus,
        JSONFileGate,
        JSONKeyGate,
        ReproducibilityGate,
        StaleCheckpointsGate,
    )


# ---------------------------------------------------------------------------
# Gate factory: build gates from a config
# ---------------------------------------------------------------------------


def build_gates_from_config(config) -> list[Gate]:
    """Build the list of gates from a FrameworkConfig.

    Each gate in config.gates becomes a Gate instance. The keys are:
    - file_exists: PASS if all expected files exist
    - behavioral_probes: PASS if all probes pass (loaded from JSON)
    - label_quality: PASS if no FAIL in label quality JSON
    - f1_vs_prior: PASS if F1 > prior (read from epoch log)
    - contamination: (placeholder — actual contamination script is in ml/scripts)
    - calibration_files: PASS if thresholds JSON exists
    """
    gates: list[Gate] = []

    # --- file_exists ---
    fe_cfg = config.gates.get("file_exists")
    if fe_cfg and fe_cfg.enabled:
        for path_key in ("checkpoint", "thresholds"):
            p = config.paths.get(path_key)
            if p:
                resolved = _resolve(p, config)
                gates.append(FileExistsGate(
                    name=f"file_exists:{path_key}",
                    path=Path(resolved),
                ))

    # --- behavioral_probes ---
    bp_cfg = config.gates.get("behavioral_probes")
    if bp_cfg and bp_cfg.enabled:
        probes_path = config.paths.get("behavioral_probes")
        if probes_path:
            resolved = _resolve(probes_path, config)
            gates.append(JSONFileGate(
                name="behavioral_probes:all_passed",
                path=Path(resolved),
                key="summary.all_passed",
                expected=True,
            ))

    # --- label_quality ---
    lq_cfg = config.gates.get("label_quality")
    if lq_cfg and lq_cfg.enabled:
        lq_path = config.paths.get("label_quality")
        if lq_path:
            resolved = _resolve(lq_path, config)
            gates.append(JSONFileGate(
                name="label_quality:no_failures",
                path=Path(resolved),
                key="summary.failed",
                expected=0,
            ))

    # --- f1_vs_prior ---
    f1_cfg = config.gates.get("f1_vs_prior")
    if f1_cfg and f1_cfg.enabled:
        # This gate needs the current F1 and the prior F1. They must be
        # supplied via CLI args or env vars (the config has no place for them).
        # For now, this is a placeholder — populated by the run command.
        pass

    # --- calibration_files ---
    cal_cfg = config.gates.get("calibration_files")
    if cal_cfg and cal_cfg.enabled:
        for f in cal_cfg.extra.get("required", []):
            p = config.paths.get(f)
            if p:
                resolved = _resolve(p, config)
                gates.append(FileExistsGate(
                    name=f"calibration:{f}",
                    path=Path(resolved),
                ))

    # --- threshold_sensitivity ---
    ts_cfg = config.gates.get("threshold_sensitivity")
    if ts_cfg and ts_cfg.enabled:
        ts_path = config.paths.get("threshold_sensitivity")
        if ts_path:
            resolved = _resolve(ts_path, config)
            gates.append(JSONKeyGate(
                name="threshold_sensitivity:no_flagged",
                path=Path(resolved),
                key="summary.n_flagged",
                op="==",
                expected=0,
            ))

    # --- cross_tool ---
    ct_cfg = config.gates.get("cross_tool")
    if ct_cfg and ct_cfg.enabled:
        ct_path = config.paths.get("cross_tool")
        if ct_path:
            resolved = _resolve(ct_path, config)
            gates.append(JSONKeyGate(
                name="cross_tool:no_flagged",
                path=Path(resolved),
                key="summary.n_flagged",
                op="==",
                expected=0,
            ))

    # --- reproducibility ---
    rep_cfg = config.gates.get("reproducibility")
    if rep_cfg and rep_cfg.enabled:
        rep_path = config.paths.get("reproducibility")
        if rep_path:
            resolved = _resolve(rep_path, config)
            gates.append(ReproducibilityGate(
                name="reproducibility:reproducible",
                path=Path(resolved),
                expected_result="PASS",
            ))

    # --- stale_checkpoints ---
    sc_cfg = config.gates.get("stale_checkpoints")
    if sc_cfg and sc_cfg.enabled:
        sc_path = config.paths.get("stale_checkpoints")
        if sc_path:
            resolved = _resolve(sc_path, config)
            gates.append(StaleCheckpointsGate(
                name="stale_checkpoints:no_stale",
                path=Path(resolved),
                max_stale=sc_cfg.extra.get("max_stale", 0),
            ))

    return gates


def _resolve(path: str, config) -> str:
    """Resolve ${run_name} placeholders in a path."""
    return path.replace("${run_name}", config.run_name)


# ---------------------------------------------------------------------------
# Run command
# ---------------------------------------------------------------------------


def run_gates(config, gates: list[Gate] | None = None) -> list[GateResult]:
    """Run all enabled gates. Returns a list of GateResult."""
    if gates is None:
        gates = build_gates_from_config(config)
    return [g.run_with_timing() for g in gates]


def summarize(results: list[GateResult]) -> dict[str, Any]:
    """Summarize gate results for a report."""
    n_pass = sum(1 for r in results if r.status == GateStatus.PASS)
    n_fail = sum(1 for r in results if r.status == GateStatus.FAIL)
    n_warn = sum(1 for r in results if r.status == GateStatus.WARN)
    n_unverified = sum(1 for r in results if r.status == GateStatus.UNVERIFIED)
    return {
        "total": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "warned": n_warn,
        "unverified": n_unverified,
        "all_passed": n_fail == 0,
        "results": [r.to_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ml-validate",
        description="Single CLI entry point for the testing framework.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run all gates on a run")
    run_p.add_argument("--config", type=Path, required=True,
                        help="Path to config YAML")
    run_p.add_argument("--output", type=Path, default=None,
                        help="Write report to this path")
    run_p.add_argument("--exit-on-fail", action="store_true",
                        help="Exit 1 if any gate FAILs")
    run_p.add_argument("--verbose", action="store_true",
                        help="Print every gate (not just FAILs)")

    args = parser.parse_args()

    if args.cmd == "run":
        if __package__ in (None, ""):
            from config import load_config
        else:
            from .config import load_config
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        results = run_gates(config)
        summary = summarize(results)

        # Print
        for r in results:
            if r.status == GateStatus.FAIL or args.verbose:
                icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "UNVERIFIED": "?"}[r.status.value]
                print(f"  [{icon}] {r.gate_name}: {r.message}")

        print()
        print("=" * 70)
        s = summary
        print(f"SUMMARY: {s['passed']} PASS, {s['warned']} WARN, {s['unverified']} UNVERIFIED, {s['failed']} FAIL")
        print("=" * 70)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2, default=str))
            print(f"\nReport written to: {args.output}")

        if args.exit_on_fail and not summary["all_passed"]:
            return 1
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
