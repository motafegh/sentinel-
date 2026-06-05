"""
run_all.py — Master runner for all per-fix smoke tests.

Runs smoke_fix{1..8}.py in phase-aware order. Exits 0 only if all pass.
Supports filtering by phase or fix number for incremental development.

Phases (matching docs/pre-run9-fixes/TODO.md execution order):
  Phase 1 (no model impact)         → Fix #6, #7, #8  (display + benchmark + docs)
  Phase 2 (schema bump to v9)        → Fix #2, #3, #4  (sequential, each invalidates cache)
  Phase 3 (data relabel)             → Fix #5           (Slither-derived ground truth)
  Phase 4 (regression — must pass first) → Fix #1       (already applied; permanent check)

Usage:
    python ml/scripts/smoke/run_all.py                   # all phases, all fixes
    python ml/scripts/smoke/run_all.py --phase 1         # Phase 1 only (#6, #7, #8)
    python ml/scripts/smoke/run_all.py --fix 2           # single fix
    python ml/scripts/smoke/run_all.py --preflight       # system pre-flight only
    python ml/scripts/smoke/run_all.py --no-fix1         # skip regression check
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

SMOKE_DIR = Path(__file__).resolve().parent

# Phase → list of fix numbers (in recommended execution order).
PHASES: dict[int, list[int]] = {
    0: [1],          # pre-flight: regression check on already-applied Fix #1
    1: [6, 7, 8],    # display + benchmark + docs (no model change)
    2: [2, 3, 4],    # schema bump v8→v9 (sequential)
    3: [5],          # Slither-derived labels
}

# Human-readable phase descriptions.
PHASE_DESCRIPTIONS: dict[int, str] = {
    0: "Phase 0 — Pre-flight (regression check on Fix #1)",
    1: "Phase 1 — Display + benchmark + docs (no model change)",
    2: "Phase 2 — Schema bump v8→v9 (sequential; each invalidates cache)",
    3: "Phase 3 — Data relabel (Slither-derived ground truth)",
}


def run_one(fix: int) -> tuple[int, float]:
    """Run a single smoke test. Returns (returncode, elapsed_seconds)."""
    script = SMOKE_DIR / f"smoke_fix{fix}.py"
    if not script.exists():
        print(f"  [SKIP] smoke_fix{fix}.py missing", file=sys.stderr, flush=True)
        return -1, 0.0
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, elapsed


def run_preflight() -> int:
    """Run system-level pre-flight checks before any fix smoke test."""
    print("=== SYSTEM PRE-FLIGHT ===", file=sys.stderr, flush=True)
    checks: list[tuple[bool, str]] = []

    py = sys.executable
    checks.append((bool(py), f"python executable: {py}"))

    try:
        import torch
        checks.append((True, f"torch {torch.__version__}"))
    except ImportError:
        checks.append((False, "torch NOT installed"))

    try:
        import numpy
        checks.append((True, f"numpy {numpy.__version__}"))
    except ImportError:
        checks.append((False, "numpy NOT installed"))

    try:
        import pandas
        checks.append((True, f"pandas {pandas.__version__}"))
    except ImportError:
        checks.append((False, "pandas NOT installed"))

    try:
        from _common import GRAPHS_DIR
        n_graphs = len(list(GRAPHS_DIR.glob("*.pt"))) if GRAPHS_DIR.exists() else 0
        checks.append((n_graphs > 100, f"graphs dir: {n_graphs} .pt files"))
    except Exception as exc:
        checks.append((False, f"graphs dir check failed: {exc}"))

    try:
        from _common import CHECKPOINTS_DIR
        n_ckpt = len(list(CHECKPOINTS_DIR.glob("*best.pt"))) if CHECKPOINTS_DIR.exists() else 0
        checks.append((n_ckpt > 0, f"checkpoints dir: {n_ckpt} .pt files"))
    except Exception as exc:
        checks.append((False, f"checkpoints dir check failed: {exc}"))

    failed = 0
    for ok, msg in checks:
        marker = "[OK]  " if ok else "[FAIL]"
        print(f"  {marker} {msg}", file=sys.stderr, flush=True)
        if not ok:
            failed += 1

    print("", file=sys.stderr, flush=True)
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SENTINEL per-fix smoke tests.")
    parser.add_argument(
        "--phase", type=int, choices=[0, 1, 2, 3], default=None,
        help="Run only one phase (0=preflight, 1=display+docs, 2=schema, 3=relabel)",
    )
    parser.add_argument(
        "--fix", type=int, choices=list(range(1, 9)), default=None,
        help="Run a single fix's smoke test (1–8)",
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="Run only the system pre-flight check (no per-fix tests)",
    )
    parser.add_argument(
        "--no-fix1", action="store_true",
        help="Skip Fix #1 (regression check on already-applied relabel)",
    )
    args = parser.parse_args()

    if args.preflight:
        return run_preflight()

    if args.fix is not None:
        fixes_to_run: list[int] = [args.fix]
    elif args.phase is not None:
        fixes_to_run = list(PHASES[args.phase])
    else:
        fixes_to_run = [f for fixes in PHASES.values() for f in fixes]

    if args.no_fix1 and 1 in fixes_to_run:
        fixes_to_run.remove(1)

    if not args.fix and not args.phase:
        print("=" * 60, file=sys.stderr, flush=True)
        print("SENTINEL Pre-Run-9 Smoke Test Suite", file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)

        pf_rc = run_preflight()
        if pf_rc != 0:
            print("\nPRE-FLIGHT FAILED — aborting before per-fix tests.", file=sys.stderr, flush=True)
            return pf_rc
        print("", file=sys.stderr, flush=True)

    results: list[tuple[int, int, float]] = []
    if args.fix is None and args.phase is None:
        current_phase: int | None = None
    else:
        current_phase = args.phase

    for fix in fixes_to_run:
        if current_phase is None:
            for ph, fixes in PHASES.items():
                if fix in fixes:
                    if current_phase != ph:
                        current_phase = ph
                        print("", file=sys.stderr, flush=True)
                        print(PHASE_DESCRIPTIONS[ph], file=sys.stderr, flush=True)
                        print("-" * 60, file=sys.stderr, flush=True)
                    break
        print("", file=sys.stderr, flush=True)
        rc, elapsed = run_one(fix)
        results.append((fix, rc, elapsed))

    print("", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    print("SUMMARY", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)

    passed = 0
    failed = 0
    skipped = 0
    for fix, rc, elapsed in results:
        if rc == 0:
            marker = "PASS"
            passed += 1
        elif rc < 0:
            marker = "SKIP"
            skipped += 1
        else:
            marker = f"FAIL(rc={rc})"
            failed += 1
        print(f"  Fix #{fix}: {marker}  ({elapsed:.2f}s)", file=sys.stderr, flush=True)

    total = passed + failed + skipped
    print("", file=sys.stderr, flush=True)
    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}",
          file=sys.stderr, flush=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
