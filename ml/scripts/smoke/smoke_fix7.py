"""
smoke_fix7.py — Smoke test for Fix #7 (SmartBugs benchmark).

Verifies that:
  - ml/scripts/manual_test_smartbugs.py exists
  - SmartBugs dataset has ≥ 10 .sol files
  - Script runs successfully on a 5-file limit
  - Per-category accuracy table is non-empty

Gates-in:
  G7.1 — ml/data/smartbugs-curated/dataset/ has ≥ 10 .sol files
  G7.2 — ml/scripts/manual_test_smartbugs.py exists
  G7.3 — Checkpoint exists

Gates-out:
  G7.4 — Script exits 0 on 5-file limit
  G7.5 — stdout contains at least one category name from SmartBugs
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    CHECKPOINTS_DIR,
    SMARTBUGS_DIR,
    SRC_DIR,
    check,
    find_checkpoint,
    pass_,
    smoke_header,
    timed,
)

# Known SmartBugs curated category names (verified from dataset layout).
SMARTBUGS_CATEGORIES: list[str] = [
    "access_control",
    "arithmetic",
    "denial_of_service",
    "front_running",
    "reentrancy",
    "time_manipulation",
    "unchecked_low_level_calls",
    "unchecked_send",
    "bad_randomness",
    "short_addresses",
]


@timed("fix7_count_files")
def count_smartbugs_files() -> int:
    """Count .sol files in SmartBugs dataset."""
    if not SMARTBUGS_DIR.exists():
        raise AssertionError(f"G7.1 SmartBugs dataset missing: {SMARTBUGS_DIR}")
    sol_files = list(SMARTBUGS_DIR.rglob("*.sol"))
    check(len(sol_files) >= 10, f"G7.1 SmartBugs has ≥ 10 .sol files (found {len(sol_files)})")
    return len(sol_files)


@timed("fix7_check_script")
def check_benchmark_script() -> Path:
    """Verify manual_test_smartbugs.py exists at the expected path."""
    script = ML_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "manual_test_smartbugs.py"
    if not script.exists():
        raise AssertionError(
            f"G7.2 benchmark script missing: {script} — create it per doc 06"
        )
    return script


@timed("fix7_run_script")
def run_benchmark_limited() -> tuple[int, str, str]:
    """Run benchmark with --limit 5 and capture output. Returns (returncode, stdout, stderr)."""
    script = check_benchmark_script()
    ckpt = find_checkpoint("Run8-v10") or find_checkpoint("v10")
    if ckpt is None:
        cands = sorted(CHECKPOINTS_DIR.glob("*best.pt"))
        if not cands:
            raise AssertionError(f"G7.3 no checkpoint in {CHECKPOINTS_DIR}")
        ckpt = str(cands[-1])

    cmd = [
        sys.executable, str(script),
        "--checkpoint", ckpt,
        "--limit", "5",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("G7.4 benchmark script timed out (>180s)") from exc

    return result.returncode, result.stdout, result.stderr


@timed("fix7_check_output")
def check_output_has_category(stdout: str) -> str:
    """Verify stdout contains at least one known SmartBugs category name."""
    text = (stdout or "").lower()
    found = [c for c in SMARTBUGS_CATEGORIES if c in text]
    if not found:
        snippet = stdout[:500] if stdout else "(empty)"
        raise AssertionError(
            f"G7.5 stdout missing any SmartBugs category name. Found categories: {SMARTBUGS_CATEGORIES}. "
            f"Stdout: {snippet}"
        )
    return found[0]


@timed("fix7_total")
def main() -> int:
    smoke_header(7, "SmartBugs benchmark (curated 143 .sol, 10 categories)")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    n_sol = count_smartbugs_files()
    check_benchmark_script()
    pass_(f"G7.2 benchmark script present")

    # ── Body ─────────────────────────────────────────────────────────────
    rc, stdout, stderr = run_benchmark_limited()
    check(rc == 0, f"G7.4 benchmark --limit 5 exits 0 (got rc={rc}, stderr: {stderr[:200]})")
    cat = check_output_has_category(stdout)
    pass_(f"G7.5 benchmark stdout contains category: {cat}")

    elapsed = time.perf_counter() - start
    pass_(f"Fix #7 smoke OK — {n_sol} .sol files available, benchmark runs, {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #7 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
