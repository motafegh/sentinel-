#!/usr/bin/env python3
"""
run_all_audits.py — Master runner for all SENTINEL v6 audit scripts.

Runs every task script in priority order and collects all reports.

Usage:
    cd /home/motafeq/projects/sentinel
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/audit/run_all_audits.py              # run all
    PYTHONPATH=. python ml/scripts/audit/run_all_audits.py --batch 1    # run batch 1 only
    PYTHONPATH=. python ml/scripts/audit/run_all_audits.py --task 09    # run single task
    PYTHONPATH=. python ml/scripts/audit/run_all_audits.py --dry-run    # show what would run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Task scripts in priority order (batch, task_id, script_name, estimated_minutes)
TASK_SCHEDULE = [
    # Batch 1: Invisible Corruption (run first)
    (1, "09", "task09_feature_range_audit.py",          15),
    (1, "10", "task10_token_graph_alignment.py",         30),
    (1, "11", "task11_file_triple_alignment.py",         10),
    (1, "12", "task12_token_integrity.py",               15),
    (1, "13", "task13_graph_integrity.py",               20),
    (1, "26", "task26_stale_v5_contamination.py",        20),

    # Batch 2: Architecture Decisions
    (2, "16", "task16_wrong_contract_selection.py",      60),
    (2, "17", "task17_safemath_viability.py",            45),
    (2, "18", "task18_solidity_version_dist.py",         30),
    (2, "21", "task21_feature_correlation.py",           30),

    # Batch 3: Label Quality
    (3, "19", "task19_timestamp_label_quality.py",       45),
    (3, "20", "task20_dos_reentrancy_separability.py",   30),

    # Batch 4: Confounds & Shifts
    (4, "22", "task22_graph_size_confound.py",           30),
    (4, "23", "task23_send_unchecked_prevalence.py",     30),
    (4, "24", "task24_token_graph_source_alignment.py",  20),
    (4, "25", "task25_split_distribution_shift.py",      30),

    # Batch 5: Extended Validation
    (5, "14", "task14_subsample_coverage.py",            30),
    (5, "15", "task15_in_unchecked_regex.py",            20),
    (5, "01", "task1_recheck_activation_split.py",       20),
    (5, "05", "task5_recheck_edge_types_full.py",        15),
]

BATCH_NAMES = {
    1: "Integrity & Alignment (highest priority)",
    2: "Architecture Decisions",
    3: "Label Quality",
    4: "Confound & Distribution Shift",
    5: "Extended Validation",
}


def main():
    parser = argparse.ArgumentParser(description="SENTINEL v6 Audit Runner")
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only this batch")
    parser.add_argument("--task", type=str,
                        help="Run only this task (e.g., '09', '16')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--continue-on-error", action="store_true", default=True,
                        help="Continue to next task if one fails (default: True)")
    args = parser.parse_args()

    # Filter tasks
    tasks = TASK_SCHEDULE
    if args.batch:
        tasks = [t for t in tasks if t[0] == args.batch]
    if args.task:
        tasks = [t for t in tasks if t[1] == args.task]

    if not tasks:
        print("No tasks match the given filters.")
        sys.exit(1)

    # Print schedule
    print("=" * 70)
    print("  SENTINEL v6 Audit — Task Schedule")
    print("=" * 70)
    total_est = 0
    current_batch = None
    for batch, tid, script, est in tasks:
        if batch != current_batch:
            current_batch = batch
            print(f"\n  Batch {batch}: {BATCH_NAMES.get(batch, '')}")
            print(f"  {'─' * 60}")
        status = "✓" if (SCRIPT_DIR / script).exists() else "✗ MISSING"
        print(f"    Task {tid:>2}: {script:<45} ~{est:>2}min  {status}")
        total_est += est

    print(f"\n  Total estimated time: ~{total_est} minutes ({total_est/60:.1f} hours)")
    print("=" * 70)

    if args.dry_run:
        print("\nDry run — no scripts executed.")
        return

    # Confirm
    print(f"\nWill run {len(tasks)} tasks. Press Ctrl+C to cancel, Enter to continue...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    # Execute
    results = []
    t_start = time.time()

    for batch, tid, script, est in tasks:
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            print(f"\n⚠ SKIP Task {tid}: {script} not found")
            results.append((tid, "SKIP", 0, "Script not found"))
            continue

        print(f"\n{'='*70}")
        print(f"  Running Task {tid}: {script}")
        print(f"  Estimated: ~{est} minutes")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(SCRIPT_DIR.parents[2]),  # project root
                capture_output=False,
                text=True,
                timeout=est * 60 * 2,  # 2x estimated time as timeout
            )
            elapsed = time.time() - t0
            status = "PASS" if result.returncode == 0 else "FAIL"
            results.append((tid, status, elapsed, f"exit_code={result.returncode}"))

            if result.returncode != 0 and not args.continue_on_error:
                print(f"\n✗ Task {tid} FAILED — stopping.")
                break

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            results.append((tid, "TIMEOUT", elapsed, f"exceeded {est*2}min"))
            if not args.continue_on_error:
                break
        except Exception as e:
            elapsed = time.time() - t0
            results.append((tid, "ERROR", elapsed, str(e)[:80]))
            if not args.continue_on_error:
                break

    # Summary
    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  AUDIT RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print()

    passed = sum(1 for _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, s, _, _ in results if s != "PASS")

    for tid, status, elapsed, detail in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} Task {tid:>2}: {status:<8} {elapsed/60:.1f}min  {detail}")

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(results)} tasks")
    print(f"  Reports saved to: ml/scripts/audit/reports/")
    print("=" * 70)


if __name__ == "__main__":
    main()
