"""
run_overnight_experiments.py

Sequential launcher for 4 overnight hyperparameter experiments.
Each run logs to MLflow as a separate named run inside sentinel-training.

Design decisions:
- Sequential not parallel: GPU can only train one model at a time.
- Each TrainConfig overrides only what changes vs baseline defaults.
- focal_alpha=0.25 is CORRECT for this dataset (vulnerable=64% majority,
  safe=36% minority). alpha applies to label=1 (vulnerable), so 0.25
  down-weights the majority and 0.75 up-weights the minority (safe).
- run-alpha-tune tests alpha=0.35: a modest nudge to see if the current
  3x penalty gap (0.25 vs 0.75) is too aggressive for a 1.8x class ratio.

Real class distribution (verified by scanning all 68,555 .pt files):
  Vulnerable (label=1): 44,099 — 64.33%  ← majority
  Safe       (label=0): 24,456 — 35.67%  ← minority

Usage — first run:
    nohup poetry run python ml/scripts/run_overnight_experiments.py \\
        > ml/logs/overnight.log 2>&1 &
    echo "PID $!"

Usage — resume after crash (e.g. restart from experiment 3):
    nohup poetry run python ml/scripts/run_overnight_experiments.py \\
        --start-from 3 > ml/logs/overnight_resume.log 2>&1 &

Morning check:
    tail -50 ml/logs/overnight.log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

# Project root → makes ml/ importable regardless of launch directory.
# parents[0] = ml/scripts/
# parents[1] = ml/
# parents[2] = ~/projects/sentinel/  ← package root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.training.trainer import TrainConfig, train


# ---------------------------------------------------------------------------
# Experiment matrix
# Each entry overrides ONLY the fields that differ from TrainConfig defaults.
# Baseline defaults: epochs=20, lr=1e-4, focal_alpha=0.25, batch_size=32
# ---------------------------------------------------------------------------
EXPERIMENTS: list[TrainConfig] = [

    # Experiment 1 — Alpha sensitivity test
    # Baseline alpha=0.25 means weight ratio is 3x (safe 0.75 vs vuln 0.25).
    # But the class ratio is only 1.8x (64% vs 36%).
    # Testing alpha=0.35 softens the ratio to 1.86x — closer to actual imbalance.
    # Hypothesis: F1-safe improves without hurting F1-vuln.
    TrainConfig(
        focal_alpha=0.35,
        run_name="run-alpha-tune",
    ),

    # Experiment 2 — More epochs
    # Baseline peaked at epoch 8 (F1-vuln=0.7133) and epoch 16 (F1-macro=0.6515).
    # Model was still improving at epoch 16 — 20 epochs may simply be too few.
    # Extending to 40 at the same lr reveals the true plateau.
    TrainConfig(
        epochs=40,
        run_name="run-more-epochs",
    ),

    # Experiment 3 — Lower learning rate
    # Baseline showed ~0.15 F1 oscillation between epochs — classic sign of
    # lr slightly too high. 3e-5 (3x smaller) should produce a smoother curve
    # and potentially a higher stable peak. 30 epochs to compensate for
    # slower convergence at the smaller step size.
    TrainConfig(
        lr=3e-5,
        epochs=30,
        run_name="run-lr-lower",
    ),

    # Experiment 4 — Combined
    # Combines alpha-tune (0.35) + lr-lower (3e-5).
    # If both individually improve results, combined should be the best run.
    # Expected: highest F1-vuln and smoothest curve of all 4.
    TrainConfig(
        focal_alpha=0.35,
        lr=3e-5,
        epochs=30,
        run_name="run-combined",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Only --start-from is supported."""
    parser = argparse.ArgumentParser(
        description="SENTINEL overnight experiment launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Normal run:          python run_overnight_experiments.py\n"
            "  Resume from exp 3:   python run_overnight_experiments.py --start-from 3\n"
        ),
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Start from experiment N (1-indexed). "
            "Use to resume after a crash without re-running completed experiments. "
            f"Valid range: 1–{len(EXPERIMENTS)}. Default: 1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Run experiments sequentially with error isolation.

    Each experiment is wrapped in try/except so a crash in one run does NOT
    abort the remaining runs. Failed runs are recorded in the final summary
    so you wake up with the maximum number of usable results.
    """
    args = parse_args()

    # Validate --start-from before touching GPU or disk
    if not 1 <= args.start_from <= len(EXPERIMENTS):
        raise ValueError(
            f"--start-from must be between 1 and {len(EXPERIMENTS)}, "
            f"got {args.start_from}"
        )

    total = len(EXPERIMENTS)  # always 4 — keeps "Run 3/4" logging consistent

    # Build the list of (original_index, config) pairs to actually run.
    # Using original index means MLflow logs and log messages stay consistent
    # whether this is a fresh run or a resume.
    experiments_to_run = [
        (i, config)
        for i, config in enumerate(EXPERIMENTS, start=1)
        if i >= args.start_from
    ]

    if args.start_from > 1:
        skipped = args.start_from - 1
        logger.info(
            f"Resuming from experiment {args.start_from}/{total} — "
            f"skipping {skipped} already-completed run(s)"
        )

    # Tracking for the final summary
    completed: list[str] = []
    failed: list[str] = []
    wall_start = time.time()

    for i, config in experiments_to_run:
        logger.info("=" * 55)
        logger.info(f"  Run {i}/{total}: {config.run_name}")
        logger.info(f"  focal_alpha : {config.focal_alpha}")
        logger.info(f"  lr          : {config.lr}")
        logger.info(f"  epochs      : {config.epochs}")
        logger.info("=" * 55)

        run_start = time.time()

        try:
            # train() handles all MLflow logging internally.
            # Each run_name creates a separate named run inside
            # the sentinel-training experiment.
            train(config)

            elapsed = time.time() - run_start
            completed.append(config.run_name)
            logger.info(
                f"✅ Run {i}/{total} complete: {config.run_name} "
                f"— {elapsed / 60:.1f} min"
            )

        except Exception:
            elapsed = time.time() - run_start
            failed.append(config.run_name)
            # logger.exception captures full stack trace — critical for 2am debugging
            logger.exception(
                f"❌ Run {i}/{total} FAILED: {config.run_name} "
                f"after {elapsed / 60:.1f} min — continuing to next run"
            )

    # -----------------------------------------------------------------------
    # Final summary — first thing you read in the morning
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - wall_start

    logger.info("=" * 55)
    logger.info(
        f"Overnight experiments complete — "
        f"{total_elapsed / 3600:.2f} hr total"
    )
    logger.info("")
    logger.info(
        f"Completed {len(completed)}/{total}: "
        f"{', '.join(completed) if completed else 'none'}"
    )

    if failed:
        logger.warning(
            f"Failed {len(failed)}/{total}: {', '.join(failed)}"
        )
        logger.warning(
            "Check the exception tracebacks above for each failed run."
        )
        logger.warning(
            f"To resume: python run_overnight_experiments.py "
            f"--start-from {EXPERIMENTS.index(next(e for e in EXPERIMENTS if e.run_name == failed[0])) + 1}"
        )
    else:
        logger.info("Failed: 0 — clean sweep ✅")

    logger.info("")
    logger.info("View results:")
    logger.info("  poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db")
    logger.info("  http://localhost:5000")
    logger.info("")
    logger.info("Reading order:")
    logger.info("  1. run-lr-lower     → did oscillation reduce? (smoother F1 curve)")
    logger.info("  2. run-more-epochs  → what epoch did it peak?")
    logger.info("  3. run-alpha-tune   → did F1-safe improve vs baseline?")
    logger.info("  4. run-combined     → did combining both beat all singles?")
    logger.info("  5. For each run: val_recall_vulnerable — that's the real signal")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
