#!/usr/bin/env python3
"""Run or resume the canonical R4 Phase-8 repaired full training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from ml.src.training.vnext_runner import run_phase8_training


def _default_representations_root() -> Path:
    env = os.getenv("SENTINEL_REPRESENTATIONS_ROOT")
    if env:
        return Path(env).expanduser()
    local = REPO_ROOT / "data_module/data/representations"
    if local.is_dir():
        return local
    return Path.home() / "projects/sentinel/data_module/data/representations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=REPO_ROOT / "data_module/data/exports/sentinel-r4-vnext-v1",
        help="G7-passed DATA vNext semantic overlay",
    )
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=_default_representations_root(),
        help="physical graph/token representation root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional run root; default is binding-derived under ml/logs/r4-phase8",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="canonical same-run resume checkpoint: <run>/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers; this value is bound into the run identity",
    )
    parser.add_argument(
        "--milestone-interval-epochs",
        type=int,
        default=10,
        help="durable recovery milestone interval; bound into run identity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_phase8_training(
        overlay_dir=args.overlay,
        representations_root=args.representations_root,
        output_dir=args.output_dir,
        resume=args.resume,
        num_workers=args.num_workers,
        milestone_interval_epochs=args.milestone_interval_epochs,
    )
    print()
    print("=== R4 PHASE 8 TRAINING RESULT ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
