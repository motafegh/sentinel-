#!/usr/bin/env python3
"""Update mlops_config.json to point at a new active checkpoint.

Usage:
    python ml/scripts/set_active_checkpoint.py GCB-P1-Run12-v3dospatched-20260613_FINAL
    python ml/scripts/set_active_checkpoint.py GCB-P1-Run13-..._FINAL --dry-run

Atomic write: writes to a .tmp file then renames, so a crash during write never
corrupts the existing config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONFIG_PATH = Path("ml/mlops_config.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update mlops_config.json to point at a new checkpoint."
    )
    parser.add_argument(
        "checkpoint_name",
        help="Filename of the .pt checkpoint (no path prefix — assumed in ml/checkpoints/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change")
    args = parser.parse_args()

    checkpoint_path = Path("ml/checkpoints") / args.checkpoint_name
    # Auto-detect extension if a bare stem was passed
    if not checkpoint_path.exists():
        for ext in (".pt",):
            candidate = checkpoint_path.with_suffix(checkpoint_path.suffix + ext)
            if candidate.exists():
                checkpoint_path = candidate
                break
    if not checkpoint_path.exists():
        print(
            f"ERROR: checkpoint not found: ml/checkpoints/{args.checkpoint_name}*",
            file=sys.stderr,
        )
        sys.exit(1)

    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with CONFIG_PATH.open() as f:
        config = json.load(f)

    config["checkpoint"] = str(checkpoint_path)

    thresholds = checkpoint_path.with_name(f"{checkpoint_path.stem}_thresholds.json")
    if thresholds.exists():
        config["thresholds"] = str(thresholds)

    if args.dry_run:
        print("DRY RUN — would update to:")
        print(json.dumps(config, indent=2))
        return

    tmp = CONFIG_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(config, f, indent=2)
    tmp.rename(CONFIG_PATH)

    print(f"Updated {CONFIG_PATH}")
    print(f"  checkpoint → {checkpoint_path}")
    if thresholds.exists():
        print(f"  thresholds → {thresholds}")


if __name__ == "__main__":
    main()
