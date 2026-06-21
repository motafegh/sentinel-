"""
check_stale_checkpoints.py — Operational hygiene for ml/checkpoints/.

WHY THIS EXISTS

Over time, ml/checkpoints/ accumulates:
- Old checkpoints that were evaluated but never promoted
- Intermediate training saves that are no longer needed
- Stale _thresholds.json / _temperatures.json companions
- "Best" checkpoints that are actually worse than newer runs

This script:
1. Lists all checkpoints in ml/checkpoints/
2. For each, reads the sidecar JSON (epoch, f1, etc.)
3. Flags checkpoints that are:
   - Older than N days AND not the current best
   - Have no sidecar JSON
   - Have lower F1 than a newer checkpoint of the same run
4. Optionally archives stale ones (moves to ml/checkpoints/_archive/)

USAGE

    # Just report
    python ml/scripts/check_stale_checkpoints.py

    # Auto-archive
    python ml/scripts/check_stale_checkpoints.py --archive

Exit codes:
    0  no stale checkpoints (or all successfully archived)
    1  stale found but --archive not passed
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class CheckpointInfo:
    path: str
    size_mb: float
    age_days: float
    has_sidecar: bool
    is_best: bool
    is_final: bool
    epoch: int | None
    f1: float | None
    f1_tuned: float | None
    status: str  # "OK", "STALE", "ORPHAN", "BEST_NEWER_EXISTS"
    notes: str = ""


def _file_age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def inspect_checkpoint(ckpt: Path) -> CheckpointInfo:
    """Inspect a single checkpoint file."""
    name = ckpt.name
    is_best = "_best" in name
    is_final = "_FINAL" in name

    # Sidecar JSON (look for _thresholds.json or _state.json with same stem)
    stem = ckpt.stem
    thresholds = ckpt.with_name(f"{stem}_thresholds.json")
    state = ckpt.with_name(f"{stem}.state.json")
    has_sidecar = thresholds.exists() or state.exists()

    sidecar = _read_json(thresholds) or _read_json(state) or {}

    epoch = sidecar.get("epoch")
    f1 = sidecar.get("f1_macro") or sidecar.get("val_f1_macro")
    f1_tuned = sidecar.get("f1_macro_tuned")

    return CheckpointInfo(
        path=str(ckpt),
        size_mb=round(_file_size_mb(ckpt), 1),
        age_days=round(_file_age_days(ckpt), 1),
        has_sidecar=has_sidecar,
        is_best=is_best,
        is_final=is_final,
        epoch=epoch,
        f1=f1,
        f1_tuned=f1_tuned,
        status="OK",  # Will be updated later
    )


def analyze_checkpoints(ckpt_dir: Path, max_age_days: int = 30) -> list[CheckpointInfo]:
    """Analyze all checkpoints, flag stale ones."""
    if not ckpt_dir.exists():
        return []

    info_list = []
    for ckpt in sorted(ckpt_dir.glob("*.pt")):
        info = inspect_checkpoint(ckpt)
        # Flag based on criteria
        if not info.has_sidecar and info.age_days > 7:
            info.status = "ORPHAN"
            info.notes = f"no sidecar, {info.age_days:.0f} days old"
        elif info.age_days > max_age_days and not info.is_final:
            info.status = "STALE"
            info.notes = f"{info.age_days:.0f} days old, not _FINAL"
        elif info.is_best and info.age_days > 14:
            info.status = "BEST_OLD"
            info.notes = f"best from {info.age_days:.0f} days ago, may have a newer replacement"
        # Check if there's a newer _FINAL with higher F1
        if info.is_final and info.f1_tuned is not None:
            for other in info_list:
                if other.is_final and other.f1_tuned is not None and other.f1_tuned > info.f1_tuned:
                    info.status = "WORSE_F1"
                    info.notes = f"F1 {info.f1_tuned:.4f} < {other.f1_tuned:.4f} in {Path(other.path).name}"
                    break
        info_list.append(info)

    return info_list


def archive_checkpoint(ckpt: Path, archive_dir: Path) -> bool:
    """Move checkpoint + sidecars to archive."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Move .pt
        shutil.move(str(ckpt), str(archive_dir / ckpt.name))
        # Move sidecars
        for suffix in ["_thresholds.json", "_temperatures.json", ".state.json", "_behavioral_probes.json", "_label_quality.json"]:
            sidecar = ckpt.with_name(f"{ckpt.stem}{suffix}")
            if sidecar.exists():
                shutil.move(str(sidecar), str(archive_dir / sidecar.name))
        return True
    except Exception as e:
        print(f"ERROR: failed to archive {ckpt}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="check_stale_checkpoints",
        description=(
            "Check ml/checkpoints/ for stale or orphaned checkpoints. "
            "Optionally archive them."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("ml/checkpoints"),
                        help="Directory to scan (default: ml/checkpoints)")
    parser.add_argument("--max-age-days", type=int, default=30,
                        help="Mark checkpoints older than this as STALE")
    parser.add_argument("--archive", action="store_true",
                        help="Auto-archive stale checkpoints")
    parser.add_argument("--exit-on-stale", action="store_true",
                        help="Exit 1 if stale checkpoints are found (unless --archive is passed)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON report to this path (for framework gates).")
    args = parser.parse_args()

    info_list = analyze_checkpoints(args.checkpoint_dir, args.max_age_days)

    if not info_list:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "summary": {"total": 0, "stale_or_orphan": 0, "all_clean": True},
                "checkpoints": [],
            }, indent=2))
        return 0

    print(f"Found {len(info_list)} checkpoints in {args.checkpoint_dir}:")
    print()
    n_stale = 0
    for info in info_list:
        marker = {
            "OK": "✓",
            "STALE": "⚠",
            "ORPHAN": "✗",
            "BEST_OLD": "⚠",
            "WORSE_F1": "⚠",
        }.get(info.status, "?")
        print(f"  {marker} {Path(info.path).name}")
        print(f"     size={info.size_mb}MB  age={info.age_days:.0f}d  "
              f"sidecar={'Y' if info.has_sidecar else 'N'}  "
              f"epoch={info.epoch}  f1={info.f1_tuned or info.f1 or '?'}")
        if info.status != "OK":
            print(f"     STATUS: {info.status} — {info.notes}")
            n_stale += 1
    print()
    print(f"Summary: {len(info_list)} total, {n_stale} stale/orphan")

    if args.archive and n_stale > 0:
        archive_dir = args.checkpoint_dir / "_archive"
        print(f"\nArchiving stale checkpoints to {archive_dir}...")
        archived = 0
        for info in info_list:
            if info.status in ("STALE", "ORPHAN", "BEST_OLD", "WORSE_F1"):
                if archive_checkpoint(Path(info.path), archive_dir):
                    print(f"  archived: {Path(info.path).name}")
                    archived += 1
        print(f"Archived {archived} checkpoints")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "summary": {
                "total": len(info_list),
                "stale_or_orphan": n_stale,
                "all_clean": n_stale == 0,
                "max_age_days": args.max_age_days,
            },
            "checkpoints": [asdict(i) for i in info_list],
        }, indent=2))
        print(f"Report written to: {args.output}")

    if args.exit_on_stale and n_stale > 0 and not args.archive:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
