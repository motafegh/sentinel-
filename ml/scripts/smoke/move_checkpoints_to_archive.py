#!/usr/bin/env python3
"""Move non-canonical checkpoints to _archive/ (Q4 MLOps Phase A.4 policy)."""
import shutil
from pathlib import Path

checkpoints_dir = Path("/home/motafeq/projects/sentinel/ml/checkpoints")
archive_dir = checkpoints_dir / "_archive"
archive_dir.mkdir(exist_ok=True)

# Files to KEEP in source (canonical Run 12 FINAL — to be DVC-tracked)
keep = {
    "GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
    "GCB-P1-Run12-v3dospatched-20260613_FINAL.state.json",
    "GCB-P1-Run12-v3dospatched-20260613_FINAL_thresholds.json",
}

# Move everything else to _archive
moved = 0
for f in sorted(checkpoints_dir.iterdir()):
    if f.is_file() and f.name not in keep:
        dest = archive_dir / f.name
        shutil.move(str(f), str(dest))
        moved += 1
        print(f"  moved: {f.name}")

print(f"\nMoved {moved} files to {archive_dir}")
print(f"\nRemaining in ml/checkpoints/:")
for f in sorted(checkpoints_dir.iterdir()):
    if f.is_file():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:60s} {size_mb:8.1f} MB")
