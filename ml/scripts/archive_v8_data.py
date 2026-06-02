"""
archive_v8_data.py — Phase 7.0: Archive all v8-era data before Phase 7 re-extraction.

Run this ONCE before re-extracting v9 graphs. All v8 artifacts are moved to
ml/data/archive/ with a manifest file listing counts and sizes.

Run 5 must train ONLY on v9 data. This script ensures no v8 artifacts remain
in active directories to prevent accidental contamination.

After this script: Run re-extraction → rebuild cache → rebuild index → Run 5.

Usage:
    PYTHONPATH=. python ml/scripts/archive_v8_data.py
    PYTHONPATH=. python ml/scripts/archive_v8_data.py --dry-run   # preview only
    PYTHONPATH=. python ml/scripts/archive_v8_data.py --skip-checkpoints  # keep checkpoints
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml" / "data"
ARCHIVE = DATA / "archive"


ARCHIVE_PLAN = [
    # (source, dest_name, description)
    (DATA / "graphs",                             "graphs_v8_pre_run5",       "v8 graph .pt files"),
    (DATA / "cached_dataset_v8.pkl",              "cached_dataset_v8.pkl",    "v8 PyG cache"),
    (DATA / "tokens_windowed",                    "tokens_windowed_v8",       "v8 token .pt files"),
    (DATA / "processed" / "multilabel_index_cleaned.csv", "multilabel_index_cleaned_v8.csv", "v8 cleaned label CSV"),
    (DATA / "processed" / "multilabel_index.csv", "multilabel_index_v8.csv",  "v8 raw label CSV"),
    (DATA / "splits" / "deduped",                 "splits_v8_deduped",        "v8 train/val/test splits"),
]

CHECKPOINT_PLAN = [
    (ROOT / "ml" / "checkpoints",  "checkpoints_pre_run5",  "pre-Run-5 checkpoints"),
    (ROOT / "ml" / "logs",         "logs_pre_run5",         "pre-Run-5 training logs"),
]


def _size_str(path: Path) -> str:
    if path.is_file():
        mb = path.stat().st_size / 1024**2
        return f"{mb:.1f} MB"
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        gb = total / 1024**3
        return f"{gb:.2f} GB ({sum(1 for _ in path.rglob('*.pt') if _.is_file())} .pt files)"
    return "not found"


def _count_pt(path: Path) -> int:
    if path.is_dir():
        return sum(1 for _ in path.rglob("*.pt") if _.is_file())
    return 1 if (path.is_file() and path.suffix == ".pt") else 0


def archive(dry_run: bool, skip_checkpoints: bool) -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = ARCHIVE / "v8_archive_manifest.txt"

    manifest_lines = [
        f"v8 Archive Manifest — created {ts}",
        f"Archived by: archive_v8_data.py",
        f"Dry run: {dry_run}",
        "",
    ]

    plan = ARCHIVE_PLAN + ([] if skip_checkpoints else CHECKPOINT_PLAN)

    all_ok = True
    for src, dest_name, desc in plan:
        dest = ARCHIVE / dest_name
        size = _size_str(src)
        exists = src.exists()

        if not exists:
            status = "SKIP (not found)"
            manifest_lines.append(f"  SKIP  {src} → {dest_name}  [{desc}] — not found")
            print(f"  [SKIP]  {src.name} — not found")
            continue

        if dest.exists():
            status = "SKIP (already archived)"
            manifest_lines.append(f"  SKIP  {src} → {dest_name}  [{desc}] — already in archive")
            print(f"  [SKIP]  {dest_name} — already in archive")
            continue

        manifest_lines.append(f"  MOVED {src} → {dest_name}  [{desc}]  {size}")
        if dry_run:
            print(f"  [DRY]   {src.name} ({size}) → {dest_name}")
        else:
            print(f"  [MOVE]  {src.name} ({size}) → archive/{dest_name}")
            try:
                shutil.move(str(src), str(dest))
            except Exception as e:
                print(f"          ERROR: {e}", file=sys.stderr)
                all_ok = False

    if not dry_run:
        manifest_path.write_text("\n".join(manifest_lines) + "\n")
        print(f"\nManifest written: {manifest_path}")

    print(f"\nArchive status: {'OK' if all_ok else 'ERRORS — check above'}")
    print(f"Archive location: {ARCHIVE}")
    print()
    if not dry_run and all_ok:
        print("Next steps:")
        print("  1. Run Gate 5.3 VRAM test: python ml/scripts/vram_gate_test.py")
        print("  2. Re-extract v9 graphs:   python ml/scripts/reextract_graphs.py --max-nodes 2048")
        print("  3. Rebuild cache:          python ml/scripts/create_cache.py")
        print("  4. Rebuild index:          python ml/scripts/build_multilabel_index.py")
        print("  5. Run Phase 8 (Run 5):    python ml/scripts/train.py --run-name v8.0-run5-<date>")


def main() -> None:
    p = argparse.ArgumentParser(description="Archive v8 data before Phase 7 re-extraction")
    p.add_argument("--dry-run",          action="store_true",
                   help="Preview what would be moved without actually moving anything")
    p.add_argument("--skip-checkpoints", action="store_true",
                   help="Skip archiving ml/checkpoints and ml/logs (keep for reference)")
    args = p.parse_args()

    if not args.dry_run:
        print("\n⚠  This will MOVE (not copy) v8 artifacts to ml/data/archive/.")
        print("   The archive is the fallback — do NOT delete it after re-extraction.\n")
        confirm = input("Type 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    archive(dry_run=args.dry_run, skip_checkpoints=args.skip_checkpoints)


if __name__ == "__main__":
    main()
