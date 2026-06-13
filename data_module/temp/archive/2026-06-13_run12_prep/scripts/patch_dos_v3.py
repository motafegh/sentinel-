"""Patch DIVE source labels: zero out DoS where DoS+Reentrancy both = 1.

The merger.py co-occurrence logic only flags T3/T4 sources (BCCC pattern).
DIVE (T2) is left as-is by the merger. The 2,655 DIVE contracts with both
DoS=1 and Reentrancy=1 are noise — the plan documented this patch on
2026-06-13 but it was never applied to the source labels.

This script:
  1. Backs up the original DIVE labels to data/_backup_pre_dos_patch_2026-06-13/
  2. Patches DoS=0 in DIVE labels where DoS+Reentrancy both = 1
  3. Reports the count of patches
"""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

LABELS_DIR = Path("data_module/data/labels/dive")
BACKUP_ROOT = Path("data_module/data/_backup_pre_dos_patch_2026-06-13")
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    files = sorted(LABELS_DIR.glob("*.labels.json"))
    print(f"DIVE labels dir: {LABELS_DIR}")
    print(f"DIVE files: {len(files)}")

    # 1. Back up the DIVE source labels (one-time, idempotent)
    if BACKUP_ROOT.exists():
        print(f"Backup dir already exists: {BACKUP_ROOT} — skipping backup (already patched?)")
    else:
        BACKUP_ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(LABELS_DIR, BACKUP_ROOT)
        print(f"Backed up DIVE labels to: {BACKUP_ROOT}")
        print(f"  ({len(files)} files)")

    # 2. Patch DoS=0 where DoS+Reentrancy both = 1
    n_patched = 0
    n_skipped = 0
    n_already_patched = 0
    sample_shas = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            n_skipped += 1
            continue
        cls = d.get("classes", {})
        dos = cls.get("DenialOfService", {})
        ree = cls.get("Reentrancy", {})
        if int(dos.get("value", 0)) == 1 and int(ree.get("value", 0)) == 1:
            # Patch
            cls["DenialOfService"] = {
                "value": 0,
                "tier": None,
                "source": dos.get("source", "dive"),
            }
            n_pos_new = sum(int(c.get("value", 0)) for c in cls.values())
            d["classes"] = cls
            d["n_pos"] = n_pos_new
            f.write_text(json.dumps(d, indent=2))
            n_patched += 1
            if len(sample_shas) < 3:
                sample_shas.append(f.stem.replace(".labels.json", ""))
        else:
            # Check if already patched (DoS=0, Reentrancy=1 is a legitimate state)
            if int(dos.get("value", 0)) == 0 and int(ree.get("value", 0)) == 1:
                n_already_patched += 1

    print()
    print(f"PATCH RESULTS:")
    print(f"  Files with DoS+Reentrancy (patched):  {n_patched}")
    print(f"  Files with DoS=0, Reentrancy=1 (already correct): {n_already_patched}")
    print(f"  Files with malformed JSON (skipped):  {n_skipped}")
    print()
    print(f"Sample patched SHAs (first 3): {sample_shas}")
    print()
    print(f"Expected post-patch DIVE DoS count: 3,750 - 2,655 = 1,095")
    print()
    print("NEXT STEPS:")
    print("  1. Force re-run merger to regenerate merged labels (passes through patched DIVE):")
    print("     python -c 'from sentinel_data.labeling.merger import run_merger; from pathlib import Path; r = run_merger(Path(\"data_module/data\"), [\"dive\", \"solidifi\", \"smartbugs_curated\"], force=True); print(r)'")
    print("  2. Re-export v3 to update labels.parquet + manifest:")
    print("     python -c 'from sentinel_data.export.chunker import chunk_export; ...'")
    print("  3. Verify DoS count drops to 1,101 (DIVE 1,095 + SmartBugs 6), DoS+Reentrancy overlap = 0")


if __name__ == "__main__":
    main()
