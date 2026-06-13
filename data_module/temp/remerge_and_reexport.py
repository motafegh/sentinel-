"""Re-run merger + re-export v3 to propagate the DoS patch.

Step 1: Force re-run merger (regenerates merged labels from patched DIVE sources).
Step 2: Re-export v3 (regenerates labels.parquet, manifest.json, .hash_cache.json).
Step 3: Verify DoS counts + 0 DoS+Reentrancy overlap.
"""
import json
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

DATA_DIR = Path("data_module/data")
EXPORT_DIR = DATA_DIR / "exports" / "sentinel-v3-smartbugs-2026-06-13"
SPLITS_DIR = DATA_DIR / "splits" / "v3"

print("=" * 70)
print("STEP 1: Force re-run merger (regenerate merged labels from patched DIVE)")
print("=" * 70)
from sentinel_data.labeling.merger import run_merger

t0 = time.monotonic()
r = run_merger(
    DATA_DIR,
    ["dive", "solidifi", "smartbugs_curated"],
    force=True,
)
print(f"Merger result: {r}")
print(f"Merger took {r.duration_s:.1f}s (wall: {time.monotonic()-t0:.1f}s)")

print()
print("=" * 70)
print("STEP 2: Re-export v3 (regenerate labels.parquet, manifest, hash_cache)")
print("=" * 70)
from sentinel_data.export.chunker import chunk_export

# Back up the current v3 export manifest + hash_cache + labels.parquet
# (just in case we need to rollback)
import shutil
BACKUP_DIR = DATA_DIR / "exports" / "sentinel-v3-PRE-DOS-PATCH-backup"
if not BACKUP_DIR.exists():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for f in ["labels.parquet", "metadata.parquet", "manifest.json", ".hash_cache.json"]:
        src = EXPORT_DIR / f
        if src.exists():
            shutil.copy2(src, BACKUP_DIR / f)
            print(f"Backed up: {f}")
print(f"Pre-patch export backed up to: {BACKUP_DIR}")

t0 = time.monotonic()
manifest = chunk_export(
    rep_root=DATA_DIR / "representations",
    preproc_root=DATA_DIR / "preprocessed",
    splits_dir=SPLITS_DIR,
    output_dir=EXPORT_DIR,
    config_path=Path("data_module/config.yaml"),
    shard_size=5000,
    source_set=["dive", "solidifi", "smartbugs_curated"],
    skipped_sources=[],
    graph_schema_version="v9",
)
print(f"Re-export took {time.monotonic()-t0:.1f}s")
print(f"  n_contracts: {manifest.n_contracts}")
print(f"  n_contracts_with_reps: {manifest.n_contracts_with_reps}")
print(f"  n_shards: {manifest.n_shards}")
print(f"  artifact_hash: {manifest.artifact_hash}")
print(f"  created_at: {manifest.created_at}")

print()
print("=" * 70)
print("STEP 3: VERIFY")
print("=" * 70)
df = pd.read_parquet(EXPORT_DIR / "labels.parquet")
print(f"Total contracts: {len(df)}")
print()

# Per-class counts
class_names = ["CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
               "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
               "TransactionOrderDependence", "UnusedReturn"]
print("Per-class positive counts per split:")
for split in ["train", "val", "test"]:
    sub = df[df["split"] == split]
    print(f"  {split}:")
    for i, cn in enumerate(class_names):
        n = int(sub[f"class_{i}"].sum())
        prev = n / len(sub) * 100
        marker = ""
        if cn == "DenialOfService":
            marker = "  ← (was 2,910 train before patch)"
        print(f"    {cn:30s} {n:5d}  ({prev:5.2f}%){marker}")
print()

# Critical check: DoS+Reentrancy overlap
both = df[(df["class_1"] == 1) & (df["class_6"] == 1)]
print(f"DoS=1 AND Reentrancy=1 contracts (MUST BE 0): {len(both)}")
if len(both) > 0:
    print("  WARNING: patch not fully applied!")
    for _, row in both.head(5).iterrows():
        print(f"    {row['contract_id']} split={row['split']} src={row['source']}")
else:
    print("  ✓ Verified: 0 DoS+Reentrancy overlap.")

# Total DoS check
dos = df[df["class_1"] == 1]
print(f"Total DoS=1 (expected 1,101 = 1,095 DIVE + 6 SmartBugs): {len(dos)}")
print(f"  train: {int((dos['split']=='train').sum())} (expected 859)")
print(f"  val:   {int((dos['split']=='val').sum())} (expected 120)")
print(f"  test:  {int((dos['split']=='test').sum())} (expected 122)")

# Sanity: ExternalBug / Reentrancy / IntegerUO counts should be unchanged
for cn, idx in [("Reentrancy", 6), ("ExternalBug", 2), ("IntegerUO", 4)]:
    n = int(df[f"class_{idx}"].sum())
    print(f"  {cn}: {n} (unchanged check)")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
