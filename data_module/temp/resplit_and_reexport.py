"""Re-split v3 (overwrite) + re-export v3, with patched DoS labels.

The DoS patch zeroed 2,655 DIVE labels but the v3 splits JSONL was built
from the un-patched labels. Re-run the splitter on the patched merged
labels to propagate the fix.
"""
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path("data_module/data")
SPLITS_DIR = DATA_DIR / "splits" / "v3"
EXPORT_DIR = DATA_DIR / "exports" / "sentinel-v3-smartbugs-2026-06-13"
DEDUP_GROUPS_PATH = DATA_DIR / "dedup_groups_graph_hash.json"

# Back up the current v3 splits before overwriting
BACKUP_SPLITS = DATA_DIR / "splits" / "v3-PRE-DOS-PATCH-backup"
if not BACKUP_SPLITS.exists():
    shutil.copytree(SPLITS_DIR, BACKUP_SPLITS)
    print(f"Backed up v3 splits to: {BACKUP_SPLITS}")

print("=" * 70)
print("STEP 1: Re-run splitter (regenerate v3 splits with patched merged labels)")
print("=" * 70)

from sentinel_data.splitting import (
    Contract, apply_dedup_enforcer, apply_nonvulnerable_cap,
    stratified_split, write_manifest, write_splits,
)

# Load dedup groups (L3-applied)
with open(DEDUP_GROUPS_PATH) as f:
    dg_data = json.load(f)
cid_to_group: dict[str, str] = dg_data.get("groups", {})
print(f"  Loaded {len(cid_to_group)} dedup groups ({dg_data.get('n_unique_groups', 0)} unique)")

# Load merged labels
merged_dir = DATA_DIR / "labels" / "merged"
contracts = []
for p in sorted(merged_dir.glob("*.labels.json")):
    try:
        lj = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    sha = lj["sha256"]
    sources = lj.get("sources") or ["unknown"]
    source = sources[0] if sources else "unknown"
    tier = "T0"
    for cls, entry in lj.get("classes", {}).items():
        if entry.get("value") == 1:
            tier = entry.get("tier") or "T0"
            break
    classes = {cls: entry.get("value", 0) for cls, entry in lj.get("classes", {}).items()}
    primary = next((c for c, e in lj.get("classes", {}).items() if e.get("value") == 1), "NonVulnerable")
    n_pos = sum(1 for e in lj.get("classes", {}).values() if e.get("value") == 1)
    contracts.append(Contract(
        sha256=sha, source=source, tier=tier,
        classes=classes, primary_class=primary, n_pos=n_pos,
        dedup_group=cid_to_group.get(sha),
    ))
print(f"  Loaded {len(contracts)} contracts from merged labels")

# Run splitter with seed=42 (matches previous v3 split)
SEED = 42
NONVULN_CAP = 3.0
print(f"\n  Splitting (strategy=stratified, seed={SEED})...")
splits = stratified_split(contracts, seed=SEED)
print(f"  After stratified: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

print("\n  Applying dedup_enforcer...")
apply_dedup_enforcer(splits)
print(f"  After dedup: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
print(f"  dedup_groups_resolved={splits.metadata.dedup_groups_resolved}")

print(f"\n  Applying NonVulnerable cap ({NONVULN_CAP}:1)...")
apply_nonvulnerable_cap(splits, cap=NONVULN_CAP, seed=SEED)
print(f"  After cap: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

# Write to v3 dir (overwrite)
print(f"\n  Writing splits to {SPLITS_DIR}...")
write_splits(splits, SPLITS_DIR)
write_manifest(splits, SPLITS_DIR)

print()
print("=" * 70)
print("STEP 2: Re-export v3 (regenerate labels.parquet + manifest + hash_cache)")
print("=" * 70)

# Back up the previously regenerated v3 export (which had pre-split patch)
BACKUP_EXPORT_PRE_RESPLIT = DATA_DIR / "exports" / "sentinel-v3-PRE-DOS-PATCH-AND-RESPLIT-backup"
if not BACKUP_EXPORT_PRE_RESPLIT.exists():
    BACKUP_EXPORT_PRE_RESPLIT.mkdir(parents=True, exist_ok=True)
    for f in ["labels.parquet", "metadata.parquet", "manifest.json", ".hash_cache.json"]:
        src = EXPORT_DIR / f
        if src.exists():
            shutil.copy2(src, BACKUP_EXPORT_PRE_RESPLIT / f)
    print(f"Backed up v3 export to: {BACKUP_EXPORT_PRE_RESPLIT}")

from sentinel_data.export.chunker import chunk_export

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
import pandas as pd
df = pd.read_parquet(EXPORT_DIR / "labels.parquet")
print(f"Total contracts: {len(df)}")
print()

class_names = ["CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
               "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
               "TransactionOrderDependence", "UnusedReturn"]
print("Per-class positive counts per split (PATCHED):")
for split in ["train", "val", "test"]:
    sub = df[df["split"] == split]
    print(f"  {split}:")
    for i, cn in enumerate(class_names):
        n = int(sub[f"class_{i}"].sum())
        prev = n / len(sub) * 100
        marker = ""
        if cn == "DenialOfService":
            marker = "  ← (was 2,910 train pre-patch; expected 859)"
        print(f"    {cn:30s} {n:5d}  ({prev:5.2f}%){marker}")
print()

# Critical check: DoS+Reentrancy overlap
both = df[(df["class_1"] == 1) & (df["class_6"] == 1)]
print(f"DoS=1 AND Reentrancy=1 contracts (MUST BE 0): {len(both)}")
if len(both) > 0:
    print("  WARNING: patch not fully applied!")
else:
    print("  ✓ Verified: 0 DoS+Reentrancy overlap.")

# Total DoS check
dos = df[df["class_1"] == 1]
print(f"Total DoS=1 (expected ~1,101 = 1,095 DIVE + 6 SmartBugs): {len(dos)}")
print(f"  train: {int((dos['split']=='train').sum())} (was 2,910, expected 859)")
print(f"  val:   {int((dos['split']=='val').sum())} (was 429, expected 120)")
print(f"  test:  {int((dos['split']=='test').sum())} (was 417, expected 122)")

# Sanity: ExternalBug / Reentrancy / IntegerUO counts should be UNCHANGED
for cn, idx in [("Reentrancy", 6), ("ExternalBug", 2), ("IntegerUO", 4), ("UnusedReturn", 9), ("Timestamp", 7)]:
    n = int(df[f"class_{idx}"].sum())
    print(f"  {cn}: {n} (unchanged check)")

print()
print("DONE")
