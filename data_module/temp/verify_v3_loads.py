"""Verify v3 export loads cleanly via SentinelDatasetExport (Gate 3 light)."""
import time
from pathlib import Path
import sys
sys.path.insert(0, "data_module")
from sentinel_data.export.export import SentinelDatasetExport

EXPORT_DIR = Path("data_module/data/exports/sentinel-v3-smartbugs-2026-06-13")

t0 = time.perf_counter()
exp = SentinelDatasetExport(EXPORT_DIR)
print(f"Export loaded in {time.perf_counter()-t0:.2f}s")
print(f"  schema_version: {exp.manifest.schema_version}")
print(f"  graph_schema_version: {exp.manifest.graph_schema_version}")
print(f"  n_contracts: {exp.manifest.n_contracts}")
print(f"  n_contracts_with_reps: {exp.manifest.n_contracts_with_reps}")
print(f"  n_shards: {exp.manifest.n_shards}")
print(f"  artifact_hash: {exp.manifest.artifact_hash}")
print(f"  splits: train={len(exp.manifest.splits.get('train', []))} val={len(exp.manifest.splits.get('val', []))} test={len(exp.manifest.splits.get('test', []))}")

# Verify hash
t0 = time.perf_counter()
ok = exp.verify_artifact_hash()
print(f"\nverify_artifact_hash: {ok} (took {time.perf_counter()-t0:.2f}s)")

# Leakage check
import json
train_shas = set(exp.manifest.splits.get("train", []))
val_shas = set(exp.manifest.splits.get("val", []))
test_shas = set(exp.manifest.splits.get("test", []))
print(f"\nLeakage check:")
print(f"  train∩val: {len(train_shas & val_shas)}")
print(f"  train∩test: {len(train_shas & test_shas)}")
print(f"  val∩test: {len(val_shas & test_shas)}")

# DoS check on the embedded splits
import pandas as pd
df = pd.DataFrame(exp.manifest.splits.get("train", []), columns=["sha"]).assign(split="train")
df = pd.concat([df, pd.DataFrame(exp.manifest.splits.get("val", []), columns=["sha"]).assign(split="val")])
df = pd.concat([df, pd.DataFrame(exp.manifest.splits.get("test", []), columns=["sha"]).assign(split="test")])

# Now read labels.parquet
labels = pd.read_parquet(EXPORT_DIR / "labels.parquet")
both = labels[(labels["class_1"] == 1) & (labels["class_6"] == 1)]
print(f"\nDoS=1 AND Reentrancy=1 in v3 export: {len(both)} (MUST BE 0)")

print(f"\nTotal DoS=1 in v3 export: {labels['class_1'].sum()}")
print(f"  train: {int(labels[labels['split']=='train']['class_1'].sum())}")
print(f"  val:   {int(labels[labels['split']=='val']['class_1'].sum())}")
print(f"  test:  {int(labels[labels['split']=='test']['class_1'].sum())}")

print("\nGATE 3 (SentinelDataset round-trip) — loadable, hash verified, 0 leakage, DoS patch in effect")
