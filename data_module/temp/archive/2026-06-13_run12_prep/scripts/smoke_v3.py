import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from ml.src.datasets.sentinel_dataset import SentinelDataset

EXPORT = "data_module/data/exports/sentinel-v3-smartbugs-2026-06-13"
print(f"Loading SentinelDataset from {EXPORT}...")
ds = SentinelDataset("train", EXPORT)
print(f"Loaded: {len(ds)} train contracts")
g, tok, y, cid, tier = ds[0]
print(f"Sample 0: y.shape={y.shape}, y.sum={int(y.sum())} positives, contract_id={cid[:16]}, tier={tier}")
print(f"Sample 0 graph: x.shape={g.x.shape}, edge_index.shape={g.edge_index.shape}")
print(f"Sample 0 tokens: input_ids.shape={tok['input_ids'].shape}")
