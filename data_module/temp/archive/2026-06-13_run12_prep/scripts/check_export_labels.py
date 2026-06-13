import json
import pandas as pd
from pathlib import Path

export_dir = Path("data/exports/sentinel-v3-smartbugs-2026-06-13")
m = json.loads((export_dir / "manifest.json").read_text())
cols = m["label_class_columns"]
dos_col = f"class_{cols.index('DenialOfService')}"
reen_col = f"class_{cols.index('Reentrancy')}"

labels = pd.read_parquet(export_dir / "labels.parquet")
dos = int((labels[dos_col] == 1).sum())
dos_reen = int(((labels[dos_col] == 1) & (labels[reen_col] == 1)).sum())
print(f"EXPORT labels.parquet: DoS=1: {dos}, DoS+Reen=1: {dos_reen}")
print(f"artifact_hash in manifest: {m['artifact_hash'][:16]}...")

# Check if hash cache is still valid
cache_path = export_dir / ".hash_cache.json"
if cache_path.exists():
    cache = json.loads(cache_path.read_text())
    print(f"cache artifact_hash:      {cache['artifact_hash'][:16]}...")
    print(f"hashes match: {cache['artifact_hash'] == m['artifact_hash']}")
