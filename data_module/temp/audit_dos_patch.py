import pandas as pd
import json
from pathlib import Path

export_dir = Path("data/exports/sentinel-v3-smartbugs-2026-06-13")
labels = pd.read_parquet(export_dir / "labels.parquet")
manifest = json.loads((export_dir / "manifest.json").read_text())

cols = manifest["label_class_columns"]
dos_idx = cols.index("DenialOfService")
reen_idx = cols.index("Reentrancy")
dos_col = f"class_{dos_idx}"
reen_col = f"class_{reen_idx}"
print(f"DoS col: {dos_col}  Reentrancy col: {reen_col}")

dos_total = int((labels[dos_col] == 1).sum())
reen_total = int((labels[reen_col] == 1).sum())
dos_and_reen = int(((labels[dos_col] == 1) & (labels[reen_col] == 1)).sum())
print(f"DoS=1 total:            {dos_total}")
print(f"Reentrancy=1 total:     {reen_total}")
print(f"DoS=1 AND Reentrancy=1: {dos_and_reen}")

splits = manifest["splits"]
train_ids = set(splits["train"])
val_ids = set(splits["val"])
test_ids = set(splits["test"])
li = labels.set_index("contract_id")
print(f"\nPer-split DoS=1:")
print(f"  train: {int((li[li.index.isin(train_ids)][dos_col]==1).sum())}")
print(f"  val:   {int((li[li.index.isin(val_ids)][dos_col]==1).sum())}")
print(f"  test:  {int((li[li.index.isin(test_ids)][dos_col]==1).sum())}")

# Check a raw merged label file for format
sample_sha = labels[(labels[dos_col]==1) & (labels[reen_col]==1)]["contract_id"].iloc[0]
print(f"\nSample SHA with DoS=1 AND Reen=1: {sample_sha}")
for source in ["dive", "solidifi", "smartbugs_curated"]:
    p = Path(f"data/labels/{source}/{sample_sha}.labels.json")
    if p.exists():
        raw = json.loads(p.read_text())
        print(f"Raw label structure keys: {list(raw.keys())}")
        print(f"Raw label (source={source}):")
        print(json.dumps(raw, indent=2)[:800])
        break

# Also check whether co-occurrence patch code exists anywhere
import subprocess
result = subprocess.run(
    ["grep", "-r", "co.occurrence\|cooccurrence\|co_occurrence\|DoS.*Reentrancy\|Reentrancy.*DoS",
     "sentinel_data/", "--include=*.py", "-l"],
    capture_output=True, text=True
)
print(f"\nFiles mentioning co-occurrence patch: {result.stdout.strip() or 'NONE'}")
