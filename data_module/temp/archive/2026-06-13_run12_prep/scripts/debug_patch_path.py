"""Debug: trace where DoS+Reentrancy lives across all data files."""
import json
from pathlib import Path

# Pick a known patched contract
sha = "001374474bd1c27f1565b536585ebb6869dac042b508281f5c4b364b3122216a"

# 1. Backup file (should have DoS=1, Reentrancy=1 - the original)
backup_path = Path(f"data_module/data/_backup_pre_dos_patch_2026-06-13/{sha}.labels.json")
with open(backup_path) as f:
    backup = json.load(f)
print(f"BACKUP DIVE label for {sha[:16]}:")
print(f"  DoS={backup['classes']['DenialOfService']['value']}  Reentrancy={backup['classes']['Reentrancy']['value']}  n_pos={backup['n_pos']}")
print()

# 2. Current DIVE source label (should have DoS=0, Reentrancy=1 after patch)
dive_path = Path(f"data_module/data/labels/dive/{sha}.labels.json")
with open(dive_path) as f:
    dive = json.load(f)
print(f"CURRENT DIVE source label for {sha[:16]}:")
print(f"  DoS={dive['classes']['DenialOfService']['value']}  Reentrancy={dive['classes']['Reentrancy']['value']}  n_pos={dive['n_pos']}")
print()

# 3. Current merged label (should be patched - merger is pass-through for single-source DIVE)
merged_path = Path(f"data_module/data/labels/merged/{sha}.labels.json")
with open(merged_path) as f:
    merged = json.load(f)
print(f"CURRENT MERGED label for {sha[:16]}:")
print(f"  DoS={merged['classes']['DenialOfService']['value']}  Reentrancy={merged['classes']['Reentrancy']['value']}  n_pos={merged['n_pos']}")
print(f"  sources={merged.get('sources')}")
print()

# 4. Splits JSONL (might have DoS baked in)
found = False
for split in ["train", "val", "test"]:
    split_path = Path(f"data_module/data/splits/v3/{split}.jsonl")
    with open(split_path) as f:
        for line in f:
            row = json.loads(line)
            if row["sha256"] == sha:
                print(f"SPLIT v3/{split} row for {sha[:16]}:")
                print(f"  classes={row.get('classes')}")
                print(f"  n_pos={row.get('n_pos')}")
                found = True
                break
    if found:
        break
if not found:
    print(f"NOT FOUND in any v3 split: {sha[:16]}")
