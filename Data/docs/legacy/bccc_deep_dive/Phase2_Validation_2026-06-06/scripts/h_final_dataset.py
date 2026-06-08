"""BCCC Phase 2 — WS-H: Final Cleaned Dataset.

Emits the production-ready cleaned BCCC dataset for SENTINEL training:
  - contracts_clean.csv     (the main deliverable; 67,311 contracts × 10 classes)
  - contracts_clean.parquet (same data, parquet format)
  - split_assignments.csv   (which contract is in train/val/test/review_pending)
  - metadata.json           (full provenance + hashes)
  - README.md               (usage guide)

Inputs:
  - ../labels/contracts_filtered.csv (67,311 contracts, 10 classes, post-D-F1)
  - ../splits/{train,val,test}.csv
  - ../labels/review_pending_ids.csv
  - ../labels/dropped_contracts.csv
  - ../integrity/dedup_map.csv
  - ../complexity/per_contract_stats.csv

Outputs (under ../outputs/):
  - contracts_clean.csv
  - contracts_clean.parquet
  - split_assignments.csv
  - metadata.json
  - README.md
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/home/motafeq/projects/sentinel")
LABELS = Path(__file__).resolve().parent.parent / "labels"
SPLITS = Path(__file__).resolve().parent.parent / "splits"
INTEG = Path(__file__).resolve().parent.parent / "integrity"
COMPLEX = Path(__file__).resolve().parent.parent / "complexity"
OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SENTINEL_V9_ORDER = [
    "Class01:ExternalBug", "Class02:GasException", "Class03:MishandledException",
    "Class04:Timestamp", "Class06:UnusedReturn", "Class08:CallToUnknown",
    "Class09:DenialOfService", "Class10:IntegerUO", "Class11:Reentrancy", "Class12:NonVulnerable",
]
SENTINEL_V9_SHORT = {
    "Class01:ExternalBug": "ExternalBug",
    "Class02:GasException": "GasException",
    "Class03:MishandledException": "MishandledException",
    "Class04:Timestamp": "Timestamp",
    "Class06:UnusedReturn": "UnusedReturn",
    "Class08:CallToUnknown": "CallToUnknown",
    "Class09:DenialOfService": "DenialOfService",
    "Class10:IntegerUO": "IntegerUO",
    "Class11:Reentrancy": "Reentrancy",
    "Class12:NonVulnerable": "NonVulnerable",
}


def main():
    print("=" * 70)
    print("WS-H: Final Cleaned Dataset")
    print("=" * 70)

    # 1. Load all sources
    print("\n[1/6] Loading all sources...")
    filtered = pd.read_csv(LABELS / "contracts_filtered.csv")
    print(f"  filtered: {len(filtered):,} contracts")
    train_ids = set(pd.read_csv(SPLITS / "train.csv")["id"].tolist())
    val_ids = set(pd.read_csv(SPLITS / "val.csv")["id"].tolist())
    test_ids = set(pd.read_csv(SPLITS / "test.csv")["id"].tolist())
    print(f"  train: {len(train_ids):,}, val: {len(val_ids):,}, test: {len(test_ids):,}")
    review_ids = set(pd.read_csv(LABELS / "review_pending_ids.csv")["id"].tolist())
    print(f"  review_pending: {len(review_ids):,}")
    dropped_ids = set(pd.read_csv(LABELS / "dropped_contracts.csv")["id"].tolist())
    print(f"  dropped: {len(dropped_ids):,}")

    # dedup_map for ID -> folder mapping
    id_to_folder = {}
    with (INTEG / "dedup_map.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            id_to_folder[row["canonical_id"]] = row["folders"].split(";")[0]
    print(f"  id->folder map: {len(id_to_folder):,}")

    # complexity stats
    complexity = pd.read_csv(COMPLEX / "per_contract_stats.csv")
    print(f"  complexity: {len(complexity):,} contracts")

    # 2. Build cleaned dataset
    print("\n[2/6] Building cleaned dataset...")
    cleaned = filtered[["id"] + SENTINEL_V9_ORDER + ["review_pending"]].copy()
    # Add primary class (first positive vuln class, or NV if no vuln)
    def primary_class(row):
        for c in SENTINEL_V9_ORDER:
            if c != "Class12:NonVulnerable" and row[c] == 1:
                return SENTINEL_V9_SHORT[c]
        return "NonVulnerable"
    cleaned["primary_class"] = cleaned.apply(primary_class, axis=1)
    # Add source folder
    cleaned["bccc_folder"] = cleaned["id"].map(lambda i: id_to_folder.get(i, ""))
    # Add bccc_file_path
    cleaned["bccc_file_path"] = cleaned.apply(
        lambda r: f"BCCC-SCsVul-2024/Source Codes/{r['bccc_folder']}/{r['id']}.sol" if r["bccc_folder"] else "",
        axis=1
    )
    # Add n_pos
    cleaned["n_pos"] = cleaned[SENTINEL_V9_ORDER].sum(axis=1).astype(int)
    # Add is_pure_nv flag
    vuln_cols = [c for c in SENTINEL_V9_ORDER if c != "Class12:NonVulnerable"]
    cleaned["is_pure_nv"] = ((cleaned["Class12:NonVulnerable"] == 1) & (cleaned[vuln_cols].sum(axis=1) == 0)).astype(int)
    print(f"  Cleaned: {cleaned.shape}")
    print(f"  Pure NV: {cleaned['is_pure_nv'].sum():,}")

    # 3. Merge complexity stats
    print("\n[3/6] Merging complexity stats...")
    # complexity is keyed by sha256(content) — but we have canonical_id (BCCC ID)
    # We need a sha256 -> id mapping. From dedup_map.
    sha_to_id = {}
    with (INTEG / "dedup_map.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            sha_to_id[row["content_sha256"]] = row["canonical_id"]
    complexity["id"] = complexity["content_sha256"].map(sha_to_id)
    complexity = complexity.dropna(subset=["id"])
    complexity["id"] = complexity["id"].astype(str)
    # Rename columns to match our schema
    complexity = complexity.rename(columns={"loc_total": "loc"})
    complexity["has_pragma"] = complexity["pragma"].notna().astype(int)
    complexity_subset = complexity[["id", "loc", "n_functions", "n_events", "n_modifiers", "has_pragma", "pragma", "spdx"]]
    cleaned = cleaned.merge(complexity_subset, on="id", how="left")
    print(f"  After merge: {cleaned.shape}")
    print(f"  Contracts with complexity stats: {cleaned['loc'].notna().sum():,}")

    # 4. Save
    print("\n[4/6] Saving...")
    csv_path = OUT / "contracts_clean.csv"
    cleaned.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path} ({cleaned.shape[0]:,} × {cleaned.shape[1]})")
    parquet_path = OUT / "contracts_clean.parquet"
    cleaned.to_parquet(parquet_path, index=False)
    print(f"  Wrote {parquet_path}")

    # Compute hash for provenance
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    parquet_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    print(f"  CSV sha256: {csv_hash}")
    print(f"  Parquet sha256: {parquet_hash}")

    # 5. Build split_assignments.csv
    print("\n[5/6] Building split_assignments.csv...")
    all_ids = cleaned["id"].tolist()
    split_rows = []
    for cid in all_ids:
        if cid in train_ids:
            split = "train"
        elif cid in val_ids:
            split = "val"
        elif cid in test_ids:
            split = "test"
        elif cid in review_ids:
            split = "review_pending"
        else:
            split = "UNASSIGNED"
        split_rows.append({"id": cid, "split": split})
    splits_df = pd.DataFrame(split_rows)
    splits_df.to_csv(OUT / "split_assignments.csv", index=False)
    print(f"  Wrote {OUT / 'split_assignments.csv'} ({len(splits_df):,} assignments)")
    print(f"  Split distribution:")
    for s, n in splits_df["split"].value_counts().items():
        print(f"    {s}: {n:,}")

    # 6. metadata.json
    print("\n[6/6] Writing metadata.json...")
    metadata = {
        "dataset_name": "BCCC-SCsVul-2024 (cleaned for SENTINEL v9)",
        "version": "v1.0",
        "created": "2026-06-06",
        "schema": "SENTINEL v9 (10 classes)",
        "class_order": SENTINEL_V9_ORDER,
        "n_contracts": int(cleaned.shape[0]),
        "n_train": int((splits_df["split"] == "train").sum()),
        "n_val": int((splits_df["split"] == "val").sum()),
        "n_test": int((splits_df["split"] == "test").sum()),
        "n_review_pending": int((splits_df["split"] == "review_pending").sum()),
        "n_pure_nv": int(cleaned["is_pure_nv"].sum()),
        "decisions_applied": {
            "D-F1": "Dropped 1,122 contracts that had only Class05/Class07 (no SENTINEL v9 equivalent).",
            "D-B2": "Held out 766 NV+vuln contradictions as review_pending (manual review needed).",
        },
        "files": {
            "contracts_clean.csv": {"sha256": csv_hash, "rows": int(cleaned.shape[0]), "cols": int(cleaned.shape[1])},
            "contracts_clean.parquet": {"sha256": parquet_hash, "rows": int(cleaned.shape[0]), "cols": int(cleaned.shape[1])},
            "split_assignments.csv": {"rows": int(splits_df.shape[0])},
        },
        "source_provenance": {
            "original_zip": "/mnt/e/Project/Foundry_Advanced/Section4 Foundry Cross Chain Rebase Token/BCCC-SCsVul-2024.zip",
            "original_csv": "BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv",
            "csv_md5": "e38a2aa1c2b8a93c6cf8b23d2d7b870a",
            "phase2_dir": "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/",
        },
        "schema_mapping": {
            "BCCC_csv_columns": "Class01..Class12 (long format, one class per row)",
            "BCCC_csv_encoding": "long format: each row is (ID, single_class) with class 0/1; same ID appears for multiple classes → multi-label",
            "SENTINEL_v9_schema": "10 classes in fixed order: " + ", ".join(SENTINEL_V9_ORDER),
        },
        "random_seed": 42,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Wrote {OUT / 'metadata.json'}")

    # 7. README.md
    print("\n  Writing README.md...")
    readme = f"""# BCCC-SCsVul-2024 (Cleaned for SENTINEL v9) — v1.0

**Date:** 2026-06-06
**Schema:** SENTINEL v9 (10 classes)
**Total contracts:** {cleaned.shape[0]:,}

## Files

| File | Format | Rows × Cols | Description |
|---|---|---|---|
| `contracts_clean.csv` | CSV | {cleaned.shape[0]:,} × {cleaned.shape[1]} | Main deliverable |
| `contracts_clean.parquet` | Parquet | {cleaned.shape[0]:,} × {cleaned.shape[1]} | Same data, faster load |
| `split_assignments.csv` | CSV | {splits_df.shape[0]:,} × 2 | Train/val/test/review_pending per ID |
| `metadata.json` | JSON | — | Full provenance + hashes |
| `README.md` | Markdown | — | This file |

## Schema

### Class columns (10, SENTINEL v9-aligned)

| # | Class | Long Name | n |
|---:|---|---|---:|
"""
    for i, c in enumerate(SENTINEL_V9_ORDER, 1):
        n = int(cleaned[c].sum())
        readme += f"| {i} | `{SENTINEL_V9_SHORT[c]}` | `{c}` | {n:,} |\n"

    readme += f"""

### Other columns

- `id` — 64-hex keccak-256 of bytecode (BCCC's original ID)
- `primary_class` — first positive vuln class, or `NonVulnerable` if none (single-label view)
- `n_pos` — number of positive classes for this contract (1 to {int(cleaned['n_pos'].max())})
- `is_pure_nv` — 1 if `Class12:NonVulnerable=1` AND no other class is positive
- `review_pending` — 1 if D-B2 flagged for manual review (NV+vuln contradiction); 0 otherwise
- `bccc_folder` — original BCCC folder name (CallToUnknown, Reentrancy, etc.)
- `bccc_file_path` — relative path to the .sol source file
- `loc`, `n_functions`, `n_events`, `n_modifiers` — complexity stats (from WS-E)
- `has_pragma` — 1 if `pragma solidity ...` directive found
- `pragma` — pragma string (e.g., `^0.4.24`)

## Splits

| Split | n | % |
|---|---:|---:|
| Train | {int((splits_df['split'] == 'train').sum()):,} | {100*(splits_df['split'] == 'train').sum()/len(splits_df):.1f}% |
| Val | {int((splits_df['split'] == 'val').sum()):,} | {100*(splits_df['split'] == 'val').sum()/len(splits_df):.1f}% |
| Test | {int((splits_df['split'] == 'test').sum()):,} | {100*(splits_df['split'] == 'test').sum()/len(splits_df):.1f}% |
| Held out (review_pending) | {int((splits_df['split'] == 'review_pending').sum()):,} | — |

Stratification: simple 2-stage on (has_vuln, primary_vuln_class). See `../splits/split_summary.md` for details.

## Usage

```python
import pandas as pd
df = pd.read_csv("contracts_clean.csv")
# Filter to train
train_ids = pd.read_csv("split_assignments.csv")
train_ids = train_ids[train_ids["split"] == "train"]["id"]
train = df[df["id"].isin(train_ids)]
```

## Decisions Applied

- **D-F1:** Dropped 1,122 contracts that had only Class05 (TransactionOrderDependence) and/or Class07 (WeakAccessMod) — these classes have no SENTINEL v9 equivalent.
- **D-B2:** Held out 766 NV+vuln contradictions as `review_pending=1`. These need manual review before re-inclusion.
- **D-D:** No byte-identical overlap with SmartBugs-curated (0 contracts) → safe to use SmartBugs as OOD test set.

## Provenance

- **Original ZIP:** `/mnt/e/Project/Foundry_Advanced/Section4 Foundry Cross Chain Rebase Token/BCCC-SCsVul-2024.zip`
- **Original CSV:** `BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv` (md5: `e38a2aa1c2b8a93c6cf8b23d2d7b870a`)
- **Phase 2 work:** `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/`

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
# Full pipeline (5 workstreams, ~10-15 min):
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/a_integrity_dedup.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/b_label_validation.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/c_compile_probe.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/d_cross_corpus.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/e_complexity_profile.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/f_class_reconciliation.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/g_stratified_split.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/h_final_dataset.py
```

## Caveats

1. **Compilation success rate: 73%** (100-contract stratified sample, solc 0.4.24/0.5.17). Larger solc version library would improve this.
2. **Stratification is approximate** — simple 2-stage on rare-positive-class. For best results, install `iterative-stratification` and re-run WS-G.
3. **Review-pending set (766 contracts)** is excluded from initial training. Resolve via manual review, then re-include.
4. **NV label treated as 10th class** — model will train on it. Alternative: drop NV and use as a separate clean test set.
"""
    (OUT / "README.md").write_text(readme)
    print(f"  Wrote {OUT / 'README.md'}")
    print("\nWS-H complete. Final dataset ready.")


if __name__ == "__main__":
    main()
