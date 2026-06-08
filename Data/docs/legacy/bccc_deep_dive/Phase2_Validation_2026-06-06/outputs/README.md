# BCCC-SCsVul-2024 (Cleaned for SENTINEL v9) — v1.0

**Date:** 2026-06-06
**Schema:** SENTINEL v9 (10 classes)
**Total contracts:** 67,311

## Files

| File | Format | Rows × Cols | Description |
|---|---|---|---|
| `contracts_clean.csv` | CSV | 67,311 × 24 | Main deliverable |
| `contracts_clean.parquet` | Parquet | 67,311 × 24 | Same data, faster load |
| `split_assignments.csv` | CSV | 67,311 × 2 | Train/val/test/review_pending per ID |
| `metadata.json` | JSON | — | Full provenance + hashes |
| `README.md` | Markdown | — | This file |

## Schema

### Class columns (10, SENTINEL v9-aligned)

| # | Class | Long Name | n |
|---:|---|---|---:|
| 1 | `ExternalBug` | `Class01:ExternalBug` | 3,604 |
| 2 | `GasException` | `Class02:GasException` | 6,879 |
| 3 | `MishandledException` | `Class03:MishandledException` | 5,154 |
| 4 | `Timestamp` | `Class04:Timestamp` | 2,674 |
| 5 | `UnusedReturn` | `Class06:UnusedReturn` | 3,229 |
| 6 | `CallToUnknown` | `Class08:CallToUnknown` | 11,131 |
| 7 | `DenialOfService` | `Class09:DenialOfService` | 12,394 |
| 8 | `IntegerUO` | `Class10:IntegerUO` | 16,740 |
| 9 | `Reentrancy` | `Class11:Reentrancy` | 17,698 |
| 10 | `NonVulnerable` | `Class12:NonVulnerable` | 26,914 |


### Other columns

- `id` — 64-hex keccak-256 of bytecode (BCCC's original ID)
- `primary_class` — first positive vuln class, or `NonVulnerable` if none (single-label view)
- `n_pos` — number of positive classes for this contract (1 to 8)
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
| Train | 46,581 | 69.2% |
| Val | 9,982 | 14.8% |
| Test | 9,982 | 14.8% |
| Held out (review_pending) | 766 | — |

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
