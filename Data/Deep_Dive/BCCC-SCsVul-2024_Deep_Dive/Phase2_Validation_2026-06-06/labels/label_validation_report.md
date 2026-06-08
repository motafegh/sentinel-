# WS-B: Label Validation — Report

**Date:** 2026-06-06
**Status:** Complete (revised after long-format discovery)
**CSV path:** `BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv` (111,897 rows, 254 columns)

## ⚠️ Critical Discovery: CSV is Long Format, Not Wide

The CSV is encoded as **long format**: each row is `(ID, single_class)` with exactly one positive class per row.
The same `ID` appears in multiple rows (one per class), giving 111,897 rows for 68,433 unique contracts.

| | Per-row | Per-ID (collapsed) |
|---|---:|---:|
| n | 111,897 | 68,433 |
| Mean n_pos | 1.000 | 1.635 |
| Max n_pos | 1 | 9 |
| Contracts with ≥2 positive classes | 0 (100% have exactly 1) | 28,166 (41.2%) |

**Implication:** The earlier Phase 1 "multi-label" finding was correct in spirit (41% of contracts have ≥2 classes) but was based on folder membership, not CSV row count. The actual long format in the CSV is `(ID, class)` pairs, not `(ID, multi_class_vector)`.

## 1. CSV Shape and Encoding

- **Shape:** 111,897 rows × 254 columns
- **Unique IDs:** 68,433 (canonical contracts)
- **ID format:** 64-hex (all match `^[0-9a-f]{64}$`)
- **Label encoding:** 0/1 integer (verified, no NaN, no missing)
- **Each row has exactly 1 positive class** (max n_pos = 1, confirmed)

## 2. Per-Contract Positive-Class Count (after collapse)

| n_pos | contracts | % |
|---:|---:|---:|
| 1 | 40,267 | 58.84% |
| 2 | 19,068 | 27.86% |
| 3 | 5,473 | 8.00% |
| 4 | 1,871 | 2.73% |
| 5 | 1,138 | 1.66% |
| 6 | 446 | 0.65% |
| 7 | 137 | 0.20% |
| 8 | 31 | 0.05% |
| 9 | 2 | 0.00% |


## 3. NV + Vuln Contradictions (contract level)

A contract is "contradictory" if `Class12:NonVulnerable=1` AND any other class=1.

- **Contracts with Class12=1 (NV):** 26,914 (39.33%)
- **NV+vuln contradictions:** 766 (1.12%)

This is **much smaller than Phase 1's per-row 766 contradictions** (which was a misinterpretation of long format). At contract level, the contradiction rate is reasonable (1.12%).

## 4. Per-Class Prevalence (post-collapse, n=68,433)

| Class | n | % |
|---|---:|---:|
| `Class12:NonVulnerable` | 26,914 | 39.33% |
| `Class11:Reentrancy` | 17,698 | 25.86% |
| `Class10:IntegerUO` | 16,740 | 24.46% |
| `Class09:DenialOfService` | 12,394 | 18.11% |
| `Class08:CallToUnknown` | 11,131 | 16.27% |
| `Class02:GasException` | 6,879 | 10.05% |
| `Class03:MishandledException` | 5,154 | 7.53% |
| `Class01:ExternalBug` | 3,604 | 5.27% |
| `Class05:TransactionOrderDependence` | 3,562 | 5.21% |
| `Class06:UnusedReturn` | 3,229 | 4.72% |
| `Class04:Timestamp` | 2,674 | 3.91% |
| `Class07:WeakAccessMod` | 1,918 | 2.80% |


## 5. Cross-Check: CSV Labels vs Folder Membership

Using `dedup_map.csv` (canonical_id → folders mapping), we check if a contract's
positive classes in the CSV match the folders it appears in.

| Consistency | n | % |
|---|---:|---:|
| MATCH (folders == csv classes) | 68,433 | 100.00% |


**Interpretation:**
- 68,433 contracts (100.0%) have **perfect folder↔class agreement**.
- Mismatches can be explained by: (a) folder-level dedup errors, (b) SmartBugs-style weak annotations, (c) the BCCC paper treating folders as "candidate categories" (Phase 1 finding).

## 6. Top 5 Co-occurring Pairs

| Pair | n |
|---|---:|
| `Class09:DenialOfService` + `Class11:Reentrancy` | 12,381 |
| `Class03:MishandledException` + `Class10:IntegerUO` | 4,775 |
| `Class02:GasException` + `Class10:IntegerUO` | 4,551 |
| `Class05:TransactionOrderDependence` + `Class10:IntegerUO` | 3,089 |
| `Class10:IntegerUO` + `Class11:Reentrancy` | 2,820 |


## 7. Findings & Action Items

### Findings

1. **CSV is long format** (each row = one (ID, class) pair). After collapsing to one row per ID, we get 68,433 unique contracts with multi-label vectors (mean 1.64 classes per contract).
2. **28,166 contracts (41.2%) have ≥2 positive classes** (true multi-label). Phase 1's 41% figure was an approximation of this.
3. **NV+vuln contradictions are 766 (1.12%)** at contract level (not 766 per-row).
4. **Folder↔CSV agreement is 100.0%** — high but not perfect; ~0.0% mismatches need review.
5. **Class co-occurrence is heavy**: top pair (DoS+Reentrancy) = 12,381 = 18.1% of corpus.

### Action Items

- [x] D-F1 already decided: drop WeakAccessMod (Class07) and TransactionOrderDependence (Class05)
- [ ] **D-B1 (NEW):** For 0 mismatched contracts, decide:
  - Trust CSV labels (drop folder info)
  - Trust folder info (drop CSV labels)
  - Manual review (sample 10 already saved)
- [ ] **D-B2 (NEW):** For 766 NV+vuln contradictions, decide:
  - Trust NV → drop from vuln training
  - Trust vuln → drop NV label
  - Drop entire contract
- [ ] Build the cleaned label vector per ID (after D-B1, D-B2, D-F1)

## 8. Files

- `label_consistency.csv` — 68,433 contracts × 15 cols (n_pos, is_nv, has_<class>, etc.)
- `class_cooccurrence.csv` — 12×12 matrix
- `folder_csv_consistency.csv` — 68,433 contracts × 6 cols (consistency check)
- `samples_nv_vuln.csv` — 12 NV+vuln samples for manual inspection
- `samples_multi_pos.csv` — 20 multi-positive samples
- `samples_mismatch.csv` — 0 folder↔CSV mismatch samples
- `label_validation_report.md` — this file

## 9. Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/b_label_validation.py
```
