# WS-J: Statistical EDA Report

**Date:** 2026-06-06  
**Source data:** BCCC-SCsVul-2024 + SENTINEL v9 cleaned v1.0  
**Author:** SENTINEL Phase 3 (deep analysis)  

## 1. Headline numbers

- BCCC raw (long format): **111,897** rows × **254** cols (68,433 unique contracts)
- SENTINEL v9 cleaned v1.0: **67,311** contracts × **24** cols
- Dropped (Class05/Class07 only): **1,122** contracts
- Review-pending (NV+vuln contradiction): **766** contracts
- Pure-NV contracts: **26148** (38.8%)
- Per-contract mean rows in long format: **1.64** (max 9, suggests multi-label encoding)

## 2. Per-class prevalence (BCCC raw, per-contract)

| Class | n contracts | % of total |
|---|---:|---:|
| Class01:ExternalBug | 3,604 | 5.27% |
| Class02:GasException | 6,879 | 10.05% |
| Class03:MishandledException | 5,154 | 7.53% |
| Class04:Timestamp | 2,674 | 3.91% |
| Class05:TransactionOrderDependence | 3,562 | 5.21% |
| Class06:UnusedReturn | 3,229 | 4.72% |
| Class07:WeakAccessMod | 1,918 | 2.80% |
| Class08:CallToUnknown | 11,131 | 16.27% |
| Class09:DenialOfService | 12,394 | 18.11% |
| Class10:IntegerUO | 16,740 | 24.46% |
| Class11:Reentrancy | 17,698 | 25.86% |
| Class12:NonVulnerable | 26,914 | 39.33% |

## 3. n_pos distribution (multi-label cardinality)

BCCC raw (per-contract):

| n_pos | n contracts | % |
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

SENTINEL v9 cleaned:

| n_pos | n contracts | % |
|---:|---:|---:|
| 1 | 40,787 | 60.59% |
| 2 | 18,499 | 27.48% |
| 3 | 5,073 | 7.54% |
| 4 | 1,740 | 2.59% |
| 5 | 879 | 1.31% |
| 6 | 278 | 0.41% |
| 7 | 50 | 0.07% |
| 8 | 5 | 0.01% |

## 4. Class co-occurrence (top 10 by joint frequency)

Full matrix: `outputs/ws_j_cooccurrence_bccc.csv` and `ws_j_cooccurrence_bccc_pct.csv`

## 5. Top 10 multi-label combinations

| Count | Label set |
|---:|---|
| 26,148 | ('Class12',) |
| 11,246 | ('Class09', 'Class11') |
| 6,255 | ('Class08',) |
| 4,584 | ('Class10',) |
| 1,846 | ('Class03', 'Class10') |
| 1,201 | ('Class08', 'Class10') |
| 959 | ('Class07',) |
| 959 | ('Class07', 'Class08') |
| 939 | ('Class01', 'Class06', 'Class11') |
| 935 | ('Class02', 'Class09', 'Class11') |

## 6. Review-pending (NV+vuln contradiction)

- Total: **766** contracts (1.14% of cleaned dataset)
- These are contracts labeled BOTH NonVulnerable AND at least one vuln class. Likely label noise.
- Distribution of n_pos within review-pending:

  - n_pos=2: 55
  - n_pos=3: 705
  - n_pos=4: 1
  - n_pos=5: 5

## 7. Dropped contracts (Class05/Class07 only)

- Total dropped: **1,122** contracts
  - Class05 (TransactionOrderDependence) only: **163**
  - Class07 (WeakAccessMod) only: **959**
  - Both Class05 + Class07: **0**
- Reason: Class05 and Class07 have no SENTINEL v9 equivalent (D-F1).
- Recovery: would need to add a SWC-114 or access-control class to SENTINEL v9.

## 8. Feature missingness (BCCC 241 non-class cols)

- Cols with 0% missing: **240**
- Cols with >50% missing: **0**
- Cols with 100% missing (all-NaN): **0**
- Numeric cols: **238**

Full missingness: `outputs/ws_j_feature_missingness.csv`

## 9. OPCODE/Bytecode feature inventory

| Feature group | n cols |
|---|---:|
| Opcode Count | 138 |
| Bytecode Character Count | 63 |
| Bytecode Length and Entropy | 2 |
| ABI Features | 11 |
| AST Features | 5 |
| Functional Features | 5 |
| Solidity Features | 8 |
| Lines of Code | 4 |

## 10. Splits

| Split | n | % |
|---|---:|---:|
| train | 46,581 | 69.20% |
| val | 9,982 | 14.83% |
| test | 9,982 | 14.83% |
| review_pending | 766 | 1.14% |

## 11. Implications for Phase 3 workstreams

- **WS-M (BCCC 242-feature test):** All 241 features are present (low missingness), so WS-M can test the full BCCC feature set on a 5,000 stratified sample.
- **WS-L (AutoML):** Class imbalance is severe (Reentrancy 17,698 vs ExternalBug 3,604 ≈ 4.9×). Use class_weight='balanced' or SMOTE.
- **WS-N (dropped review):** 1,122 dropped contracts = 1.64% of BCCC. Recovery would expand SENTINEL's coverage to 100% of BCCC.
- **WS-T (multi-label structure):** n_pos=1 covers 58.8% of BCCC. Multi-label structure is meaningful for the remaining 41.2%.
