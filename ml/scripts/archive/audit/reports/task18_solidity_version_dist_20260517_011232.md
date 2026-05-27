# Task 18: Solidity Version Distribution Audit
**Files scanned:** 2000  **Files with labels:** 571  **Unique version buckets:** 4

## Version Distribution
| Version Bucket | Count | Percentage |
|----------------|-------|------------|
| 0.4.x | 1758 | 87.9% |
| 0.5.x | 160 | 8.0% |
| 0.8.x | 1 | 0.1% |
| no_pragma | 81 | 4.0% |

## Per-Class Label Distribution by Version Bucket

| Version | CallToUnknown | DenialOfService | ExternalBug | GasException | IntegerUO | MishandledException | Reentrancy | Timestamp | TransactionOrderDependence | UnusedReturn | Total Labeled |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------------|
| 0.4.x | 8.2% | 0.9% | 8.0% | 9.5% | 34.7% | 10.0% | 11.7% | 5.5% | 8.2% | 5.8% | 548 |
| 0.5.x | 15.0% | 0.0% | 35.0% | 35.0% | 70.0% | 25.0% | 35.0% | 5.0% | 20.0% | 30.0% | 20 |
| 0.8.x | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1 |
| no_pragma | 0.0% | 0.0% | 0.0% | 50.0% | 50.0% | 50.0% | 0.0% | 50.0% | 0.0% | 0.0% | 2 |

## 0.8.x Analysis
**Total 0.8.x files:** 1  **With `unchecked {}`:** 0 (0.0%) 
**With `using SafeMath`:** 0 (0.0%)

### Interpretation
- **`unchecked {}` in 0.8.x** signals potential IntegerUO: Solidity 0.8.0+ has built-in overflow/underflow checks; `unchecked` blocks bypass them.- **`using SafeMath` in 0.8.x** is typically redundant (SafeMath is unnecessary with built-in checks) and may indicate code migrated from older versions without cleanup.
- 0 of 1 0.8.x files (0.0% contain `unchecked {}` blocks — these are the primary IntegerUO candidates in 0.8.x.

## Train/Val/Test Split Version Distribution
*No split files found in `ml/data/processed/`.*

## Key Findings
1. **Dominant version:** 0.4.x (87.9% of files)
2. **Notable class rate differences across versions:**   - **CallToUnknown**: ranges from 0.0% (no_pragma) to 15.0% (0.5.x)   - **ExternalBug**: ranges from 0.0% (no_pragma) to 35.0% (0.5.x)   - **GasException**: ranges from 0.0% (0.8.x) to 50.0% (no_pragma)   - **IntegerUO**: ranges from 0.0% (0.8.x) to 70.0% (0.5.x)   - **MishandledException**: ranges from 0.0% (0.8.x) to 50.0% (no_pragma)   - **Reentrancy**: ranges from 0.0% (no_pragma) to 100.0% (0.8.x)   - **Timestamp**: ranges from 0.0% (0.8.x) to 50.0% (no_pragma)   - **TransactionOrderDependence**: ranges from 0.0% (no_pragma) to 20.0% (0.5.x)   - **UnusedReturn**: ranges from 0.0% (no_pragma) to 30.0% (0.5.x)