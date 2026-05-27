# Task 20: DoS vs Reentrancy Separability Audit
**Total DoS=1 contracts:** 377  **DoS + Reentrancy:** 370  **DoS-only:** 7

## DoS-Only: Source Pattern Analysis
### DoS Patterns Found
| Pattern | Count | Percentage |
|---------|-------|------------|
| unbounded_loop | 6 | 85.7% |
| external_call | 3 | 42.9% |
| loop_with_call | 2 | 28.6% |
| loop_with_send | 2 | 28.6% |

### Reentrancy Patterns Also Present in DoS-Only (Source)
These DoS-only contracts have reentrancy-like patterns in source but are NOT labelled Reentrancy=1:
| Pattern | Count |
|---------|-------|

## DoS+Reentrancy: Sample Analysis (10 contracts)
- **`94dbf94afd60...`**: DoS patterns=[external_call], Reentrancy patterns=[none detected]- **`102c7a7132d0...`**: DoS patterns=[none detected], Reentrancy patterns=[none detected]- **`04a747dfcdd3...`**: DoS patterns=[external_call], Reentrancy patterns=[none detected]- **`30a16a58137d...`**: DoS patterns=[none detected], Reentrancy patterns=[none detected]- **`29a05fbb441c...`**: DoS patterns=[unbounded_loop, external_call], Reentrancy patterns=[none detected]- **`2674f567fa7e...`**: DoS patterns=[none detected], Reentrancy patterns=[none detected]- **`19382507e0bd...`**: DoS patterns=[none detected], Reentrancy patterns=[none detected]- **`0f0e79df2356...`**: DoS patterns=[external_call], Reentrancy patterns=[external_call_before_write]- **`ab09df979fc2...`**: DoS patterns=[external_call], Reentrancy patterns=[none detected]- **`6d9156766ffa...`**: DoS patterns=[none detected], Reentrancy patterns=[none detected]

### Interpretation
- 1/10 DoS+Reentrancy contracts show both pattern types in source (expected for dual-labelled).

## Feature Comparison: DoS-Only vs DoS+Reentrancy
| Feature | DoS-Only (mean±std) | DoS+Re (mean±std) | Difference |
|---------|--------------------|-------------------|------------|
| n_nodes | 20.43±7.96 | 102.30±49.74 | -81.87 |
| n_edges | 33.29±13.58 | 151.90±74.08 | -118.61 |
| n_cfg_nodes | 14.29±5.77 | 68.80±35.96 | -54.51 |
| n_function_nodes | 3.29±2.12 | 20.30±11.82 | -17.01 |
| mean_has_loop | 0.36±0.31 | 0.02±0.07 | +0.34 |
| mean_ext_call_count | 0.03±0.08 | 0.06±0.03 | -0.03 |
| mean_complexity | 5.56±2.49 | 3.64±1.51 | +1.92 |
| graph_size | 53.71±21.53 | 254.20±123.12 | -200.49 |

*Note: DoS-only has limited sample size; statistics may not be robust.*

## DoS-Only Contract Split Distribution
| Split | Count |
|-------|-------|
| train | 0 |
| val | 0 |
| test | 0 |
| unknown | 7 |

## DoS-Only Contract Details
| # | Stem | Contract | DoS Patterns | Re Patterns | n_nodes | mean_loop | mean_ext_call |
|---|------|----------|-------------|-------------|---------|-----------|---------------|
| 1 | 1f39978ecb7e | DosLoop | unbounded_loop | - | 31 | 0.17 | 0.00 |
| 2 | 59a86f6d07e1 | DosArray | unbounded_loop | - | 31 | 0.17 | 0.00 |
| 3 | 94513bf8fb7b | Refunder | loop_with_call, loop_with_send, unbounded_loop | - | 15 | 0.50 | 0.00 |
| 4 | 95fc8e424291 | Refunder | loop_with_call, loop_with_send, unbounded_loop | - | 15 | 0.50 | 0.00 |
| 5 | ae8313104cef | DosGas | unbounded_loop | - | 26 | 0.20 | 0.00 |
| 6 | d7c71805cb6c | DosOnePush | unbounded_loop | - | 14 | 1.00 | 0.00 |
| 7 | dd3f8807cfcc | DosRequire | external_call | - | 11 | 0.00 | 0.23 |

## Recommendation
**DoS-only contracts are extremely rare** (7 out of 377 total DoS=1). This means:
1. **Keep separate**: The classes are largely overlapping — most DoS contracts are also Reentrancy. Merging would dilute the DoS signal.2. **Augment**: Consider generating synthetic DoS-only examples (e.g., loop-with-transfer without reentrancy pattern) to give the model more training signal for DoS-specific patterns.