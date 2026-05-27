# Task 12: Token Integrity Audit
**Sample size:** 100  
**Successfully loaded:** 100  
**Skipped (load errors):** 0

## Check Results
| Check | Pass | Fail | Rate |
|-------|------|------|------|
| input_ids_shape | 100 | 0 | 100.0% |
| attention_mask_shape | 100 | 0 | 100.0% |
| padding_windows_mask0 | 100 | 0 | 100.0% |
| real_windows_mask_non0 | 100 | 0 | 100.0% |
| input_ids_range | 100 | 0 | 100.0% |
| num_tokens_match | 100 | 0 | 100.0% |
| no_nan | 100 | 0 | 100.0% |
| no_negative | 100 | 0 | 100.0% |
| schema_v4 | 100 | 0 | 100.0% |

## Distribution of num_windows
| num_windows | Count |
|-------------|-------|
| 1 | 7 |
| 2 | 5 |
| 3 | 6 |
| 4 | 82 |

## Distribution of num_tokens
| Statistic | Value |
|-----------|-------|
| min | 175 |
| max | 2044 |
| mean | 1737.0 |
| median | 1902.5 |
| p5 | 486.1 |
| p95 | 2008.3 |

### Token Count Histogram
| Range | Count |
|-------|-------|
| 0-255 | 1 |
| 256-511 | 6 |
| 512-767 | 0 |
| 768-1023 | 5 |
| 1024-1535 | 6 |
| 1536-2047 | 82 |
| 2048+ | 0 |

✅ All checks passed on all sampled files.
