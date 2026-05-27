# Task 11: File Triple Alignment Audit
## Set Sizes
| Set | Count |
|-----|-------|
| CSV (multilabel_index_deduped.csv) | 44470 |
| Graph .pt files | 44470 |
| Token_windowed .pt files | 44470 |
| Token (legacy) .pt files | 44470 |

## Intersections
| Intersection | Count |
|--------------|-------|
| CSV ∩ Graphs | 44470 |
| CSV ∩ Tokens_windowed | 44470 |
| Graphs ∩ Tokens_windowed | 44470 |
| CSV ∩ Graphs ∩ Tokens_windowed | 44470 |

## Coverage Rates
| Metric | Rate |
|--------|------|
| CSV→Graphs coverage | 100.0% |
| CSV→Tokens coverage | 100.0% |
| CSV→Triple coverage | 100.0% |
| Graphs→Tokens coverage | 100.0% |

## Stems in CSV but not in Graphs (first 20)
None — all CSV stems have graph files.

## Stems in Graphs but not in Tokens_windowed (first 20)
None — all graph stems have token files.

## Stems in Graphs but not in CSV (first 20)
None — all graph stems are in CSV.

## Retokenization Checkpoint
- **Path:** `/home/motafeq/projects/sentinel/ml/data/tokens_windowed/checkpoint.json`
- **Exists:** True
- **total:** 44470
- **completed:** True
- **timestamp:** 2026-05-16T23:12:22.780490

✅ Retokenization appears **completed**.
