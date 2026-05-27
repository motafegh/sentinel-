# SENTINEL v6 Audit Scripts

Scripts for comprehensive pre-fix analysis of the graph and token datasets.

## Quick Start

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate

# Run a single task
PYTHONPATH=. python ml/scripts/audit/task09_feature_range_audit.py

# Run all tasks
PYTHONPATH=. python ml/scripts/audit/run_all_audits.py

# Run a specific batch
PYTHONPATH=. python ml/scripts/audit/run_all_audits.py --batch 1

# Dry run (show schedule without executing)
PYTHONPATH=. python ml/scripts/audit/run_all_audits.py --dry-run
```

## Task Scripts

### Batch 1: Integrity & Alignment (run first — invisible data corruption)

| Script | Task | What it checks | Est. |
|--------|------|----------------|------|
| `task09_feature_range_audit.py` | 9 | Feature values in expected ranges, declaration vs CFG split | 15m |
| `task10_token_graph_alignment.py` | 10 | Graph↔token contract_hash/path match | 30m |
| `task11_file_triple_alignment.py` | 11 | CSV∩graphs∩tokens file count alignment | 10m |
| `task12_token_integrity.py` | 12 | Token .pt shape, padding, vocab bounds | 15m |
| `task13_graph_integrity.py` | 13 | Graph .pt dimensions, edge bounds, NaN, self-loops | 20m |
| `task26_stale_v5_contamination.py` | 26 | 8-dim v5.0 graphs still in dataset | 20m |

### Batch 2: Architecture Decisions

| Script | Task | What it checks | Est. |
|--------|------|----------------|------|
| `task16_wrong_contract_selection.py` | 16 | Wrong contract in multi-contract files (7-28% rate) | 60m |
| `task17_safemath_viability.py` | 17 | Can SafeMath be detected? Replacement for in_unchecked? | 45m |
| `task18_solidity_version_dist.py` | 18 | Version breakdown, feature implications | 30m |
| `task21_feature_correlation.py` | 21 | Feature redundancy, MI with labels, PCA | 30m |

### Batch 3: Label Quality

| Script | Task | What it checks | Est. |
|--------|------|----------------|------|
| `task19_timestamp_label_quality.py` | 19 | Timestamp mislabelling rate (4/5 had no block.timestamp) | 45m |
| `task20_dos_reentrancy_separability.py` | 20 | Can DoS be distinguished from Reentrancy? (98% overlap) | 30m |

### Batch 4: Confounds & Distribution Shift

| Script | Task | What it checks | Est. |
|--------|------|----------------|------|
| `task22_graph_size_confound.py` | 22 | Is "big graph = vulnerable" a spurious signal? | 30m |
| `task23_send_unchecked_prevalence.py` | 23 | .send() unchecked return miss rate | 30m |
| `task24_token_graph_source_alignment.py` | 24 | Retokenization alignment verification | 20m |
| `task25_split_distribution_shift.py` | 25 | Feature distribution shift between train/val/test | 30m |

### Batch 5: Extended Validation

| Script | Task | What it checks | Est. |
|--------|------|----------------|------|
| `task14_subsample_coverage.py` | 14 | Vulnerability code survives window sub-sampling? | 30m |
| `task15_in_unchecked_regex.py` | 15 | in_unchecked regex false positive test | 20m |
| `task1_recheck_activation_split.py` | 1-R | Feature activation rates (declaration vs CFG split) | 20m |
| `task5_recheck_edge_types_full.py` | 5-R | EMITS/INHERITS full dataset confirmation | 15m |

## Output

Each script saves a markdown report to `ml/scripts/audit/reports/`.

## Already-Known Bugs (reference only — do NOT re-discover)

| ID | Severity | Description |
|----|----------|-------------|
| BUG-1 | HIGH | CFG node `loc` [6] raw, not log-normalized |
| BUG-2 | HIGH | `complexity` [5] raw, not normalized |
| BUG-3 | MEDIUM | `visibility` [1] ordinal 0/1/2 (private=2, not public) |
| BUG-5 | CRITICAL | `in_unchecked` [9] dead — always 0 (0.4.x dataset) |
| BUG-6 | HIGH | Wrong contract selection 7-28% per class |
| BUG-7 | MEDIUM | EMITS edges (type 3) never generated |
| BUG-8 | MEDIUM | INHERITS edges (type 4) never generated |
| BUG-9 | LOW | `return_ignored` misses `.send()` unchecked returns |
