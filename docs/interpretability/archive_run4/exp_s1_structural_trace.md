# EXP-S1: Structural Trace Audit

**Layer:** 1 — Structure
**Priority:** P0
**Status:** FAIL
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_s1_structural_trace.py`
**Output:** `ml/logs/interpretability/exp_s1_structural_trace.json`

**Note:** Script had a `GraphExtractionConfig(schema_version="v8")` bug — `schema_version` is not a valid constructor argument (fixed prior to run). Test contract extraction failed because `solc` (Solidity compiler) is not installed in the WSL environment.

---

## Purpose

This experiment checks whether known-vulnerable test Solidity contracts (CEI pattern violations) produce graphs containing the expected structural signatures: CALL_ENTRY edges, RETURN_TO edges, and state-write nodes after RETURN_TO. It also measures the pattern-detection rate for real reentrancy-positive contracts in the val split.

## Method

The script attempts to compile and extract graphs from 12 test `.sol` contracts (covering reentrancy, integer overflow, timestamp, unused return, etc.) and checks each extracted graph for the target structural pattern. For val-split contracts, it checks reentrancy positives for CEI signature presence. Pass criteria: ≥7/12 test contracts find the expected pattern (P0 gate), and ≥50% of val reentrancy positives contain the pattern.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_s1_structural_trace.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 100 \
  --out ml/logs/interpretability/exp_s1_structural_trace.json
```

## Results

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| Test contracts: pattern found | 0/12 | ≥7/12 | FAIL |
| P0 gate (test contracts) | 0/12 | ≥7/10 | FAIL |

**Failure cause:** `solc` (Solidity compiler) not found in PATH — all test contract extractions failed with `[Errno 2] No such file or directory: 'solc'`.

**Val split results (using cached graphs, not fresh extraction):**
| Class | Positives (sampled) | With CEI pattern | Pattern % | Status |
|-------|---------------------|-----------------|-----------|--------|
| Reentrancy | 7 | 1 | 14.3% | FAIL |
| IntegerUO | 27 | 25 | 92.6% | PASS |
| Timestamp | 1 | 1 | 100.0% | PASS |
| UnusedReturn | 2 | 0 | 0.0% | FAIL |
| MishandledException | 13 | 7 | 53.9% | PASS |

## Interpretation

The test contract portion of exp_s1 failed entirely due to missing `solc` — this is an environment prerequisite, not a code or data issue. The val-split pattern detection (which uses the cached pre-extracted graphs) ran successfully and reveals mixed results: IntegerUO and Timestamp show high pattern detection rates, but Reentrancy (14.3%) and UnusedReturn (0%) are very low.

The low reentrancy pattern rate in val (14.3%) is in sharp contrast to the exp_s4 finding (76% with CALL_ENTRY). This discrepancy may arise from the different "pattern" definition used by exp_s1 (which likely requires a stricter full CEI violation trace) vs exp_s4 (which only requires CALL_ENTRY edge presence).

## Pass/Fail Analysis

- P0 gate FAILED due to `solc` absence — this is an environment issue, not an architecture issue.
- Val-split reentrancy pattern rate (14.3%) is concerning — needs investigation.
- IntegerUO val pattern rate (92.6%) is strong, consistent with integer overflow being a syntactic-level pattern.

## Recommended Next Steps

1. **Install `solc`:** `pip install solc-select && solc-select install 0.8.0 && solc-select use 0.8.0` — or use the Docker-based extractor for WSL compatibility.
2. After `solc` installation, re-run to validate test contract extraction.
3. Investigate the discrepancy between exp_s1 reentrancy pattern rate (14.3%) and exp_s4 CALL_ENTRY rate (76%) — check what pattern definition exp_s1 uses for reentrancy.
4. Cross-reference UnusedReturn 0% pattern rate with the exp_s2 finding that EMITS edges are highly enriched for UnusedReturn — the pattern definition may be too strict.
