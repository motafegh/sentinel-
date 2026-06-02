# EXP-S4: ICFG-Lite Path Audit

**Layer:** 1 — Structure
**Priority:** P0
**Status:** PASS
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_s4_icfg_path_audit.py`
**Output:** `ml/logs/interpretability/exp_s4_icfg_path_audit.json`

**Note:** Script had a duplicate `--split` and `--n-contracts` argparse conflict (fixed prior to run). Test contracts (01_reentrancy_classic.sol, 02_reentrancy_tricky.sol) not found — extraction failed for those synthetic tests.

---

## Purpose

This experiment audits whether reentrancy-positive contracts in the val split contain the ICFG-Lite structural signature for reentrancy: a CALL_ENTRY edge (function call entry) followed by a RETURN_TO edge back to the caller, with a state-write occurring after the return. This is the programmatic representation of the CEI (Checks-Effects-Interactions) pattern violation.

## Method

The script loads reentrancy-positive contracts from the val split and checks each graph for three structural properties: (1) presence of CALL_ENTRY edges, (2) full chain: CALL_ENTRY + RETURN_TO path (call-then-return semantics), and (3) a state-write node appearing after the RETURN_TO endpoint. Pass criteria: ≥6 of the first 10 audited contracts must have CALL_ENTRY edges.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_s4_icfg_path_audit.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 100 \
  --out ml/logs/interpretability/exp_s4_icfg_path_audit.json
```

## Results

Audited 100 reentrancy-positive contracts from val split.

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| First 10 with CALL_ENTRY | 7/10 (70%) | ≥6/10 | PASS |
| First 10 with full chain (CALL_ENTRY+RETURN_TO) | 7/10 (70%) | ≥4/10 | PASS |
| First 10 with WRITE after RETURN_TO | 7/10 (70%) | — | — |
| All 100 with CALL_ENTRY | 76/100 (76%) | — | — |
| All 100 with full chain | 69/100 (69%) | — | — |

**Test contract extraction:** FAILED (file_not_found for both test reentrancy .sol files — test_contracts directory missing).

## Interpretation

76% of reentrancy-positive contracts in the val split have CALL_ENTRY edges, and 69% have the complete structural reentrancy chain. This confirms that the SENTINEL graph extraction successfully captures inter-procedural call structure for the majority of reentrancy cases. The 24% gap (contracts with CALL_ENTRY absent) likely represents reentrancy patterns where the reentrant call is more deeply nested or uses delegation patterns not captured as CALL_ENTRY in the current extractor.

The 7/10 pass rate (70%) matches the overall 76% rate, suggesting consistent extraction quality. The fact that full chain rate (69%) matches CALL_ENTRY presence (7/10 for first batch) indicates that when CALL_ENTRY exists, RETURN_TO is almost always present too.

## Pass/Fail Analysis

Both formal criteria passed:
- CALL_ENTRY ≥ 6/10: PASS (7/10)
- Full chain ≥ 4/10: PASS (7/10)

The 31% of reentrancy contracts missing the full chain are a data gap — the GNN's Phase 2 specifically targets CALL_ENTRY/RETURN_TO paths. If 31% of reentrancy positives lack these edges, the model must rely on other signals for those contracts.

## Recommended Next Steps

1. Add test contract files (`01_reentrancy_classic.sol`, `02_reentrancy_tricky.sol`) to `ml/scripts/test_contracts/` for end-to-end extraction validation.
2. Investigate the 24% of reentrancy positives missing CALL_ENTRY — are these delegate-call patterns? If so, add DELEGATECALL extraction to the graph extractor.
3. Cross-reference with model performance: do the 31% without full chain have lower reentrancy recall?
4. After IMP-D1 re-extraction, re-run to check if CALL_ENTRY coverage improves.
