# Pre-Run-9 Fixes — Overview and Priority

**Status:** Draft (2026-06-06)
**Owner:** Ali (SENTINEL)
**Context:** Run 8 (GCB-P1-Run8-v10-20260605) killed at ep29 with test F1=0.2307. Manual audit
confirmed five distinct failure modes — only one is a real schema gap, the others are local code bugs
or label noise. This folder contains the fix proposals, each with proper source-code referencing.

---

## Why Pre-Run-9 (not just resume Run 8)

Run 8 plateaued at F1=0.30 (vs Run 7 0.34). Pure architecture iteration will not beat Run 7 without
fixing what the audit calls label noise + extraction bugs + one schema gap. Run 8 already
explored 4 architectures (baseline, no-complexity, APPNP, prefix-warmup) on the same broken labels.

**Pre-Run-9 = stop the architecture bleeding by fixing the data and ground-truth first.**

---

## Priority Matrix (verified by 2026-06-05 audit)

| Number | Fix | Effort | Classes Impacted | Re-extract Needed |
|--------|-----|--------|------------------|-------------------|
| 1 | Run --relabel-timestamp on v10 CSV | 10 min, no re-extract | Timestamp | No |
| 2 | Fix _compute_uses_block_globals (catch now keyword, library wrappers) | 30 min + re-extract | Timestamp, TOD | Yes |
| 3 | Add CALL_ENTRY/RETURN_TO for HighLevelCall/LowLevelCall (external calls) | 2 hr + re-extract | DoS, ExternalBug, Reentrancy | Yes |
| 4 | Add unchecked-block feature (Solidity 0.8+) + CFG_NODE_ARITH type | 4 hr + re-extract | IntegerUO | Yes |
| 5 | Re-derive labels from Slither detectors (ml/data/slither_results/) | 1 day | All 10 classes | Partial |

### Bonus / Cleanup (not blocking Run 9)

| Number | Fix | Effort | Classes Impacted | Re-extract Needed |
|--------|-----|--------|------------------|-------------------|
| 6 | Fix predictor tier-threshold bug (hardcoded 0.55 vs per-class tuned) | 30 min | Manual eval display only | No |
| 7 | Add manual_test_smartbugs.py benchmark (swap OOD synth contracts) | 1 hr | Eval methodology | No |
| 8 | Document feat[5] complexity complexity-proxy bias fix | Already done in Run 8 | -- | No |

---

## Verified Audit Findings (2026-06-05)

| Letter | Finding | Source |
|--------|---------|--------|
| A | --relabel-timestamp NEVER applied to v10 CSV | ml/scripts/archive/dedup_multilabel_index.py:226 (never invoked) |
| B | Only 27.5% of Timestamp=1 graphs fire uses_block_globals (feat[2]) | Audit sample n=200 Timestamp=1 vs 8% baseline |
| C | Test-set precision degenerate for 9/10 classes (only IntegerUO passes) | ml/checkpoints/GCB-P1-Run8-v10-20260605_best_thresholds.json |
| D | CALL_ENTRY (edge type 8) only iterates node.internal_calls -- external calls get nothing | ml/src/preprocessing/graph_extractor.py:858-863 (_add_icfg_edges) |
| E | _compute_uses_block_globals misses now keyword (Solidity 0.4.x alias) | ml/src/preprocessing/graph_extractor.py:459 |
| F | IntegerUO unlearnable: arithmetic IR ops collapse into CFG_NODE_OTHER(12) bucket | ml/src/preprocessing/graph_extractor.py:_cfg_node_type |
| G | Manual test contracts are OOD (bottom 1st-7th percentile by size, all 0.8+ syntax) | ml/scripts/test_contracts/ vs training median |
| H | Predictor _format_result() ignores per-class tuned thresholds (hardcoded 0.55) | ml/src/inference/predictor.py:150-151, :712-715 |
| I | 87.9% of BCCC dataset is pre-0.8 Solidity (0.4-0.7) -- in_unchecked rightly dropped | ml/src/preprocessing/graph_schema.py:119-120 |
| J | Model fires near-constant ~0.30-0.45 baseline on safe contracts (Run 7 + Run 8) | Run 7 audit L4 + Run 8 manual test |

---

## Re-extract Trigger -- what counts as a schema change

**Re-extract needed** (bump FEATURE_SCHEMA_VERSION):
- Adding a new node type (Fix #4: CFG_NODE_ARITH)
- Adding a new feature dimension (Fix #4: in_unchecked_block)
- Adding a new edge type to the on-disk schema (Fix #3: CALL_ENTRY for external)

**Re-extract NOT needed**:
- Changing how an existing feature is computed (Fix #2: _compute_uses_block_globals)
- Label cleanup (Fix #1: --relabel-timestamp)
- Predictor display logic (Fix #6)

---

## Run-9 Gating Criteria (after all fixes applied)

1. Re-derive IntegerUO labels from Slither integer-overflow detector + structural guard (re-extract IF Slither matches, drop false positives).
2. Re-derive Timestamp labels via --relabel-timestamp + now/library extraction fix.
3. Re-validate CALL_ENTRY/RETURN_TO coverage: at least 5% of training graphs should now have an EXTERNAL CALL_ENTRY edge (current: 0%).
4. SmartBugs Curated baseline: per-class precision > 0.3 on 6/10 classes (currently 1/10).
5. Manual safe contracts: no class above 0.50 (currently 5/10 classes fire > 0.50 on 12_safe_contract.sol).
