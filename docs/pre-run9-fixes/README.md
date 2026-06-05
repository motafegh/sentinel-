# Pre-Run-9 Fix Proposals

**Status:** Draft proposals — none applied yet
**Date:** 2026-06-06
**Context:** Run 8 plateaued at test F1=0.2307 (well below Run 7's 0.3423). Audit of the
data + graph schema surfaced 10 findings (A-J) that explain the regression. The fixes below
target the root causes before launching Run 9.

## Reading Order

1. **[00-overview.md](00-overview.md)** — priority matrix, audit findings A-J, decision flow
2. **[01-relabel-timestamp.md](01-relabel-timestamp.md)** — drop 50% of Timestamp noise (already
   applied; this is the spec for documentation + Run 9 integration)
3. **[02-block-globals-extraction.md](02-block-globals-extraction.md)** — add `now` and library
   wrapper detection to feat[2]
4. **[03-external-call-entry.md](03-external-call-entry.md)** — new edge type 11 for external
   CALL_ENTRY / RETURN_TO
5. **[04-integeruo-schema-gap.md](04-integeruo-schema-gap.md)** — add CFG_NODE_ARITH type 13 +
   re-introduce `in_unchecked_block` feature
6. **[05-slither-derived-labels.md](05-slither-derived-labels.md)** — re-derive all 10 class
   labels from Slither detectors (largest impact, most invasive)
7. **[06-bonus-fixes.md](06-bonus-fixes.md)** — predictor tier-threshold display, SmartBugs
   benchmark, complexity-bias docs (no model change)

## Priority Matrix

| Fix | Effort | Impact | Risk | Blocking Run 9? |
|-----|--------|--------|------|-----------------|
| #1 Timestamp relabel | 1 hr (done) | High (49.9% noise removed) | None | No — already applied |
| #2 Block-globals extraction | 2 hr | High (27.5% of contracts fire feat[2]) | Low | Recommended |
| #3 External CALL_ENTRY edge | 3 hr | Medium (Reentrancy class) | Low | Recommended |
| #4 IntegerUO schema gap | 4 hr | High (IntegerUO + unchecked) | High | Optional |
| #5 Slither-derived labels | 1 day | Highest (all 10 classes) | Medium | Recommended |
| #6 Predictor tier threshold | 30 min | Display only | None | Optional |
| #7 SmartBugs benchmark | 1 hr | Eval methodology | None | Optional |
| #8 Complexity-bias doc | 0 hr | None | None | Documentation only |

**Minimum viable Run 9 launch:** Apply Fix #2 + Fix #3 + use the deduped splits from Fix #1.
**Full quality Run 9:** Apply #2 + #3 + #4 + #5 (skip none of the data fixes).
**Audit-only Run 9:** Apply #2 + #3, then re-audit to see if precision improves.

## Cross-Cutting Concerns

- **Cache invalidation:** Fixes #2, #3, #4 change the graph schema (v8 → v9). All
  `cached_dataset_v*.pkl` files are invalidated. Re-extract + retokenize + rebuild cache.
- **Splits:** Re-use `ml/data/splits/deduped/` from Fix #1 (41,576 rows, 0.7/0.15/0.15 split,
  stratified on 10 classes).
- **Schema version bump:** Bump `FEATURE_SCHEMA_VERSION = v8` to `v9` in
  `ml/src/preprocessing/graph_schema.py:160` after applying #2, #3, #4.
- **Checkpoints:** All v8 checkpoints invalid after schema change. Run 9 starts from
  `ml/checkpoints/GCB-P1-Run8-v10-20260605_best.pt` as a cold start, not a resume.

## Validation Workflow

After all fixes applied, before launching Run 9:

```bash
# 1. Schema consistency
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py --check-all

# 2. Feature distribution
PYTHONPATH=. python ml/scripts/inspect_features.py --split train

# 3. Label distribution
PYTHONPATH=. python ml/scripts/inspect_labels.py --source slither

# 4. Manual contract sanity (after re-extract)
PYTHONPATH=. python ml/scripts/manual_test_smartbugs.py \
  --checkpoint ml/checkpoints/GCB-P1-Run9-v11-2026060X_best.pt

# 5. Compare against Run 7 baseline (F1=0.3423) and Run 8 plateau (F1=0.2307)
```

## Open Questions

- Should Fix #5 (Slither-derived labels) be applied to the FULL BCCC dataset, or only to
  test/val splits for cleaner eval?
- Do we have ground-truth access to a labelled subset of SmartBugs (not just folder names)?
- Should we add a "label provenance" field to the dataset for retrospective re-labeling?

## Files Outside This Folder That Will Change

| File | Why |
|------|-----|
| `ml/src/preprocessing/graph_extractor.py` | #2 (feat[2]), #3 (edge type 11), #4 (arith + unchecked) |
| `ml/src/preprocessing/graph_schema.py` | #3 (edge type 11), #4 (node type 13, feature dim 12) |
| `ml/scripts/archive/dedup_multilabel_index.py` | #1 (already done — line 64 parents[3]) |
| `ml/scripts/derive_slither_labels.py` | NEW for #5 |
| `ml/scripts/validate_graph_dataset.py` | New check flags for #3, #4 |
| `ml/src/inference/predictor.py` | #6 (tier display) |
| `ml/scripts/manual_test_smartbugs.py` | NEW for #7 |
