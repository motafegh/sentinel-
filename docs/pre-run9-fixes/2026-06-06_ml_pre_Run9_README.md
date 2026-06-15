# Pre-Run-9 Fix Proposals

**Status:** Fixes #1, #2, #3, #4, #8 applied. v9 schema live, v9 graphs extracted, Run 9 launched 2026-06-06. Fixes #5, #6, #7 pending (post-training eval).
**Date:** 2026-06-06
**Context:** Run 8 plateaued at test F1=0.2307 (well below Run 7's 0.3423). Audit of the
data + graph schema surfaced 10 findings (A-J) that explain the regression. The fixes below
target the root causes before launching Run 9.

## Reading Order

1. **[PIPELINE.md](PIPELINE.md)** — exact execution steps for fresh re-extraction (start here)
2. **[TODO.md](TODO.md)** — actionable checklist with progress tracking
3. **[00-overview.md](00-overview.md)** — priority matrix, audit findings A-J, decision flow
4. **[01-relabel-timestamp.md](01-relabel-timestamp.md)** — drop 50% of Timestamp noise (already applied)
5. **[02-block-globals-extraction.md](02-block-globals-extraction.md)** — add `now` and library wrapper detection to feat[2]
6. **[03-external-call-entry.md](03-external-call-entry.md)** — new edge type 11 for external calls
7. **[04-integeruo-schema-gap.md](04-integeruo-schema-gap.md)** — add CFG_NODE_ARITH type 13 + `in_unchecked_block` feature
8. **[05-slither-derived-labels.md](05-slither-derived-labels.md)** — re-derive all 10 class labels from Slither
9. **[06-bonus-fixes.md](06-bonus-fixes.md)** — predictor tier-threshold, SmartBugs benchmark, complexity-bias docs

## Priority Matrix

| Fix | Effort | Impact | Risk | Blocking Run 9? |
|-----|--------|--------|------|-----------------|
| #1 Timestamp relabel | 1 hr (done) | High (49.9% noise removed) | None | No — already applied |
| #2 Block-globals extraction | 2 hr | High (27.5% of contracts fire feat[2]) | Low | Yes — before re-extract |
| #3 External CALL_ENTRY edge | 3 hr | Medium (Reentrancy class) | Low | Yes — before re-extract |
| #4 IntegerUO schema gap | 4 hr | High (IntegerUO + unchecked) | High | Yes — before re-extract |
| #5 Slither-derived labels | 1 day | Highest (all 10 classes) | Medium | Recommended |
| #6 Predictor tier threshold | 30 min | Display only | None | Optional |
| #7 SmartBugs benchmark | 1 hr | Eval methodology | None | Optional |
| #8 Complexity-bias doc | 0 hr | None | None | Documentation only |

**Execution path:** Apply #2 → #3 → #4 → re-extract → retokenize → build index → dedup → cache → train.
See **[PIPELINE.md](PIPELINE.md)** for exact commands and gates.

## Cross-Cutting Concerns

- **Fresh build:** All graphs, tokens, splits, and cache get rebuilt from scratch. No old artifacts survive.
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
