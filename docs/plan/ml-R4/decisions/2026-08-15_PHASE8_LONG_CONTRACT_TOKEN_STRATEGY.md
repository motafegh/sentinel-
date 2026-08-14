# Phase-8 Decision — Long-Contract Token Strategy

**Date:** 2026-08-15
**Status:** DEFERRED DECISION — EVIDENCE COLLECTION IMPLEMENTED
**Scope:** repaired Phase-8 GraphCodeBERT token representations

## Decision

Do **not** change the frozen `[4, 512]` token tensor contract or the `four_eye_v8` / `v8.1` architecture during the repository-only real-DATA repair.

The repaired extractor keeps the historical deterministic four-window linspace selection so DATA/provenance repairs are not confounded with an architecture/input-capacity redesign. It now records exact pre-subsampling token/window counts, selected window indices/ranges, retained unique-token count, and retained-token ratio. These fields are evidence only; no retained-ratio threshold is approved.

The 2026-08-14 real-data audit found 18,491 / 21,657 represented contracts exceed four windows and 612 / 852 optimizer cells attach to over-cap inputs. Therefore shape validity is explicitly **not** evidence that four windows are adequate.

## Alternatives to test locally

1. **Historical bounded linspace — control**
   - exactly four windows distributed across the full source;
   - no architecture change;
   - current repaired implementation.
2. **Target-contract-aware bounded windows**
   - resolve the application contract first;
   - prioritize windows overlapping that declaration/body while retaining bounded global context;
   - still `[4, 512]`, so potentially backward-compatible if evidence supports it.
3. **Evidence/site-aware bounded windows**
   - where source-native evidence provides reliable line/site locations, prioritize windows covering those sites;
   - cannot be universal because many current source claims have class-level rather than site-level provenance;
   - must never use model-selection/acceptance labels to choose windows adaptively.
4. **Hierarchical / more-than-four-window encoder**
   - encode more windows and aggregate them before fusion;
   - changes model input/architecture/compute and invalidates the initial architecture-freeze experiment;
   - requires a separate architecture version and is not authorized in this repair tranche.

## Exact local experiment required before changing selection

After the repaired physical corpus and representation target binding are available:

1. Preserve the repaired four-window representations as the control lineage.
2. On the **same final leakage groups**, select a deterministic analysis sample containing:
   - every optimizer/model-selection contract with `pre_subsampling_window_count > 4`, subject to practical runtime limits;
   - a deterministic stratified sample of <=4-window contracts as a control;
   - no untouched-acceptance data, because that role is unsupported.
3. For each candidate bounded strategy, record per contract:
   - pre-subsampling code tokens/windows;
   - selected token ranges;
   - retained unique-token ratio;
   - target-contract body token coverage where declaration boundaries can be resolved;
   - source-evidence site coverage only where site provenance exists;
   - tokenizer/runtime/version identity.
4. Run a **bounded diagnostic GPU experiment**, not the 100-epoch run, using identical model initialization/seed/optimizer settings and a fixed small step budget for each strategy.
5. Compare only currently authorized evidence:
   - positive-only training/model-selection NLL and positive probability diagnostics;
   - stability/non-finite behavior;
   - representation/coverage metrics above.
   Do not use false-positive/F1/AUC/calibration claims because confirmed-negative support remains absent.
6. Adopt a different bounded selector only if it materially improves target/evidence coverage without introducing provenance leakage or unstable optimization. Record the selector as a new extractor version and repeat physical binding + bounded GPU smoke.
7. If bounded four-window strategies remain inadequate, open a separate architecture decision for hierarchical/more-window encoding rather than silently changing `four_eye_v8`.

The repository-provided read-only comparison command is:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_compare_bounded_window_strategies.py \
  --output data_module/data/r4-v2-build/bounded_window_experiment.json
```

This command does not rewrite representations and cannot promote a selector by itself.

## Current implementation contract

- token shape: `[4, 512]` — frozen;
- selection: historical deterministic linspace — retained as control;
- coverage schema: `r4-token-coverage-v1`;
- repaired representation extractor: `v2.2-r4-repaired`;
- coverage interpretation: `diagnostic_only_no_adequacy_threshold`;
- 100-epoch training: **not authorized** until physical repaired-DATA acceptance and the bounded repaired-data GPU smoke are reviewed.
