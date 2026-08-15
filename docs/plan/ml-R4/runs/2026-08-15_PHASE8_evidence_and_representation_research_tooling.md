# Phase 8 Evidence and Representation Research Tooling

**Date:** 2026-08-15
**Scope:** repository-side tooling after R4-D-008 repaired-v2 physical DATA acceptance
**DATA status:** accepted repaired-v2 physical lineage remains immutable
**Training status:** 100-epoch Phase-8 run remains NOT AUTHORIZED
**G8:** OPEN

## Purpose

R4-D-008 established that the repaired-v2 physical DATA lineage is usable for
bounded research, but not that the current model can discriminate positives from
negatives. The next work therefore separates four evidence questions instead of
jumping directly to a new loss or a long training run:

1. can class-specific confirmed-negative evaluation evidence be created without
   turning unlabeled cells into fake negatives;
2. does source-scoped shared-address grouping create overly broad leakage
   families;
3. can target-aware token selection improve target-code coverage without
   silently changing the accepted representation lineage;
4. how sensitive bounded model behavior and GPU feasibility are to compatibility
   graphs, multi-component file-union graphs, and worst-case active samples.

## Implemented bounded-research surfaces

### 1. Confirmed-negative evaluation pilot

Repository code:

- `data_module/sentinel_data/vnext/confirmed_negative_evaluation.py`
- `docs/plan/ml-R4/specs/confirmed_negative_evaluation_v1.json`
- `docs/plan/ml-R4/specs/confirmed_negative_adjudication_template_v1.json`
- `docs/plan/ml-R4/scripts/p8_build_confirmed_negative_review_queue.py`
- `docs/plan/ml-R4/scripts/p8_validate_confirmed_negative_adjudications.py`

The queue is deterministic and starts only from currently `TRAIN_UNLABELED`
groups. Queue membership means **review candidate**, not negative truth.

A `CONFIRMED_NEGATIVE` cell is accepted only when:

- the claim is class-specific;
- complete relevant code scope and every file-graph component were reviewed;
- contradictory positive evidence was not found;
- at least one direct class-specific evidence type is recorded;
- a second, distinct reviewer independently agrees and records evidence.

Accepted cells are `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY`. They do not mutate
`sentinel-r4-vnext-v2`, do not authorize target `0` for optimizer loss, and do
not authorize threshold/calibration fitting.

The default queue is 25 candidates per enabled class. This is intentionally a
pilot for adjudication yield/review cost. A simple zero-false-positive binomial
planning bound for false-positive rate below 5% at 95% confidence is 59 actually
confirmed negative examples per class if zero false positives are observed.
That bound is planning evidence only; group dependence, selection bias,
multiple classes, threshold selection, and nonzero false-positive counts require
additional design.

### 2. Leakage-group breadth audit

Repository code:

- `data_module/sentinel_data/preprocessing/r4_grouping_audit.py`
- `docs/plan/ml-R4/scripts/p8_audit_grouping_breadth.py`

The profiler measures group-size distributions, evidence-edge reasons,
source-scoped address frequency, address-only groups, large address-connected
groups, and multi-address transitive components. Diagnostic thresholds request
review but do not automatically declare repaired-v2 grouping defective.

Any grouping-policy change must receive a new grouping/partition version; the
accepted `r4-leakage-groups-v2` evidence is not edited in place.

### 3. Guarded target-aware bounded selector

Repository code:

- `ml/src/data_extraction/bounded_window_selector.py`
- `data_module/sentinel_data/representation/r4_target_spans.py`
- `docs/plan/ml-R4/scripts/p8_compare_bounded_window_selector_v1.py`
- `docs/plan/ml-R4/scripts/p8_run_selector_gpu_compare.py`

Three research strategies are explicit:

- `historical_linspace_v1` — exact production control;
- `target_aware_greedy_v1` — greedily maximizes requested target-declaration
  token coverage;
- `target_aware_guarded_v1` — uses the greedy choice only when requested
  target-declaration coverage is strictly better than control; equal or worse
  coverage falls back to the historical control.

The accepted `[4,512]` token artifacts are not rewritten. Repaired-v2
preprocessing already removed comments before the representation stage, and the
research path preserves that tokenizer-input contract. The CUDA comparison
requires the dynamically regenerated historical-control tensors to equal the
already bound token tensors for every accessed control sample.

The target-coverage metric is deliberately narrower than full semantic coverage:
it measures the requested file-graph target declaration spans. Inherited base
code, library code, and other semantic dependencies outside those spans are not
claimed as covered merely because the target-declaration ratio is high. That is
one reason the bounded GPU/model comparison remains mandatory before promotion.

The CUDA launcher also:

- uses the same repaired-v2 active train/model-selection populations;
- copies one exact initial model state into both strategy runs;
- verifies the initial-state digest for both runs;
- uses the same seed, deterministic group sampler, optimizer construction,
  batch limits, and positive-only loss;
- writes no checkpoint and loads no Run12 weights;
- can forward-probe active worst-case samples from the sensitivity profile.

This is selector evidence only; no selector promotion is implied.

### 4. Representation sensitivity profile

Repository code:

- `data_module/sentinel_data/representation/r4_sensitivity.py`
- `docs/plan/ml-R4/scripts/p8_profile_representation_sensitivity.py`

It produces exact comparison sets for:

- optimizer-active compatibility-mode contracts;
- model-selection compatibility-mode contracts;
- optimizer-active file-union contracts;
- model-selection file-union contracts;
- worst-case active GPU candidates interleaved across node-count, edge-count,
  component-count, and token-window extremes.

Current repaired-v2 physical binding already permits one narrow legacy
provenance inference: sidecars byte-reused from the successful portion of a
failed-tail recovery build may omit `graph_extraction_mode`, in which case the
binder records them as inferred standard `slither_full_analysis`. The research
profiler now follows that same accepted binding rule and reports the inferred
count explicitly. Sidecars that record a source transform cannot use that
inference, and newly generated/recovered compatibility artifacts still require
explicit mode provenance.

## Local execution order

Use the canonical `main` worktree and the same accepted ML virtual environment
that produced the repaired-v2 CUDA evidence. Do not substitute the system
Python or upgrade packages for this experiment.

```bash
cd ~/projects/sentinel

git pull --ff-only origin main

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=".:data_module"

./ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_audit_grouping_breadth.py

./ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_profile_representation_sensitivity.py

./ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_compare_bounded_window_selector_v1.py

./ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_build_confirmed_negative_review_queue.py

./ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_run_selector_gpu_compare.py
```

The expected local/git-ignored outputs are:

- `data_module/data/r4-v2-build/grouping_breadth_audit_v1.json`
- `data_module/data/r4-v2-build/representation_sensitivity_v1.json`
- `data_module/data/r4-v2-build/bounded_window_selector_v1.json`
- `data_module/data/r4-v2-build/confirmed_negative_review_queue_v1.json`
- `data_module/data/r4-v2-build/selector_gpu_compare_v1.json`

Only after real review/adjudication records exist should this be run:

```bash
./ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_validate_confirmed_negative_adjudications.py
```

That validator expects:
`data_module/data/r4-v2-build/confirmed_negative_adjudications_v1.jsonl`.

## Review boundary

Review the five evidence surfaces together before changing any objective or
representation lineage:

1. grouping breadth and the largest address-connected families;
2. exact compatibility/file-union/worst-case active sets;
3. control vs guarded requested-target declaration coverage;
4. identical-initialization bounded CUDA behavior and memory;
5. confirmed-negative pilot yield and reviewer disagreement/exclusion rate.

Only then decide whether the next version should use ordinary supervised
learning with newly acquired negatives, a Positive-Unlabeled (PU) objective, or
another evidence-honest formulation.

## Stop lines

This tranche does **not** authorize:

- a 100-epoch training run;
- a PU objective;
- promotion of the guarded selector;
- mutation of accepted repaired-v2 physical artifacts;
- changing graph schema v9 or the four-eye architecture;
- treating `TRAIN_UNLABELED` as safe/negative;
- using confirmed-negative evaluation cells as optimizer targets;
- threshold fitting, calibration fitting, or untouched-acceptance claims.

Those require later explicit decisions after the local evidence is reviewed.
