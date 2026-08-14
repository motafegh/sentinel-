# R4 Phase 8 — Pre-Training Launch Handoff

**Date:** 2026-08-14  
**Canonical branch:** `main`  
**Phase:** 8 — Existing Architecture Retraining  
**Gate:** G8  
**State at handoff:** implementation and launch preflight complete; full repaired training run has **not** been launched yet.

> **Post-handoff launch hold — 2026-08-14:** The completed live data audit found 65 compile-valid distinct positives removed by address equality, one valid contract blocked by a legacy-incompatible compiler flag, five recoverable Timestamp positives, at least 790 DIVE and seven SolidiFI normalized outputs corrupted after their compile gate, 341 graphs selecting a library/non-contract declaration, and 18,491 represented contracts omitting code tokens under the four-window cap. Only 275 strong + 577 weak cells are optimizer-active; 612 / 852 are over the token cap. The bounded GPU path passes, but that proves execution rather than evidence adequacy. **Do not launch the 100-epoch run on `sentinel-r4-vnext-v1`; repair and re-freeze a new DATA version first.** See `2026-08-14_PHASE8_real_data_readiness_audit.md`. G7 remains the valid binding result for the historical v1 artifact; this hold does not retroactively alter it.

## Purpose

This handoff was the durable restart point immediately before the expensive Phase-8 full retrain. The post-handoff real-data audit now supersedes its immediate-launch instruction: the runner remains the canonical execution path, but DATA remediation/re-freezing is the next boundary before the expensive run.

It does **not** replace executable source, machine-readable R4 policy/manifests, or `PLAN_STATUS_MATRIX.md`. Those remain higher authority. This file exists so a future session does not repeat the investigation that already established launch readiness or accidentally reinterpret old Run12 semantics.

## Canonical execution line

Phase-8 implementation was initially developed on `r4/phase8-existing-model-retraining`, then intentionally adopted onto canonical `main` on 2026-08-14. The old branch/worktree is provenance only; it has no higher authority than `main`.

Important lineage points:

- Phase-8 implementation/runtime-provenance baseline: `14ebc4e1aa58e7ed631e7d1456ba6d3aff134c51`;
- Phase-8 branch and `main` were verified identical before adoption;
- pre-memory-sync canonical `main` launch-readiness head: `a882d825c3bb8099f3aa6ad8072b27de59975b87`;
- this handoff and any other documentation-only synchronization commits necessarily move `main` again, so the **actual training source commit must be taken from `git rev-parse HEAD` after pulling the final synchronization commit and must be bound by the runner at launch**.

Do not hard-code `a882d825...` as the future training source SHA. Re-run the short preflight after the final documentation sync.

## DATA vNext authority entering Phase 8

The repaired training input authority is:

- dataset/export: `sentinel-r4-vnext-v1`;
- export schema: `v2`;
- graph schema: `v9`;
- Phase-6 role authority: `r4-vnext-roles-v1`;
- representation binding digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`;
- represented contracts checked: 21,657 / 21,657;
- representation files checked: 64,971 / 64,971;
- missing files: 0;
- representation mismatches: 0;
- positive targets: 1,007;
- confirmed-negative targets: 0.

The permanent semantic boundary remains:

- unknown / unsupported / historical absence is not a negative;
- no target `0` is authorized without confirmed-negative evidence;
- `GasException` and `UnusedReturn` remain supervision-disabled under policy v1;
- threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty for the first repaired baseline;
- Run12 labels, thresholds, calibration, optimizer state, scheduler state, and learned weights are historical and are not reused as repaired-vNext truth.

## Frozen Phase-8 optimizer population

Frozen role population:

- `TRAIN_STRONG`: 275 contracts;
- `TRAIN_WEAK`: 773 contracts;
- total frozen training population: 1,048 contracts.

Actual optimizer-bearing population:

- active strong: 275;
- active weak: 577;
- optimizer-bearing contracts: 852;
- weak no-signal siblings excluded from loss: 196;
- active/frozen training leakage groups: 703.

MODEL_SELECTION:

- 56 contracts;
- 51 leakage groups;
- positive-only limited support.

Training sampling is one deterministically rotating member per frozen leakage group per epoch. Related siblings therefore do not receive multiplicative influence merely because a group contains more files.

## Frozen model and optimization settings

Architecture remains deliberately unchanged for this repaired baseline:

- architecture: `four_eye_v8`;
- model version: `v8.1`;
- ten output classes remain in the locked class order;
- new run starts from the accepted pretrained GraphCodeBERT base plus fresh/current Phase-8 trainable components; it does **not** load Run12 learned weights.

Frozen settings:

- random seed: `20260813`;
- epochs: 100;
- batch size: 8;
- gradient accumulation: 8;
- base learning rate: `2e-4`;
- weight decay: `1e-2`;
- warmup percentage: `0.10`;
- GNN LR multiplier: `2.5`;
- LoRA LR multiplier: `0.3`;
- fusion LR multiplier: `0.5`;
- prefix LR multiplier: `5.0`;
- weak-positive loss weight: `0.25`;
- auxiliary loss weight: `0.3` with 8-epoch warmup;
- phase-2 loss weight: `0.2`;
- JK entropy regularization: `0.005`;
- gradient clip: `1.0`;
- fixed diagnostic threshold: `0.5` only; no threshold tuning.

With 703 sampled groups per epoch:

- micro-batches per epoch: 88;
- optimizer/scheduler steps per epoch: 11;
- total optimizer steps for the fixed 100-epoch horizon: 1,100.

OneCycleLR is retained only as a valid historical optimization mechanic; its horizon is derived from the actual grouped sampler rather than old contract-row counts.

## Model-selection and checkpoint semantics

Historical Run12 F1/threshold-based checkpoint selection is not valid under repaired partial-label semantics.

Phase 8 therefore:

- runs the predetermined 100-epoch horizon unless a runtime safety failure aborts it;
- does not early-stop on MODEL_SELECTION;
- uses positive-only NLL as a **limited positive-fit diagnostic**;
- may retain `best_positive_nll`, but that checkpoint must not be called generally best, calibrated, acceptance-ready, or false-positive validated;
- treats the fixed-horizon `final` checkpoint as the primary G8 completion artifact;
- preserves `latest` every completed epoch and milestone full checkpoints every 10 epochs;
- preserves deterministic JSONL epoch metrics and raw MODEL_SELECTION probabilities;
- never uses threshold/calibration/acceptance data in Phase 8.

## Reproducibility/runtime provenance proven before launch

The accepted pretrained backbone identity is:

- model: `microsoft/graphcodebert-base`;
- accepted Hugging Face snapshot: `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`.

The local launch runtime preflight at pre-memory-sync head `a882d825...` reported:

- Python: CPython `3.12.3`;
- PyTorch: `2.5.1+cu124`;
- PyTorch compiled CUDA: `12.4`;
- cuDNN: `90100`;
- Transformers: `4.46.3`;
- PEFT: `0.15.2`;
- PyTorch Geometric: `2.7.0`;
- NumPy: `1.26.4`;
- Pandas: `2.3.3`;
- PyArrow: `19.0.1`.

`runtime_binding_metadata()` fails closed if the mutable local GraphCodeBERT name does not resolve to the accepted snapshot and records the effective runtime package provenance. The generic TransformerEncoder loader itself is not revision-pinned; the Phase-8 fail-close provenance check plus offline launch mode is the current accepted control for this run.

## Evidence already completed

Do not repeat the basic GPU micro-smoke unless model/data/training mechanics materially change.

The earlier end-to-end GPU micro-smoke passed through:

`DATA vNext -> loader -> grouped sampler -> frozen model -> masked losses -> backprop -> optimizer -> positive-only MODEL_SELECTION`

without loading Run12 learned weights.

Observed micro-smoke evidence included:

- `PHASE8_END_TO_END_MICRO_SMOKE_PASS`;
- BF16 CUDA execution;
- 2 training batches / 2 optimizer steps;
- CUDA peak about 970 MiB for the tiny smoke configuration;
- finite training losses and positive-only selection NLL/probabilities.

This proves path compatibility, not full-run quality or full-run memory sufficiency.

## CI / governance state before launch

Before this handoff:

- Phase-8 implementation was adopted onto `main`;
- `PLAN_STATUS_MATRIX.md` records Phase 8 as `IN_PROGRESS` on canonical `main` and G8 still open;
- the dedicated `R4 Phase 8 vNext training compatibility` workflow was changed to run on pushes to `main` and passed on the main lineage;
- the canonical Handbook validator was updated from the old `Phase 8 READY` assumption to the current `Phase 8 IN_PROGRESS` state and its final run passed;
- tracked launch worktree preflight was clean;
- runtime/backbone provenance preflight passed.

## Local worktree facts

Canonical worktree:

`~/projects/sentinel` on `main`.

The user's main worktree contains several pre-existing **untracked** audit/plan files. Phase-8 `git_source_commit()` deliberately checks tracked modifications with `--untracked-files=no`, so those untracked files do not contaminate the source binding and must not be deleted merely to launch training.

Old Phase-8 worktree:

`~/projects/sentinel-r4-phase8` at historical/detached Phase-8 provenance. Do not use it as the canonical launch worktree now that `main` is the chosen execution line.

## Exact next action after this synchronization commit is pulled

First re-run the minimal launch preflight because documentation synchronization moves the exact source SHA:

```bash
cd ~/projects/sentinel

git pull --ff-only origin main

echo "=== HEAD ==="
git rev-parse HEAD

echo "=== TRACKED STATUS ==="
git status --short --untracked-files=no

echo "=== RUNTIME BINDING ==="
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
PYTHONPATH=.:data_module \
./ml/.venv/bin/python - <<'PY'
import json
from ml.src.training.vnext_binding import runtime_binding_metadata
print(json.dumps(runtime_binding_metadata(), indent=2, sort_keys=True))
PY
```

Required outcome:

- current `main` HEAD printed and later bound by the runner;
- no tracked-status output;
- GraphCodeBERT resolves to `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`;
- effective runtime metadata is populated and materially unchanged from the accepted preflight unless separately reviewed.

Then launch the real full run:

```bash
cd ~/projects/sentinel
export SENTINEL_REPRESENTATIONS_ROOT="$HOME/projects/sentinel/data_module/data/representations"
docs/plan/ml-R4/scripts/p8_run_training.sh
```

The first output through the beginning/completion of epoch 1 should be reviewed before treating the long run as healthy.

## During-run controls

Once the full run is launched from its bound source commit:

- do not `git pull`, checkout another commit, or modify tracked files in the active training worktree;
- do not change packages in `ml/.venv`;
- do not delete/edit the generated `ml/logs/r4-phase8/` run directory;
- on an error, inspect the exact terminal output, run manifest, and checkpoint state before deciding whether to restart;
- after a completed epoch, same-run recovery must use that run's `checkpoints/latest.pt` with the same source/runtime binding;
- do not change frozen batch size, architecture, label semantics, thresholds, or other Phase-8 policy merely because a runtime result is inconvenient.

Canonical resume form after a completed epoch:

```bash
docs/plan/ml-R4/scripts/p8_run_training.sh \
  --resume ml/logs/r4-phase8/run-<binding>/checkpoints/latest.pt
```

## What remains unknown until the expensive run executes

Launch readiness does **not** prove model quality. The full run must reveal, rather than assume:

- whether positive-only supervision produces useful class separation or broad overprediction;
- whether the fixed training horizon is numerically stable;
- whether full-size batches fit the RTX 3070 Laptop 8 GiB runtime throughout real training;
- whether MODEL_SELECTION positive NLL improves meaningfully;
- what checkpoint behavior emerges under the repaired semantics.

These are Phase-8 results. They must not be pre-filled with assumptions or repaired by inventing negative labels.

## Handoff sentence

**SENTINEL R4 is at Phase 8 on canonical `main`: G0–G7 are passed; DATA vNext and representation binding are frozen; the existing Four-Eye v8.1 retraining path, durable checkpoints/resume, runtime provenance, main-branch CI, and launch preflight are validated; the real 100-epoch / 1,100-optimizer-step repaired retrain has not yet been launched, and the next action is to pull this synchronization state, re-run the minimal source/runtime preflight, then start `p8_run_training.sh` and inspect startup/epoch-1 output.**
