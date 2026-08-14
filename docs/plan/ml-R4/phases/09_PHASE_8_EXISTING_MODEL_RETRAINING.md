# Phase 8 — Existing Architecture Retraining

**Status:** IN_PROGRESS — G7 SATISFIED  
**Gate:** G8

## Objective

Retrain the existing architecture on DATA vNext.

## Allowed code changes

Only changes required for:

- export-schema compatibility;
- class masks;
- weak/strong role handling;
- grouped samplers;
- correct metric masking;
- artifact binding;
- deterministic logging;
- raw probability output.

## Training sequence

1. reproduce the historical bundle where feasible;
2. create a DATA vNext compatibility test;
3. train the unchanged architecture;
4. use only training/model-selection roles;
5. bind checkpoint to code, export, split, seed, and config;
6. preserve logs and intermediate checkpoints;
7. analyze failures without touching acceptance.

## Durable full-run control — 2026-08-14

The Phase-8 compatibility path and committed GPU micro-smoke have established that DATA vNext can execute end-to-end through the frozen model, masked optimizer path, and positive-only MODEL_SELECTION path. The remaining Phase-8 work is therefore run control and reproducibility, not another training-path redesign.

Historical Run12 mechanics are retained only where their meaning survives the repaired label semantics:

- retain AdamW parameter-group learning-rate multipliers and the historical OneCycleLR/cosine schedule shape;
- compute scheduler progress from the actual grouped Phase-8 sampler, not from all optimizer-bearing contract rows;
- restore model, optimizer, scheduler, epoch/global-step, and Python/NumPy/Torch/CUDA RNG state on resume;
- write checkpoints atomically and fail closed when checkpoint/run binding differs;
- preserve deterministic epoch metrics and raw MODEL_SELECTION probability records.

With the frozen Phase-8 baseline, the grouped sampler emits one rotating contract from each of 703 training groups per epoch. At batch size 8 and gradient accumulation 8 this is 88 micro-batches and 11 optimizer/scheduler steps per full epoch, for 1,100 planned optimizer steps over 100 epochs. These counts are derived at runtime and are included in the run binding; a population/config change must not silently reuse the old schedule horizon.

The historical Run12 model-selection policy does **not** survive R4. Phase 8 therefore:

- does not use historical binary-label F1, AUC, Brier, ECE, tuned thresholds, or historical calibration for checkpoint selection;
- does not early-stop on MODEL_SELECTION because the role contains positive-only limited support and cannot measure false-positive discrimination;
- runs the predetermined training horizon unless a runtime safety failure aborts the run;
- preserves a `best_positive_nll` companion checkpoint using minimum MODEL_SELECTION positive NLL as a **limited positive-fit diagnostic only**;
- treats the fixed-horizon `final` checkpoint as the primary G8 completion artifact; `best_positive_nll` must not be described as a generally best or promotion-ready model.

Checkpoint/run identity must include the exact source commit, G7 DATA manifest and representation binding, Phase-6 role/partition identity, seed, weak-positive weight, optimizer/scheduler configuration, mixed-precision mode, sampler population, and planned optimizer-step horizon. Resume is permitted only when this binding matches exactly.

Durable runtime outputs are local run artifacts rather than DATA truth: an immutable run binding inside the manifest/checkpoints, deterministic JSONL epoch metrics, deterministic raw MODEL_SELECTION records, an atomic latest checkpoint, periodic intermediate recovery checkpoints, the limited `best_positive_nll` checkpoint, and the final checkpoint. Threshold fitting, calibration fitting, acceptance access, and pseudo-negative construction remain unavailable.

## Prohibited

- architecture search;
- acceptance-set inspection;
- threshold tuning on model-selection metrics;
- silent fallback to historical zeros;
- selecting a checkpoint using historical corrupted-target scores alone.

## G8 pass criteria

A reproducible current-architecture checkpoint is bound to DATA vNext and developed without acceptance leakage.

## Phase-7 handoff

Training input authority is `sentinel-r4-vnext-v1` with local representation binding digest `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`. Phase 8 must preserve class order/schema, Phase-6 roles, nullable target semantics, STRONG/WEAK distinction, disabled-class masking, and the unsupported threshold/calibration/acceptance boundaries.
