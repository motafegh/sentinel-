# Phase 8 — Existing Architecture Retraining

**Status:** READY — G7 SATISFIED  
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

