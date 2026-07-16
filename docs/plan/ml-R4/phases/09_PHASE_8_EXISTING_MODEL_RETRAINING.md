# Phase 8 — Existing Architecture Retraining

**Status:** WAITING FOR G7  
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
