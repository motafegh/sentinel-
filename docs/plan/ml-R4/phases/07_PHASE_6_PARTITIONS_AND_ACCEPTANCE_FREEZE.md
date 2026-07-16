# Phase 6 — Dataset Roles, Leakage-Safe Partitions, and Acceptance Freeze

**Status:** WAITING FOR G5  
**Gate:** G6

## Objective

Create role-isolated partitions before DATA vNext training begins.

## Roles

- train strong;
- train weak;
- train unlabeled;
- model selection;
- threshold fit;
- calibration fit;
- internal audit;
- untouched acceptance;
- case study;
- excluded.

## Group constraints

Keep exact/near duplicates, project families, templates, injected pairs, compiler variants, and other defined leakage groups in one compatible role.

## Acceptance freeze

- finalize and hash the acceptance manifest;
- restrict routine access;
- record any prior exposure;
- never use acceptance to select checkpoints, hyperparameters, thresholds, or calibration.

## Support table

Per class and role report:

- confirmed positives;
- confirmed negatives;
- groups;
- sources;
- compiler eras;
- prevalence;
- evidence categories;
- limitations.

## G6 pass criteria

No incompatible role leakage; acceptance is frozen; unsupported roles/classes are explicit.
