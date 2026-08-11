# Phase 6 — Dataset Roles, Leakage-Safe Partitions, and Acceptance Freeze

**Status:** READY — G5 SATISFIED  
**Gate:** G6

## Objective

Create role-isolated partitions before DATA vNext training begins.

## Entry contract

Phase 5 has fixed the semantic policy. Phase 6 must partition according to `data-vnext-policy-v1`; it may not strengthen source authority or synthesize missing negatives.

Controlling constraints include:

- SolidiFI injected class and approved SmartBugs direct categories are strong-positive eligibility candidates;
- DIVE is unlabeled except weak TOD;
- DIVE weak TOD cannot be model-selection, threshold-fit, calibration-fit, or untouched-acceptance evidence;
- GasException and UnusedReturn supervision are disabled pending evidence;
- no blanket confirmed-negative source exists;
- evaluation-only candidate corpora require explicit exposure/leakage accounting before role assignment;
- historical v1 partitions are lineage evidence, not a vNext role assignment.

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
- never use acceptance to select checkpoints, hyperparameters, thresholds, or calibration;
- if trustworthy untouched acceptance support is unavailable, record that explicitly rather than relabeling exposed data as untouched.

## Support table

Per class and role report:

- confirmed positives;
- confirmed negatives;
- weak signals;
- unlabeled rows;
- groups;
- sources;
- compiler eras;
- prevalence where outcome evidence permits it;
- evidence categories;
- limitations.

## G6 pass criteria

No incompatible role leakage; acceptance is frozen or explicitly declared unsupported with a controlled empty/blocked manifest; unsupported roles/classes are explicit; no source authority exceeds Phase-5 policy.
