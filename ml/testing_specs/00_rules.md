# 00 — Universal Validation Rules

These rules apply to **every procedure in `ml/testing_specs/`**.

The suite contains useful mechanics from multiple project eras, especially Run12. It is not a stronger source of current DATA/ML truth than executable source, R4 policy/roles, or canonical current status.

## Rule 0 — Resolve current authority before running a procedure

For current work, read facts in this order:

1. executable source/config/tests;
2. current machine-readable R4 policy/manifests;
3. canonical handbook/current status;
4. active R4 ADR/decision/risk records;
5. this validation procedure;
6. dated historical findings/memory for historical context only.

For DATA-vNext/retraining work, mandatory current inputs are:

- `docs/plan/ml-R4/specs/data_vnext_policy_v1.json`;
- `docs/plan/ml-R4/manifests/p6_partition_manifest.json`;
- `docs/plan/ml-R4/manifests/p6_role_support_table.json`;
- `docs/plan/ml-R4/manifests/p6_untouched_acceptance_manifest.json`;
- `docs/handbook/16_current_status.md`.

If an individual spec conflicts with those artifacts, **the spec is historical at that point**. Update/adapt it rather than weakening R4.

## Rule 1 — Never recreate the historical label defect

Permanent R4 invariants:

- historical zero ≠ confirmed negative;
- unknown/unsupported/absent/dropped/tool-silent ≠ negative;
- masked/disabled cells must not be filled with zero;
- weak evidence ≠ strong or metric-grade evidence;
- GasException and UnusedReturn remain supervision-disabled under policy v1;
- DIVE Front Running→TOD is weak-positive only;
- no blanket confirmed-negative source exists in policy v1.

Any validation procedure that computes negatives or full binary metrics must first prove that its selected role actually contains authorized confirmed-negative evidence.

## Rule 2 — Dataset roles are evidence boundaries

Current first-baseline role authority:

- `TRAIN_STRONG` — optimization allowed for strong targets;
- `TRAIN_WEAK` — optimization allowed only with explicit weak handling;
- `TRAIN_UNLABELED` — unlabeled use only;
- `MODEL_SELECTION` — positive-only limited diagnostics;
- `INTERNAL_AUDIT` — internal/exposed audit evidence;
- `THRESHOLD_FIT` — `UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT` — `UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE` — `UNSUPPORTED_EMPTY_FROZEN`.

Do not borrow one role to fill another because a legacy script expects data.

In particular:

- positive-only model selection does not support trusted F1/AUC/FPR claims;
- a threshold script running successfully does not authorize threshold fitting;
- a calibration script running successfully does not authorize calibration;
- exposed historical/manual/quickstart data cannot be renamed untouched acceptance.

## Rule 3 — Run12 is historical operational baseline

Run12 weights, thresholds, calibration, old splits, and old release/staging state are historical evidence.

A future repaired checkpoint must receive new lineage bound to:

- exact DATA vNext artifact/policy/roles;
- training config/seed/initialization;
- numeric strong/weak optimization behavior;
- checkpoint SHA;
- checkpoint-selection evidence and limitations;
- any later separately authorized threshold/calibration evidence.

Do not automatically reuse Run12 decision artifacts because output dimensions match.

## Rule 4 — Read before claiming

Never assert a value/state/result from memory when the source exists.

For every important claim, identify its source:

- schema constants → canonical source;
- DATA semantics/roles → R4 machine artifacts;
- checkpoint/runtime path → current config/source/artifact lineage;
- current phase/gate → R4 status matrix/current status;
- metric result → exact run/report/artifact;
- protocol behavior → executable source/tests.

If evidence is absent or ambiguous, record `UNVERIFIED`/unsupported rather than estimating or reconstructing a convenient answer.

## Rule 5 — Validate the validation

Every decision-driving procedure needs three layers:

### Layer 1 — explicit result

Write PASS / FAIL / UNVERIFIED (or a domain-specific explicit state) with exact artifact/commit/config/role identity.

### Layer 2 — independent cross-check

Cross-check decision-driving counts/metrics/hashes using independent sources where possible. Disagreement is a finding, not something to average away.

### Layer 3 — completion record

Persist the procedure result, skipped steps, limitations, new findings, and output artifact identities before closing the task.

A skipped step without reason is a gap.

## Rule 6 — No floating findings

Consequential findings/decisions must move from conversation into the appropriate durable place:

- source bug → issue/finding/test;
- DATA/ML semantic decision → R4 ADR/decision/register;
- run result → versioned run/evaluation report;
- unsupported evidence → risk/blocker/status;
- current-state change → canonical current status.

Do not create new governance artifacts merely for ceremony; use the existing controlling record where one exists.

## Rule 7 — Procedures are not knowledge

A procedure describes **how to check**. It does not permanently define the answer.

Avoid hardcoding transient run names, thresholds, checkpoint paths, or current metric totals into reusable specs unless the file is explicitly versioned as historical for that run.

Current answers belong in source/config/artifact/run reports/current status, not in generic procedure rules.

## Rule 8 — Separate implementation checks from evidence authority

A test can prove that code behaves as written. It cannot by itself prove that:

- labels are ground truth;
- a threshold/calibration corpus is valid;
- a checkpoint is promotion-ready;
- an on-chain record is a vulnerability label;
- a ZK proof covers upstream teacher/source/AGENTS execution.

Always state whether an output is an implementation test, data/evidence validation, model-quality evaluation, protocol proof, or release/promotion decision.

## Rule 9 — Current phase sequencing

Stable `main` is through R4 G6. Phase 7 must finish local representation binding/G7 before Phase 8 retraining.

Do not use this spec suite to bypass an R4 gate because the next-stage code/script already exists.

## Before using any spec

1. Read this file.
2. Read current status and the relevant R4 machine artifacts.
3. Identify which assumptions in the selected spec are historical versus still valid.
4. Bind exact commit/DATA/checkpoint/config/role identities.
5. Run the procedure without weakening failure/evidence gates.
6. Persist PASS/FAIL/UNVERIFIED plus limitations and artifact hashes.

If a procedure cannot be made compatible with current R4 evidence without inventing data or authority, stop and record that limitation instead of forcing it to pass.
