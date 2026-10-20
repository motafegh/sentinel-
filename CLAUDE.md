# SENTINEL AI Working Instructions

SENTINEL is Ali's learning-by-doing smart-contract security project. AI assistance may write substantial code, but project ownership, architecture reasoning, evidence quality, reproducibility, and educational value must remain explicit.

Project root: `~/projects/sentinel` in the primary WSL environment.

## Current project authority

Use this order when facts conflict:

1. **Executable source/config/tests** for actual behavior.
2. **Committed machine-readable governance/artifacts** for semantics and gate state — especially `docs/plan/ml-R4/` for current DATA/ML work.
3. **Canonical handbook** under `docs/handbook/` for current explanatory/navigation context.
4. ADRs/decision/registers for rationale and controlled decisions.
5. Supplementary guides/labs and dated historical reports for learning/history only.

Do not treat a stale comment, docstring, README, old plan, old test count, or historical artifact as stronger than current source/policy.

### Current stable technical baseline — reconciled 2026-10-20

- canonical `main` has passed historical R4 G0–G7 and remains the active Phase-8 line;
- Phase 8 is `IN_PROGRESS`; G8 is not passed;
- R4-D-009 is accepted logical V3 authority (`r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3`);
- R4-D-011 accepts the exact immutable V10/V2.6 graph/control root with digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- R4-D-013 accepts the exact immutable guarded-token successor `r4-v10-v26-guarded-tokens-v1` with digest `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- accepted token tensor contract remains `[4,512]` and architecture remains `four_eye_v8` / `v8.1` unless a later explicit decision changes it;
- R4-D-014 accepts `masked_positive_bce_control_v1` only as bounded positive-only control/evaluation authority; MODEL_SELECTION remains positive-fit diagnostic only;
- ML M0-M4 are complete;
- R4-D-015 records `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD` after the fixed 8-epoch / 120-step M4-C pilot;
- M4 checkpoints are bounded research evidence, not an operational replacement for Run12;
- R4-D-016 is accepted `CallToUnknown` semantic authority: class 0 is the raw/dynamic low-level unknown-target interaction class; `call`, `delegatecall`, legacy `callcode`, applicable `staticcall`, and `send` are in scope; `transfer`, ordinary typed high-level calls, in-scope library calls, contract creation, and selfdestruct/suicide destinations are not positive solely because their destination is external/unverified;
- R4-D-016 preserves the 10-class identity/order and taxonomy schema v1;
- M5/full training remains unauthorized and G8 remains open;
- confirmed negatives remain zero;
- candidate #1 historical `NOT_CONFIRMED` rationale is definition-stale under R4-D-016; its authoritative queue state remains `UNKNOWN / PENDING_REVIEW / target None`, and it requires a new primary review if reused;
- candidate #2 primary review is semantically compatible with R4-D-016 but authoritative state remains `UNKNOWN / PENDING_REVIEW / target None` until admissible independent agreement exists;
- three recent external candidate #2 review attempts are non-authoritative because of prior-candidate-context contamination and/or insufficient governing material;
- blind-review bundle v1 is historical only and retired for future adjudication;
- the current R4-GAP-007 action is to build/verify a self-contained candidate #2 v2 blind bundle and use a genuinely isolated reviewer with no candidate-specific prior context;
- threshold-fit, calibration-fit and untouched-acceptance roles remain unsupported/empty;
- accepted confirmed negatives, if any, are evaluation-only unless a later versioned decision grants optimizer authority;
- target `0` remains forbidden without complete class-specific confirmed-negative evidence plus admissible independent agreement;
- PU/nnPU, pseudo-negatives, threshold/calibration fitting, production promotion and Run12 learned-state reuse remain unauthorized shortcuts;
- Run12 remains the historical operational ML baseline;
- audit MCP is read-only;
- V3 signing/broadcast is outside the analysis MCP security domain;
- retained EZKL proof scope is proxy computation only; V3 context attestation is separate.

### Exact current DATA/ML restart boundary

Read in this order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
2. `docs/plan/ml-R4/adrs/ADR-R4-016-call-to-unknown-semantic-clarification.md` and `docs/plan/ml-R4/evidence/2026-10-20_call_to_unknown_semantic_alignment/decision.json`;
3. `docs/plan/system-finalization/technical-completion/02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`;
4. `docs/plan/ml-R4/runs/2026-10-17_PHASE8_m4_bounded_pilot_review_and_full_training_hold.md` and `docs/plan/ml-R4/adrs/ADR-R4-015-bounded-pilot-pass-full-training-hold.md`;
5. `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md`, `RISK_AND_BLOCKER_REGISTER.md`, `CLAIM_STATUS_MATRIX.md`, and `DECISION_REGISTER.md`;
6. R4-D-011/R4-D-013/R4-D-014 records when exact graph/token/objective authority is needed;
7. the accepted hardened V3 evidence snapshot when logical/evaluation population detail is needed.

The dated `working/2026-10-03_ml_m4_bounded_pilot.md` record contains historical in-progress stop lines from execution; do not use those as the current restart instruction after R4-D-015. The 2026-10-20 semantic-alignment working record is closed evidence under R4-D-016, not an active definition task.

The current next action is **build and verify `r4_gap007_candidate2_independent_review_v2.zip`, then obtain a genuinely isolated independent review**. Do not repeat accepted DATA work, repeat M4, use v1 for future adjudication, count contaminated reviews, or launch M5/full training.

## Approval model

Ali has delegated **routine technical and governance approvals** to the assistant. Do not stop merely to ask for approval of a normal plan, implementation slice, refactor, test, ADR, branch, PR, or evidence-preserving decision when the technically best option is clear.

Instead:

1. make the best evidence-based decision;
2. record important decisions in the appropriate plan/ADR/register;
3. execute and validate them;
4. surface consequential tradeoffs/results to Ali.

Still stop for genuinely non-routine decisions that cannot be inferred safely, such as destructive loss of protected data/history, real production deployment/funds/keys, irreversible external actions, or choices that change the project's intended product/career direction rather than implementation quality.

## Plan before code

For non-trivial work, write or update the appropriate plan/working record before implementation. The plan is an execution aid, not an approval ceremony.

Avoid excessive governance ceremony. Do not create new specs/registers/ADRs unless the decision actually needs durable control or later implementation could otherwise invent semantics.

## Working memory

In Ali's primary Claude Code setup, project memory may exist under:

`/home/motafeq/.claude/projects/-home-motafeq-projects-sentinel/memory/`

When available and relevant, read `MEMORY.md` plus only the referenced working memories needed for the task. Do not let private/local memory override current committed source or machine-readable governance.

For the **current Phase-8 DATA/ML boundary**, private/local memory must follow the restart order above, beginning with `PLAN_STATUS_MATRIX.md`, R4-D-016, the current ML completion plan, and R4-D-015. Older V2.5/V2.6 staging records, M4 in-progress working stop lines, the now-closed semantic-alignment task, and blind-review v1 are historical context only.

For **portfolio-professionalization work**, use `docs/plan/portfolio-professionalization/CURRENT_STATUS.md` as the live program status. The dated P0/P1 records are historical phase evidence and must not override that current status file.

The accepted pre-pilot V3 boundary remains `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`, governed by R4-D-009 / ADR-R4-009. Protected-local stage counts are evidence only when tied to the report hashes recorded in the current evidence chain/artifact index.

For long analysis/implementation sessions, preserve incremental findings in a working file rather than relying on conversation context alone. Promote durable conclusions into the repository only when they belong there.

## Professional engineering rules

### 1. Single responsibility

Prefer focused modules/functions with one clear reason to change. Split growing god-files instead of appending unrelated behavior because it is convenient.

Heuristics, not laws:

- functions around 50 lines deserve review when much larger;
- files around 200–400 lines are easier to own;
- >500 lines needs a reason;
- >1000 lines normally needs decomposition.

### 2. Decision numbers require evidence

Thresholds, weights, confidence cutoffs, acceptance gates, and similar numbers are policy. Externalize/version them, change them only with measured evidence, and bind them to the artifact/config that used them.

Tests prove code behavior. Evals/evidence justify decision quality. For R4, do **not** invent thresholds/calibration data to fill unsupported roles.

### 3. No silent failures or silent skips

A failure must be represented as one of:

1. precise eager error;
2. structured degraded result containing the reason/status;
3. explicit state/tool-status field consumed downstream.

Never make “tool did not run” indistinguishable from “tool ran and found nothing.” Do not silently convert missing dependencies/tools into clean evidence or contaminate metrics with unproven fallback values.

### 4. Preserve semantic uncertainty

Permanent R4 rule:

- unknown ≠ negative;
- unavailable ≠ clean;
- unsupported ≠ negative;
- weak evidence ≠ strong/metric-grade evidence;
- historical label ≠ current DATA vNext truth;
- on-chain record ≠ vulnerability ground truth;
- valid proxy proof ≠ proof of teacher/source/AGENTS execution.

Encode uncertainty/status explicitly rather than forcing a convenient downstream type.

### 5. Version rather than overwrite history

Historical v1 DATA, accepted repaired-v2 physical artifacts, superseded V2 grouping/roles, accepted V3 logical lineage, Run12 artifacts, V1/V2 registry history, and retained proxy artifacts are reproducibility/rollback roots.

For semantic/protocol changes: create a new versioned artifact/path, bind hashes/lineage, validate before promotion, and rollback by artifact selection rather than reverse-editing history.

### 6. Source-first review, docs-currentness second

When changing code:

1. inspect executable source/tests;
2. inspect current machine policy/ADR if semantics are governed;
3. check canonical docs for expected boundary/intent;
4. fix canonical documentation contradictions in the same change when practical.

Dated audits, experimental reports, `docs/learning/`, and handbook exercises may intentionally describe historical mechanics. Preserve them unless they falsely present themselves as current authority.

## Module-specific boundaries

### DATA / ML

- R4 controls current semantic authority.
- Historical Phase-6/v1 roles remain frozen and immutable.
- Repaired-v2/v9 remains historical physical/reproducibility evidence; v9 is not eligible for a new full training run under R4-D-010.
- Current accepted logical authority is R4-D-009 V3; arbitrary address literals may not create leakage-group edges.
- Explicit source-native family/project identifiers must be source-namespaced before grouping authority is applied.
- R4-D-011 exact V10/V2.6 graph/control bytes are immutable accepted physical authority.
- R4-D-013 exact guarded-token successor is immutable accepted token authority.
- no target `0` without complete confirmed-negative evidence plus admissible independent agreement;
- GasException/UnusedReturn supervision remains disabled under policy v1;
- Run12 threshold/calibration artifacts are historical only;
- architecture remains frozen absent separate governance;
- do not overwrite historical or accepted DATA/representation artifacts;
- R4-GAP-007 queue membership is review reservation only, not negative truth or optimizer supervision;
- R4-D-016 governs `CallToUnknown` as the low-level unknown-target class; do not silently broaden or narrow it;
- candidate #1 historical `NOT_CONFIRMED` rationale is definition-stale; queue state remains UNKNOWN/PENDING_REVIEW/target `None`, and a fresh primary review is required if reused;
- candidate #2 remains UNKNOWN/PENDING_REVIEW/target `None`; its primary review is compatible with R4-D-016 but it still needs genuinely isolated independent agreement;
- v1 blind bundle and contaminated/under-specified reviewer attempts are historical/diagnostic only and must not be counted as independent evidence;
- R4-GAP-008 is resolved only for the exact R4-D-011 root/digest;
- accepted confirmed negatives remain evaluation-only unless later policy grants optimizer authority;
- PU learning remains a future objective-design candidate, not current implementation authority;
- R4-D-014 is bounded positive-only objective/evaluation authority;
- R4-D-015 closes M4 but explicitly keeps M5/full training prohibited until later governance re-authorizes it.

### AGENTS

- gateway/LangGraph is off-chain;
- audit MCP is read-only;
- tool status must distinguish unavailable from clean;
- V3 feedback does not auto-promote while policy is unavailable.

### ZKML / Contracts

- retained proof scope is proxy-only;
- V3 context attestation is separate;
- no raw private-key runtime submission helper in analysis/ZKML MCP code;
- preserve UUPS storage and historical V1/V2 reads;
- signing/broadcast is a separate trust domain.

## Validation discipline

Use the narrowest meaningful test first, then expand according to blast radius. Preserve raw failures; do not weaken gates to turn failures green.

For Phase-8 repository DATA/logical-V3 work, the dedicated CI contract is:

`.github/workflows/r4-phase8-data-repair.yml`

It must compile repaired/V3 DATA and ML entry points, run repaired/V3 regressions including V3 snapshot-coherence tests, prove historical G6 still validates, and pass `git diff --check` from the audit handoff base. R4-D-016 adds focused `CallToUnknown` semantic-scope regressions and compiles the candidate #2 v2 blind-bundle builder in this gate.

For canonical documentation/current-state changes:

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
python3 docs/handbook/tools/verify_handbook.py inventory
```

For historical R4 G6 semantics:

```bash
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Later phase-specific gates are authoritative only after their phase/artifacts exist on the branch being validated.

## Learning lens

When helping Ali understand code, architecture, or decisions:

- avoid harmful oversimplification;
- name real technical terms and abbreviations;
- distinguish what must be learned now from what can be deferred;
- explain why an approach exists and what failure it prevents;
- connect code syntax to architecture/data flow when useful;
- preserve evidence versus inference distinctions.

AI-generated implementation is acceptable; unexamined implementation is not the goal.
