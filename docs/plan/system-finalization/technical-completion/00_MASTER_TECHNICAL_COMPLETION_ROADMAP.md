# SENTINEL Technical Completion Roadmap

**Status:** ACTIVE — cross-module execution controller  
**Created:** 2026-09-14  
**Current-state reconciliation:** 2026-10-20  
**Scope:** remaining technical completion, capability hardening, model evidence/promotion, and downstream lineage reconciliation  

## 1. Purpose

This roadmap coordinates the remaining engineering work across SENTINEL. Executable source/config/tests remain the highest authority for actual behavior, while R4 machine-readable governance, registers and accepted ADRs remain authoritative for DATA/ML semantics and gate state.

This file does **not** replace `docs/plan/ml-R4/00_MASTER_PLAN.md`, the R4 decision register, accepted ADRs, or module-specific plans. It defines cross-module sequencing and the conditions under which work may move from one subsystem to another.

Subordinate plans:

1. `01_DATA_REPRESENTATION_COMPLETION_PLAN.md`
2. `02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`
3. `03_INFERENCE_DEPLOYMENT_PROMOTION_PLAN.md`
4. `04_AGENTS_ORCHESTRATION_CAPABILITY_COMPLETION_PLAN.md`
5. `05_ZKML_LINEAGE_PROOF_COMPLETION_PLAN.md`
6. `06_CONTRACTS_ONCHAIN_COMPLETION_PLAN.md`

## 2. Current technical baseline

### DATA / representations — CLOSED FOR CURRENT ML INPUT

- historical G0–G7 remain passed and immutable;
- logical V3 grouping/roles are accepted under R4-D-009;
- exact V10/V2.6 graph/control lineage is accepted under R4-D-011 with 22,540 identities and binding digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- R4-D-012 promotes `target_aware_guarded_v1` for the successor token lineage;
- R4-D-013 accepts the exact 22,540-identity guarded-token successor `r4-v10-v26-guarded-tokens-v1` with digest `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- accepted token tensor shape remains `[4,512]`;
- R4-D-011 and R4-D-013 roots are immutable; no current DATA construction task blocks ML.

### ML — M0-M4 COMPLETE, M5 HELD

- architecture remains frozen at `four_eye_v8` / `v8.1`;
- R4-D-014 accepts `masked_positive_bce_control_v1` only as bounded positive-only control/evaluation authority;
- exact guarded-lineage M3 integration/no-step proof is complete;
- M4-A/B/C mechanics, checkpoint/recovery and bounded dynamics are complete;
- R4-D-015 records `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD` after the fixed 8-epoch / 120-optimizer-step pilot;
- M4 reached very strong positive fit, but confirmed-negative evidence remains zero, so no discrimination/false-positive claim follows;
- G8 remains open and M5/full training is unauthorized;
- threshold-fit, calibration-fit and untouched-acceptance populations remain unsupported/empty;
- Run12 remains the historical operational baseline and its learned state must not seed repaired R4 truth.

### Active ML evidence blocker — R4-GAP-007

- candidate #1 is durably `NOT_CONFIRMED`;
- candidate #2 primary review supports a class-specific negative, but authoritative state remains `UNKNOWN / PENDING_REVIEW / target None`;
- three recent external review attempts are non-authoritative because of candidate-context contamination and/or insufficient neutral governing material;
- those attempts exposed a material `CallToUnknown` semantic inconsistency:
  - locked taxonomy includes unchecked low-level calls **or calls to unverified external addresses**;
  - DIVE/SolidiFI operational mappings frame the class around raw/unchecked low-level `call` / `send` / `delegatecall` behavior;
- the neutral class boundary must be resolved before another candidate #2 independent adjudication;
- active working record: `working/2026-10-20_ml_gap007_call_to_unknown_semantic_alignment.md`.

### Inference

- the FastAPI inference runtime is substantive and strict about checkpoint/schema compatibility;
- the default operational checkpoint remains historical Run12;
- repaired R4 model promotion has not occurred;
- no M4 checkpoint may be treated as operational.

### Agents

- LangGraph orchestration, tool adapters, provenance contracts, evaluation, persistence, ingestion/feedback, RAG, static analysis and synthesis layers are substantive source modules;
- execution status/provenance is fail-closed for missing, degraded, mock or mutated tool output;
- some capability labels still require real-contract capability audit/hardening, including generic Halmos assumptions.

### ZKML

- proxy distillation, ONNX export, calibration, EZKL setup/prove/verify, bundle validation and verifier generation are implemented;
- current proof lineage remains historical-teacher-bound;
- retained proof scope is proxy computation only, not the full audit pipeline;
- `check_mode="UNSAFE"` remains a separate assurance limitation to investigate before stronger claims.

### Contracts / on-chain

- AuditRegistry V3, EIP-712 context binding, staking, verifier integration, upgrade behavior, invariants and Foundry tests are substantive;
- V3 separates proxy-proof verification from policy/provenance attestation;
- analysis/MCP code intentionally contains no production private-key/sign/broadcast/finality authority;
- production finality remains an optional isolated capability, not an implied current feature.

## 3. Governing completion principles

1. **Source before prose.** Audit actual source/tests before changing implementation.
2. **Preserve accepted history.** R4-D-009/R4-D-011/R4-D-013 and earlier accepted roots are immutable evidence.
3. **Unknown is not negative.** No target-zero, pseudo-negative, threshold, calibration or acceptance population may be invented.
4. **Resolve semantics before adjudication.** A class-definition ambiguity must not be silently narrowed/broadened to obtain a desired candidate verdict.
5. **Evidence before policy.** Thresholds, objectives, promotion gates and assurance claims require measured evidence and explicit identity.
6. **No silent capability inflation.** Wired code is not automatically a validated generic capability.
7. **No downstream rebind before upstream acceptance.** Inference/ZKML/contracts must not move to an unaccepted repaired model.
8. **Training is gated.** DATA acceptance does not authorize training; M4 success does not authorize M5; training does not authorize promotion.
9. **Plans may discover work.** Bounded audit/investigation is allowed when evidence is incomplete; unresolved uncertainty must remain explicit.

## 4. Current dependency order

```text
R4-D-009 logical V3
        +
R4-D-011 V10 graph parent
        +
R4-D-013 guarded tokens
        |
        v
R4-D-014 bounded objective/evaluation
        |
        v
M3 exact guarded integration        COMPLETE
        |
        v
M4 bounded mechanics/recovery       COMPLETE
        |
        v
R4-D-015 full-training HOLD
        |
        v
R4-GAP-007 class-definition alignment   <-- ACTIVE
        |
        v
self-contained v2 blind bundle
        |
        v
genuinely independent negative review
        |
        v
explicit objective/evaluation/M5 re-review
        |
        v
repaired full training only if authorized
        |
        v
evidence-qualified checkpoint/promotion
        |
        v
Inference migration from Run12
        |
        v
new teacher-bound ZKML lineage
        |
        v
verifier/registry reconciliation
```

Agents capability hardening may proceed in parallel when it neither changes R4 DATA/ML semantics nor depends on an unaccepted repaired model.

## 5. Program stages

### T0 — Re-anchor and source audit

For each module, inspect current source/config/tests, compare behavior with claims, identify stale assumptions, and record substantial unresolved questions in a dated working record.

**Exit:** no critical uncertainty is silently treated as fact.

### T1 — Guarded-token physical lineage

**State: COMPLETE / ACCEPTED.** R4-D-013 accepts the guarded-token physical successor. Do not repeat this work absent concrete contradiction.

### T2 — Learning/evaluation regime resolution

**State: ACTIVE / EVIDENCE-LIMITED.** R4-D-014 fixed the bounded positive-only contract, but R4-GAP-007 remains open.

Current sub-order:

1. resolve `CallToUnknown` semantic-definition authority neutrally;
2. if material definition change is required, use R4 ADR/schema/version governance;
3. rebuild candidate #2 blind review as a self-contained v2 package;
4. obtain genuinely isolated independent review;
5. assess whether the resulting negative evidence is sufficient for any later evaluation/objective decision.

**Stop line:** no pseudo-negatives, ungoverned PU learning, thresholds, calibration or M5 launch to bypass missing evidence.

### T3 — Training authorization and execution

**State: BOUNDED PILOT COMPLETE; FULL EXECUTION BLOCKED.**

M3 and M4 are complete. R4-D-015 keeps full training on HOLD. A later M5 launch requires a new explicit decision after T2 evidence changes materially.

### T4 — Evidence-qualified model promotion

**State: WAITING.** Promotion requires an authorized repaired checkpoint and evidence-qualified evaluation. Full, partial, restricted, research-only, rejected and HOLD outcomes remain possible.

### T5 — Inference migration

**State: WAITING ON MODEL PROMOTION.** Run12 remains operational baseline until M7-style model promotion explicitly hands off a repaired checkpoint/policy bundle.

### T6 — Agents capability hardening

**State: PARALLEL-ELIGIBLE.** Continue source-level/real-contract validation of consequential node/tool paths and fail-closed unavailable/degraded/mock semantics.

### T7 — ZKML lineage rebuild

**State: WAITING ON ACCEPTED OPERATIONAL TEACHER.** Re-distill/rebuild only after repaired model promotion; preserve exact proof-scope language and investigate unsafe proof settings before stronger assurance claims.

### T8 — Contracts and optional production finality

**State: WAITING ON DOWNSTREAM LINEAGE where applicable.** Validate new verifier/bundle compatibility only when a new ZKML lineage exists. Signing/broadcast/finality remains a separate optional capability decision.

## 6. Investigation and working-record protocol

Substantial pre-implementation analysis belongs under:

`docs/plan/system-finalization/technical-completion/working/`

A working record should capture:

- exact question;
- source/tests inspected;
- observations/evidence;
- rejected interpretations;
- unresolved items;
- decision/output needed;
- next executable step.

Working records are navigation/evidence aids, not semantic authority. Durable policy/class-definition decisions still use the existing R4 decision/ADR/schema system when required.

## 7. Module-plan status discipline

Use these states where appropriate:

- `NOT_STARTED`
- `AUDITING`
- `INVESTIGATING`
- `DESIGN_READY`
- `IMPLEMENTING`
- `VALIDATING`
- `BLOCKED`
- `ACCEPTED`
- `DEFERRED`
- `REJECTED`

A module may contain work packages at different states. “Code exists” is not equivalent to `ACCEPTED`.

## 8. Cross-module acceptance rules

A downstream module may consume an upstream artifact only when all are explicit:

- version/name;
- source commit;
- configuration/policy identity;
- input lineage;
- artifact hash/digest where applicable;
- validation evidence;
- acceptance authority/status;
- rollback or historical-control identity.

If one is missing, the artifact may be bounded research evidence but not silently promoted.

## 9. Completion definition

The technical-completion program is complete only when:

1. accepted DATA/representation roots remain reproducible;
2. R4 reaches an evidence-honest terminal model decision rather than merely completing a training run;
3. the operational inference model is explicitly identified and reproducible;
4. consequential agent capabilities have validation commensurate with their claims;
5. ZKML artifacts are rebound to the accepted operational teacher or explicitly retained as historical-only;
6. contract/verifier compatibility matches the selected ZKML lineage;
7. intentionally deferred capabilities are named rather than implied;
8. immutable historical roots remain reproducible.

## 10. Immediate next action

The active cross-module workstream is **ML evidence completion**, specifically R4-GAP-007 `CallToUnknown` semantic-definition alignment.

Read:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
2. `02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`;
3. `working/2026-10-20_ml_gap007_call_to_unknown_semantic_alignment.md`;
4. `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md` and `RISK_AND_BLOCKER_REGISTER.md`.

Then audit the neutral class-definition provenance/authority. Do **not** repeat DATA acceptance or M4, do not send candidate #2 for another review until the definition is resolved, and do not launch M5/full training.
