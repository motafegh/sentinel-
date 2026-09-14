# SENTINEL Technical Completion Roadmap

**Status:** ACTIVE — cross-module execution controller  
**Created:** 2026-09-14  
**Baseline:** `main` at `796b2e809cc9956b1228930bf45ba5d8a97f5616`  
**Scope:** remaining technical completion, capability hardening, model repair/promotion, and downstream lineage reconciliation  

## 1. Purpose

This roadmap coordinates the remaining engineering work across SENTINEL after the portfolio-professionalization program. It is intentionally source-driven: executable source/config/tests remain the highest authority for actual behavior, while the R4 machine-readable governance and accepted ADRs remain authoritative for DATA/ML semantics and gate state.

This file does **not** replace `docs/plan/ml-R4/00_MASTER_PLAN.md`, the R4 decision register, accepted ADRs, or module-specific plans. It defines cross-module sequencing and the conditions under which work may move from one subsystem to another.

The completion program uses six subordinate plans:

1. `01_DATA_REPRESENTATION_COMPLETION_PLAN.md`
2. `02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`
3. `03_INFERENCE_DEPLOYMENT_PROMOTION_PLAN.md`
4. `04_AGENTS_ORCHESTRATION_CAPABILITY_COMPLETION_PLAN.md`
5. `05_ZKML_LINEAGE_PROOF_COMPLETION_PLAN.md`
6. `06_CONTRACTS_ONCHAIN_COMPLETION_PLAN.md`

## 2. Current technical baseline

### DATA / representations

- historical G0–G7 remain passed and immutable;
- logical V3 grouping/roles are accepted;
- exact V10 V2.6 physical graph lineage is accepted under R4-D-011;
- accepted V10 identity count is 22,540 contracts with the exact R4-D-011 binding digest;
- R4-D-012 promotes `target_aware_guarded_v1` only for a **new versioned token lineage**;
- the guarded-token physical candidate has not yet been built or accepted;
- current executable tokenization still preserves the historical four-window linspace selector unless an accepted token tensor is copied.

### ML

- `four_eye_v8` / `v8.1` remains frozen for the first repaired-data baseline;
- the Phase-8 runner, deterministic grouped sampling, BF16 execution, checkpoint/resume, run binding, and positive-only masked loss/selection diagnostics are implemented;
- current Phase-8 supervision is positive-only and explicitly rejects authorized target-zero cells;
- confirmed negatives remain zero;
- threshold-fit, calibration-fit, and untouched-acceptance populations remain unsupported/empty;
- G8 is open and the 100-epoch repaired run remains unauthorized;
- the durable runner is not yet promoted onto a final accepted V3/V10/guarded-token training lineage.

### Inference

- the FastAPI inference runtime is real and strict about checkpoint/schema compatibility;
- the default operational checkpoint remains historical Run12;
- repaired R4 model promotion has not occurred.

### Agents

- the LangGraph orchestration, tool adapters, provenance contracts, evaluation, persistence, ingestion/feedback, RAG, static analysis and synthesis layers are substantive source modules;
- execution status/provenance is fail-closed for missing, degraded, mock, or mutated tool output;
- some capability labels are stronger than their generic implementation today: notably the Halmos harness assumes specific target methods and needs a dedicated capability repair/audit.

### ZKML

- proxy distillation, ONNX export, calibration, EZKL setup/prove/verify, bundle validation and verifier generation are implemented;
- the current proof path remains bound to the historical teacher/proxy lineage;
- the retained proof scope is proxy-computation only and does not prove the full audit pipeline;
- retained EZKL settings include `check_mode="UNSAFE"`, which requires explicit investigation before any stronger assurance claim.

### Contracts / on-chain

- `AuditRegistry` V3, EIP-712 context binding, staking, verifier integration, upgrade behavior, invariants and Foundry tests are substantive;
- V3 separates proxy-proof verification from policy/provenance attestation;
- analysis/MCP code deliberately contains no production private key, signing, transaction broadcast, or receipt/finality authority;
- autonomous production finality is therefore a separate optional capability, not an implied current feature.

## 3. Governing completion principles

1. **Source before prose.** Every module plan starts with a source/test audit before new implementation.
2. **Preserve accepted history.** R4-D-011 and all earlier accepted lineages are immutable evidence roots. New behavior creates new versioned lineage.
3. **Unknown is not negative.** No target-zero, pseudo-negative, threshold, calibration, or acceptance population may be invented to unblock training.
4. **Evidence before policy.** Thresholds, weights, promotion gates and assurance claims require measured evidence and explicit artifact identity.
5. **No silent capability inflation.** A wired node/tool is not automatically a working generic capability; real-contract behavior must be tested.
6. **No downstream rebind before upstream identity exists.** Inference, ZKML and contracts must not be rebound to a repaired teacher before that teacher is actually accepted.
7. **Training is a gated action.** Building the selector candidate does not authorize training; training does not authorize promotion; promotion does not authorize on-chain finality.
8. **Plans may discover work.** Each module plan explicitly authorizes bounded audit/investigation before design or implementation when the evidence is incomplete.

## 4. Dependency order

```text
R4-D-011 immutable V10 physical parent
        |
        v
DATA: implement D-012 guarded selector
        |
        v
fresh guarded-token physical candidate
        |
        v
binding + full-population validation + physical acceptance
        |
        +------------------------------+
        |                              |
        v                              v
ML objective/evaluation evidence     Agents capability hardening
        |
        v
explicit training authorization
        |
        v
repaired training + reproducible checkpoint
        |
        v
evidence-qualified evaluation / promotion decision
        |
        v
Inference migration from Run12
        |
        v
new teacher-bound proxy distillation / ZKML bundle
        |
        v
verifier / registry compatibility reconciliation
        |
        v
optional isolated signer/broadcaster/finality capability
```

Agents auditing and generic-capability repairs may proceed in parallel when they do not change R4 DATA/ML semantics or depend on an unaccepted repaired model.

## 5. Program stages

### T0 — Re-anchor and source audit

For each module:

- inspect current source, config, tests and runtime boundaries;
- compare actual behavior with current public/architectural claims;
- identify stale or misleading assumptions;
- record unresolved questions in a dated working record if the investigation is substantial.

**Exit:** module plan has a source-grounded implementation boundary and no critical uncertainty is being silently treated as fact.

### T1 — Guarded-token physical lineage

Owned by the DATA plan.

- implement `target_aware_guarded_v1` as a separate, versioned selector;
- retain the historical linspace selector as explicit control;
- bind selector/fallback/coverage evidence into artifact metadata;
- generate the full 22,540-contract candidate locally;
- prove deterministic behavior and historical-control equivalence where required;
- compute new physical identity/digest;
- perform a separate physical acceptance review.

**Stop line:** no repaired full training while this candidate is unaccepted.

### T2 — Learning/evaluation regime resolution

Owned by the ML plan.

Investigate and decide, from evidence rather than convenience:

- whether confirmed-negative evidence can be expanded legitimately;
- what evaluation claims are possible with positive-only evidence;
- whether Positive–Unlabeled learning or another objective is justified;
- what populations can support model selection, threshold fitting, calibration and acceptance;
- what classes must remain provisional/unsupported if evidence stays insufficient.

**Stop line:** do not implement PU learning, pseudo-negatives, thresholds or calibration merely to fill empty roles.

### T3 — Training authorization and execution

Only after T1 and T2 produce explicit accepted inputs:

- bind the accepted physical/data lineage and exact code/config;
- validate training/inference architecture compatibility;
- authorize a bounded smoke/pilot before any long run;
- run the repaired training only when its objective/evaluation contract is explicit;
- preserve deterministic/resumable evidence and resource telemetry.

**Exit:** reproducible checkpoint candidate bound to source, data, representation, objective and config identities.

### T4 — Evidence-qualified model promotion

Evaluate only claims the evidence supports. Promotion may be full, partial, restricted, or rejected.

Required outputs include:

- class-level support status;
- discrimination/positive-response evidence appropriate to available truth;
- calibration/threshold evidence only where valid populations exist;
- regression/compatibility evidence;
- rollback path;
- explicit decision on whether Run12 remains operational baseline.

### T5 — Inference migration

Owned by the inference plan.

Only after a model is accepted:

- validate checkpoint architecture/schema/representation compatibility;
- bind thresholds/policy artifacts separately from raw model output;
- perform offline parity and live-service regression tests;
- update deployment configuration only after acceptance;
- keep rollback to Run12 or the prior accepted bundle rehearsable.

### T6 — Agents capability hardening

Owned by the agents plan and may overlap earlier stages.

Primary targets:

- real-contract audit of each consequential node/tool path;
- repair Halmos generic-contract/property generation or narrow its advertised capability honestly;
- test degraded/unavailable/mock semantics end-to-end;
- validate routing, evidence fusion, consensus, synthesis and report language against actual tool status;
- eliminate any remaining case where “tool absent/failed” can look equivalent to “tool ran clean.”

### T7 — ZKML lineage rebuild

Only after a repaired teacher is accepted and inference identity is stable:

- re-distill proxy from the exact accepted teacher;
- regenerate ONNX/calibration manifests;
- investigate and resolve the `UNSAFE` proof-setting limitation before stronger assurance claims;
- generate a new EZKL setup bundle, keys, proof evidence and verifier as required;
- preserve the exact proof scope: proxy computation is not the entire audit.

### T8 — Contracts and optional production finality

- validate new verifier/bundle compatibility with V3 registry semantics;
- rerun unit, upgrade, invariant, golden-digest and real-proof tests;
- preserve storage and EIP-712 identity guarantees;
- separately decide whether an isolated signer/broadcaster/finality service belongs in Sentinel’s intended product scope.

A signer/broadcaster is **not** required to declare the analysis system technically coherent. It is required only if autonomous production submission/finality becomes an explicit capability goal.

## 6. Investigation and working-record protocol

When a stage needs substantial analysis before implementation, create a dated working record under:

`docs/plan/system-finalization/technical-completion/working/`

Recommended naming:

`YYYY-MM-DD_<module>_<question>_working.md`

A working record should capture:

- exact question;
- source/tests inspected;
- observations and evidence;
- rejected interpretations;
- unresolved items;
- decision/output needed;
- next executable step.

Working records are evidence/navigation aids, not new semantic authorities. Durable policy decisions still belong in the appropriate existing R4 decision/ADR system or the relevant module plan when no new ADR is necessary.

## 7. Module-plan status discipline

Each subordinate plan should use these states:

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

A module may contain multiple work packages at different states. “Code exists” is not equivalent to `ACCEPTED`.

## 8. Cross-module acceptance rules

A downstream module may consume an upstream artifact only when all of the following are explicit:

- version/name;
- source commit;
- configuration/policy identity;
- input lineage;
- artifact hash/digest where applicable;
- validation evidence;
- acceptance authority/status;
- rollback or historical-control identity.

If one of these is missing, the artifact may be used for bounded research but not silently promoted as current authority.

## 9. Completion definition

The technical-completion program is complete only when:

1. the guarded-token physical lineage is accepted or explicitly rejected with a successor decision;
2. R4 reaches an evidence-honest terminal model decision rather than merely completing a training run;
3. the operational inference model is explicitly identified and reproducible;
4. consequential agent capabilities have source-level and real-contract validation commensurate with their claims;
5. ZKML artifacts are either rebound to the accepted operational teacher or explicitly retained as historical-only;
6. contract/verifier compatibility matches the selected ZKML lineage;
7. every intentionally deferred capability is named rather than implied;
8. immutable historical roots remain reproducible.

## 10. Immediate next action

Begin `01_DATA_REPRESENTATION_COMPLETION_PLAN.md` at its source-audit/implementation boundary for R4-D-012. Do **not** begin the repaired full training run, threshold/calibration fitting, or downstream ZKML rebinding first.
