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

### Current stable technical baseline

As of the 2026-09-02 V10 V2.6 physical-acceptance decision, with R4-D-010 governing representation semantics, R4-D-011 governing the exact physical V10 lineage, and the 2026-08-16 hardened V3 evidence snapshot retained as the accepted pre-pilot logical baseline:

- canonical `main` has passed historical R4 G0–G7 and remains the active Phase-8 execution line;
- Phase 8 is `IN_PROGRESS`, not G8-passed;
- current semantic supervision policy remains `data-vnext-policy-v1`;
- historical `sentinel-r4-vnext-v1` / `r4-vnext-roles-v1` / graph-schema-v9 G7 evidence remains immutable and reproducible; historical G7 implementation merge `81d9c547d` and tracked candidate status `VALIDATED_G7_CANDIDATE` remain historical anchors;
- R4-D-008 physically accepts repaired-v2 preprocessing/representations as the reusable physical DATA root: 22,540 contracts / 67,620 graph-token-sidecar files, all physically valid, binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- repaired physical identities remain `sentinel-preprocessed-r4-v2`, `r4-provenance-v1`, `evidence-ledger-r4-v2`, `representations-r4-v2`, extractor `v2.2-r4-repaired`;
- V2 address-literal grouping is superseded after a 10,327-contract DIVE component demonstrated over-grouping;
- R4-D-009 accepts logical authority `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3`, logical build `r4-logical-lineage-v3`;
- protected local V3 acceptance passed: 22,394 groups, max group size 7, 146 normalized-code edges, zero address-authority edges, unchanged target/strength semantics, all 67,620 physical files valid, and the exact repaired-v2 physical binding digest preserved;
- V3 family authority is hardened: normalized-code identity is global; exact artifact identity is global; explicit source-native family/project IDs are source-namespaced as `<source>:<field>:<value>`; arbitrary Ethereum address literals remain diagnostic-only;
- the accepted V3 population had zero explicit-family edges, so source-namespacing hardening does not invalidate the accepted V3 grouping artifact;
- V3 active optimizer supervision remains 932 positive-only effective loss cells;
- hardened acceptance durably establishes **143 contracts / 142 unique groups as the combined outcome-metric population across `MODEL_SELECTION` + `INTERNAL_AUDIT`, not the MODEL_SELECTION population**;
- active `MODEL_SELECTION = 71 contracts / 71 groups`; active `INTERNAL_AUDIT = 72 contracts / 71 groups`;
- the ML dataset adapter allows `MODEL_SELECTION` only for model-selection loading and does not load `INTERNAL_AUDIT`; the old defect was governance/reporting and research-population labeling, not trainer leakage;
- the post-acceptance V3 evidence-hardening tranche is complete: acceptance, sensitivity, CPU selector, negative-review queue, and CUDA comparison were regenerated under source commit `83bd566b9c4f4f653e530c2c0f5c990858dd759d`;
- the final Git-safe evidence snapshot at `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/` passed `coherence=PASS`, all listed JSON SHA-256 checks verified `OK`, and was committed at `44fbb9c1d2033be8002fe404d650cf09f08b0f29`;
- hardened CPU selector evidence reproduced 1,018 analyzed records / 737 over-cap / 476 improved / 261 equal / 0 regressed / 0 failures;
- hardened CUDA selector evidence completed identical initialization and 4/4 required worst-case probes, with no Run12 weights, no checkpoint, no selector promotion, and no full-training authorization;
- the hardened V3 confirmed-negative queue contains 200 `PENDING_REVIEW` cells across 200 globally unique leakage groups, all target `None`, all `TRAIN_UNLABELED`, `negative_truth_claim=false`;
- R4-GAP-007 is `IN_PROGRESS`: candidate #1 is durably `NOT_CONFIRMED`; candidate #2 primary review supports `CONFIRMED_NEGATIVE`, but target `None` / UNKNOWN / PENDING_REVIEW remain authoritative until a genuinely independent verifier agrees; confirmed negatives remain zero;
- candidate #2 exposed R4-GAP-008: v9 type-11 external-call semantics materially conflated library calls and omitted substantial Transfer/Send/low-level-call behavior;
- R4-D-010 preserves v9/repaired-v2 as immutable historical and physical-reproducibility evidence but withdraws v9 from eligibility for the new full training run; graph schema v10 is the required future physical lineage;
- the original V10 `v2.3-r4-call-semantics` lineage is now a frozen structural-reference diagnostic, not the current future-candidate extractor;
- the 26-contract parse-only remediation is complete in the later compatibility lineage. The protected V2.4 diagnostic candidate has 22,540 identities, exact accepted-V9 token bytes, zero parse-only outputs, zero unclassified call IR, and the required 22,539 primary Slither-0.10 + one identity-bound Slither-0.11.5 runtime split;
- the later V2.5 structural correction used extractor `v2.5-r4-call-semantics-deterministic-cfg`. Three fresh bounded generations closed all 20 previously unexpected structural-drift identities: 8 exact node-index-invariant labelled graph-equivalence identities and 12 deterministic persistent-storage `CFG_NODE_WRITE` corrections; the final bounded verifier reported zero unexplained drift and no blockers;
- the current V2.6 correction uses extractor `v2.6-r4-call-semantics-deterministic-cfg-mutators` and additionally recognizes only persistent-storage collection `push`/`pop` calls while preserving call-node priority; memory receivers and arbitrary member calls remain excluded;
- the V2.5 evidence chain is SHA-bound to the original transition audit and merged semantic evidence; full-gate compile/tests and evidence-chain preflight pass;
- heterogeneous-runtime full-candidate staging is approved and preflighted. Exact partition is 22,539 ordinary identities under Slither 0.10.0 plus one declared exception `dive/caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9` under Slither 0.11.5;
- the historical V2.5 Stages A-D passed, but Stage E correctly rejected that complete candidate with `PASS_BASE_MECHANICS_WITH_STRUCTURAL_EVIDENCE_BLOCKER`: it found 311 raw non-parse-only drifts, re-proved the bounded 8+12 classes, and left 298 full-population identities outside those approved evidence classes;
- fresh protected-local V2.6 Stages A-D pass: Stage A produced all 22,539 ordinary Slither-0.10 identities with zero unexpected failures, Stage B staged only validated triples, Stage C filled only the declared Slither-0.11.5 exception, and Stage D bound all 22,540 identities with exact accepted-V9 token bytes and digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- the V2.6 population changed by +52/-8 relative to the historical 311-case set, so three fresh generations and three semantic-evidence passes covered the exact current 355 identities. The final V4 audit passes all 22,540 mechanics and re-proves 349 persistent-storage WRITE corrections plus 6 exact index-equivalent graphs with zero unexplained drift;
- R4-D-011 physically accepts only the exact protected-local V2.6 root and binding digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd` after refreshed binding and current-commit V4 review; R4-B008 / R4-GAP-008 are closed for that identity, while selector promotion and training remain unauthorized;
- the selector historical-control equivalence gate passes 22,540/22,540: dynamic `historical_linspace_v1` reconstruction exactly matches every R4-D-011 bound token tensor and selected-window index. This is prerequisite evidence only; `target_aware_guarded_v1` remains unpromoted pending a separate decision;
- target `0` remains forbidden without complete class-specific confirmed-negative evidence plus independent agreeing verification;
- accepted confirmed negatives are evaluation-only unless a later versioned policy grants optimizer authority;
- Positive–Unlabeled (PU) learning is a future objective-design candidate, not current implementation authority;
- threshold-fit, calibration-fit, and untouched-acceptance roles remain unsupported/empty;
- the guarded selector remains unpromoted and requires separate full-population bound-token control-equivalence evidence plus a promotion decision;
- accepted historical/repaired representations remain graph schema `v9`; the future candidate is graph schema `v10` / extractor V2.6. The accepted token tensor contract remains `[4,512]`, and architecture remains `four_eye_v8` / `v8.1` unless a later explicit decision changes it;
- Run12 is the historical operational ML baseline, not repaired-v2/V3 truth; do not reuse its learned weights, optimizer/scheduler state, thresholds, or calibration as current Phase-8 truth;
- the 100-epoch Phase-8 run remains unauthorized; no model-quality improvement is claimed;
- V3 is the current registry submission protocol; V1/V2 are historical compatibility;
- audit MCP is read-only;
- V3 signing/broadcast is outside the analysis MCP security domain;
- the retained EZKL proof proves only the proxy computation; V3 context attestation is separate.

For the exact current DATA/ML restart boundary, read in order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
2. `docs/plan/ml-R4/runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`;
3. `docs/plan/ml-R4/adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md`;
4. `docs/plan/ml-R4/runs/2026-09-02_PHASE8_selector_control_equivalence_plan.md` for the current executable prerequisite;
5. `docs/plan/ml-R4/runs/2026-08-30_PHASE8_v10_v25_full_population_structural_evidence_plan.md`;
6. `docs/plan/ml-R4/runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md` for historical staging context;
7. `docs/plan/ml-R4/reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`;
8. `docs/plan/ml-R4/adrs/ADR-R4-010-versioned-external-call-representation-correction.md` for the accepted semantic decision;
9. `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md` for the accepted pre-pilot evidence boundary;
10. `docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md` for the separate negative-evidence track;
11. `docs/plan/ml-R4/DECISION_REGISTER.md` and `docs/plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md` as decision context.

The August 21 V10 implementation records, the August 23 parse-only plan, and the August 23 structural-drift handoff are historical execution records and must not be used as the current restart boundary when they describe blockers already closed. Candidate #1 is durably closed `NOT_CONFIRMED`; do not repeat it. Candidate #2 primary review supports a class-specific negative, but the primary reviewer must not self-verify it. Selector promotion and objective/evaluation design remain separate later tracks. Do not infer negatives, silently promote the selector, fit unsupported threshold/calibration roles, reuse Run12 state, overwrite protected V10 history/reference roots, or launch full training.

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

For the current Phase-8 boundary, point private/local memory first to `docs/plan/ml-R4/runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`, then the 2026-08-30 full-population structural-evidence plan. The accepted pre-pilot V3 boundary remains `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`, governed by R4-D-009 / ADR-R4-009. R4-D-008 remains the physical-data reproducibility root. Protected-local Stage A-E counts are evidence only when tied to the report hashes recorded in the checkpoint and artifact index.

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
- Repaired-v2 source/representation artifacts remain the accepted immutable physical/reproducibility root, but v9 is no longer eligible for a new full training run under R4-D-010.
- V2 grouping/roles are historical/superseded for future model research.
- Current accepted logical authority is V3 grouping/roles/publication; no arbitrary address literal may create a leakage-group edge.
- Explicit source-native family/project identifiers must be source-namespaced before grouping authority is applied.
- no target `0` without confirmed-negative evidence;
- GasException/UnusedReturn supervision disabled under policy v1;
- Run12 threshold/calibration artifacts are historical only;
- architecture remains frozen; R4-D-010 is a scoped representation-schema unfreeze for a new v10 candidate, not permission for an unrelated model-architecture change;
- do not overwrite historical or accepted physical DATA/representation artifacts;
- V3 acceptance proved unchanged semantic counts and the exact same repaired-v2 physical binding digest;
- hardened post-acceptance V3 research reports have been regenerated coherently and committed in the final Git-safe evidence snapshot;
- R4-GAP-007 queue membership is review reservation only, not negative truth or optimizer supervision; use only the committed hardened queue for pilot review;
- R4-GAP-007 is in progress; candidate #1 is durably `NOT_CONFIRMED`. Candidate #2 primary review supports `CONFIRMED_NEGATIVE`, but it remains UNKNOWN/target `None` pending genuinely independent agreement; do not fabricate that verification;
- R4-GAP-008 root-cause evidence is resolved under R4-D-010. The 26 parse-only repair, bounded 20-identity V2.5 tranche, and complete 355-identity V2.6 structural-evidence tranche are complete. R4-D-011 accepts only the exact V2.6 root/digest after refreshed binding and current-commit V4 review; v9 must never be patched in place, and this acceptance grants no training authority;
- accepted confirmed negatives remain evaluation-only unless a later versioned decision grants optimizer authority;
- PU learning remains a later objective-design candidate, not current implementation authority;
- selector promotion requires a separate ADR/versioned extractor decision and bound-token control-equivalence evidence;
- full Phase-8 training remains prohibited until explicit later governance re-authorizes it.

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

It must compile repaired/V3 DATA and ML entry points, run repaired/V3 regressions including V3 snapshot-coherence tests, prove historical G6 still validates, and pass `git diff --check` from the audit handoff base.

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
