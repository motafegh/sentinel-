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

As of the 2026-08-16 V3 evidence-hardening boundary:

- canonical `main` has passed historical R4 G0–G7 and remains the active Phase-8 execution line;
- Phase 8 is `IN_PROGRESS`, not G8-passed;
- current semantic supervision policy remains `data-vnext-policy-v1`;
- historical `sentinel-r4-vnext-v1` / `r4-vnext-roles-v1` / graph-schema-v9 G7 evidence remains immutable and reproducible;
- R4-D-008 physically accepts repaired-v2 preprocessing/representations as the reusable physical DATA root: 22,540 contracts / 67,620 graph-token-sidecar files, all physically valid, binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- repaired physical identities remain `sentinel-preprocessed-r4-v2`, `r4-provenance-v1`, `evidence-ledger-r4-v2`, `representations-r4-v2`, extractor `v2.2-r4-repaired`;
- V2 address-literal grouping is superseded after a 10,327-contract DIVE component demonstrated over-grouping;
- R4-D-009 accepts logical authority `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3`, logical build `r4-logical-lineage-v3`;
- protected local V3 acceptance passed: 22,394 groups, max group size 7, 146 normalized-code edges, zero address-authority edges, unchanged target/strength semantics, all 67,620 physical files valid, and the exact repaired-v2 physical binding digest preserved;
- V3 family authority is now hardened: normalized-code identity is global; exact artifact identity is global; explicit source-native family/project IDs are source-namespaced as `<source>:<field>:<value>`; arbitrary Ethereum address literals remain diagnostic-only;
- the accepted V3 population had zero explicit-family edges, so source-namespacing hardening does not invalidate the accepted V3 grouping artifact;
- V3 active optimizer supervision remains 932 positive-only effective loss cells;
- **143 contracts / 142 unique groups is the previously observed combined outcome-metric population across `MODEL_SELECTION` + `INTERNAL_AUDIT`, not the MODEL_SELECTION population**;
- protected-local audit observed active `MODEL_SELECTION = 71 contracts / 71 groups` and active `INTERNAL_AUDIT = 72 contracts / 71 groups`; these counts must be reproduced by the hardened acceptance rerun before final snapshotting;
- the ML dataset adapter itself allows `MODEL_SELECTION` only for model-selection loading and does not load `INTERNAL_AUDIT`; the defect was governance/reporting and research-population labeling, not trainer leakage;
- the earlier V3 sensitivity/CPU-selector/negative-queue/CUDA reports are **pre-hardening observations**. Repository code has been hardened and those affected reports must be regenerated locally before a final durable V3 snapshot;
- snapshotting now fails closed on cross-report manifest/version/binding/queue/selector/GPU coherence and writes `snapshot_coherence_v1.json` only after a coherent tranche passes;
- representation sensitivity is now self-identifying by dataset/grouping/partition versions, publication manifest SHA, physical binding digest, and source commit; `MODEL_SELECTION` and `INTERNAL_AUDIT` sensitivity sets are separated;
- CPU selector evidence is now publication/binding/source-lineage bound;
- V3 CUDA comparison now requires a sensitivity report from the exact same V3 manifest, physical binding and source commit and records its SHA;
- confirmed-negative queue generation now enforces global leakage-group uniqueness across enabled classes and fails closed if the requested class balance cannot be satisfied;
- R4-GAP-007 remains the approved confirmed-negative pilot gap, but **do not adjudicate the pre-hardening queue**; regenerate and inspect it first;
- no confirmed-negative target exists yet; target `0` remains forbidden without class-specific confirmed-negative evidence;
- threshold-fit, calibration-fit, and untouched-acceptance roles remain unsupported/empty;
- the guarded selector remains unpromoted. Before promotion, add/execute full-population verification that the historical control selector reproduces the currently bound representation token tensors exactly;
- graph schema remains `v9`, token tensor contract `[4,512]`, architecture `four_eye_v8` / `v8.1`;
- Run12 is the historical operational ML baseline, not repaired-v2/V3 truth; do not reuse its learned weights, optimizer/scheduler state, thresholds, or calibration as current Phase-8 truth;
- the 100-epoch Phase-8 run remains unauthorized; no model-quality improvement is claimed;
- V3 is the current registry submission protocol; V1/V2 are historical compatibility;
- audit MCP is read-only;
- V3 signing/broadcast is outside the analysis MCP security domain;
- the retained EZKL proof proves only the proxy computation; V3 context attestation is separate.

For the exact current DATA/ML restart boundary, read in order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
2. `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`;
3. `docs/plan/ml-R4/DECISION_REGISTER.md`, `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md`, and `docs/plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md`;
4. `docs/plan/ml-R4/runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md` as the **pre-hardening historical checkpoint**;
5. `docs/plan/ml-R4/runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md` for the physical repaired-v2 acceptance boundary.

Do not create the final V3 snapshot from the old local reports. The current controlled work is: sync hardened `main`, regenerate corrected acceptance → sensitivity → CPU selector → globally unique negative queue → CUDA comparison, then run the coherence-gated final snapshot. Do not adjudicate negatives, promote the selector, or launch full training before that regeneration is reviewed.

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

For the current Phase-8 boundary, committed restart authority is `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`, governed by accepted R4-D-009 / ADR-R4-009. R4-D-008 remains the physical-data reproducibility root. R4-GAP-007 governs any later confirmed-negative review. If private/local memory is maintained, it should point to these committed boundaries rather than duplicate transient generated counts.

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
- Repaired-v2 source/representation artifacts remain the accepted physical root.
- V2 grouping/roles are historical/superseded for future model research.
- Current accepted logical authority is V3 grouping/roles/publication; no arbitrary address literal may create a leakage-group edge.
- Explicit source-native family/project identifiers must be source-namespaced before grouping authority is applied.
- no target `0` without confirmed-negative evidence;
- GasException/UnusedReturn supervision disabled under policy v1;
- Run12 threshold/calibration artifacts are historical only;
- architecture remains frozen unless a later approved architecture decision changes it;
- do not overwrite historical or accepted physical DATA/representation artifacts;
- V3 acceptance proved unchanged semantic counts and the exact same repaired-v2 physical binding digest;
- post-acceptance research reports must be regenerated under the hardened evidence-lineage code before final snapshot;
- R4-GAP-007 queue membership is review reservation only, not negative truth or optimizer supervision;
- selector promotion requires a separate ADR/versioned extractor decision and bound-token control equivalence evidence;
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
