# SENTINEL AI Working Instructions

SENTINEL is Ali's learning-by-doing smart-contract security project. AI assistance may write substantial code, but project ownership, architecture reasoning, evidence quality, reproducibility, and educational value must remain explicit.

Project root: `~/projects/sentinel` in the primary WSL environment.

## Current project authority

Use this order when facts conflict:

1. **Executable source/config/tests** for actual behavior.
2. **Committed machine-readable governance/artifacts** for semantics and gate state — especially `docs/plan/ml-R4/` for current DATA/ML work.
3. **Canonical handbook** under `docs/handbook/` for current explanatory/navigation context.
4. ADRs/decision/risk registers for rationale and controlled decisions.
5. Supplementary guides/labs and dated historical reports for learning/history only.

Do not treat a stale comment, docstring, README, old plan, old test count, or historical artifact as stronger than current source/policy.

### Current stable technical baseline

As of the 2026-08-14 Phase-8 launch-readiness reconciliation:

- canonical `main` has passed R4 G0–G7 and is the active Phase-8 execution line;
- Phase 8 is `IN_PROGRESS`: the existing-architecture vNext compatibility path, masked training/evaluation semantics, grouped sampler, durable checkpoint/resume runner, runtime provenance binding, and bounded GPU execution are validated, but the real fixed-horizon retrain has **not** been launched and is on a DATA/representation hold pending a repaired, versioned, re-frozen lineage;
- `sentinel-r4-vnext-v1` / export schema v2 is the canonical repaired DATA input, physically bound to 21,657 required representations / 64,971 graph-token-sidecar files with zero missing files and zero mismatches;
- `data-vnext-policy-v1` and `r4-vnext-roles-v1` govern current DATA/ML semantics;
- historical zero/absence/unsupported state is not confirmed-negative truth, and the current repaired baseline has zero confirmed-negative target cells;
- threshold-fit, calibration-fit, and untouched-acceptance roles remain unsupported/empty for the first repaired baseline;
- Run12 is the **historical operational ML baseline**, not a repaired-vNext model; its learned weights, optimizer/scheduler state, thresholds, and calibration are not reused as current Phase-8 truth;
- the first repaired retrain keeps the existing `four_eye_v8` / `v8.1` architecture frozen so data/label repair can be evaluated without simultaneous architecture search;
- the old `r4/phase8-existing-model-retraining` branch/worktree is provenance only after adoption into `main`;
- V3 is the current registry submission protocol; V1/V2 are historical compatibility;
- live audit MCP is read-only;
- V3 signing/broadcast is outside the analysis MCP security domain;
- the retained EZKL proof proves only the proxy computation; V3 context attestation is separate.

For the current pre-training boundary, read `docs/plan/ml-R4/runs/2026-08-14_PHASE8_real_data_readiness_audit.md` before `docs/plan/ml-R4/runs/2026-08-14_PHASE8_pretraining_launch_handoff.md`. Always check `docs/handbook/16_current_status.md` and the relevant machine-readable R4 artifacts before making a current-state claim.

## Approval model

Ali has delegated **routine technical and governance approvals** to the assistant. Do not stop merely to ask for approval of a normal plan, implementation slice, refactor, test, ADR, branch, PR, or evidence-preserving decision when the technically best option is clear.

Instead:

1. make the best evidence-based decision;
2. record important decisions in the appropriate plan/ADR/register;
3. execute and validate them;
4. surface consequential tradeoffs/results to Ali.

Still stop for genuinely non-routine user decisions that cannot be inferred safely, such as destructive loss of protected data/history, real production deployment/funds/keys, irreversible external actions, or choices that change the project's intended product/career direction rather than implementation quality.

## Plan before code

For non-trivial work, write or update the appropriate plan/working record before implementation. The plan is an execution aid, not an approval ceremony.

Avoid excessive governance ceremony. Do not create new specs/registers/ADRs unless the decision actually needs durable control or later implementation could otherwise invent semantics.

## Working memory

In Ali's primary Claude Code setup, project memory may exist under:

`/home/motafeq/.claude/projects/-home-motafeq-projects-sentinel/memory/`

When available and relevant, read `MEMORY.md` plus only the referenced working memories needed for the task. Do not let private/local memory override current committed source or machine-readable governance.

For the current R4 Phase-8 pre-training boundary, the committed restart record is `docs/plan/ml-R4/runs/2026-08-14_PHASE8_pretraining_launch_handoff.md`. If private/local `MEMORY.md` is maintained, it should point to that handoff rather than duplicate all transient launch details.

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

Thresholds, weights, confidence cutoffs, acceptance gates, and similar numbers are policy.

They should be:

- externalized/versioned rather than buried in implementation;
- changed only with measured evidence;
- bound to the artifact/config that used them.

Tests prove code behavior. Evals/evidence justify decision quality.

For R4 specifically, do **not** invent thresholds/calibration data to fill currently unsupported roles.

### 3. No silent failures or silent skips

A failure must be represented as one of:

1. precise eager error;
2. structured degraded result containing the reason/status;
3. explicit state/tool-status field consumed downstream.

Never make “tool did not run” indistinguishable from “tool ran and found nothing.”

Forbidden patterns include:

- `except ...: return []/{}/None` when caller interprets empty as clean;
- debug-log-only failure reporting;
- optional dependency/tool absence silently converted into valid evidence;
- fallback values that contaminate metrics without provenance.

### 4. Preserve semantic uncertainty

This rule is permanent after R4:

- unknown ≠ negative;
- unavailable ≠ clean;
- unsupported ≠ negative;
- weak evidence ≠ strong/metric-grade evidence;
- historical label ≠ current DATA vNext truth;
- on-chain record ≠ vulnerability ground truth;
- valid proxy proof ≠ proof of teacher/source/AGENTS execution.

Encode uncertainty/status explicitly rather than forcing a convenient downstream type.

### 5. Version rather than overwrite history

Historical v1 DATA, Run12 artifacts, V1/V2 registry history, and retained proxy artifacts are reproducibility/rollback roots.

For semantic/protocol changes:

- create a new versioned artifact/path;
- bind hashes and lineage;
- validate before promotion;
- rollback by selecting an older compatible bundle, not reverse-editing history.

### 6. Source-first review, docs-currentness second

When changing code:

1. inspect executable source/tests;
2. inspect current machine policy/ADR if semantics are governed;
3. check canonical docs for expected boundary/intent;
4. flag and fix any canonical documentation contradiction in the same change when practical.

Dated audits, redesign notes, experimental reports, `docs/learning/`, and handbook technical/lab exercises may intentionally describe historical mechanics. Preserve them unless they falsely present themselves as current authority.

## Module-specific boundaries

### DATA / ML

- R4 controls current semantic authority.
- Phase-6 roles are frozen; later phases must not rebalance them implicitly.
- no target `0` without confirmed-negative evidence;
- GasException/UnusedReturn supervision disabled under policy v1;
- Run12 threshold/calibration artifacts are historical only;
- first repaired retrain keeps architecture frozen unless a later approved architecture decision changes that;
- Phase-8 full retraining must bind the exact `main` source commit, G7 DATA/representation lineage, runtime provenance, seed, optimizer/scheduler horizon, and frozen role/population identities;
- while a bound full run is active, do not mutate tracked files, change the ML environment, or reinterpret positive-only MODEL_SELECTION as general validation evidence.

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

For canonical documentation/current-state changes:

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
python3 docs/handbook/tools/verify_handbook.py inventory
```

For R4 G6 semantics:

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
