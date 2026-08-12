# 16 — Current status and gap ledger

**Read this when:** you need to know what is canonical, verified, pending, unsupported, historical, or local-only today.

**Skip this if:** never before operational, training, evaluation, or production-readiness decisions.

**Estimated reading time:** 10 minutes.

## 30-second summary

The stable source/runtime baseline for this handbook reconciliation is `main` merge **`91f795885`** (2026-08-12): R4 DATA/ML repair has passed **G0 through G6**, the V3 registry/context protocol and read-only audit-MCP boundary are implemented, and Run12 remains the historical operational teacher. Phase 7 DATA vNext implementation is active on `r4/phase7-data-vnext-implementation`; its remote semantic build/validation is green, but **G7 remains pending local physical binding of 21,657 representations**. No repaired teacher has been retrained/promoted yet.

This page intentionally does **not** carry the old July module-suite totals. They were measured against an obsolete source/architecture baseline and are now historical evidence only. New volatile suite totals should be added only when they are rerun against a named current commit/environment.

## Just-enough mental model

```text
canonical main (91f795885)
  R4 G0–G6 PASS
  V3 registry/read-only observation boundaries
  Run12 historical inference baseline
        ↓
Phase 7 branch
  deterministic DATA vNext v2 semantic overlay
  remote semantic checks PASS
  local representation binding PENDING
        ↓
G7
        ↓
Phase 8 retrain existing architecture
        ↓
Phase 9 evaluation/policy
        ↓
Phase 10 promotion/rollback
```

Three current limitations are deliberate, not hidden TODOs:

```text
confirmed-negative support: absent in policy v1
threshold/calibration fitting: unsupported/empty
untouched acceptance: unsupported/empty/frozen
```

## Actual runtime/source walkthrough

### Canonical R4 gate state

| Phase | State | Meaning |
|---:|---|---|
| 0 | G0 PASS | historical baseline/evidence locations frozen |
| 1 | G1 PASS | prior evidence recovered |
| 2 | G2 PASS | label-corruption mechanisms reconstructed |
| 3 | G3 PASS | 22,493×10 evidence ledger materialized/validated |
| 4 | G4 PASS | decision-critical DIVE source authority adjudicated |
| 5 | G5 PASS | DATA vNext policy/schema/ADRs accepted |
| 6 | G6 PASS | leakage-safe roles frozen; acceptance support explicitly bounded |
| 7 | pending G7 | implementation branch exists; local physical representation binding still required |
| 8–10 | waiting | retraining/evaluation/promotion not authorized until preceding gates |

### R4 DATA foundation

Canonical evidence/policy facts on `main`:

- contracts: **22,493**;
- contract×class ledger rows: **224,930**;
- represented historical contracts: **21,657**;
- incomplete-representation contracts: **836**;
- historical-zero rows remain unresolved/unknown rather than confirmed negative;
- ten-class order remains locked/v9-compatible;
- GasException and UnusedReturn are supervision-disabled pending evidence;
- DIVE Front Running→TransactionOrderDependence is weak-positive only;
- no blanket confirmed-negative source exists.

Key lineage roots include the Phase-3 evidence ledger SHA-256:

`3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`

and accepted machine policy `data-vnext-policy-v1`.

### Frozen Phase-6 roles

`r4-vnext-roles-v1` assigns every active contract/leakage group exactly once:

| Role | Groups | Contracts | Current authority |
|---|---:|---:|---|
| TRAIN_STRONG | 238 | 275 | strong-positive training |
| MODEL_SELECTION | 51 | 56 | positive-only limited diagnostics |
| INTERNAL_AUDIT | 51 | 62 | internal strong-positive audit |
| TRAIN_WEAK | 465 | 773 | DIVE TOD weak-positive only |
| TRAIN_UNLABELED | 11,869 | 20,491 | represented, no authorized target |
| EXCLUDED | 835 | 836 | incomplete representation group |

Unsupported/frozen roles:

- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`

No confirmed-negative row is synthesized to fill these roles.

### Phase 7 candidate state

Branch: `r4/phase7-data-vnext-implementation`  
Latest documentation-audit-visible branch head: **`95c339edf`**.

Remote Phase-7 work has produced an additive v2 semantic implementation under `data_module/sentinel_data/vnext` plus a deterministic semantic overlay. Remote checks establish:

- vNext unit/compile checks pass;
- full 224,930-row semantic build is byte-deterministic under the pinned CI toolchain;
- committed semantic overlay validates independently;
- generated target count is 1,007 positive targets, **0 negative targets**;
- 403 strong and 604 weak training signals reconcile with frozen evidence/roles;
- manifest state is `SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING`;
- v2 loader rejects silent historical-v1 fallback.

G7 is **not** passed yet. The required local gate must verify the existing physical representation root for all 21,657 required contracts: graph + tokens + sidecar = **64,971 physical files**. Only a successful representation-bound final validator may promote the Phase-7 manifest to a G7 candidate for PR/merge.

### Current ML state

Run12 remains the historical operational checkpoint and comparison baseline. It was trained using the pre-R4 binary target semantics. Therefore:

- current Run12 inference remains usable for runtime continuity/historical comparison;
- Run12 weights are not the repaired DATA-vNext model;
- Run12 thresholds/calibration are historical companions and are not valid defaults for a future retrain;
- no Phase-8 repaired teacher checkpoint exists yet.

Architecture remains frozen through the initial repaired retrain so R4 can measure the effect of data/label repair before redesigning the model.

### Current V3 / chain state

`AuditRegistry` now contains V1/V2 historical compatibility plus the V3 context-attested protocol.

V3:

- appends storage safely;
- `initializeV3` configures V3 verifier/policy signer and disables new legacy V1/V2 writes;
- binds target runtime bytecode and agent/round/model/proxy/DATA/schema/proof/signal/score/deadline identities through EIP-712;
- rejects replayed request digests;
- verifies configured policy signer and retained proxy proof;
- preserves V1/V2/V3 reads/history.

The retained proof still proves only the proxy computation. `check_mode="UNSAFE"` remains an explicit production-assurance limitation.

### Current audit MCP / feedback state

Live audit MCP :8012 is **read-only** and exposes exactly:

- `get_latest_audit`
- `get_audit_history`
- `check_audit_exists`

It observes V1/V2/V3 history. Historical mutable submission code is not exposed by the live analysis service.

V3 feedback observation is implemented with an explicit policy boundary. Current V3 promotion policy is intentionally unavailable, so V3 observations may be durably pending but do not automatically enter RAG/DATA truth.

## Interfaces, data shapes, and configuration

### Current authority hierarchy

1. executable source and committed machine-readable policy/manifests;
2. this canonical handbook/current-status page;
3. active R4 plans/ADRs/registers for DATA/ML work;
4. supplementary technical guides/labs for mechanics;
5. historical plans/reports/learning files for context only.

### Fresh-clone / local-only distinction

Tracked in Git:

- source/config/tests;
- R4 evidence ledger/policy/schema/role manifests and reports;
- contract V3 tests;
- retained proxy/ONNX/settings/compiled/VK/verifier artifacts where already tracked.

Potentially local/protected/regenerated:

- Run12 teacher checkpoint/companions depending on checkout/acquisition;
- large physical DATA representations;
- proving key/SRS/runtime proof workspaces;
- RAG indexes and runtime databases;
- secrets, RPC credentials, signing keys.

## Failure modes and current limitations

### DATA/ML

- no trustworthy class-specific confirmed-negative population in policy v1;
- two classes supervision-disabled;
- model-selection is positive-only limited;
- threshold/calibration/untouched acceptance unsupported;
- Phase 7 still requires local physical representation binding;
- no repaired teacher retrain/promoted checkpoint yet.

### ZK/V3

- retained proof scope is proxy-only;
- `UNSAFE` EZKL check mode remains;
- policy signature authenticates context but does not expand the circuit;
- no production signing/broadcast service is claimed;
- owner/verifier/signer rotation remains governance/security trust.

### Runtime/feedback

- gateway audit completion is off-chain only;
- audit MCP is read-only;
- V3 feedback promotion remains disabled pending measured policy;
- a versioned/on-chain record is not automatically vulnerability ground truth.

### Test evidence

The July 13 full-module counts previously published in this page are **superseded current-status evidence**. Do not cite them as the present project state. The R4/V3 work has its own green gate/targeted CI evidence; a fresh whole-repository suite census should be recorded only after rerunning it against a named current post-merge commit and environment.

## Common change recipe

To update this page:

1. record exact commit/date and which branch is canonical versus candidate;
2. update only evidence actually rerun/verified;
3. keep gate states, artifact hashes, role limitations, and local-only blockers explicit;
4. do not carry forward old test counts automatically;
5. if a previously unsupported role becomes supported, name the new evidence/decision/version that changed it;
6. run handbook static validation and relevant R4 regression gates.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

For Phase 7, use the active branch’s `p7_run_local_gate.py` command when the local representation root is available. Full module/live suites are separate evidence and should be recorded here only after an intentional current rerun.

## Optional deep references

- [R4 plan status matrix](../plan/ml-R4/PLAN_STATUS_MATRIX.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)
- [R4 risk/blocker register](../plan/ml-R4/RISK_AND_BLOCKER_REGISTER.md)
- [DATA artifacts](04_data_artifacts.md)
- [Runtime flows](02_runtime_flows.md)
- [Security and trust](12_security_and_trust.md)

## Technical mastery layer

### Prerequisite knowledge

Know commit binding, artifact hashes, gate-based development, historical-versus-current evidence, partial-label semantics, role isolation, and local/protected artifact boundaries.

### Source map and reading order

Read R4 status/decision/risk registers, current source for V3/audit MCP, this page, then the relevant subsystem. For Phase 7, inspect its branch and gate scripts rather than assuming candidate code is already canonical main.

### Execution trace and worked example

Today a correct statement is: “R4 G6 is canonical; Phase 7 remote semantics are green but local representation binding is pending; Run12 is historical operational inference; no retrained vNext teacher or untouched-acceptance claim exists.” A statement like “the current model has passed final vNext test/calibration” is false.

### Implementation practice

When a phase/role becomes available, change the machine-readable plan/manifests first, validate, merge, then update this page. Do not make status prose the only place where a project-state transition exists.

### Review and ownership check

Can you distinguish canonical main from Phase-7 candidate state, state all unsupported evaluation roles, identify the current teacher/proof/MCP protocol versions, and name the exact local blocker before G7?
