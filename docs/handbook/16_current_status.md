# 16 — Current status and gap ledger

**Read this when:** you need to know what is canonical, verified, pending, unsupported, historical, or local-only today.

**Skip this if:** never before operational, training, evaluation, or production-readiness decisions.

**Estimated reading time:** 10 minutes.

## 30-second summary

The canonical post-G7 baseline includes DATA vNext implementation merge **`81d9c547d`** (2026-08-12): R4 DATA/ML repair has passed **G0 through G7**, the V3 registry/context protocol and read-only audit-MCP boundary remain implemented, and Run12 remains the historical operational teacher. The v2 semantic overlay is now physically bound to all 21,657 required representations / 64,971 graph-token-sidecar files with zero missing files and zero mismatches. **Phase 8 retraining is READY**, but no repaired teacher checkpoint has been trained or promoted yet.

The evidence limitations remain explicit: no confirmed-negative source exists in policy v1, threshold/calibration roles are unsupported/empty, and untouched acceptance is unsupported/empty/frozen. Historical July suite totals remain historical evidence rather than current-state proof.

## Just-enough mental model

```text
canonical main (91f795885)
  R4 G0–G7 PASS
  V3 registry/read-only observation boundaries
  Run12 historical inference baseline
        ↓
DATA vNext v2
  deterministic semantic overlay PASS
  local representation binding PASS
  G7 PASS
        ↓
Phase 8 retrain existing architecture (READY)
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
| 7 | G7 PASS | v2 implementation merged; 21,657 representations / 64,971 files physically bound with zero mismatches |
| 8 | READY | existing-architecture retraining authorized against the exact G7-passed v2 lineage |
| 9–10 | waiting | evaluation/promotion remain gated by preceding phases |

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

### Phase 7 G7 state

DATA vNext v2 is canonical through implementation merge `81d9c547d` and the G7 closeout records the locally bound publication.

Final G7 evidence:

- manifest status: `VALIDATED_G7_CANDIDATE`;
- contracts: 22,493;
- contract×class rows: 224,930;
- positive targets: 1,007; negative targets: 0;
- STRONG signals: 403; WEAK signals: 604;
- effective loss cells: 852; outcome-metric cells: 118;
- required/checked representation contracts: 21,657/21,657;
- required/checked physical files: 64,971/64,971;
- missing files: 0; representation mismatches: 0;
- physical local path recorded: false;
- representation binding digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`.

The v2 loader rejects silent historical-v1 fallback. Historical v1 artifacts remain immutable. Phase 8 may adapt the frozen training consumer to this exact lineage; it may not invent negatives, rebalance frozen roles, or manufacture unsupported threshold/calibration/acceptance populations.

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
- Phase 7 physical representation binding passed; the remaining DATA/ML limitations are evidence limitations, not G7 implementation blockers;
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

For G7 evidence, use the committed vNext manifest, representation-binding report, and final G7 validation report. The local gate remains available for reproducibility, not because G7 is pending. Full module/live suites are separate evidence and should be recorded here only after an intentional current rerun.

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

Read R4 status/decision/risk registers, current source for V3/audit MCP, this page, then the relevant subsystem. For DATA vNext, inspect the canonical vNext package, G7 manifest/reports, and R4 decisions rather than historical v1 label/export assumptions.

### Execution trace and worked example

Today a correct statement is: “R4 G7 is canonical; DATA vNext v2 is representation-bound; Phase 8 retraining is ready; Run12 remains historical operational inference; no retrained vNext teacher or untouched-acceptance claim exists.” A statement like “the current model has passed final vNext test/calibration” is false.

### Implementation practice

When a phase/role becomes available, change the machine-readable plan/manifests first, validate, merge, then update this page. Do not make status prose the only place where a project-state transition exists.

### Review and ownership check

Can you identify the exact G7 DATA vNext manifest/binding lineage, state all unsupported evaluation roles, distinguish Run12 from the future repaired teacher, and explain what Phase 8 is and is not authorized to change?
