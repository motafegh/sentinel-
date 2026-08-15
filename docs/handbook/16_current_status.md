# 16 — Current status and gap ledger

**Read this when:** you need to know what is canonical, verified, pending, unsupported, historical, or local-only today.

**Skip this if:** never before operational, training, evaluation, or production-readiness decisions.

**Estimated reading time:** 10 minutes.

## 30-second summary

The canonical DATA foundation includes DATA vNext implementation merge **`81d9c547d`** (2026-08-12): R4 DATA/ML repair has passed **G0 through G7**, the V3 registry/context protocol and read-only audit-MCP boundary remain implemented, and Run12 remains the historical operational teacher. The v2 semantic overlay is physically bound to all 21,657 required representations / 64,971 graph-token-sidecar files with zero missing files and zero mismatches.

As of the 2026-08-15 local rebuild, **Phase 8 is IN_PROGRESS with repaired-v2 DATA physically accepted but full training on an evidence/objective hold**. The accepted local publication contains 22,540 contracts and all 67,620 required graph/token/sidecar files pass physical binding. The 20-check repaired-lineage audit and bounded CUDA smoke pass. The fixed-horizon teacher run has **not** been launched: every one of the 899 effective supervised cells is positive, there are no confirmed negatives, and the four-window control has measured target-code omissions. G8 remains open.

The evidence limitations remain explicit: no confirmed-negative source exists in policy v1, threshold/calibration roles are unsupported/empty, and untouched acceptance is unsupported/empty/frozen. Historical July suite totals remain historical evidence rather than current-state proof.

## Just-enough mental model

```text
canonical main — active Phase-8 execution line
  R4 G0–G7 PASS
  V3 registry/read-only observation boundaries
  Run12 historical inference baseline
        ↓
DATA vNext v2
  deterministic semantic overlay PASS
  local representation binding PASS
  G7 PASS
        ↓
Phase 8 retrain existing architecture (IN_PROGRESS)
  repaired-v2 physical DATA + bounded smoke accepted locally
  full launch held for objective/evaluation + token-selector evidence
  full 100-epoch repaired run not yet launched
        ↓
G8 checkpoint/evidence review
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
| 8 | IN_PROGRESS | repaired-v2 physical binding and bounded GPU smoke passed locally; full launch is held because supervision is positive-only and token-selection adequacy remains unresolved; repaired run/G8 checkpoint not complete |
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

The v2 loader rejects silent historical-v1 fallback. Historical v1 artifacts remain immutable. Phase 8 consumes this exact lineage; it may not invent negatives, rebalance frozen roles, or manufacture unsupported threshold/calibration/acceptance populations.

### Repaired-v2 local physical acceptance

The versioned repaired candidate is accepted as a physically complete local
research dataset, separate from immutable historical G7/v1:

- DATA publication: `sentinel-r4-vnext-v2`;
- 22,540 contracts / 225,400 contract-class rows;
- roles: 310 `TRAIN_STRONG`, 10,926 `TRAIN_WEAK`, 11,124
  `TRAIN_UNLABELED`, 105 `MODEL_SELECTION`, 75 `INTERNAL_AUDIT`;
- targets: 1,080 positive, 224,320 unknown, zero confirmed negative;
- 22,540 / 22,540 contracts and 67,620 files physically valid;
- representation binding digest:
  `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- extraction modes: 22,512 full analysis, 26 provenance-visible
  parse-only, two full-analysis graph-only constant-array folds;
- all 20 repaired-lineage acceptance checks pass;
- bounded two-step CUDA smoke passes and writes no checkpoint.

This is DATA acceptance, not model-quality acceptance. There are 899 effective
loss cells and every one targets `1`; 21,641 published contracts have no active
loss cell. The current objective can measure positive fitting but cannot learn
or evaluate false-positive discrimination.

Token evidence is also material: 19,451 / 22,540 contracts exceed four
pre-subsampling windows. Across all 11,341 optimizer/model-selection role
records, a target-aware bounded candidate raises median target-contract
coverage from 0.2760 to 0.5119, but regresses 342 records and is not promoted.
See the [repaired-DATA acceptance and launch decision](../plan/ml-R4/runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md).

### Current ML state

Run12 remains the historical operational checkpoint and comparison baseline. It was trained using the pre-R4 binary target semantics. Therefore:

- current Run12 inference remains usable for runtime continuity/historical comparison;
- Run12 weights are not the repaired DATA-vNext model;
- Run12 learned weights, optimizer/scheduler state, thresholds, and calibration are not inputs to the repaired Phase-8 training run;
- no Phase-8 repaired teacher checkpoint exists yet.

Architecture remains frozen through the initial repaired retrain so R4 can measure the effect of data/label repair before redesigning the model.

The historical Phase-8 execution configuration remains explicit, but its old
population figures are not authority for repaired-v2 and full launch is held:

- architecture/model: `four_eye_v8` / `v8.1`, ten outputs;
- training starts from the accepted pretrained GraphCodeBERT base plus fresh/current Phase-8 trainable components, not Run12 learned weights;
- repaired frozen training roles: 310 `TRAIN_STRONG` + 10,926 `TRAIN_WEAK`;
- repaired optimizer-bearing population: 298 strong + 601 weak = 899 contracts;
- grouped sampler: 831 active leakage groups, one deterministically rotating member per group/epoch;
- repaired MODEL_SELECTION: 105 frozen / 103 active contracts, 61 active groups, positive-only limited support;
- epochs: 100; batch size: 8; gradient accumulation: 8;
- 88 micro-batches and 11 optimizer/scheduler steps per epoch;
- fixed run horizon: 1,100 optimizer steps;
- no F1 early stopping, threshold tuning, calibration fitting, acceptance access, or pseudo-negative construction;
- fixed-horizon `final` checkpoint is the primary G8 completion artifact; `best_positive_nll` is only a limited positive-fit diagnostic.

The accepted pretrained backbone snapshot is `microsoft/graphcodebert-base` revision `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`. Runtime binding records Python, PyTorch/CUDA/cuDNN, Transformers, PEFT, PyTorch Geometric, NumPy, Pandas, and PyArrow versions and fails closed on backbone provenance mismatch.

The 2026-08-14 audit motivated the repaired lineage and remains historical
diagnostic evidence. The completed local rebuild recovered the predicted 71
strong semantic cells, physically bound 883 more contracts than v1, and
measured graph size through a maximum of 16,065 nodes / 166,459 edges. The
remaining launch hold is no longer physical incompleteness; it is the inability
of all-positive supervision to support discrimination claims, plus the
unpromoted token selector.

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
- Phase-8 training run logs/checkpoints under `ml/logs/r4-phase8/`;
- proving key/SRS/runtime proof workspaces;
- RAG indexes and runtime databases;
- secrets, RPC credentials, signing keys.

## Failure modes and current limitations

### DATA/ML

- no trustworthy class-specific confirmed-negative population in policy v1;
- two classes supervision-disabled;
- model-selection is positive-only limited;
- threshold/calibration/untouched acceptance unsupported;
- repaired-v2 physical representation binding and bounded CUDA smoke passed locally; generated artifacts remain local/Git-ignored;
- Phase-8 implementation is technically executable, but full execution is held for a credible objective/evaluation contract and a reviewed token selector; no repaired full-run/final teacher checkpoint exists;
- positive-only supervision may produce broad overprediction; that must be measured as a result, not hidden by invented negatives.

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

For Phase 8 specifically, the dedicated vNext training compatibility workflow now runs on canonical `main`; the main-line compatibility gate and canonical Handbook validation were green before the pre-training handoff was written. Documentation synchronization after that point changes the exact source SHA and must be followed by the short source/runtime preflight before launch.

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

For G7 evidence, use the committed vNext manifest, representation-binding report, and final G7 validation report. The local gate remains available for reproducibility, not because G7 is pending. For the current Phase-8 launch boundary, use the committed pre-training handoff and the Phase-8 compatibility tests/runner controls. Full module/live suites are separate evidence and should be recorded here only after an intentional current rerun.

## Optional deep references

- [R4 plan status matrix](../plan/ml-R4/PLAN_STATUS_MATRIX.md)
- [Phase-8 execution plan](../plan/ml-R4/runs/2026-08-13_PHASE8_existing_model_retraining_plan.md)
- [Phase-8 pre-training launch handoff](../plan/ml-R4/runs/2026-08-14_PHASE8_pretraining_launch_handoff.md)
- [Phase-8 real-data readiness audit](../plan/ml-R4/runs/2026-08-14_PHASE8_real_data_readiness_audit.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)
- [R4 risk/blocker register](../plan/ml-R4/RISK_AND_BLOCKER_REGISTER.md)
- [DATA artifacts](04_data_artifacts.md)
- [Runtime flows](02_runtime_flows.md)
- [Security and trust](12_security_and_trust.md)

## Technical mastery layer

### Prerequisite knowledge

Know commit binding, artifact hashes, gate-based development, historical-versus-current evidence, partial-label semantics, role isolation, and local/protected artifact boundaries.

### Source map and reading order

Read R4 status/decision/risk registers, the Phase-8 pre-training handoff when execution state matters, current source for V3/audit MCP, this page, then the relevant subsystem. For DATA vNext, inspect the canonical vNext package, G7 manifest/reports, and R4 decisions rather than historical v1 label/export assumptions.

### Execution trace and worked example

Today a correct statement is: “R4 G7 remains the valid binding result for DATA vNext v2; Phase 8 is IN_PROGRESS on `main`; its training implementation, micro-smoke, and runtime preflight are validated; the full launch is held because a later live audit found material recoverable supervision omitted by preprocessing; the 100-epoch retrain has not been launched; Run12 remains historical operational inference; no retrained vNext teacher or untouched-acceptance claim exists.” A statement like “the current model has passed final vNext test/calibration” is false.

### Implementation practice

When a phase/role becomes available, change the machine-readable plan/manifests first, validate, merge, then update this page. Do not make status prose the only place where a project-state transition exists.

### Review and ownership check

Can you identify the exact G7 DATA vNext manifest/binding lineage, state all unsupported evaluation roles, distinguish Run12 from the future repaired teacher, explain why Phase 8 keeps the architecture frozen, and state exactly what remains unproven until the full training run executes?
