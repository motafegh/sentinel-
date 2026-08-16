# 16 — Current status and gap ledger

**Read this when:** you need the canonical current project state before DATA/ML execution, evaluation, promotion, or training decisions.

**Skip this if:** you only need a historical implementation walkthrough.

**Estimated reading time:** 8 minutes

## 30-second summary

SENTINEL historical R4 **G0–G7 remain PASSED** and immutable. **Phase 8 is IN_PROGRESS; G8 is open.** Run12 is the historical operational ML baseline, not current repaired training truth.

R4-D-008 physically accepts repaired-v2 DATA: **22,540 contracts**, **225,400 contract×class rows**, **67,620 graph/token/sidecar files**, and physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

R4-D-009 accepts corrected logical V3 after V2 address-literal grouping produced a 10,327-contract DIVE component. Accepted V3 contains **22,394 groups**, maximum group size **7**, **146 normalized-code edges**, and **zero address-authority edges**, while preserving the repaired-v2 semantic counts and exact physical binding digest.

A later protected-local audit found five post-acceptance research/reporting defects: combined `MODEL_SELECTION` + `INTERNAL_AUDIT` outcome metrics had been mislabeled as model selection; final snapshotting lacked cross-report coherence checks; sensitivity lacked immutable lineage; negative-review queue uniqueness was not enforced across classes; and explicit source-native family IDs were not source-namespaced. These findings do **not** reverse physical-v2 or logical-V3 acceptance. Repository hardening is implemented, but the affected local reports must now be regenerated before a final durable V3 snapshot.

There are still **zero confirmed-negative examples**. R4-GAP-007 remains approved, but do not adjudicate the pre-hardening V3 queue. Threshold fitting, calibration, untouched acceptance, selector promotion, and the 100-epoch Phase-8 run remain unauthorized.

## Just-enough mental model

```text
historical G0–G7
  PASSED / immutable
        ↓
repaired-v2 physical DATA
  ACCEPTED / digest 16dd4a3f...
        ↓
V2 grouping defect
  10,327-contract address-connected component
        ↓
logical V3
  ACCEPTED / 22,394 groups / max 7 / address edges 0
        ↓
post-acceptance research audit
  reporting + lineage/invariant defects found
        ↓
repository hardening
  IMPLEMENTED
        ↓
LOCAL REGENERATION REQUIRED
  acceptance → sensitivity → CPU selector → queue → CUDA
        ↓
coherence-gated final snapshot
        ↓
negative adjudication / separate selector promotion decision
        ↓
possible later G8 training authorization
```

The key distinction is:

```text
valid physical DATA
≠ valid leakage split
≠ coherent research evidence
≠ sufficient supervision
≠ trustworthy model quality
```

## Actual runtime/source walkthrough

### Canonical R4 gate state

| Phase | State | Current meaning |
|---:|---|---|
| 0–7 | PASSED | historical G0–G7 remain immutable/reproducible |
| 8 | IN_PROGRESS | repaired-v2 physical DATA and logical V3 accepted; V3 research code hardened; protected-local research reports require coherent regeneration; full training unauthorized |
| 9–10 | WAITING | evaluation/calibration/promotion remain gated by G8 and missing evidence |

### Repaired-v2 physical acceptance

Current physical root:

- preprocessing: `sentinel-preprocessed-r4-v2`;
- provenance: `r4-provenance-v1`;
- role-independent ledger: `evidence-ledger-r4-v2`;
- representations: `representations-r4-v2`;
- extractor: `v2.2-r4-repaired`;
- graph schema: `v9`;
- token tensor: `[4,512]`;
- contracts: 22,540;
- contract×class rows: 225,400;
- representation files: 67,620;
- positive / unknown / confirmed-negative targets: 1,080 / 224,320 / 0;
- STRONG / WEAK semantic cells: 474 / 606;
- physical binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

Physical compatibility provenance remains explicit: 22,512 full Slither analyses, 26 parse-only recoveries, and two graph-only constant-array compatibility transformations.

### Why V2 logical grouping is superseded

V2 grouped contracts when arbitrary same-source Ethereum address literals matched. Full-population audit showed:

- V2 groups: 11,551;
- largest group: 10,327 contracts;
- 18,213 address-derived edges in that component;
- common router, dead, zero and WETH addresses connected unrelated artifacts.

Therefore address coincidence is diagnostic correlation, not family identity.

### Accepted logical V3

Accepted identifiers:

- grouping: `r4-leakage-groups-v3`;
- partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

Accepted population:

- groups: 22,394;
- maximum group size: 7;
- normalized-code edges: 146;
- explicit-family edges in the accepted population: 0;
- address-authority edges: 0;
- physical binding unchanged.

Current hardened grouping authority:

- exact artifact identity: global;
- normalized-code identity: global;
- explicit source-native family/project ID: **source-namespaced** as `<source>:<field>:<value>`;
- Ethereum address literal: diagnostic-only.

Because the accepted V3 population has zero explicit-family edges, source-namespacing hardening does not alter its accepted group population.

### Corrected role/outcome terminology

Frozen V3 role contract counts remain:

- `TRAIN_STRONG`: 343;
- `TRAIN_WEAK`: 602;
- `TRAIN_UNLABELED`: 21,449;
- `MODEL_SELECTION`: 73;
- `INTERNAL_AUDIT`: 73.

Optimizer-active supervision remains 932 positive-only contracts/cells across 932 groups.

A previous acceptance report incorrectly called every metric-active row “model selection.” In V3, outcome metric masks intentionally exist for both `MODEL_SELECTION` and `INTERNAL_AUDIT`.

Protected-local audit observed:

- active `MODEL_SELECTION`: **71 contracts / 71 groups**;
- active `INTERNAL_AUDIT`: **72 contracts / 71 groups**;
- combined outcome-metric/audit population: **143 contracts / 142 unique groups**.

Those active counts must be reproduced by the hardened acceptance rerun before final snapshotting. The ML adapter itself loads `MODEL_SELECTION` only for model selection and does not load `INTERNAL_AUDIT`; this was reporting/research-population contamination, not trainer leakage.

### Evidence hardening now implemented

Repository code now enforces:

- acceptance reports separate model-selection and internal-audit populations and record manifest/binding/source lineage;
- representation sensitivity records dataset/grouping/partition versions, publication manifest SHA, binding digest and source commit;
- sensitivity separates `MODEL_SELECTION` from `INTERNAL_AUDIT`; selector worst-case GPU candidates use optimizer-active or actual MODEL_SELECTION rows only;
- CPU selector reports are publication/binding/source-lineage bound;
- confirmed-negative queue groups are globally unique across enabled classes and queue generation fails closed if balanced uniqueness cannot be satisfied;
- explicit family IDs are source-namespaced;
- CUDA selector comparison rejects a sensitivity report from another manifest/binding/source commit and records the sensitivity-report SHA;
- final snapshotting validates cross-report coherence **before** copying evidence and writes `snapshot_coherence_v1.json` only on PASS.

### Pre-hardening research observations

The earlier V3 research remains useful as historical observations, but must be reproduced under the hardened source before becoming final durable evidence.

Earlier CPU selector observation:

- records analyzed: 1,018;
- over four windows: 737;
- guarded improved: 476;
- equal/fallback: 261;
- regressed: 0;
- median target coverage: historical ~63.01%, guarded ~87.94%.

Earlier CUDA observation:

- identical initialization: true;
- four train + four model-selection batches per strategy;
- 4/4 worst-case probes completed;
- control vs guarded positive NLL: ~0.6847 vs ~0.6601;
- mean positive probability: ~0.5061 vs ~0.5189;
- peak CUDA allocation: ~967 MB vs ~957 MB;
- no Run12 weights; no checkpoint.

These observations do not promote `target_aware_guarded_v1` and do not establish false-positive discrimination.

### Confirmed-negative state

Current source authority still contains zero confirmed negatives.

R4-GAP-007 authorizes class-specific dual-review negative evaluation work. The pre-hardening queue observed 200 `PENDING_REVIEW` cells across 200 distinct groups, but it must be regenerated because the implementation now **guarantees** global group uniqueness and records stronger lineage.

Queue membership remains review reservation only. Never infer negative truth from:

- historical zero;
- unlabeled state;
- source silence;
- static-tool silence;
- queue membership.

Any accepted negative is initially `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY`.

## Interfaces, data shapes, and configuration

### Current DATA/ML authority stack

| Surface | Current authority |
|---|---|
| semantic policy | `data-vnext-policy-v1` |
| historical gate baseline | G0–G7 / `sentinel-r4-vnext-v1` |
| physical DATA root | R4-D-008 / repaired-v2 |
| logical grouping/role authority | R4-D-009 / accepted V3 |
| current execution restart | `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md` |
| confirmed-negative review | R4-GAP-007, adjudication not started |
| selector | historical control remains bound; guarded candidate unpromoted |
| training authorization | HOLD / none |

### Stable shapes

- vulnerability classes: 10;
- graph schema: `v9`;
- node feature dimension: 12;
- node types: 14;
- edge types: 12;
- model token input: `[4,512]`;
- model outputs: 10;
- architecture: `four_eye_v8` / `v8.1`.

### Unsupported evaluation roles

```text
THRESHOLD_FIT        = UNSUPPORTED_EMPTY
CALIBRATION_FIT      = UNSUPPORTED_EMPTY
UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN
confirmed negatives  = 0
```

## Failure modes and current limitations

### DATA / grouping

- repaired-v2 physical artifacts are accepted;
- V2 grouping/roles are superseded;
- accepted V3 remains valid;
- explicit family identifiers must remain source-namespaced;
- arbitrary address literals must never regain grouping authority.

### Evidence coherence

The final V3 snapshot must fail if acceptance, sensitivity, queue, selector or GPU evidence references a different publication manifest, physical binding, version, or required dependency. Do not hand-edit reports to make coherence pass.

### Supervision / evaluation

- zero confirmed-negative rows;
- no trustworthy general specificity/FPR/F1 claim;
- threshold and calibration fitting unavailable;
- untouched acceptance unavailable.

### Representation / selector

- `[4,512]` capacity omits material code on long contracts;
- guarded target-aware selection remains promising but unpromoted;
- before promotion, verify across the relevant full population that the historical control selector exactly reproduces the **currently bound token tensors**. Coverage/CUDA evidence alone is insufficient to alter the bound extractor policy.

### Training

- no repaired full-run checkpoint exists;
- Run12 state is historical only;
- no current full-run horizon is authorized;
- no F1 early stopping, pseudo-negatives, threshold fitting or calibration under current evidence.

## Common change recipe

For the current Phase-8 boundary:

1. synchronize local `main`;
2. read `PLAN_STATUS_MATRIX.md` and `runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`;
3. preserve repaired-v2 physical roots and accepted V3 publication/grouping;
4. regenerate corrected V3 acceptance reporting;
5. regenerate lineage-bound representation sensitivity;
6. regenerate lineage-bound CPU selector evidence;
7. regenerate the globally group-unique V3 negative queue;
8. re-run CUDA comparison using the newly bound sensitivity report;
9. run final snapshot helper and require `coherence=PASS`;
10. inspect/commit only sanitized final evidence;
11. then proceed to R4-GAP-007 adjudication and/or a separate selector-promotion decision.

Never fix generated parquet/JSON evidence by hand.

## Verification commands

Repository Phase-8 CI contract:

`.github/workflows/r4-phase8-data-repair.yml`

Canonical handbook validation:

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
python3 docs/handbook/tools/verify_handbook.py inventory
```

Historical G6 validation:

```bash
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Protected local regeneration commands are intentionally centralized in:

`docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`.

## Optional deep references

- [R4 plan status matrix](../plan/ml-R4/PLAN_STATUS_MATRIX.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)
- [R4 evidence gap register](../plan/ml-R4/EVIDENCE_GAP_REGISTER.md)
- [ADR-R4-008 repaired-v2 physical DATA acceptance](../plan/ml-R4/adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md)
- [ADR-R4-009 logical V3 grouping correction](../plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md)
- [V3 evidence-hardening handoff](../plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md)
- [Pre-hardening V3 research checkpoint](../plan/ml-R4/runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md)

## Technical mastery layer

The current work illustrates five distinct concepts:

1. **Physical integrity** — files reconstruct, deserialize and hash-bind correctly.
2. **Semantic supervision integrity** — labels mean only what evidence authorizes.
3. **Leakage partition integrity** — related artifacts stay together without collapsing unrelated projects.
4. **Evidence-lineage integrity** — reports used in one decision all belong to the same versioned publication/binding/source state.
5. **Statistical learning adequacy** — available positive/negative/evaluation evidence supports the quality claims we want to make.

A useful rule:

```text
valid file ≠ valid label
valid label ≠ valid split
valid split ≠ coherent experiment evidence
coherent experiment evidence ≠ sufficient supervision
successful optimization ≠ trustworthy model quality
```

## Prerequisite knowledge

To own this phase technically, understand:

- multi-label classification and nullable/masked targets;
- positive-only versus confirmed-negative supervision;
- leakage groups and group-atomic splits;
- union-find / connected-component grouping;
- provenance versus heuristic correlation;
- deterministic dataset/hash binding;
- train vs MODEL_SELECTION vs INTERNAL_AUDIT roles;
- token truncation/window selection;
- why a CUDA smoke/comparison proves mechanics rather than model discrimination.

PU-learning theory remains deferred until negative-evaluation evidence and objective design are ready.

## Source map and reading order

Current reading order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
2. `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`
3. `docs/plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md`
4. `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md`
5. `data_module/sentinel_data/preprocessing/r4_grouping_v3.py`
6. `docs/plan/ml-R4/scripts/p8_audit_logical_v3_acceptance.py`
7. `docs/plan/ml-R4/scripts/p8_profile_representation_sensitivity.py`
8. `data_module/sentinel_data/vnext/confirmed_negative_evaluation.py`
9. `docs/plan/ml-R4/scripts/p8_snapshot_logical_v3_evidence.py`

## Execution trace and worked example

Two unrelated contracts may both contain WETH or a router address. Under V3 that overlap is diagnostic only and creates no group edge.

Two contracts with identical normalized code may group globally.

Two contracts in the same source with `project_id=7` may group using `source:project_id:7`.

A contract in another source that also uses `project_id=7` does **not** group through that local identifier alone.

Similarly, `outcome_metric_mask=true` does not itself mean “MODEL_SELECTION”: the role must be inspected. `INTERNAL_AUDIT` is a separate population.

## Implementation practice

Useful regression fixtures now include:

- two contracts sharing a common Ethereum address → separate groups;
- same normalized code across sources → same group;
- same explicit family ID within one source → same group;
- same explicit family ID across different sources → separate groups;
- queue candidates across multiple classes → globally distinct groups;
- snapshot inputs with a stale manifest hash → snapshot rejected;
- snapshot inputs with incomplete GPU probes → snapshot rejected.

## Review and ownership check

Before proceeding, be able to answer:

- Why does physical repaired-v2 acceptance survive logical/reporting corrections?
- Why was 143/142 not a MODEL_SELECTION count?
- Why does the trainer remain leakage-safe from INTERNAL_AUDIT despite that reporting bug?
- Why must explicit source-native IDs be source-namespaced?
- Why must the queue reserve groups globally, not only per class?
- Why must final snapshotting validate manifest/binding/report coherence?
- Why can positive-only selector CUDA evidence not establish false-positive discrimination?
- What must be regenerated before the final V3 snapshot?

If any answer is unclear, use the V3 evidence-hardening handoff as the restart point rather than guessing from pre-hardening reports.
