# 16 — Current status and gap ledger

**Read this when:** you need the canonical current project state before DATA/ML execution, evaluation, promotion, or training decisions.

**Skip this if:** you only need a historical implementation walkthrough.

**Estimated reading time:** 8 minutes

## 30-second summary

SENTINEL historical R4 **G0–G7 remain PASSED** and immutable. **Phase 8 is IN_PROGRESS; G8 is open.** Run12 is the historical operational ML baseline, not current repaired training truth.

R4-D-008 physically accepts repaired-v2 DATA: **22,540 contracts**, **225,400 contract×class rows**, **67,620 graph/token/sidecar files**, and physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

R4-D-009 accepts corrected logical V3 after V2 address-literal grouping produced a 10,327-contract DIVE component. Accepted V3 contains **22,394 groups**, maximum group size **7**, **146 normalized-code edges**, and **zero address-authority edges**, while preserving repaired-v2 semantic counts and the exact physical binding digest.

A later protected-local audit found five post-acceptance research/reporting defects. Repository hardening fixed them, the affected acceptance/sensitivity/selector/queue/CUDA reports were regenerated under source commit `83bd566b9c4f4f653e530c2c0f5c990858dd759d`, and the final Git-safe V3 evidence snapshot completed with `coherence=PASS` and verified SHA-256 checksums. The durable snapshot was committed at `44fbb9c1d2033be8002fe404d650cf09f08b0f29`.

There are still **zero confirmed-negative examples**. R4-GAP-007 has moved from pilot-ready to **pilot in progress** using the hardened 200-cell / 200-group queue. Candidate #1 (`CallToUnknown`) has only a partial primary review and remains `UNKNOWN` / `PENDING_REVIEW` with target `None`; no independent verification or negative verdict exists. Threshold fitting, calibration, untouched acceptance, selector promotion, and the 100-epoch Phase-8 run remain unauthorized.

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
post-acceptance evidence audit
  reporting + lineage/invariant defects found
        ↓
repository hardening
  IMPLEMENTED
        ↓
protected-local regeneration
  acceptance → sensitivity → CPU selector → queue → CUDA
  COMPLETE
        ↓
coherence-gated final snapshot
  PASS / COMMITTED
        ↓
R4-GAP-007 negative pilot
  IN PROGRESS / candidate #1 partial primary review
        ↓
separate selector-promotion decision / objective design
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
| 8 | IN_PROGRESS | repaired-v2 physical DATA and logical V3 accepted; evidence hardening/regeneration closed with a coherent committed snapshot; R4-GAP-007 pilot started and candidate #1 is under partial primary review; full training unauthorized |
| 9–10 | WAITING | evaluation/calibration/promotion remain gated by G8 and missing evidence |

### Historical G7 validation anchors

The handbook intentionally preserves the historical G7 machine-validation anchors because later Phase-8 lineages must not erase them:

- historical G7 implementation merge: `81d9c547d`;
- historical tracked publication status: `VALIDATED_G7_CANDIDATE`;
- historical G7 representation binding digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`;
- historical G7 contracts / representation files: 22,493 / 64,971.

These anchors are historical reproducibility facts, not current Phase-8 DATA/ML authority.

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
- explicit-family edges in accepted population: 0;
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

The hardened acceptance rerun durably establishes:

- active `MODEL_SELECTION`: **71 contracts / 71 groups**;
- active `INTERNAL_AUDIT`: **72 contracts / 71 groups**;
- combined outcome-metric population: **143 contracts / 142 unique groups**;
- total outcome-metric cells recorded by the V3 publication: **143**.

`143/142` is therefore not a MODEL_SELECTION population count. The ML adapter loads `MODEL_SELECTION` only for model selection and does not load `INTERNAL_AUDIT`; the old defect was reporting/research-population contamination, not trainer leakage.

### Evidence-hardening closeout

The audit found five issues:

1. MODEL_SELECTION and INTERNAL_AUDIT were conflated in reporting.
2. Final snapshotting did not prove cross-report coherence.
3. Sensitivity lacked immutable lineage metadata.
4. Negative-review queue group uniqueness was observed but not guaranteed across classes.
5. Explicit source-native family IDs were not source-namespaced.

All five are now hardened in source/tests.

The protected-local evidence tranche was regenerated from source commit:

`83bd566b9c4f4f653e530c2c0f5c990858dd759d`

Final durable snapshot:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Snapshot commit:

`44fbb9c1d2033be8002fe404d650cf09f08b0f29`

Snapshot result:

- `coherence=PASS`;
- all JSON files in `SHA256SUMS.txt` verified `OK` before commit;
- no physical repaired-v2 rebuild;
- no Run12 state;
- no selector promotion;
- no training authorization.

### Hardened selector evidence

CPU selector rerun:

- records analyzed: 1,018;
- over four windows: 737;
- guarded improved: 476;
- equal: 261;
- fallback: 261;
- regressed: 0;
- failures: 0.

CUDA rerun:

- status: `LOGICAL_V3_BOUNDED_RESEARCH_COMPLETE`;
- identical initialization: true;
- 4/4 required worst-case probes completed;
- no Run12 weights;
- no checkpoint;
- `selector_promotion_authorized=false`;
- `full_training_authorized=false`.

This is durable mechanical/coverage evidence. It still cannot establish false-positive discrimination because the relevant supervised/model-selection evidence remains positive-only.

Before any promotion ADR, verify across the relevant full population that the historical control selector exactly reproduces the currently bound representation token tensors.

### Confirmed-negative state

Current source authority still contains zero confirmed negatives.

R4-GAP-007 authorizes class-specific dual-review negative evaluation work. The committed hardened queue contains:

- 200 `PENDING_REVIEW` cells;
- 25 candidates for each of eight enabled classes;
- 200 globally unique leakage groups;
- `group_uniqueness_scope=GLOBAL_ACROSS_ENABLED_CLASSES`;
- all target `None`;
- all roles `TRAIN_UNLABELED`;
- `negative_truth_claim=false`.

The pilot has now started with deterministic candidate #1 for `CallToUnknown`. Partial primary review found a typed callback to a caller-supplied `_spender` (`spender.receiveApproval(...)`) and a legacy Solidity Ether transfer (`msg.sender.transfer(...)`). Neither interaction establishes positive or negative class truth by itself. Candidate #1 remains `UNKNOWN` / `PENDING_REVIEW`; complete source/representation review and any genuinely independent verification are still pending.

Queue membership remains review reservation only. Never infer negative truth from historical zero, unlabeled state, source silence, static-tool silence, or queue membership. Any accepted negative is initially `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY` and requires class-specific primary review plus independent agreeing verification.

## Interfaces, data shapes, and configuration

### Current DATA/ML authority stack

| Surface | Current authority |
|---|---|
| semantic policy | `data-vnext-policy-v1` |
| historical gate baseline | G0–G7 / `sentinel-r4-vnext-v1` |
| physical DATA root | R4-D-008 / repaired-v2 |
| logical grouping/role authority | R4-D-009 / accepted V3 |
| durable research evidence | coherent snapshot `44fbb9c1d...` under `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/` |
| current execution restart | `docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate1_local_handoff.md` |
| confirmed-negative review | R4-GAP-007 / pilot IN_PROGRESS / candidate #1 partial primary review |
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

Any future replacement/recomputed V3 report must still fail closed if acceptance, sensitivity, queue, selector or GPU evidence references a different publication manifest, physical binding, source commit, version, or required dependency. Never hand-edit reports to make coherence pass.

### Supervision / evaluation

- zero confirmed-negative rows;
- no trustworthy general specificity/FPR/F1 claim;
- threshold and calibration fitting unavailable;
- untouched acceptance unavailable.

### Representation / selector

- `[4,512]` capacity omits material code on long contracts;
- guarded target-aware selection remains promising but unpromoted;
- full-population bound-token equivalence remains required before promotion.

### Training

- no repaired full-run checkpoint exists;
- Run12 state is historical only;
- no current full-run horizon is authorized;
- no F1 early stopping, pseudo-negatives, threshold fitting or calibration under current evidence.

## Common change recipe

For the current Phase-8 boundary:

1. synchronize local `main`;
2. read `PLAN_STATUS_MATRIX.md`, the 2026-08-16 hardened snapshot closeout, and `runs/2026-08-21_PHASE8_gap007_candidate1_local_handoff.md`;
3. preserve repaired-v2 physical roots, accepted V3 publication/grouping, and the committed coherent evidence snapshot;
4. continue candidate #1 complete primary review from the committed hardened queue only;
5. keep every candidate UNKNOWN/PENDING_REVIEW until class-specific primary review plus independent agreeing verification establishes negative evidence;
6. keep any accepted negative evaluation-only unless later policy explicitly grants optimizer authority;
7. separately design/execute the full-population control-selector → bound-token equivalence check before any guarded-selector promotion ADR;
8. revisit objective/evaluation/training authorization, including any PU-learning decision, only after new evidence supports it.

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

## Optional deep references

- [R4 plan status matrix](../plan/ml-R4/PLAN_STATUS_MATRIX.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)
- [R4 evidence gap register](../plan/ml-R4/EVIDENCE_GAP_REGISTER.md)
- [ADR-R4-008 repaired-v2 physical DATA acceptance](../plan/ml-R4/adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md)
- [ADR-R4-009 logical V3 grouping correction](../plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md)
- [Current GAP-007 candidate #1 local handoff](../plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate1_local_handoff.md)
- [Hardened V3 evidence snapshot closeout](../plan/ml-R4/runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md)
- [Completed V3 evidence-hardening procedure](../plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md)
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

To own this phase technically, understand multi-label classification and nullable/masked targets; positive-only versus confirmed-negative supervision; leakage groups and group-atomic splits; union-find grouping; provenance versus heuristic correlation; deterministic dataset/hash binding; train vs MODEL_SELECTION vs INTERNAL_AUDIT roles; token window selection; and why a CUDA smoke/comparison proves mechanics rather than model discrimination.

PU-learning theory remains deferred until negative-evaluation evidence and objective design are ready.

## Source map and reading order

Current reading order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
2. `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`
3. `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md`
4. `docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate1_local_handoff.md`
5. `docs/plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md`
6. `docs/plan/ml-R4/CLAIM_STATUS_MATRIX.md`
7. `data_module/sentinel_data/vnext/confirmed_negative_evaluation.py`
8. `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/snapshot_coherence_v1.json`
9. `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/confirmed_negative_review_queue_v1.json`
10. `ml/src/data_extraction/bounded_window_selector.py`

## Execution trace and worked example

Two unrelated contracts may both contain WETH or a router address. Under V3 that overlap is diagnostic only and creates no group edge. Two contracts with identical normalized code may group globally. Two contracts in the same source with `project_id=7` may group using `source:project_id:7`; another source using `project_id=7` does not group through that local identifier alone.

Similarly, `outcome_metric_mask=true` does not itself mean “MODEL_SELECTION”: the role must be inspected. `INTERNAL_AUDIT` is a separate population.

## Implementation practice

Useful regression fixtures include:

- shared common Ethereum address → separate groups;
- same normalized code across sources → same group;
- same explicit family ID within one source → same group;
- same explicit family ID across different sources → separate groups;
- queue candidates across multiple classes → globally distinct groups;
- snapshot inputs with stale manifest/source commit → rejected;
- snapshot inputs with incomplete GPU probes → rejected.

## Review and ownership check

Before proceeding, be able to answer:

- Why does physical repaired-v2 acceptance survive logical/reporting corrections?
- Why is 143/142 not a MODEL_SELECTION count?
- Why does the trainer remain leakage-safe from INTERNAL_AUDIT despite the old reporting bug?
- Why must explicit source-native IDs be source-namespaced?
- Why must the queue reserve groups globally, not only per class?
- What exactly did `coherence=PASS` prove, and what did it not prove?
- Why can positive-only selector CUDA evidence not establish false-positive discrimination?
- What evidence is still missing before G8 can authorize full training?

If any answer is unclear, use the 2026-08-16 hardened V3 evidence snapshot closeout for the accepted pre-pilot baseline and the 2026-08-21 GAP-007 candidate #1 handoff for the current execution point rather than guessing from historical pre-hardening reports.