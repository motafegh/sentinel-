# 16 — Current status and gap ledger

**Read this when:** you need the canonical current project state before DATA/ML execution, evaluation, promotion, or training decisions.

**Skip this if:** you only need a historical implementation walkthrough.

**Estimated reading time:** 8 minutes

## 30-second summary

SENTINEL historical R4 **G0–G7 remain PASSED** and immutable. **Phase 8 is IN_PROGRESS; G8 is open.** Run12 is the historical operational ML baseline, not current repaired training truth.

R4-D-008 physically accepts repaired-v2 DATA: **22,540 contracts**, **225,400 contract×class rows**, **67,620 graph/token/sidecar files**, and physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

R4-D-009 accepts corrected logical V3 after V2 address-literal grouping produced a 10,327-contract DIVE component. Accepted V3 contains **22,394 groups**, maximum group size **7**, **146 normalized-code edges**, and **zero address-authority edges**, while preserving repaired-v2 semantic counts and the exact physical binding digest.

A later protected-local audit found five post-acceptance research/reporting defects. Repository hardening fixed them, the affected acceptance/sensitivity/selector/queue/CUDA reports were regenerated under source commit `83bd566b9c4f4f653e530c2c0f5c990858dd759d`, and the final Git-safe V3 evidence snapshot completed with `coherence=PASS` and verified SHA-256 checksums. The durable snapshot was committed at `44fbb9c1d2033be8002fe404d650cf09f08b0f29`.

There are still **zero confirmed-negative examples**. R4-GAP-007 is **pilot in progress** using the hardened 200-cell / 200-group queue. Candidate #1 (`CallToUnknown`) is `NOT_CONFIRMED`. Candidate #2's source-first primary review supports `CONFIRMED_NEGATIVE`, but it remains `UNKNOWN` / `PENDING_REVIEW` with target `None` until a genuinely independent reviewer agrees.

Candidate #2 also exposed a real v9 representation defect: all 30 of its type-11 `EXTERNAL_CALL` edges describe same-file `SafeMath` library calls, while its actual Ether `transfer` has no type-11 edge. The R4-GAP-008 audit reproduced this semantic mismatch across all 22,540 graphs. R4-D-010 therefore preserves v9 as immutable historical/reproducibility evidence but makes it ineligible for the new full run. The later V2.6 lineage passes all 22,540 mechanics and reconciles its exact 355-identity drift population as 349 persistent-storage WRITE corrections plus 6 exact index-equivalent graphs. R4-D-011 physically accepts only the exact protected-local V2.6 root and digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`. Threshold fitting, calibration, untouched acceptance, selector promotion, and the 100-epoch run remain unauthorized.

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
  IN PROGRESS / candidate #1 NOT_CONFIRMED
  candidate #2 primary supports negative / independent review pending
        ↓
R4-GAP-008 v10 representation remediation
  POLICY + FULL-POPULATION PROOF COMPLETE
        ↓
versioned V2.6 protected-local candidate
  PHYSICALLY ACCEPTED / 22,540 / DIGEST d9f925...
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
| 8 | IN_PROGRESS | repaired-v2 physical DATA and logical V3 remain accepted evidence; exact V2.6 physical representation is accepted under R4-D-011; candidate #1 is `NOT_CONFIRMED`; candidate #2 primary supports a negative pending independent review; full training unauthorized |
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

This acceptance proves physical integrity and reproducibility of those exact v9 bytes. R4-D-010 now explicitly prevents using them as the graph lineage for a new full training run; it does not erase the earlier acceptance evidence.

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

The pilot started with deterministic candidate #1 for `CallToUnknown`. Complete whole-source and bound-representation primary review found no raw/unchecked low-level call, and Solidity 0.4.18 semantics plus targeted Slither analysis corroborated that the legacy `msg.sender.transfer(...)` reverts on failure. However, the typed callback to caller-supplied `_spender` (`spender.receiveApproval(...)`) is a real call to an unverified external target, which conflicts with the governing taxonomy's broader class definition. Candidate #1 was therefore adjudicated `NOT_CONFIRMED` and remains `UNKNOWN` / `PENDING_REVIEW`, target `None`. Validation passed with zero accepted negatives and no new authorization. Independent verification is required only for a primary `CONFIRMED_NEGATIVE`, so none was needed here.

Candidate #2 for `CallToUnknown` (`r4neg-bfe90ef82e33a324d612256a5d4053c6`) then received complete whole-source, graph, token-window, sidecar, and targeted Slither primary review. Its only actual external interaction is `_customerAddress.transfer(_dividends)` after state update; no raw low-level call, send, typed callback, assembly, creation, or self-destruct was found. The primary result therefore supports `CONFIRMED_NEGATIVE` for this class, but it is not accepted truth yet. The primary reviewer must not self-verify; the deterministic blind source bundle under `docs/plan/ml-R4/review_bundles/` is the handoff for a genuinely independent reviewer.

Queue membership remains review reservation only. Never infer negative truth from historical zero, unlabeled state, source silence, static-tool silence, or queue membership. Any accepted negative is initially `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY` and requires class-specific primary review plus independent agreeing verification.

### Representation semantic integrity

Candidate #2's bound v9 graph contains 30 type-11 edges and all 30 are provable same-file `SafeMath` library calls. Its actual `Transfer` IR is present in graph metadata but does not receive type 11. The read-only R4-GAP-008 audit then found across all 22,540 repaired-v2 graphs:

- 217,490 total type-11 edges;
- at least 11,702 provable same-file declared-library type-11 edges in 1,489 graphs;
- 7,057 / 13,413 raw-low-level nodes with type 11;
- 40 / 4,215 send nodes with type 11;
- 6,557 / 80,927 transfer nodes with type 11;
- 9,013 / 13,025 transfer-containing graphs without a transfer-linked type-11 edge;
- 12,653 graphs retaining less than 50% of unique code tokens.

The library classifier is conservative and can undercount imported, aliased, or `using for` libraries. Transfer/send name matching can include token-interface methods. These counts are therefore representation diagnostics and proven lower bounds, not vulnerability labels or a complete false-positive rate.

R4-D-010 requires graph schema v10 with distinct typed-high-level,
raw-low-level, `Transfer`, `Send`, `LibraryCall`, and contract-creation kinds.
The accepted extractor is
`v2.6-r4-call-semantics-deterministic-cfg-mutators`. Its full protected-local
candidate passes generation, heterogeneous staging, exception fill, binding,
exact accepted-v9 token-byte identity, and runtime reconciliation for all
22,540 identities. Three fresh generations and evidence passes cover its exact
355-identity drift census; V4 independently re-proves 349 persistent-storage
WRITE corrections plus 6 index-equivalent graphs with zero unexplained drift.
R4-D-011 accepts only the recorded root/digest. Regeneration or training is not
the current authority.

## Interfaces, data shapes, and configuration

### Current DATA/ML authority stack

| Surface | Current authority |
|---|---|
| semantic policy | `data-vnext-policy-v1` |
| historical gate baseline | G0–G7 / `sentinel-r4-vnext-v1` |
| physical DATA root | R4-D-008 / repaired-v2; immutable reproduction evidence, not new-full-training eligible |
| logical grouping/role authority | R4-D-009 / accepted V3 |
| durable research evidence | coherent snapshot `44fbb9c1d...` under `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/` |
| current execution restart | `runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`; candidate #2 review is a separate pending-independent track |
| confirmed-negative review | R4-GAP-007 / candidate #1 `NOT_CONFIRMED` / candidate #2 primary-support only, independent pending |
| physical representation for a future authorized run | R4-D-011 exact v10 / extractor `v2.6-r4-call-semantics-deterministic-cfg-mutators` / protected-local `representations-r4-v3-candidate`; digest `d9f925...`; physically accepted, not training-authorized |
| selector | historical control remains bound; guarded candidate unpromoted |
| training authorization | HOLD / none |

### Stable accepted-v9 shapes

- vulnerability classes: 10;
- graph schema: `v9`;
- node feature dimension: 12;
- node types: 14;
- edge types: 12;
- model token input: `[4,512]`;
- model outputs: 10;
- architecture: `four_eye_v8` / `v8.1`.

These describe accepted v9 artifacts. V10 has 17 exact edge kinds. The exact
V2.6 candidate passes all mechanics for 22,540 identities and is physically
accepted under R4-D-011 after complete 355-identity structural reconciliation.

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

- v9 type-11 edges conflate library calls and omit most Transfer/Send nodes; v9 is prohibited for the new full run;
- exact V2.6 protected-local physical acceptance is complete under R4-D-011; it grants no selector or training authority;
- `[4,512]` capacity omits material code on long contracts;
- guarded target-aware selection remains promising but unpromoted;
- full-population historical-control → R4-D-011 bound-token equivalence passes 22,540/22,540; guarded promotion still requires a separate source-first decision.

### Training

- no repaired full-run checkpoint exists;
- Run12 state is historical only;
- no current full-run horizon is authorized;
- no F1 early stopping, pseudo-negatives, threshold fitting or calibration under current evidence.

## Common change recipe

For the current Phase-8 boundary:

1. synchronize local `main`;
2. read `PLAN_STATUS_MATRIX.md`, the current restart checkpoint, the 2026-08-30 full-population evidence plan, ADR-R4-010, candidate #2 primary review, and the hardened snapshot closeout;
3. preserve v9/repaired-v2 physical roots, accepted V3 publication/grouping, the committed coherent snapshot, and the passed protected-local V2.5 Stage A-D candidate;
4. build duplicate-safe repeated semantic evidence for all 311 raw drift identities; never patch v9 or mutate the protected candidate;
5. have a genuinely independent reviewer evaluate candidate #2 from the blind bundle; keep target `None` unless dual review agrees;
6. run a new versioned full transition audit only after the evidence validator accounts for every drift with zero duplicate ambiguity and zero unexplained change;
7. keep any accepted negative evaluation-only unless later policy explicitly grants optimizer authority;
8. separately execute control-selector → bound-token equivalence before any guarded-selector promotion ADR;
9. revisit objective/evaluation/training authorization, including PU learning, only after new evidence supports it.

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
- [ADR-R4-010 versioned external-call representation correction](../plan/ml-R4/adrs/ADR-R4-010-versioned-external-call-representation-correction.md)
- [Current GAP-008 external-call semantics audit](../plan/ml-R4/runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md)
- [V10 implementation and bounded-local regression](../plan/ml-R4/runs/2026-08-21_PHASE8_v10_implementation_and_local_regression.md)
- [V10 repository/local implementation handoff](../plan/ml-R4/runs/2026-08-21_PHASE8_v10_external_call_implementation_handoff.md)
- [Current GAP-007 candidate #2 primary review](../plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md)
- [Current GAP-007 candidate #1 primary-review closeout](../plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate1_primary_review.md)
- [Prior GAP-007 candidate #1 local handoff](../plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate1_local_handoff.md)
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
4. `docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`
5. `docs/plan/ml-R4/adrs/ADR-R4-010-versioned-external-call-representation-correction.md`
6. `docs/plan/ml-R4/runs/2026-08-21_PHASE8_v10_implementation_and_local_regression.md`
7. `docs/plan/ml-R4/runs/2026-08-21_PHASE8_v10_external_call_implementation_handoff.md`
8. `docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`
9. `docs/plan/ml-R4/CLAIM_STATUS_MATRIX.md`
10. `data_module/sentinel_data/representation/graph_extractor.py`
11. `data_module/sentinel_data/verification/semantic_checker.py`
12. `data_module/sentinel_data/vnext/confirmed_negative_evaluation.py`
13. `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/snapshot_coherence_v1.json`
14. `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/confirmed_negative_review_queue_v1.json`
15. `ml/src/data_extraction/bounded_window_selector.py`

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
- Why can v9 remain valid historical evidence while being ineligible for a new training run?

If any answer is unclear, use the 2026-08-16 hardened V3 evidence snapshot closeout for the accepted pre-pilot baseline, then the 2026-09-02 V2.6 physical-acceptance/no-launch record and ADR-R4-011 for the current execution point. Use candidate #2's primary-review record only for its pending-independent review track.
