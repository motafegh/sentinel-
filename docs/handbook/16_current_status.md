# 16 — Current status and gap ledger

**Read this when:** you need the canonical current project state, especially before DATA/ML execution, evaluation, or promotion decisions.

**Skip this if:** you only need a historical implementation walkthrough and do not need current authority or gate state.

**Estimated reading time:** 8 minutes

## 30-second summary

SENTINEL historical R4 **G0–G7 remain PASSED** and immutable. The historical G7 implementation merge is `81d9c547d`, and its tracked publication status remains `VALIDATED_G7_CANDIDATE` as a reproducibility anchor. **Phase 8 is IN_PROGRESS; G8 is open.** Run12 remains the historical operational ML baseline and is not current repaired training truth.

R4-D-008 physically accepted the repaired-v2 DATA source/representation layer: **22,540 contracts**, **225,400 contract×class rows**, and **67,620 graph/token/sidecar files**, with zero missing/invalid required representation files and physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

A later full-population research audit found a separate logical defect: `r4-leakage-groups-v2` used arbitrary same-source Ethereum address literals as family/group authority. One DIVE component contained **10,327 contracts**, driven largely by common addresses such as the Uniswap V2 router, dead/zero addresses, WETH, and other ubiquitous constants. R4-D-009 therefore **supersedes V2 grouping/roles for future model research while preserving repaired-v2 physical artifacts as accepted evidence**.

The corrected logical lineage `r4-leakage-groups-v3` → `r4-vnext-roles-v3` → `sentinel-r4-vnext-v3` has now passed protected local generation, same-byte rebinding, and V2→V3 acceptance. V3 contains **22,394 groups**, maximum group size **7**, **146 normalized-code edges**, and **zero address-authority edges** while preserving the exact repaired-v2 physical digest. R4-D-009 is therefore **ACCEPTED**.

Role-dependent research has also been regenerated under V3. Representation sensitivity is stable; the guarded four-window selector improved target coverage on **476 of 737** over-cap records, was equal/fallback on 261, and regressed on **0**. The bounded identical-initialization CUDA comparison completed all **4/4 mandatory worst-case probes** on the RTX 3070 Laptop GPU. This makes the guarded selector evidence-ready for a separate promotion ADR/versioned extractor decision, but it is **not yet promoted**.

There are still **zero confirmed-negative examples**. R4-GAP-007 authorizes a 200-cell V3 pilot review queue (25 candidates per enabled class), but every candidate remains `PENDING_REVIEW`, target `None`, and `negative_truth_claim=false`. Threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty. **The 100-epoch Phase-8 run is not authorized.**

## Just-enough mental model

```text
historical R4 G0–G7
  PASSED / immutable evidence
        ↓
repaired-v2 physical DATA
  22,540 contracts
  67,620 representation files
  physical acceptance PASS
  binding digest 16dd4a3f...
        ↓
V2 grouping/roles audit
  10,327-contract address-connected DIVE component
  arbitrary address coincidence proven too broad
        ↓
R4-D-009 logical V3 correction
  r4-leakage-groups-v3
  r4-vnext-roles-v3
  sentinel-r4-vnext-v3
  address literals = diagnostics only
        ↓
LOCAL V3 acceptance PASS
  22,394 groups / max 7
  zero address-authority edges
  same 67,620 physical files
  same physical digest
        ↓
V3 role-dependent research COMPLETE
  representation sensitivity
  selector CPU coverage
  confirmed-negative pilot queue
  CUDA comparison + 4/4 worst-case probes
        ↓
next controlled decisions
  final Git-safe evidence snapshot
  selector promotion ADR/versioning
  R4-GAP-007 negative adjudication
  objective/evaluation design
        ↓
possible later G8 training authorization
```

The key distinction is:

```text
physical DATA integrity       accepted under R4-D-008
logical V2 grouping/roles     superseded for future research
logical V3 grouping/roles     accepted under R4-D-009
selector evidence             strong enough for separate promotion decision, not promoted
model discrimination evidence still unavailable because confirmed negatives = 0
```

## Actual runtime/source walkthrough

### Canonical R4 gate state

| Phase | State | Current meaning |
|---:|---|---|
| 0 | G0 PASS | historical baseline/evidence locations frozen |
| 1 | G1 PASS | prior evidence recovered |
| 2 | G2 PASS | historical label-corruption mechanisms reconstructed |
| 3 | G3 PASS | historical 22,493×10 evidence ledger materialized/validated |
| 4 | G4 PASS | decision-critical source authority adjudicated |
| 5 | G5 PASS | DATA vNext policy/schema/ADRs accepted |
| 6 | G6 PASS | historical v1 leakage-safe role manifests frozen; unsupported evaluation roles explicit |
| 7 | G7 PASS | historical `sentinel-r4-vnext-v1` bound to 21,657 representations / 64,971 files with zero mismatches |
| 8 | IN_PROGRESS | repaired-v2 physical DATA accepted; V2 grouping superseded; logical V3 accepted; V3 selector/sensitivity/negative-queue/CUDA research completed; confirmed-negative adjudication, selector promotion, objective/evaluation design, and full training remain pending |
| 9–10 | WAITING | evaluation/calibration/promotion remain gated by G8 and missing evidence |

### Historical G7 foundation

Historical `sentinel-r4-vnext-v1` remains an immutable reproducibility root:

- implementation merge: `81d9c547d`;
- tracked publication status: `VALIDATED_G7_CANDIDATE`;
- contracts: 22,493;
- contract×class rows: 224,930;
- positive targets: 1,007;
- negative targets: 0;
- STRONG signals: 403;
- WEAK signals: 604;
- required representations: 21,657;
- checked files: 64,971;
- missing/mismatched files: 0;
- representation binding digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`.

Historical Phase-6 `r4-vnext-roles-v1` remains frozen and is not rewritten by later Phase-8 lineages. The handbook metadata intentionally continues to validate those historical G7 machine-readable facts.

### Repaired-v2 physical acceptance

R4-D-008 accepts the local repaired physical lineage as a bounded-research source/representation root:

- preprocessing: `sentinel-preprocessed-r4-v2`;
- provenance/source claims: `r4-provenance-v1`;
- role-independent semantic evidence ledger: `evidence-ledger-r4-v2`;
- representation root: `representations-r4-v2`;
- extractor: `v2.2-r4-repaired`;
- graph schema: `v9`;
- token tensor shape: `[4,512]`;
- contracts: 22,540;
- contract×class rows: 225,400;
- graph/token/sidecar files: 67,620;
- required physical files missing/invalid: 0;
- physical representation binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- confirmed-negative rows: 0.

The physical acceptance includes explicit compatibility provenance rather than silent skips: 22,512 full-analysis-classified representations, 26 parse-only recoveries, and two graph-only constant-array compatibility transformations. These physical artifacts remain reusable under V3 because R4-D-009 changes logical grouping/roles, not graph/token/sidecar bytes.

### Why V2 logical grouping is superseded

The V2 full-population grouping audit found:

- artifacts: 22,540;
- V2 groups: 11,551;
- largest group: 10,327 contracts;
- largest-group address keys: 999;
- address-derived edges in that component: 18,213;
- Uniswap V2 router address present in 8,225 DIVE artifacts;
- dead address present in 1,240;
- zero address present in 519;
- WETH address present in 504.

The V2 implementation joined same-source artifacts whenever they shared **any** Ethereum address literal. The audit demonstrated that common protocol/sentinel/constants transitively connect unrelated contracts. Therefore “same address literal” is not acceptable family identity evidence.

R4-D-009 does not retroactively erase V2 evidence. It changes future logical authority:

```text
V2 physical source/representations    retained
V2 grouping/roles                     historical/superseded
V2 role-dependent research outputs   historical/population-specific
```

Do not use the V2 200-cell confirmed-negative review queue for manual adjudication. Its group reservations are tied to the superseded V2 partition.

### Accepted logical V3 lineage

The corrected V3 identifiers are:

- grouping: `r4-leakage-groups-v3`;
- role partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

V3 grouping authority permits:

- exact artifact identity;
- identical normalized-code identity;
- explicit source-provided family/project identifiers such as `base_family_id`, `family_id`, `project_group_id`, or `project_id`.

V3 explicitly forbids arbitrary address literals as grouping authority. Address overlaps remain measurable diagnostics and create **zero** union edges.

Protected local acceptance proved:

- contracts / contract×class rows unchanged: 22,540 / 225,400;
- positive / unknown / confirmed-negative targets unchanged: 1,080 / 224,320 / 0;
- STRONG / WEAK semantic cells unchanged: 474 / 606;
- V3 groups: 22,394;
- maximum group size: 7;
- normalized-code grouping edges: 146;
- address literals observed: 14,851;
- address-authority grouping edges: 0;
- V2 giant group removed;
- physical representation files checked: 67,620;
- V3 physical binding digest exactly equals repaired-v2: `16dd4a3f...`;
- physical rebuild performed: false.

V3 role contract counts:

- `TRAIN_STRONG`: 343;
- `TRAIN_WEAK`: 602;
- `TRAIN_UNLABELED`: 21,449;
- `MODEL_SELECTION`: 73;
- `INTERNAL_AUDIT`: 73.

V3 active supervision/evaluation:

- effective loss cells: 932;
- optimizer-active contracts/groups: 932 / 932;
- model-selection outcome cells: 143;
- model-selection contracts/groups: 143 / 142.

All of those supervised/model-selection cells remain positive-only.

Repository implementation includes:

- `data_module/sentinel_data/preprocessing/r4_grouping_v3.py`;
- `data_module/sentinel_data/vnext/r4_logical_v3.py`;
- `ml/src/datasets/vnext_logical_v3_dataset.py`;
- V3 rebuild/binding/acceptance scripts;
- V3 confirmed-negative queue builder;
- V3 selector CUDA comparison with mandatory worst-case probes;
- Git-safe V3 evidence snapshotting;
- focused V3 regression tests and Phase-8 CI coverage.

### Current model/evaluation state

Run12 remains the historical operational checkpoint and comparison baseline. Its learned weights, optimizer/scheduler state, thresholds, and calibration are not current repaired training truth.

The architecture remains frozen for this evidence tranche:

- architecture/model: `four_eye_v8` / `v8.1`;
- ten outputs;
- GraphCodeBERT base snapshot: `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`;
- graph schema: `v9`;
- token tensor: `[4,512]`.

The current V3 partition contains 932 optimizer-bearing positive cells/contracts across 932 active groups. The acceptance script also reported planning-only batch-8/accum-8 arithmetic of 117 micro-batches/epoch, 15 optimizer steps/epoch, and 1,500 optimizer steps over a hypothetical 100 epochs. **Those numbers are not an authorized training horizon.**

The supervision problem remains independent of grouping: current policy has zero confirmed-negative source authority. Unknown cells remain masked, not target `0`. A future objective cannot claim trustworthy false-positive discrimination until suitable negative/evaluation evidence or an explicitly justified alternative evaluation design exists.

### V3 research evidence now regenerated

#### Representation sensitivity

Physical telemetry remains:

- graph nodes p50/p95/p99/max: 264 / 883 / 1,606 / 16,065;
- edges: 685 / 1,969 / 4,269 / 166,459;
- components: 1 / 2 / 5 / 28;
- pre-subsampling windows: 17 / 54 / 97 / 403.

Representation modes remain 22,512 full analysis, 26 parse-only, and 2 constant-array-fold compatibility cases. Under V3 roles, MODEL_SELECTION contains zero compatibility-mode contracts; seven optimizer-active compatibility cases remain, all `TRAIN_WEAK` parse-only.

#### Selector CPU population comparison

V3 selector experiment:

- records analyzed: 1,018;
- over four windows: 737;
- guarded improved target coverage: 476;
- equal/control fallback: 261;
- regressions: 0;
- median target coverage: historical ~63.01%, guarded ~87.94%.

The V3 population differs materially from the superseded V2 partition, so V2 improvement percentages are historical rather than a baseline the V3 percentages must match.

#### Confirmed-negative pilot queue

R4-GAP-007 is APPROVED. The current V3 queue contains:

- 200 candidate cells;
- 25 candidates for each of the eight enabled classes;
- 200 reserved leakage groups;
- all `PENDING_REVIEW`;
- all current targets `None`;
- all role `TRAIN_UNLABELED`;
- `negative_truth_claim=false`.

The planning-only one-sided zero-FP bound is 59 confirmed negatives/class for a 5% maximum FPR at 95% confidence if zero false positives are observed. The 25/class queue is only a pilot to estimate adjudication yield, not a final quality gate.

#### Identical-initialization CUDA selector comparison

The bounded V3 comparison ran on the NVIDIA GeForce RTX 3070 Laptop GPU with BF16 autocast:

- identical initialization: true;
- initial-state digest: `ad1987633e72d74fa3350d9e20cd1c01ada67d257ccba1691ba4b58e88ea7606` for both strategies;
- 4 train batches + 4 selection batches per strategy;
- 4/4 mandatory worst-case probes completed;
- no Run12 weights loaded;
- no checkpoint written.

Positive-only model selection:

- control NLL / mean positive probability: 0.68474 / 0.50607;
- guarded NLL / mean positive probability: 0.66014 / 0.51887;
- positive recall at fixed 0.5 threshold: 0.5 for both;
- metric cells: 4.

Peak CUDA allocation:

- control: 967.36 MB;
- guarded: 956.68 MB.

Worst-case probes included a 353-window `TRAIN_WEAK` contract whose target coverage increased from ~1.64% to ~3.79%, a 78-window `TRAIN_STRONG` contract from ~18.38% to ~35.53%, and a 51-window `TRAIN_STRONG` contract from ~12.31% to ~19.42%. A four-window case correctly fell back to control and preserved 100% coverage.

This evidence is sufficient to consider a **separate selector-promotion ADR/versioned extractor decision**. It does not itself promote `target_aware_guarded_v1`, and positive-only CUDA behavior does not establish vulnerability discrimination.

## Interfaces, data shapes, and configuration

### Current DATA/ML authority stack

| Surface | Current authority |
|---|---|
| semantic policy | `data-vnext-policy-v1` |
| historical gate baseline | G0–G7 / `sentinel-r4-vnext-v1` |
| repaired physical source/representation root | R4-D-008 / repaired-v2 physical artifacts |
| current logical authority | R4-D-009 / accepted logical V3 |
| confirmed-negative review authority | R4-GAP-007 / V3 pilot, adjudication not started |
| current restart checkpoint | `docs/plan/ml-R4/runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md` |
| training authorization | HOLD / none |

### Stable shapes

- vulnerability classes: 10, fixed order;
- graph schema: `v9`;
- node feature dimension: 12;
- node types: 14;
- edge types: 12;
- GraphCodeBERT windows delivered to the frozen model: `[4,512]`;
- model outputs: 10.

Logical V3 changes group and role identities. It does not change any of these shapes.

### Unsupported evaluation roles

```text
THRESHOLD_FIT        = UNSUPPORTED_EMPTY
CALIBRATION_FIT      = UNSUPPORTED_EMPTY
UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN
confirmed negatives  = 0
```

A new evidence/decision/version is required before any of these states changes.

### Fresh-clone versus protected local artifacts

Tracked in Git:

- source/config/tests;
- historical G0–G7 evidence/policy/manifests;
- V2 repair/V3 logical tooling and governance;
- sanitized decisive research snapshots;
- ADRs, status matrix, evidence-gap register, and handoffs/checkpoints.

Local/protected/generated:

- repaired-v2 preprocessing tree;
- repaired-v2 representation tree;
- repaired-v2/V3 generated publications where ignored;
- V3 local build/evidence reports before final sanitization;
- Run12 checkpoint companions depending on acquisition;
- GPU training logs/checkpoints;
- secrets, RPC credentials, signing keys, proving-key/SRS workspaces.

A fresh clone may cite committed acceptance evidence, but it must not claim local protected trees are physically present until they are acquired/rebuilt and validated.

## Failure modes and current limitations

### DATA / grouping

- repaired-v2 physical artifacts are accepted;
- V2 grouping/roles are superseded for future research because address coincidence over-grouped unrelated contracts;
- V3 grouping/roles are accepted after protected local validation;
- V3 preserved the exact repaired-v2 physical representation binding digest;
- source acquisition portability remains imperfect for some external corpora.

### Supervision / evaluation

- zero confirmed-negative rows;
- R4-GAP-007 pilot candidates are review reservations, not negatives;
- all historical/repaired negative-like absence remains unknown unless independently confirmed;
- threshold fitting unavailable;
- calibration fitting unavailable;
- untouched acceptance unavailable;
- no trustworthy general specificity/FPR/F1 claim can currently be made for a new repaired model.

### Representation / selector

- fixed `[4,512]` input capacity omits material code for many long contracts;
- target-aware guarded selection has strong V3 CPU + bounded CUDA evidence but remains unpromoted pending a separate ADR/versioned extractor lineage;
- compatibility-mode and multi-component/file-union representations remain explicit sensitivity risks;
- all four required worst-case graph/token GPU probes completed successfully in the current bounded experiment.

### Training

- no repaired full-run checkpoint exists;
- no current V3 training horizon is authorized;
- do not reuse Run12 learned weights as repaired truth;
- do not use F1 early stopping, threshold tuning, calibration fitting, pseudo-negatives, or untouched acceptance under current evidence;
- positive-only supervision may encourage broad overprediction and cannot establish discrimination by itself.

### ZK / runtime boundaries

- retained EZKL proof scope remains proxy-only;
- `check_mode="UNSAFE"` remains explicit;
- V3 context attestation does not expand the circuit;
- audit MCP on port 8012 remains read-only;
- an on-chain/versioned record is not automatically vulnerability ground truth.

## Common change recipe

For current Phase-8 DATA/ML work:

1. start from synchronized `main`;
2. read `PLAN_STATUS_MATRIX.md` and `runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`;
3. preserve historical V1/V2 evidence, repaired-v2 physical artifacts, and accepted V3 logical authority;
4. run the final Git-safe V3 evidence snapshot helper before treating the current local research tranche as fully packaged in Git;
5. use a new ADR/versioned extractor lineage for selector promotion; do not mutate repaired-v2 bound tokens in place;
6. conduct R4-GAP-007 review only under class-specific primary evidence plus independent agreeing verification;
7. decide objective/evaluation design after negative-evidence yield is known;
8. reconsider a full training horizon only after selector/objective/evaluation authority is explicit and bound.

Never “fix” current state by editing generated parquet/JSON artifacts by hand.

## Verification commands

Focused V3 repository tests:

```bash
PYTHONPATH=.:data_module python -m pytest -q -c /dev/null \
  data_module/tests/test_preprocessing/test_r4_grouping_v3.py \
  data_module/tests/test_vnext/test_r4_logical_v3.py \
  data_module/tests/test_vnext/test_logical_v3_ml_adapter.py
```

The canonical Phase-8 repository CI contract is:

`.github/workflows/r4-phase8-data-repair.yml`

Historical G6 validation:

```bash
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Canonical handbook validation:

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
python3 docs/handbook/tools/verify_handbook.py inventory
```

Protected local V3 acceptance is complete and is recorded in the current checkpoint. CI still cannot replace protected local physical validation because CI does not contain the Git-ignored repaired-v2 physical trees.

## Optional deep references

- [R4 plan status matrix](../plan/ml-R4/PLAN_STATUS_MATRIX.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)
- [R4 evidence gap register](../plan/ml-R4/EVIDENCE_GAP_REGISTER.md)
- [ADR-R4-008 repaired-v2 physical DATA acceptance](../plan/ml-R4/adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md)
- [ADR-R4-009 accepted logical V3 grouping correction](../plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md)
- [Repaired-v2 acceptance/no-launch decision](../plan/ml-R4/runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md)
- [Logical V3 execution handoff/history](../plan/ml-R4/runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md)
- [Logical V3 acceptance/research checkpoint](../plan/ml-R4/runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md)
- [V2 research evidence snapshot](../plan/ml-R4/evidence/2026-08-15_phase8_research/)
- [V3 interim research evidence](../plan/ml-R4/evidence/2026-08-15_phase8_logical_v3_interim/)

## Technical mastery layer

The current Phase-8 issue illustrates four separate engineering concepts that must not be conflated:

1. **Physical integrity** — can every contract and representation file be reconstructed, deserialized, hash-bound, and traced?
2. **Semantic supervision integrity** — does each contract×class target mean what the evidence actually authorizes?
3. **Leakage partition integrity** — are related/duplicate artifacts prevented from crossing roles without collapsing unrelated projects?
4. **Statistical learning adequacy** — does the available positive/negative/evaluation evidence support the claims we want to make about the model?

R4-D-008 substantially solved (1) and repaired much of (2). Accepted R4-D-009 corrects the discovered defect in (3). The project still has major unresolved work in (4), especially confirmed-negative evaluation and later objective/threshold/calibration design.

A useful mental rule is:

```text
valid file ≠ valid label
valid label ≠ valid split
valid split ≠ sufficient learning objective
successful optimization ≠ trustworthy model quality
```

## Prerequisite knowledge

To own this phase technically, understand at least:

- multi-label classification and nullable/masked targets;
- positive-only versus confirmed-negative supervision;
- data leakage and group-atomic splitting;
- connected components / union-find grouping;
- provenance versus heuristic similarity;
- deterministic dataset/version/hash binding;
- token truncation/window selection;
- train/model-selection/threshold/calibration/acceptance role separation;
- why a GPU smoke test proves execution mechanics rather than model quality.

You do not need to master PU-learning theory yet. That decision is intentionally deferred until confirmed-negative pilot yield and the later objective/evaluation design are understood.

## Source map and reading order

For current Phase-8 ownership, read in this order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
2. `docs/plan/ml-R4/runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`
3. `docs/plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md`
4. `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md` / R4-GAP-007
5. `data_module/sentinel_data/preprocessing/r4_grouping_v3.py`
6. `data_module/sentinel_data/vnext/r4_logical_v3.py`
7. `ml/src/datasets/vnext_logical_v3_dataset.py`
8. selector/negative-evaluation source only when working on those next decisions.

For historical context, read R4-D-008, the completed V3 handoff, and the V2 evidence snapshot afterward rather than treating them as the current restart point.

## Execution trace and worked example

Suppose two unrelated DIVE contracts both contain the Uniswap V2 router address.

Under V2 grouping:

```text
Contract A ── shared router address ── Contract B
          → same leakage component
          → one group-atomic role
```

If many contracts also share WETH, zero/dead addresses, proxy slots, or other common constants, union-find transitively creates a giant component even though the projects are unrelated.

Under V3:

```text
Contract A ── shared address ── Contract B
          diagnostic only
          NO union edge

Contract A ── same normalized-code SHA ── Contract C
          authoritative leakage edge

Contract D ── explicit family_id ── Contract E
          authoritative leakage edge
```

After V3 grouping was built, role assignment was recomputed group-atomically. The physical graph/token files were not regenerated. The V3 binder checked the same 22,540 physical contracts / 67,620 representation files and recovered the exact repaired-v2 physical binding digest. That invariant passed, so the logical V3 migration was accepted.

## Implementation practice

A safe exercise before changing V3 grouping is to add a synthetic regression fixture containing:

- two different contracts sharing a common Ethereum address;
- two contracts with identical normalized code;
- two contracts sharing an explicit family ID.

Expected V3 behavior:

```text
shared address only      → separate groups
normalized-code identity → same group
explicit family ID       → same group
address edge count       → 0
```

Then run the focused grouping tests before the broader Phase-8 CI suite. This preserves the policy distinction between **diagnostic correlation** and **grouping authority**.

## Review and ownership check

Before claiming the current state is understood, you should be able to answer:

- Why does repaired-v2 physical acceptance remain valid even though V2 grouping is superseded?
- Why can an address shared by thousands of contracts not be treated as family identity?
- Which evidence types are allowed to create V3 leakage-group edges?
- Which local invariants allowed R4-D-009 to become ACCEPTED?
- Why is the old V2 200-cell negative-review queue obsolete, while the new V3 200-cell queue is still not negative truth?
- Why does the guarded selector now have stronger evidence without yet being promoted?
- Why can a successful positive-only CUDA run not establish false-positive discrimination?
- What evidence would be needed before the 100-epoch run could be reconsidered?

If any answer is unclear, use the current V3 acceptance/research checkpoint plus ADR-R4-009 as the next reading point rather than guessing from historical V2 counts.
