# 16 — Current status and gap ledger

**Read this when:** you need the canonical current project state, especially before DATA/ML execution, evaluation, or promotion decisions.

**Skip this if:** you only need a historical implementation walkthrough and do not need current authority or gate state.

**Estimated reading time:** 8 minutes

## 30-second summary

SENTINEL historical R4 **G0–G7 remain PASSED** and immutable. **Phase 8 is IN_PROGRESS; G8 is open.** Run12 remains the historical operational ML baseline and is not current repaired training truth.

R4-D-008 physically accepted the repaired-v2 DATA source/representation layer: **22,540 contracts**, **225,400 contract×class rows**, and **67,620 graph/token/sidecar files**, with zero missing/invalid required representation files and physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

A later full-population research audit found a separate logical defect: `r4-leakage-groups-v2` used arbitrary same-source Ethereum address literals as family/group authority. One DIVE component contained **10,327 contracts**, driven largely by common addresses such as the Uniswap V2 router, dead/zero addresses, WETH, and other ubiquitous constants. R4-D-009 therefore **supersedes V2 grouping/roles for future model research while preserving repaired-v2 physical artifacts as accepted evidence**.

The active logical candidate is now `r4-leakage-groups-v3` → `r4-vnext-roles-v3` → `sentinel-r4-vnext-v3`. V3 removes arbitrary address literals from grouping authority and reuses the accepted repaired-v2 physical source/representation bytes. Repository tooling and tests exist; **protected local V3 generation, same-byte rebinding, acceptance, and regenerated research evidence are still pending**.

There are still **zero confirmed-negative examples** in policy v1. Threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty. The target-aware four-window selector is not promoted. **The 100-epoch Phase-8 run is not authorized.**

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
LOCAL V3 acceptance pending
  grouping → roles/publication → same-byte rebinding
  → V2→V3 acceptance audit
        ↓
regenerate role-dependent research evidence
  selector coverage
  representation sensitivity
  confirmed-negative pilot queue
  CUDA comparison + mandatory worst-case probes
        ↓
objective/evaluation decision
        ↓
possible later G8 training authorization
```

The key distinction is:

```text
physical DATA integrity       accepted under R4-D-008
logical V2 grouping/roles     superseded for future research
logical V3 implementation     repository-complete, local acceptance pending
model discrimination evidence still unavailable
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
| 8 | IN_PROGRESS | repaired-v2 physical DATA accepted; V2 grouping superseded; logical V3 implemented but local V3 acceptance/evidence regeneration pending; full training unauthorized |
| 9–10 | WAITING | evaluation/calibration/promotion remain gated by G8 and missing evidence |

### Historical G7 foundation

Historical `sentinel-r4-vnext-v1` remains an immutable reproducibility root:

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

### Logical V3 candidate

The corrected V3 identifiers are:

- grouping: `r4-leakage-groups-v3`;
- role partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

V3 grouping authority permits:

- exact artifact identity;
- identical normalized-code identity;
- explicit source-provided family/project identifiers such as `base_family_id`, `family_id`, `project_group_id`, or `project_id`.

V3 explicitly forbids arbitrary address literals as grouping authority. Address overlaps remain measurable diagnostics and must create **zero** union edges.

Repository implementation includes:

- `data_module/sentinel_data/preprocessing/r4_grouping_v3.py`;
- `data_module/sentinel_data/vnext/r4_logical_v3.py`;
- `ml/src/datasets/vnext_logical_v3_dataset.py`;
- local V3 rebuild/binding/acceptance scripts;
- V3 confirmed-negative queue builder;
- V3 selector CUDA comparison with mandatory worst-case probes;
- Git-safe V3 evidence snapshotting;
- focused V3 regression tests and Phase-8 CI coverage.

Local acceptance must prove semantic counts are unchanged from repaired-v2, address-union edges are zero, the giant V2 component disappears, every required physical representation still validates, and the physical binding digest remains exactly `16dd4a3f...`.

### Current model/evaluation state

Run12 remains the historical operational checkpoint and comparison baseline. Its learned weights, optimizer/scheduler state, thresholds, and calibration are not current repaired training truth.

The architecture remains frozen for this evidence tranche:

- architecture/model: `four_eye_v8` / `v8.1`;
- ten outputs;
- GraphCodeBERT base snapshot: `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`;
- graph schema: `v9`;
- token tensor: `[4,512]`.

The V2 role freeze previously produced 899 optimizer-bearing positive contracts across 831 active groups. **Those are historical V2 partition facts now, not V3 training authority.** The V3 acceptance script derives fresh active optimizer contracts/groups and fresh batch/accumulation planning arithmetic from the corrected partition. Do not carry V2 `831`, `104`, `13`, or a hypothetical `1,300`-step horizon forward automatically.

The supervision problem remains independent of grouping: current policy has zero confirmed-negative source authority. Unknown cells remain masked, not target `0`. A future objective cannot claim trustworthy false-positive discrimination until suitable negative/evaluation evidence or an explicitly justified alternative evaluation design exists.

### Research evidence that must be regenerated under V3

V2 research established useful hypotheses but not future population authority:

- guarded target-aware selection improved requested-target declaration coverage on the V2 active-role population with zero guarded regressions in the committed summary;
- the small identical-initialization V2 CUDA comparison ran successfully;
- the original V2 CUDA report did **not** execute its intended worst-case forward probes because the sensitivity report did not yet exist;
- representation telemetry showed maxima of 16,065 nodes, 166,459 edges, 28 components, and 403 pre-subsampling token windows.

After V3 local acceptance, regenerate:

1. representation sensitivity under V3 roles;
2. bounded-window selector population statistics under V3 roles;
3. V3 confirmed-negative pilot queue;
4. identical-initialization CUDA selector comparison with all requested worst-case probes completed.

Selector promotion, compatibility exclusion/down-weighting, PU learning, and a full-run horizon remain separate later decisions.

## Interfaces, data shapes, and configuration

### Current DATA/ML authority stack

| Surface | Current authority |
|---|---|
| semantic policy | `data-vnext-policy-v1` |
| historical gate baseline | G0–G7 / `sentinel-r4-vnext-v1` |
| repaired physical source/representation root | R4-D-008 / repaired-v2 physical artifacts |
| current logical candidate | R4-D-009 / logical V3 |
| current local execution handoff | `docs/plan/ml-R4/runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md` |
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
- ADRs, status matrix, risk register, and handoffs.

Local/protected/generated:

- repaired-v2 preprocessing tree;
- repaired-v2 representation tree;
- repaired-v2/V3 generated publications where ignored;
- V3 local build/evidence reports before sanitization;
- Run12 checkpoint companions depending on acquisition;
- GPU training logs/checkpoints;
- secrets, RPC credentials, signing keys, proving-key/SRS workspaces.

A fresh clone may cite committed acceptance evidence, but it must not claim local protected trees are physically present until they are acquired/rebuilt and validated.

## Failure modes and current limitations

### DATA / grouping

- repaired-v2 physical artifacts are accepted;
- V2 grouping/roles are superseded for future research because address coincidence over-grouped unrelated contracts;
- V3 repository tooling exists, but protected local V3 acceptance has not yet occurred;
- if V3 changes the physical representation binding digest, stop: a logical-only migration has violated its core invariant;
- source acquisition portability remains imperfect for some external corpora.

### Supervision / evaluation

- zero confirmed-negative rows;
- all historical/repaired negative-like absence remains unknown unless independently confirmed;
- threshold fitting unavailable;
- calibration fitting unavailable;
- untouched acceptance unavailable;
- no trustworthy general specificity/FPR/F1 claim can currently be made for a new repaired model.

### Representation / selector

- fixed `[4,512]` input capacity omits material code for many long contracts;
- target-aware selection is promising but unpromoted and must be regenerated/reviewed under V3;
- compatibility-mode and multi-component/file-union representations remain explicit sensitivity risks;
- worst-case graph/token GPU probes must actually complete before long-run assumptions.

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
2. read `PLAN_STATUS_MATRIX.md`, R4-D-009/ADR-R4-009, and the logical-V3 handoff;
3. preserve repaired-v2 physical artifacts and historical V1/V2 logical artifacts;
4. execute the V3 logical build stages in order and stop on any failed invariant;
5. accept V3 only if semantic counts and physical hashes remain invariant while grouping defects are removed;
6. regenerate all role-dependent research evidence under V3;
7. review the evidence before changing selector/objective/training policy;
8. use a new ADR/version for any selector promotion, negative-training authority, PU objective, architecture change, calibration policy, or full-run authorization.

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

Protected local V3 validation is governed by the handoff and cannot be replaced by CI because CI does not contain the Git-ignored repaired-v2 physical trees.

## Optional deep references

- [R4 plan status matrix](../plan/ml-R4/PLAN_STATUS_MATRIX.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)
- [ADR-R4-008 repaired-v2 physical DATA acceptance](../plan/ml-R4/adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md)
- [ADR-R4-009 logical V3 grouping correction](../plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md)
- [Repaired-v2 acceptance/no-launch decision](../plan/ml-R4/runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md)
- [Logical V3 local execution handoff](../plan/ml-R4/runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md)
- [V2 research evidence snapshot](../plan/ml-R4/evidence/2026-08-15_phase8_research/)

## Technical mastery layer

The current Phase-8 issue illustrates four separate engineering concepts that must not be conflated:

1. **Physical integrity** — can every contract and representation file be reconstructed, deserialized, hash-bound, and traced?
2. **Semantic supervision integrity** — does each contract×class target mean what the evidence actually authorizes?
3. **Leakage partition integrity** — are related/duplicate artifacts prevented from crossing roles without collapsing unrelated projects?
4. **Statistical learning adequacy** — does the available positive/negative/evaluation evidence support the claims we want to make about the model?

R4-D-008 substantially solved (1) and repaired much of (2). R4-D-009 corrects a defect in (3). The project still has major unresolved work in (4).

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

You do not need to master PU-learning theory yet. That decision is intentionally deferred until corrected V3 roles and negative-evaluation evidence are available.

## Source map and reading order

For current Phase-8 ownership, read in this order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
2. `docs/plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md`
3. `docs/plan/ml-R4/runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md`
4. `data_module/sentinel_data/preprocessing/r4_grouping_v3.py`
5. `data_module/sentinel_data/vnext/r4_logical_v3.py`
6. `docs/plan/ml-R4/scripts/p8_audit_logical_v3_acceptance.py`
7. `ml/src/datasets/vnext_logical_v3_dataset.py`
8. V3 research scripts only after local V3 acceptance.

For historical context, read R4-D-008 and the V2 evidence snapshot afterward rather than treating them as the current logical restart point.

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

After V3 grouping is built, role assignment is recomputed group-atomically. The physical graph/token files are not regenerated. The V3 binder must check the same 22,540 physical contracts / 67,620 representation files and recover the exact same physical binding digest as repaired-v2. If it does not, the migration stops.

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
- Which evidence types are still allowed to create V3 leakage-group edges?
- Why must the V3 physical binding digest equal the V2 physical digest?
- Why is the old 200-cell negative-review queue obsolete?
- Why do V2 selector improvements need to be rerun under V3 even though the selector algorithm itself did not change?
- Why can a successful positive-only CUDA run not establish false-positive discrimination?
- What evidence would be needed before the 100-epoch run could be reconsidered?

If any answer is unclear, use the logical-V3 ADR/handoff and the grouping source as the next reading point rather than guessing from historical V2 counts.
