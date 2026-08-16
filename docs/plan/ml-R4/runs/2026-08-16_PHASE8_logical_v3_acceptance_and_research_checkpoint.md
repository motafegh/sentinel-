# Phase-8 logical V3 acceptance and research checkpoint

**Date:** 2026-08-16
**Branch:** `main`
**Logical decision:** R4-D-009 / ADR-R4-009
**State:** logical V3 accepted locally; V3 research regeneration completed through bounded CUDA comparison
**Training:** NOT AUTHORIZED
**G8:** OPEN

## Outcome first

The logical V3 repair has now completed its protected local acceptance sequence over the unchanged repaired-v2 physical DATA. The V2 grouping defect is removed without rebuilding or mutating the accepted physical source/graph/token/sidecar artifacts. The corrected V3 grouping, role partition, publication, same-byte physical binding, and V2→V3 acceptance gate all passed.

Role-dependent Phase-8 research was then regenerated under the corrected V3 partition. Representation sensitivity remains physically consistent with repaired-v2. The guarded target-aware four-window selector improves target-contract coverage on the corrected V3 active-role population with zero observed coverage regressions, and its bounded identical-initialization CUDA comparison completed successfully including all required worst-case forward probes. This is sufficient evidence to consider a separate selector-promotion ADR/version, but **does not itself promote the selector**.

The central model-quality blocker remains unchanged: there are still zero confirmed-negative targets. A new V3 pilot review queue contains 200 unlabeled review reservations, not negative truth. Threshold fitting, calibration fitting, untouched acceptance, trustworthy specificity/FPR claims, and the 100-epoch Phase-8 run remain unsupported/unauthorized.

## Repository/evidence commits entering this checkpoint

- V3 implementation + handbook/CI baseline before protected local execution: `204f5059f20ae564aa9be92c77b9a7bbfbd6f167`.
- Interim V3 representation-sensitivity evidence: `5e19fdf3a134ef2eb5b72df166a157c421fa811b`.
- Interim V3 selector-coverage summary: `a51f28e0684f63cec69af2e76efcfc518035a21a`.

The remaining generated V3 reports described below are still local/protected until the final Git-safe evidence snapshot helper is run. This checkpoint records their observed decision-level results before that snapshot step.

## 1. Logical V3 acceptance — PASS

### Versions

- grouping: `r4-leakage-groups-v3`;
- partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`;
- source evidence ledger reused: `evidence-ledger-r4-v2`.

### Physical population preserved

- contracts: 22,540;
- contract×class rows: 225,400;
- representation triples/files: 22,540 / 67,620;
- physical rebuild performed: **false**;
- required physical files invalid/missing: 0;
- repaired-v2/V3 physical binding digest:
  `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- V3 binding matched parent repaired-v2: **true**.

### Semantic population preserved

- positive targets: 1,080;
- unknown targets: 224,320;
- confirmed-negative targets: 0;
- STRONG semantic cells: 474;
- WEAK semantic cells: 606;
- target counts unchanged from repaired-v2: **true**;
- training-strength counts unchanged from repaired-v2: **true**.

### Corrected grouping result

- artifacts: 22,540;
- V3 leakage groups: 22,394;
- multi-member groups: 117;
- maximum V3 group size: 7;
- normalized-code identity edges: 146;
- explicit family edges observed in this population: 0;
- Ethereum address literals observed: 14,851;
- address-authority union edges: **0**;
- arbitrary address-literal grouping authority: **false**.

For comparison, superseded V2 contained 11,551 groups and one 10,327-contract DIVE component driven by common address literals. That giant component no longer exists under V3.

### V3 role freeze

Contract counts:

- `TRAIN_STRONG`: 343;
- `TRAIN_WEAK`: 602;
- `TRAIN_UNLABELED`: 21,449;
- `MODEL_SELECTION`: 73;
- `INTERNAL_AUDIT`: 73.

Group counts:

- `TRAIN_STRONG`: 331;
- `TRAIN_WEAK`: 601;
- `TRAIN_UNLABELED`: 21,320;
- `MODEL_SELECTION`: 71;
- `INTERNAL_AUDIT`: 71.

Active supervision/evaluation:

- effective loss cells: 932;
- optimizer-active contracts: 932;
- optimizer-active groups: 932;
- `TRAIN_STRONG` optimizer contracts: 331;
- `TRAIN_WEAK` optimizer contracts: 601;
- model-selection outcome cells: 143;
- model-selection contracts: 143;
- model-selection groups: 142.

Planning-only batch-8 / accumulation-8 arithmetic reported 117 micro-batches/epoch, 15 optimizer steps/epoch, and 1,500 optimizer steps over a hypothetical 100 epochs. **This arithmetic is not an authorized training horizon.**

## 2. Representation sensitivity under V3 — PASS / risk remains explicit

The V3 sensitivity report used the unchanged repaired-v2 representations and corrected V3 roles.

Physical telemetry remained consistent with repaired-v2:

| Metric | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| graph nodes | 264 | 883 | 1,606 | 16,065 |
| graph edges | 685 | 1,969 | 4,269 | 166,459 |
| graph components | 1 | 2 | 5 | 28 |
| pre-subsampling token windows | 17 | 54 | 97 | 403 |

Representation modes:

- Slither full analysis: 22,512;
- Slither parse-only compatibility: 26;
- constant-array-fold compatibility: 2.

Role-sensitive compatibility result:

- MODEL_SELECTION compatibility contracts: **0**;
- optimizer-active compatibility contracts: **7**;
- all seven optimizer-active compatibility contracts are `TRAIN_WEAK` parse-only cases;
- the two constant-array-fold cases remain in `TRAIN_UNLABELED`.

A major active worst case is contract
`f50cd5d7df9ab644a02eb760ceab56548d327984db313015a66bca85513fa3c5`:

- role: `TRAIN_WEAK`;
- nodes: 16,065;
- edges: 67,106;
- graph components: 21;
- pre-subsampling windows: 353;
- optimizer-active: true.

No representation policy change is made by this checkpoint. Compatibility/file-union exposure remains explicit evidence for later policy decisions rather than an implicit exclusion.

Committed interim evidence:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3_interim/representation_sensitivity_v1.json`

## 3. V3 bounded-window selector CPU comparison — PASS as research evidence

Population:

- records requested/analyzed: 1,018 / 1,018;
- failures: 0;
- contracts with more than four windows: 737.

Guarded versus historical coverage on the over-cap population:

- guarded improved target coverage: 476 records (~64.6%);
- guarded equal/control fallback: 261 records (~35.4%);
- guarded target-coverage regressions: **0**.

Median target-contract coverage:

- `historical_linspace_v1`: 0.6300634456 (~63.01%);
- `target_aware_greedy_v1`: 0.8687164470 (~86.87%);
- `target_aware_guarded_v1`: 0.8794466403 (~87.94%).

Median retained-token ratio:

- historical: ~60.10%;
- guarded: ~57.79%.

The corrected V3 population differs materially from the superseded V2 role population, so the lower V3 percentage of “improved records” is not treated as a regression against V2. The relevant V3 facts are substantial median target-coverage improvement and zero observed guarded regressions.

Committed interim evidence:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3_interim/bounded_window_selector_v1.summary.json`

The full per-contract report remains local and is SHA-bound by the summary.

## 4. V3 confirmed-negative pilot queue — CLEAN, NOT NEGATIVE TRUTH

Generated local report:

`data_module/data/r4-v3-logical-build/confirmed_negative_review_queue_v1.json`

Observed queue state:

- status: `PILOT_REVIEW_QUEUE_NOT_NEGATIVE_TRUTH`;
- dataset: `sentinel-r4-vnext-v3`;
- partition: `r4-vnext-roles-v3`;
- queued cells: 200;
- reserved leakage groups: 200;
- all candidate statuses: `PENDING_REVIEW`;
- all current target values: `None`;
- all roles at queue creation: `TRAIN_UNLABELED`;
- `negative_truth_claim`: false.

The eight enabled classes each received 25 pilot candidates:

- CallToUnknown;
- DenialOfService;
- ExternalBug;
- IntegerUO;
- MishandledException;
- Reentrancy;
- Timestamp;
- TransactionOrderDependence.

Planning-only one-sided zero-false-positive bound:

- target maximum FPR: 5%;
- confidence: 95%;
- minimum confirmed negatives per class if zero false positives are observed: 59.

The default 25/class queue is therefore a **pilot to estimate adjudication yield**, not a final evaluation population. No candidate may become target `0` without class-specific primary evidence and independent verification under the confirmed-negative adjudication contract.

The obsolete V2 queue must not be adjudicated.

## 5. Identical-initialization V3 CUDA selector comparison — PASS as bounded research

Generated local report:

`data_module/data/r4-v3-logical-build/selector_gpu_compare_v1.json`

Runtime scope:

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU;
- mixed precision: BF16 autocast;
- batch size: 1;
- gradient accumulation: 1;
- train batches per strategy: 4;
- model-selection batches per strategy: 4;
- worst-case probes required/completed: 4 / 4;
- Run12 weights loaded: false;
- checkpoint written: false;
- identical initialization verified: true;
- initial-state digest for both strategies:
  `ad1987633e72d74fa3350d9e20cd1c01ada67d257ccba1691ba4b58e88ea7606`.

### Historical-control strategy

Training smoke:

- total loss: 0.8341597319;
- main loss: 0.6096345782;
- auxiliary loss: 2.0642062426;
- phase-2 loss: 0.7351235449;
- optimizer steps: 4.

Positive-only model selection:

- positive NLL: 0.6847369671;
- mean positive probability: 0.5060739517;
- positive recall at fixed 0.5 threshold: 0.5;
- metric cells: 4.

CUDA:

- peak allocated: 967.36 MB;
- allocated at report: 290.34 MB;
- reserved: 1,008 MB.

### Guarded target-aware candidate

Training smoke:

- total loss: 0.9046758115;
- main loss: 0.6797058731;
- auxiliary loss: 2.2484135330;
- phase-2 loss: 0.7029369622;
- optimizer steps: 4.

Positive-only model selection:

- positive NLL: 0.6601404548;
- mean positive probability: 0.5188665390;
- positive recall at fixed 0.5 threshold: 0.5;
- metric cells: 4.

CUDA:

- peak allocated: 956.68 MB;
- allocated at report: 289.84 MB;
- reserved: 1,000 MB.

Probability delta over the four positive model-selection cells:

- mean signed delta: +0.0127926469;
- mean absolute delta: 0.0127926469;
- max absolute delta: 0.0180904269.

The candidate’s four-step training loss is higher than control. This bounded run is not a converged training comparison, so that short-run loss difference is not used as a selector-quality conclusion. The useful evidence is control identity, CUDA safety, target coverage, positive-only selection behavior, and successful worst-case execution.

### Mandatory worst-case forward probes

All four required probes completed with finite forward output.

1. `f50cd5...` — `TRAIN_WEAK`, 353 windows, guarded target coverage 0.03789286 vs control 0.01638309, no fallback, peak 559.82 MB.
2. `bf8aff...` — `TRAIN_STRONG`, 78 windows, guarded 0.35533879 vs control 0.18376589, no fallback, peak 327.99 MB.
3. `7fdee9...` — `TRAIN_STRONG`, 51 windows, guarded 0.19421173 vs control 0.12309596, no fallback, peak 319.64 MB.
4. `bec1a6...` — `TRAIN_WEAK`, 4 windows, guarded/control both 1.0, correct control fallback, peak 323.74 MB.

This fixes the incompleteness of the earlier V2 CUDA evidence, where intended worst-case probes silently had no sensitivity input population.

## 6. Decision state at this checkpoint

### Accepted now

- repaired-v2 physical DATA acceptance under R4-D-008 remains valid;
- R4-D-009 logical V3 invariants have passed protected local validation;
- corrected V3 grouping/roles/publication are the current logical authority for future Phase-8 model research;
- V2 grouping/roles remain immutable historical evidence but are superseded for future training/evaluation authority.

### Evidence-ready but NOT yet promoted

`target_aware_guarded_v1` now has:

- corrected V3 population coverage evidence;
- zero observed guarded target-coverage regressions;
- identical-initialization CUDA comparison;
- successful mandatory worst-case probes;
- no observed CUDA-memory penalty.

This is enough to move to a **separate explicit selector-promotion decision/ADR and extractor version/binding step**. It is not itself a promotion. Current production/bound representations still use the historical selector until that decision is made.

### Still blocked / unsupported

- confirmed-negative targets: 0;
- negative adjudication: not started;
- false-positive discrimination / specificity: unsupported;
- threshold fit: unsupported/empty;
- calibration fit: unsupported/empty;
- untouched acceptance: unsupported/empty/frozen;
- PU objective: not selected/authorized;
- full 100-epoch Phase-8 training: NOT AUTHORIZED;
- G8: OPEN.

## 7. Next controlled steps

1. Run the Git-safe final V3 evidence snapshot helper and commit only its sanitized evidence directory.
2. Record selector promotion, if chosen, as a new ADR/decision and a new extractor/representation lineage rather than silently mutating repaired-v2 bound tokens.
3. Open/approve the confirmed-negative evidence gap before any manual candidate adjudication; retain all queue cells as UNKNOWN/PENDING_REVIEW until the explicit evidence contract is satisfied.
4. Decide the later objective/evaluation design only after negative-evidence yield is known; do not invent pseudo-negatives or threshold/calibration populations.
5. Reconsider a full training horizon only after the objective/evaluation and selector lineage are explicitly governed and bound.

## Stop lines retained

Do not:

- infer target `0` from unlabeled/source silence;
- manually adjudicate the obsolete V2 queue;
- reuse Run12 learned weights/optimizer/scheduler/threshold/calibration as repaired/V3 truth;
- mutate accepted repaired-v2 physical artifacts in place;
- silently promote the guarded selector;
- start the 100-epoch run;
- claim general FPR/F1/discrimination from the positive-only CUDA smoke.
