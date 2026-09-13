# 13 — Evaluation and release evidence

**Read this when:** you need to judge DATA quality, ML quality, AGENTS behavior, reliability weights, or release/promotion readiness.

**Skip this if:** you are only browsing architecture; do not skip it before claiming model quality or promotion.

**Estimated reading time:** 15 minutes.

## 30-second summary

Evaluation must follow evidence authority, not merely available scripts. R4 repaired DATA truth, leakage grouping/roles, and physical representation eligibility before any repaired full run. For the current repaired path, model-selection evidence remains **positive-only limited**, while threshold-fit, calibration-fit, and untouched-acceptance roles are deliberately unsupported/empty because no trustworthy confirmed-negative/unexposed corpus exists. AGENTS evaluation infrastructure remains useful, but model/release metrics must never treat unknown/masked cells as negatives or reuse historical Run12 roles/thresholds as if they were current R4 evidence.

The accepted logical authority is V3 under D-009; the accepted physical graph authority for a possible future repaired run is D-011 V10 V2.6; D-012's guarded token successor still requires separate physical acceptance. Full repaired training remains unauthorized.

## Just-enough mental model

```text
R4 semantic policy + logical V3 role/group authority
      +
accepted D-011 physical representation
      +
pending D-012 guarded-token successor acceptance
      ↓
later repaired training only if authorized
      ↓
positive-only checkpoint-selection diagnostics
      ↓
future evaluation constrained by evidence that actually exists
      ↓
NO threshold/calibration/untouched/model-quality claim without new authority
```

A test pass means implementation consistency. A metric is trustworthy only when its labels, role, exposure history, representation/checkpoint identity, and semantics justify that metric.

## Actual runtime/source walkthrough

### DATA / R4 evidence quality

The current DATA evaluation foundation is layered:

- historical-source/crosswalk reconstruction and the historical **224,930-row** Phase-3 evidence ledger;
- accepted `data-vnext-policy-v1` semantic policy;
- D-008 repaired-v2 physical DATA population;
- D-009 accepted logical V3 grouping/roles (`r4-vnext-roles-v3`);
- D-010 withdrawal of v9 from new-full-training eligibility;
- D-011 exact V10 V2.6 physical representation acceptance;
- D-012 guarded-selector promotion only for a fresh successor candidate;
- explicit unsupported threshold/calibration/acceptance roles.

Historical G6 `r4-vnext-roles-v1` and G7 artifacts remain reproducibility evidence, not the latest logical role authority.

Historical `test`, `NonVulnerable`, all-zero, source-absence, queue membership, or tool-silence states do not become trusted negatives merely because an evaluation utility expects them.

### ML evaluation for a repaired checkpoint

The repaired path must keep separate:

- optimization roles/cells (`TRAIN_STRONG`, `TRAIN_WEAK`, optional authorized unlabeled handling);
- checkpoint-selection evidence (`MODEL_SELECTION`, positive-only limited);
- internal audit/case-study evidence;
- threshold fitting — currently unavailable;
- calibration fitting — currently unavailable;
- untouched acceptance — currently unavailable.

Current Phase-8 mechanics can perform bounded positive-selection evaluation, but that does not authorize a full repaired training launch or convert positive-only evidence into trustworthy binary discrimination evidence.

Positive-only model-selection can support sensitivity/positive-loss/regression diagnostics. It cannot honestly estimate false-positive rate, full F1, ROC-AUC, PR-AUC against trusted negatives, or calibrated decision probabilities.

Run12 metrics remain historical baseline evidence and may be compared descriptively only with explicit corpus/semantic differences. They are not an unbiased repaired holdout.

### Confirmed-negative state

Confirmed negatives remain zero. Candidate #2's primary review supports a class-specific negative, but accepted truth still requires genuinely independent agreement. Until that happens and a later versioned policy/role decision authorizes use, the candidate remains UNKNOWN / target `None` and cannot enter binary metrics as a negative.

### AGENTS evaluation

AGENTS evaluation code under [`agents/src/eval`](../../agents/src/eval) evaluates orchestration/report behavior, failure semantics, path coverage, evidence use, Fβ, behavioral gates, and reliability. Tool-status rules remain critical: a tool that did not run is excluded/degraded rather than recorded as a clean/negative result.

Reliability fitting is meaningful only for evidence whose outcome labels and execution status are actually trustworthy. DATA/ML truth restrictions do not disappear when evidence reaches AGENTS.

### V3/chain evidence

A valid V3 registry record establishes successful contract/protocol verification for the stored proof/context/signature. It is not a ground-truth vulnerability label and cannot by itself become an ML evaluation target or automatic feedback promotion.

Current V3 feedback promotion policy remains unavailable; observations can remain pending rather than being treated as accepted outcome evidence.

## Interfaces, data shapes, and configuration

A valid evaluation/release record should name:

- commit/date/environment/mode;
- semantic policy and accepted logical V3 role/group identities;
- exact representation/token candidate and binding identity;
- checkpoint/training-config identity;
- which classes are supervised/disabled;
- exact role used for each metric or decision;
- strong/weak/masked support and exposure history;
- any threshold/calibration artifact and the authorized fitting evidence behind it;
- AGENTS tool-status coverage/modes where system metrics are reported;
- proxy/V3 identities only when proof/chain behavior is in scope;
- explicit unsupported metrics/claims.

### Current repaired-path role authority

| Evidence purpose | Current status |
|---|---|
| train strong | supported subject to accepted role/artifact binding |
| train weak | supported for policy-authorized weak positives |
| train unlabeled | explicit structural/unlabeled state; not negative truth |
| model selection | supported, **positive-only limited** |
| internal audit | supported, exposed/internal |
| threshold fit | `UNSUPPORTED_EMPTY` |
| calibration fit | `UNSUPPORTED_EMPTY` |
| untouched acceptance | `UNSUPPORTED_EMPTY_FROZEN` |

GasException and UnusedReturn remain supervision-disabled under policy v1.

## Failure modes and current limitations

- Unknown/masked rows counted as TN/negative recreate the original R4 corruption.
- Positive-only model-selection cannot support ordinary discrimination/false-positive metrics.
- Historical G6 role identity cannot silently replace accepted logical V3 authority.
- A threshold/calibration script can execute successfully while its fitting evidence remains unauthorized.
- Reusing Run12 thresholds/calibration after repaired training is invalid.
- Exposed/manual/quickstart corpora cannot be relabeled untouched acceptance.
- BCCC/tool silence is not class-specific confirmed-negative evidence.
- AGENTS benchmark gates do not substitute for missing ML acceptance data.
- V3 on-chain verification does not create vulnerability ground truth.
- D-011 physical acceptance or D-012 selector promotion does not itself establish model quality.

## Common change recipe

For any model/policy evaluation change:

1. freeze semantic, role/group, representation/token, checkpoint, and exposure identities before metrics;
2. state which outcome states/classes the metric requires;
3. verify that required positive/negative/acceptance support actually exists;
4. mask unknown/disabled/weak metric-ineligible cells according to policy;
5. preserve logical V3 group-role isolation;
6. report unsupported metrics instead of synthesizing data;
7. compare with Run12 only with explicit historical-semantic/corpus caveats;
8. if new negative/acceptance evidence is introduced, create a new versioned evidence/role decision before using it;
9. update [Current status](16_current_status.md) before increasing release/model-quality claims.

For AGENTS reliability/evaluation, continue to exclude `ran=false` tools and preserve deterministic/LLM/live/mock provenance.

## Verification commands

```bash
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
python3 docs/handbook/tools/verify_handbook.py static
cd agents && poetry run pytest -q -k 'eval or gate or reliability'
```

The G6 validator is historical compatibility evidence. Current repaired evaluation claims must additionally bind accepted logical/physical/current-run authority.

## Optional deep references

- [Architecture](01_architecture.md)
- [ML training and quality](06_ml_training_quality.md)
- [DATA artifacts](04_data_artifacts.md)
- [Cross-module contracts](11_cross_module_contracts.md)
- [Current status](16_current_status.md)
- [R4 Phase 9 plan](../plan/ml-R4/phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md)
- [R4 Phase 10 plan](../plan/ml-R4/phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md)

## Technical mastery layer

### Prerequisite knowledge

Know confusion matrices, precision/recall/Fβ, calibration, partial/positive-unlabeled labels, leakage/exposure, checkpoint selection, and evaluation-role separation.

### Source map and reading order

Start with current R4 status/policy, D-009 logical V3, D-011/D-012, and role support before reading ML/AGENTS metric utilities. Then inspect current Phase-8 selection mechanics, historical trainer/evaluation code, AGENTS metrics/gates/reliability, and later evaluation plans. Let evidence roles constrain which utilities are meaningful.

### Execution trace and worked example

If a model predicts all authorized positive model-selection examples correctly, a positive-selection/sensitivity diagnostic can be reported. Because that role has no trusted negatives, false-positive rate and full F1 cannot. A later separately reviewed negative corpus would require a versioned evidence/role decision before those metrics become authorized.

### Implementation practice

Make metric eligibility a machine-readable mask/role check, not reviewer memory. Unit-test that unknown rows, weak metric-ineligible rows, disabled classes, unsupported roles, and mismatched artifact/checkpoint identities cannot enter outcome metrics.

### Review and ownership check

Can you say which current evidence may train weights, select checkpoints, fit thresholds, fit calibration, or provide untouched acceptance—and identify which common metrics remain impossible because trusted negative/acceptance evidence is absent?