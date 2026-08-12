# 13 — Evaluation and release evidence

**Read this when:** you need to judge DATA quality, ML quality, AGENTS behavior, reliability weights, or release/promotion readiness.

**Skip this if:** you are only browsing architecture; do not skip it before claiming model quality or promotion.

**Estimated reading time:** 15 minutes.

## 30-second summary

Evaluation must follow evidence authority, not merely available scripts. R4 repaired the DATA truth layer and froze purpose-specific roles before retraining. For the first repaired baseline, model-selection evidence is **strong-positive but positive-only**, while threshold-fit, calibration-fit, and untouched-acceptance roles are deliberately unsupported/empty because no trustworthy confirmed-negative/unexposed corpus exists. AGENTS evaluation infrastructure remains useful, but model/release metrics must never treat unknown/masked cells as negatives or reuse historical Run12 evaluation roles as if they were vNext evidence.

## Just-enough mental model

```text
R4 DATA evidence/policy
      ↓
leakage-safe roles
      ↓
Phase 8 retrain + positive-only model-selection diagnostics
      ↓
Phase 9 evaluation constrained by available evidence
      ↓
Phase 10 promotion decision

unsupported evidence role ≠ permission to borrow another role
```

A test pass means implementation consistency. A metric is trustworthy only when its labels, role, exposure history, and semantics justify that metric.

## Actual runtime/source walkthrough

### DATA / R4 evidence quality

The current DATA evaluation foundation is the R4 chain:

- historical-source/crosswalk reconstruction;
- 224,930-row contract×class evidence ledger;
- targeted DIVE semantic review;
- accepted `data-vnext-policy-v1`;
- frozen `r4-vnext-roles-v1` leakage-group roles;
- explicit unsupported threshold/calibration/acceptance manifests.

This evidence is stronger than old split names or binary labels. Historical `test`, `NonVulnerable`, all-zero, source-absence, or tool-silence states do not become trusted negatives merely because an evaluation utility expects them.

### ML evaluation after repaired retraining

Phase 8/9 must separate:

- optimization roles (`TRAIN_STRONG`, `TRAIN_WEAK`, optional unlabeled handling);
- checkpoint-selection evidence (`MODEL_SELECTION`, positive-only limited);
- internal audit/case-study evidence;
- threshold fitting — currently unavailable;
- calibration fitting — currently unavailable;
- untouched acceptance — currently unavailable.

Positive-only model-selection can support sensitivity/positive-loss/regression diagnostics. It cannot honestly estimate false-positive rate, full F1, ROC-AUC, PR-AUC against trusted negatives, or calibrated decision probabilities.

Run12 metrics remain historical baseline evidence and may be compared descriptively only with clear corpus/semantic differences. They are not an unbiased vNext holdout.

### AGENTS evaluation

AGENTS evaluation code under [`agents/src/eval`](../../agents/src/eval) still evaluates orchestration/report behavior, failure semantics, path coverage, evidence use, Fβ, behavioral gates, and reliability. Tool-status rules remain critical: a tool that did not run is excluded rather than recorded as a negative.

Reliability fitting may be appropriate only for evidence whose outcome labels and execution status are actually trustworthy. R4 restrictions on DATA/ML truth do not disappear when evidence reaches AGENTS.

### V3/chain evidence

A valid V3 registry record establishes successful contract/protocol verification for the stored proof/context/signature. It is not a ground-truth vulnerability label and cannot by itself become an ML evaluation target or automatic feedback promotion.

Current V3 feedback promotion policy remains unavailable; observations can remain pending rather than being treated as accepted outcome evidence.

## Interfaces, data shapes, and configuration

A valid evaluation/release record should name:

- commit/date/environment/mode;
- DATA vNext policy/schema/manifest and Phase-6 role identities;
- checkpoint/training-config identity;
- which classes are supervised/disabled;
- exact role used for each metric or decision;
- strong/weak/masked support counts;
- exposure history and leakage groups;
- any threshold/calibration artifact and the authorized fitting evidence behind it;
- AGENTS tool-status coverage/modes where system metrics are reported;
- proxy/V3 identities only when chain/proof behavior is in scope;
- explicit unsupported metrics/claims.

### Current first-baseline role authority

| Evidence purpose | Current status |
|---|---|
| train strong | supported |
| train weak | supported for DIVE TOD only |
| train unlabeled | supported |
| model selection | supported, **positive-only limited** |
| internal audit | supported, exposed/internal |
| threshold fit | `UNSUPPORTED_EMPTY` |
| calibration fit | `UNSUPPORTED_EMPTY` |
| untouched acceptance | `UNSUPPORTED_EMPTY_FROZEN` |

GasException and UnusedReturn remain supervision-disabled under policy v1.

## Failure modes and current limitations

- Unknown/masked rows counted as TN/negative recreate the original R4 corruption.
- Positive-only model-selection cannot support ordinary discrimination/false-positive metrics.
- A threshold/calibration script can execute successfully while the fitting evidence is unauthorized.
- Reusing Run12 thresholds/calibration after retraining is invalid.
- Exposed manual/quickstart corpora cannot be relabeled untouched acceptance.
- Quickstart historical `NonVulnerable` semantics include invalid mappings and are not trusted negatives.
- BCCC/tool silence is not class-specific confirmed-negative evidence.
- AGENTS benchmark gates do not substitute for missing ML acceptance data.
- V3 on-chain verification does not create vulnerability ground truth.

## Common change recipe

For any model/policy evaluation change:

1. freeze artifact and role identities before metrics;
2. state which outcome states/classes the metric requires;
3. verify that required positive/negative/acceptance support actually exists;
4. mask unknown/disabled/weak cells according to policy;
5. preserve group-role isolation;
6. report unsupported metrics instead of synthesizing data;
7. compare with Run12 only with explicit historical-semantic caveats;
8. if new negative/acceptance evidence is introduced, create a new versioned evidence/role decision before using it.

For AGENTS reliability/evaluation, continue to exclude `ran=false` tools and preserve deterministic/LLM/live/mock provenance.

## Verification commands

```bash
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
python3 docs/handbook/tools/verify_handbook.py static
cd agents && poetry run pytest -q -k 'eval or gate or reliability'
```

After repaired retraining begins, Phase-8/9 commands and evidence bundles become the controlling ML evaluation artifacts.

## Optional deep references

- [ML training and quality](06_ml_training_quality.md)
- [DATA artifacts](04_data_artifacts.md)
- [Current status](16_current_status.md)
- [R4 Phase 9 plan](../plan/ml-R4/phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md)
- [R4 Phase 10 plan](../plan/ml-R4/phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md)

## Technical mastery layer

### Prerequisite knowledge

Know confusion matrices, precision/recall/Fβ, calibration, partial/positive-unlabeled labels, data leakage, exposure, checkpoint selection, and evaluation-role separation.

### Source map and reading order

Start with R4 policy and frozen support table before reading ML/AGENTS metric utilities. Then inspect historical trainer/evaluation code, AGENTS pipeline metrics/gates/reliability, and the Phase-9/10 plans. Let evidence roles constrain which utilities are meaningful.

### Execution trace and worked example

If a model predicts all strong-positive model-selection examples correctly, positive recall can be reported. Because that role has no trusted negatives, false-positive rate and full F1 cannot. A later separately reviewed negative corpus would require a new role/version before those metrics become authorized.

### Implementation practice

Make metric eligibility a machine-readable mask/role check, not reviewer memory. Unit-test that unknown rows, weak metric-ineligible rows, disabled classes, and unsupported roles cannot enter outcome metrics.

### Review and ownership check

Can you say which current roles may train weights, select checkpoints, fit thresholds, fit calibration, or provide untouched acceptance—and identify which common metrics are impossible today because negative evidence is absent?
