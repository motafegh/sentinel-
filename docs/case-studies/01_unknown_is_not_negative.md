# Case Study 01 — Unknown Is Not Negative

## Decision in one sentence

SENTINEL stopped treating historical binary `0` values, source absence, unsupported categories, or analyzer silence as evidence that a vulnerability class was absent; repaired supervision preserves those cells as unknown/unreviewed unless class-specific evidence actually supports a negative conclusion.

## Problem

The historical ML seam used ten binary class outputs. That representation was convenient for training, but a binary `0` did not consistently mean **confirmed absence of that vulnerability**.

Across the underlying sources, a zero-like state could instead mean that:

- the source did not label that class;
- the class was unsupported by that source;
- a category had been dropped or did not map cleanly into SENTINEL's ten-class taxonomy;
- no relevant assertion was present;
- a historical parser/default encoded absence as zero;
- a tool had not produced class-specific negative evidence.

If those states are consumed as trustworthy negatives, the training and evaluation pipeline manufactures information that the evidence never established.

## Evidence

The accepted R4 DATA policy makes the distinction explicit at the **contract × class** level.

`docs/plan/ml-R4/specs/data_vnext_policy_v1.json` defines:

- explicit outcome states such as `CONFIRMED_POSITIVE`, `CONFIRMED_NEGATIVE`, `UNKNOWN`, `CONFLICTING_EVIDENCE`, and `NOT_REVIEWED`;
- a nullable `target_value`;
- separate training signal and training strength;
- loss and outcome-metric eligibility;
- source claims and evidence identities.

Its invariants include:

- `historical_zero_never_implies_confirmed_negative`;
- `source_absence_never_implies_negative`;
- `unsupported_class_never_implies_negative`;
- `target_zero_requires_confirmed_negative_evidence`;
- `no_global_nonvulnerable_label_is_synthesized`.

The same policy records **no blanket negative source** for the first baseline and requires direct class-specific evidence that meaningfully assessed absence before `target_value=0` is legal.

This remains more than a design preference. The current logical-V3 acceptance record still checks that confirmed-negative rows are zero and explicitly says its PASS decision does not create confirmed-negative truth or authorize full training.

## Shortcut rejected

The easy compatibility shortcut would have been:

```text
historical zero / missing assertion / source silence
→ target = 0
→ ordinary binary loss and metrics
```

That would have preserved the old trainer interface and made conventional F1, false-positive-rate, ROC-AUC, PR-AUC, threshold, and calibration workflows easier to run.

It was rejected because the convenience comes from silently changing the meaning of the data. An unknown label cannot become a trustworthy negative merely because downstream code expects a dense binary tensor.

A second rejected shortcut was to create a global `NonVulnerable` label from all-zero or absence patterns. The accepted policy explicitly forbids that synthesis.

## Decision

Repair the semantics **before** repaired retraining rather than weakening the evidence contract to fit historical ML code.

For each contract/class cell:

- keep the canonical outcome state explicit;
- use `target_value = 0` only for `CONFIRMED_NEGATIVE`;
- use `target_value = null` when no authorized target exists;
- keep weak training signal separate from confirmed outcome truth;
- carry loss eligibility and outcome-metric eligibility separately;
- preserve source claims, evidence IDs, policy decision identity, and limitations;
- require downstream ML adapters to consume the repaired masks/strength rather than silently falling back to historical zero semantics.

Historical artifacts remain preserved for Run12 compatibility/reproduction. They are not retroactively rewritten to pretend they always carried the repaired semantics.

## Implementation and validation

The repaired DATA/ML seam uses a versioned policy/export contract rather than mutating the historical binary export.

The policy requires a repaired ML projection to carry, per class:

- nullable target;
- training strength;
- loss eligibility;
- outcome-metric eligibility;
- outcome state;
- policy decision identity.

The Phase-8 training mechanics were subsequently designed around explicit target/mask/strength handling instead of making DATA conform to the legacy dense-binary contract.

Current machine-readable validation also checks the later logical-V3 authority and confirms that the accepted state still has zero confirmed-negative rows. The canonical handbook keeps the consequence visible at the DATA/ML and evaluation seams.

## Result

The important result is a **more truthful supervision contract**, not a model-quality number.

SENTINEL can now distinguish:

```text
confirmed positive
confirmed negative
unknown / unsupported / not reviewed
weak training evidence
metric-eligible outcome evidence
```

rather than forcing all cells into `0` or `1`.

That prevents unsupported zero-like states from entering repaired training or evaluation as trusted negatives and keeps uncertainty visible through the pipeline.

This decision also explains why the current repaired path deliberately refuses several attractive evaluation claims: with zero accepted confirmed negatives, positive-only model-selection evidence cannot honestly establish false-positive rate, full binary F1, ROC-AUC, PR-AUC, or calibrated decision probabilities.

## Remaining limitation

The semantic repair does **not** solve the shortage of trustworthy negative evidence.

Confirmed negatives remain zero in the current accepted state. Candidate review work may eventually establish a class-specific negative, but primary-review support alone does not change accepted truth; genuinely independent agreement is still required before such a case can become authorized negative evidence.

Therefore:

- threshold fitting remains unsupported;
- calibration fitting remains unsupported;
- untouched acceptance remains unsupported/empty;
- no repaired teacher has been promoted;
- full repaired training remains unauthorized.

The project chose to expose that limitation rather than manufacture a conventional evaluation set from unsupported zeros.

## Evidence trail

Primary machine-readable authority:

- [`data_vnext_policy_v1.json`](../plan/ml-R4/specs/data_vnext_policy_v1.json) — accepted semantic policy, state model, target rules, source authority, negative-authority boundary, aggregation rules, and export contract.
- [`logical_v3_acceptance.json`](../plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_acceptance.json) — current logical-V3 acceptance checks, including zero confirmed-negative rows and no full-training authorization.
- [`current_r4.json`](../handbook/_meta/current_r4.json) — compact machine contract used by current handbook CI.

Canonical explanations:

- [DATA artifacts and the ML seam](../handbook/04_data_artifacts.md)
- [ML training, quality, interpretability, and MLOps](../handbook/06_ml_training_quality.md)
- [Evaluation and release evidence](../handbook/13_evaluation.md)
- [Current status and gap ledger](../handbook/16_current_status.md)

Historical compatibility remains intentionally separate from this repaired semantic authority; Run12 remains a historical operational baseline rather than an R4-retrained teacher.
