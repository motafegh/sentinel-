# Label State and Dataset Role Policy

## Unit of truth

The canonical audit unit is:

```text
contract_id × vulnerability_class
```

A contract does not receive one global confidence tier that silently applies to all classes.

## Outcome states

- `CONFIRMED_POSITIVE`: sufficient evidence that the class applies.
- `CONFIRMED_NEGATIVE`: sufficient evidence that the class was meaningfully assessed and does not apply.
- `UNKNOWN`: evidence is insufficient.
- `NOT_APPLICABLE`: the class cannot meaningfully apply under the defined scope.
- `CONFLICTING_EVIDENCE`: credible evidence supports incompatible conclusions.
- `NOT_REVIEWED`: no adequate review has occurred.
- `INVALID_RECORD`: source or artifact integrity prevents interpretation.

## Historical state

Historical binary labels remain recorded separately:

- `HISTORICAL_POSITIVE`
- `HISTORICAL_ZERO`
- `HISTORICAL_MISSING`

A historical zero must never be silently promoted to `CONFIRMED_NEGATIVE`.

## Evidence strength

Evidence strength and dataset use are separate.

Suggested evidence categories:

- `EXPLOIT_OR_INJECTION_VERIFIED`
- `EXPERT_MANUAL_REVIEW`
- `REPRODUCIBLE_STATIC_REASONING`
- `DYNAMIC_TOOL_SUPPORT`
- `STATIC_TOOL_SUPPORT`
- `SOURCE_ASSERTION`
- `TRANSFORMATION_DEFAULT`
- `NO_EVIDENCE`

No category automatically determines the outcome without class-specific interpretation.

## Dataset roles

Contract-class outcomes may be eligible for:

- `TRAIN_STRONG`
- `TRAIN_WEAK`
- `TRAIN_UNLABELED`
- `MODEL_SELECTION`
- `THRESHOLD_FIT`
- `CALIBRATION_FIT`
- `INTERNAL_AUDIT`
- `UNTOUCHED_ACCEPTANCE`
- `CASE_STUDY`
- `EXCLUDE_OUTCOME_METRICS`

## Minimum default eligibility

- Confirmed positive/negative with traceable evidence: potentially eligible.
- Unknown/not-reviewed/conflicting: masked for supervised outcome loss and metrics.
- Tool-only evidence: not eligible for untouched acceptance by default.
- Historical conclusion without raw evidence: may inform gap prioritization but not acceptance.
- Source-native injected vulnerabilities: class-specific review required to establish what the negative counterpart means.
- Unlabeled but structurally valid contracts: may remain available for unsupervised or semi-supervised use under an approved plan.

## Partition independence

A leakage group must not cross incompatible roles. Leakage groups should include, where applicable:

- exact duplicates;
- normalized-source duplicates;
- project/repository family;
- injected vulnerable/fixed pair;
- compiler-generated variants;
- template/token family;
- source family when source shortcuts are material.

## Policy changes

Any change to these states or role rules requires an ADR and schema version bump.
