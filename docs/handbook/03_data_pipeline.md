# 03 — DATA pipeline

**Read this when:** you need to understand Solidity acquisition/representation, historical labels, or the repaired DATA vNext path.

**Skip this if:** you only consume a hash-bound approved export and do not change DATA semantics.

**Estimated reading time:** 15 minutes.

## 30-second summary

SENTINEL still has the historical ten-stage DATA lifecycle, but **new DATA/ML work is governed by R4 rather than trusting the historical binary label path**. R4 reconstructed the 22,493-contract population into a contract×class evidence ledger, accepted `data-vnext-policy-v1`, froze leakage-safe roles, and is implementing an additive v2 semantic overlay. Historical v1 exports remain immutable compatibility evidence. The old label CLI is still incomplete and must not be described as the repaired vNext build path.

## Just-enough mental model

```text
Historical acquisition/representation lifecycle
upstream → ingest → preprocess → represent
                         ↓
                 graph/token artifacts

Historical label/export path (v1 compatibility)
label/crosswalk/merge → split/export → Run12 training

Current repair path (R4)
historical labels + source evidence
→ contract×class evidence ledger
→ data-vnext-policy-v1
→ r4-vnext-roles-v1
→ DATA vNext v2 semantic overlay
→ later masked/strength-aware ML retraining
```

**Historical zero is not a confirmed negative.** More generally, absence, unsupported classes, unknown states, dropped categories, and other historical-zero mechanisms cannot be silently converted into negative training targets.

## Actual runtime/source walkthrough

### Historical ten-stage lifecycle

The historical CLI registry remains:

1. ingest
2. preprocess
3. represent
4. label
5. verify
6. split
7. register
8. analyze
9. export
10. freshness

`sentinel-data run` walks the nine `STAGES`; freshness remains separate. `cli.py::_run_label` is still a placeholder. Lower-level labeling libraries/parsers exist, but this does not make the old one-command label stage an authoritative repaired build.

### What R4 changed

R4 did not delete old contracts or representations. It repaired what DATA is allowed to **claim** about each contract/class:

- Phase 0–2 reconstructed source/crosswalk/zero semantics;
- Phase 3 materialized 224,930 contract×class ledger rows;
- Phase 4 reviewed the decision-critical DIVE gap;
- Phase 5 accepted explicit outcome/training-state semantics;
- Phase 6 froze one leakage-safe role per group;
- Phase 7 builds the v2 semantic/export overlay and is pending final local representation binding/G7.

### Accepted source policy for the first repaired baseline

- SolidiFI injected class: strong positive for that class only; other classes unknown.
- approved SmartBugs Curated in-taxonomy category: strong positive for that category only; other classes unknown.
- DIVE: unlabeled/masked except Front Running→TransactionOrderDependence as weak-positive training signal.
- historical DIVE zeros remain unknown.
- SmartBugs `bad_randomness`, `short_addresses`, and `other` produce no canonical vNext target.
- Web3Bugs/DISL unavailable; BCCC/DeFiHackLabs deferred for the first baseline.
- GasException and UnusedReturn supervision are disabled pending evidence.
- no blanket confirmed-negative source exists.

## Interfaces, data shapes, and configuration

The graph/token representation contract remains v9 and unchanged by R4. The new semantic unit is `contract_id × class_index`, carrying explicit outcome/training/provenance state instead of only `0/1`.

The frozen role layer uses groups rather than old train/val/test names as semantic authority. Roles include `TRAIN_STRONG`, `TRAIN_WEAK`, `TRAIN_UNLABELED`, `MODEL_SELECTION`, `INTERNAL_AUDIT`, and `EXCLUDED`; threshold/calibration/untouched-acceptance roles are controlled empty/unsupported in policy v1.

The controlling repair artifacts live under [`docs/plan/ml-R4`](../plan/ml-R4): evidence ledger, policy/schema, ADRs, role manifests, decisions, risks, and gate status.

## Failure modes and current limitations

- Running historical label/merge code does not produce DATA vNext semantics.
- Historical `0` is not reusable as a negative target.
- Historical train/val/test membership is lineage evidence, not the vNext role freeze.
- 836 contracts belong to incomplete-representation groups and are frozen `EXCLUDED` for the first baseline.
- no confirmed-negative population means threshold-fit, calibration-fit, and untouched acceptance are unavailable for the first repaired baseline.
- Phase 7 is not G7-complete until 21,657 required physical representations are locally bound and validated.

## Common change recipe

For a DATA semantic change:

1. identify the controlling R4 decision/policy/ADR;
2. never edit historical v1 artifacts in place;
3. preserve source-native claim and evidence lineage;
4. make unknown/no-target/masked states explicit;
5. update leakage-group roles if eligibility changes;
6. regenerate a new versioned vNext artifact and validate hashes/counts;
7. reassess ML retraining/evaluation roles before promotion.

For representation-only changes, use the historical ten-stage source ownership but version graph/token schema independently from label policy.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Phase-7 vNext validation commands belong to the active R4 Phase-7 branch until G7 is merged.

## Optional deep references

- [DATA artifacts](04_data_artifacts.md)
- [ML training and quality](06_ml_training_quality.md)
- [Evaluation](13_evaluation.md)
- [R4 master plan](../plan/ml-R4/00_MASTER_PLAN.md)
- [R4 DATA vNext specification](../plan/ml-R4/findings/07_data_vnext_policy_and_design_specification.md)

## Technical mastery layer

### Prerequisite knowledge

Know content hashes, multi-label classification, unknown-vs-negative semantics, leakage grouping, masks, provenance, and versioned artifacts.

### Source map and reading order

For historical representation mechanics: `cli.py` → preprocessing → representation. For current DATA semantics: R4 evidence ledger → `specs/data_vnext_policy_v1.json` → `manifests/p6_partition_manifest.json`. After G7, the additive `sentinel_data.vnext` package becomes the implementation owner for v2 semantics.

### Execution trace and worked example

A DIVE contract historically labeled DoS no longer automatically becomes `DenialOfService=1`, and its other nine cells never become negatives. Under vNext policy, the source assertion can be masked while the contract remains useful as unlabeled structure. By contrast, a SolidiFI injected Reentrancy contract can contribute a strong positive Reentrancy target while its non-injected classes remain unknown.

### Implementation practice

Never “repair” DATA by editing a Parquet target value manually. Repair source/evidence policy, regenerate a versioned semantic artifact, validate group/role invariants, then retrain consumers explicitly.

### Review and ownership check

Can you distinguish the historical ten-stage mechanics from the current R4 semantic authority, and explain why the same Solidity/representation corpus can support a different trustworthy training contract without rewriting history?
