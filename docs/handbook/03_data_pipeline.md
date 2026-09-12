# 03 — DATA pipeline

**Read this when:** you need to understand Solidity acquisition/representation mechanics, historical labels, or the current repaired R4 DATA lifecycle.

**Skip this if:** you only consume an already accepted, hash-bound artifact and do not change DATA semantics or representations.

**Estimated reading time:** 15 minutes.

## 30-second summary

SENTINEL still retains its historical ten-stage DATA lifecycle, but **current DATA/ML authority is the R4 evidence/semantic/representation chain**, not the old binary-label pipeline. Historical G0–G7 remain immutable reproducibility evidence. R4-D-008 accepts repaired-v2 physical DATA, R4-D-009 accepts logical V3 grouping/roles, R4-D-010 withdraws v9 from eligibility for a new full training run, R4-D-011 accepts the exact V10 V2.6 physical representation lineage, and R4-D-012 allows `target_aware_guarded_v1` only in a fresh successor candidate requiring separate physical acceptance.

Full repaired training remains unauthorized. Historical source/label/export tooling is useful for lineage and mechanics but must not be mistaken for the current R4 build authority.

## Just-enough mental model

```text
Historical mechanics / reproducibility
upstream → ingest → preprocess → represent → label/verify/split/export
                                      ↓
                              historical G0–G7 artifacts

Current R4 authority
historical evidence + source evidence
→ explicit contract×class semantic state
→ data-vnext-policy-v1
→ repaired-v2 physical DATA (D-008)
→ logical V3 grouping / roles (D-009)
→ v9 withdrawn from new-full-training eligibility (D-010)
→ exact V10 V2.6 physical representation accepted (D-011)
→ fresh guarded-selector successor required (D-012)
→ later objective/evaluation/training work only if separately authorized
```

**Historical zero is not a confirmed negative.** Absence, unsupported class, source silence, dropped/out-of-taxonomy mapping, queue membership, or analyzer silence cannot silently become a negative target.

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

`sentinel-data run` walks the historical stage registry; freshness remains separate. The old CLI lifecycle is not a one-command constructor for the currently accepted R4-D-011 physical lineage.

### Historical R4/G7 foundations that remain evidence

R4 originally reconstructed the 22,493-contract historical population into a contract×class evidence model. The Phase-3 ledger contained **224,930** historical contract×class rows. G5 accepted `data-vnext-policy-v1`; G6 froze the historical first repaired role partition; G7 published the additive DATA vNext v2 overlay and representation binding.

Those records remain immutable evidence and compatibility anchors. They are **not** the latest physical/logical authority for a future full training run.

### Current repaired physical DATA — R4-D-008

R4-D-008 accepts the repaired-v2 physical DATA population as immutable reproducibility evidence:

- 22,540 contracts;
- 225,400 contract×class semantic rows;
- 67,620 graph/token/sidecar files;
- repaired-v2 physical binding digest recorded by the current R4 authority.

The repaired-v2 physical graph lineage remains historical/reproducibility evidence after D-010; its semantic/evidence repairs are not erased by later representation-version changes.

### Current logical authority — R4-D-009

The old V2 leakage grouping was superseded after arbitrary Ethereum address coincidence produced an invalid 10,327-contract connected component. Accepted logical V3 uses defensible grouping authority:

- exact artifact identity: global;
- normalized-code identity: global;
- explicit source-native family/project IDs: source-namespaced;
- Ethereum address literals: diagnostic-only, never grouping authority.

Current logical identities are `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3`, and `r4-logical-lineage-v3`.

### Current physical representation authority — R4-D-010 / D-011 / D-012

R4-GAP-008 showed that historical graph schema v9 materially conflated/omitted important call semantics. The resolution is versioned rather than patched in place:

1. **R4-D-010:** v9 remains immutable historical/reproducibility evidence but is ineligible for the new full training run.
2. **R4-D-011:** accepts only the exact 22,540-identity V10 V2.6 physical representation root produced by extractor `v2.6-r4-call-semantics-deterministic-cfg-mutators` and its recorded binding digest.
3. **R4-D-012:** authorizes `target_aware_guarded_v1` only for a fresh versioned successor token lineage. The accepted D-011 root is not mutated in place.
4. The successor still requires generation, binding, review, and separate physical acceptance before later objective/evaluation/training decisions.

### Current semantic supervision boundary

Policy v1 still preserves uncertainty:

- approved SolidiFI / SmartBugs source assertions may establish class-specific strong positives;
- DIVE Front Running→TransactionOrderDependence remains weak-positive authority only under the accepted policy;
- GasException and UnusedReturn supervision remain disabled pending evidence;
- confirmed negatives remain zero;
- candidate #2 has primary-review support only and remains UNKNOWN / target `None` until genuinely independent agreement;
- threshold-fit, calibration-fit, and untouched-acceptance roles remain unsupported/empty.

The DATA layer therefore cannot manufacture the missing negative/evaluation evidence merely to satisfy downstream training APIs.

## Interfaces, data shapes, and configuration

Two representation generations must be distinguished:

| Boundary | Meaning |
|---|---|
| historical v9 graph + `[4,512]` tokens | immutable G7/repaired-v2/Run12 compatibility and reproducibility evidence |
| accepted V10 V2.6 graph lineage | current R4-D-011 physical representation authority for a possible future repaired run |
| guarded-selector successor tokens | R4-D-012-required fresh candidate; not yet separately physically accepted |

The semantic unit is `contract_id × class_index`, carrying explicit outcome/training/provenance state rather than only `0/1`. Current leakage-safe dataset role authority is logical V3, not the historical G6 `r4-vnext-roles-v1` partition.

The controlling repair artifacts live under [`docs/plan/ml-R4`](../plan/ml-R4): policies, evidence ledgers, logical/physical lineage records, ADRs, role manifests, decision registers, reviews, and gate state.

## Failure modes and current limitations

- Running the historical label/merge/export CLI does not reproduce current R4 semantics or D-011 bytes automatically.
- Historical `0` is not reusable as a negative target.
- Historical G6/G7 roles remain reproducibility evidence but are not the current logical V3 role authority.
- Treating v9 as eligible for a new full run violates R4-D-010.
- Treating D-011 physical acceptance as training authorization violates its acceptance scope.
- Treating D-012 selector promotion as an in-place mutation of D-011 destroys rollback/evidence identity.
- No confirmed-negative population means threshold/calibration/untouched acceptance remain unsupported for the repaired path.
- Heavy accepted physical representations are protected/local artifacts and are not promised by a fresh Git clone.

## Common change recipe

For a DATA semantic or representation change:

1. identify the controlling R4 decision/policy/ADR;
2. classify whether the change affects historical mechanics, semantic state, logical grouping/roles, graph representation, token selection, or more than one layer;
3. never edit historical/accepted artifacts in place;
4. preserve source/evidence lineage and explicit unknown/no-target states;
5. generate a new versioned candidate when semantics/bytes change;
6. bind exact artifact identity/digests and validate the full affected population;
7. require separate acceptance/promotion before downstream training/evaluation consumes the successor;
8. update [Current status](16_current_status.md) and cross-module contracts if the accepted boundary changes.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

The G6 validator proves historical partition compatibility only. Current V3/D-011/D-012 acceptance claims require their current R4 evidence/review records; do not interpret an old gate validator as proof of the latest lineage.

## Optional deep references

- [Architecture](01_architecture.md)
- [DATA artifacts / ML seam](04_data_artifacts.md)
- [ML training and quality](06_ml_training_quality.md)
- [Evaluation](13_evaluation.md)
- [Current status](16_current_status.md)
- [R4 master plan](../plan/ml-R4/00_MASTER_PLAN.md)

## Technical mastery layer

### Prerequisite knowledge

Know content hashes, multi-label/partial-label semantics, unknown-vs-negative state, leakage grouping, masks/roles, graph/token representations, provenance, and versioned artifact acceptance.

### Source map and reading order

For historical mechanics: `data_module/sentinel_data/cli.py` → preprocessing → historical representation/export. For current DATA authority: `PLAN_STATUS_MATRIX.md` → `data-vnext-policy-v1` → D-008 → D-009 → D-010 → D-011 → D-012, then the current `sentinel_data.vnext` / representation sources needed for the specific task.

### Execution trace and worked example

A historical DIVE cell may preserve its original source assertion while remaining `target=null` under the repaired policy. A contract can belong to the accepted logical V3 grouping while its historical v9 representation remains only a reproduction root. For a possible future repaired run, the exact D-011 V10 graph bytes are the accepted physical graph lineage, and a new guarded-selector token candidate must be accepted separately before use.

### Implementation practice

Never “repair” DATA by manually editing a target, group, graph, or token artifact. Repair the owning policy/implementation, create a versioned candidate, bind it to exact inputs/code/runtime, validate it, and promote it through the appropriate decision boundary.

### Review and ownership check

Can you distinguish historical mechanics, G7 compatibility evidence, D-008 repaired physical DATA, D-009 logical V3 authority, D-011 accepted V10 physical representation, and the still-pending D-012 guarded-token successor—and state what each one does **not** authorize?