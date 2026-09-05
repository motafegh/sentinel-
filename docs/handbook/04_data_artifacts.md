# 04 — DATA artifacts and the ML seam

**Read this when:** you need to distinguish historical labels/representations from current R4 semantic state, logical V3 roles, D-011 physical representations, or the Phase-8 ML dataset boundary.

**Skip this if:** you only need current Run12 source inference.

**Estimated reading time:** 13 minutes.

## 30-second summary

SENTINEL has multiple intentionally preserved DATA generations, and their meanings must not be mixed. **Historical v1** binary exports and v9 representations remain the Run12/reproducibility seam. **DATA vNext v2** introduced explicit contract×class state, nullable targets, strength/masks, provenance, and role-aware loading. Later R4 decisions repaired the physical and logical authority further: D-008 accepts the 22,540-contract repaired physical DATA population, D-009 accepts logical V3 grouping/roles, D-010 makes v9 ineligible for a new full run, and D-011 accepts the exact V10 V2.6 physical graph lineage. D-012 requires a fresh guarded-selector successor token lineage before a future repaired training candidate can use that selector.

No repaired teacher has been promoted and full training remains unauthorized.

## Just-enough mental model

```text
historical runtime/reproduction seam
v1 binary labels + v9 graph/tokens
→ legacy SentinelDataset / Run12

repaired semantic seam
historical/source evidence
→ DATA vNext v2 explicit contract×class state
→ data-vnext-policy-v1
→ logical V3 grouping / roles (D-009)

current possible future-training physical seam
D-011 exact V10 V2.6 graph lineage
+ D-012-required fresh guarded-selector token successor (pending acceptance)
→ Phase-8 dataset/trainer only after exact candidate binding + authorization
```

For policy v1, target `0` requires real class-specific `CONFIRMED_NEGATIVE` evidence. Confirmed negatives remain zero today.

## Actual runtime/source walkthrough

### Historical v1 compatibility

The legacy export/dataset stack remains available for Run12 reproduction/runtime continuity:

- historical graph schema v9;
- historical token windows `[4,512]`;
- ten non-nullable binary label columns;
- `ml/src/datasets/sentinel_dataset.py` returns legacy `y[10]` semantics;
- historical collate/loss paths do not carry repaired masks/strength/provenance.

That seam is **not** the interface for a new repaired full training run.

### DATA vNext v2 semantic compatibility layer

The G7-published vNext implementation remains an important compatibility/evidence layer. It carries explicit per contract×class fields such as:

- canonical outcome state;
- nullable target;
- training signal/strength;
- source-policy loss eligibility;
- outcome-metric eligibility;
- provenance/evidence identity;
- dataset-role identity.

The tracked G7 export remains reproducibility evidence. Its historical G6 role identity (`r4-vnext-roles-v1`) is no longer the latest logical authority after D-009, but the semantic policy `data-vnext-policy-v1` remains current.

### Repaired physical DATA — D-008

D-008 accepts repaired-v2 physical DATA for 22,540 identities and 225,400 class cells. This is the reusable repaired physical/evidence population from which later logical/representation work proceeds.

D-008 does not mean its historical v9 graph semantics are eligible for a new full run after D-010.

### Logical V3 grouping/roles — D-009

Current leakage-group/role authority is V3:

- `r4-leakage-groups-v3`;
- `r4-vnext-roles-v3`;
- `sentinel-r4-vnext-v3`;
- logical build `r4-logical-lineage-v3`.

V3 removes arbitrary address literals from grouping authority and preserves exact artifact/normalized-code/source-namespaced family identity rules. The accepted population has 22,394 groups and maximum group size 7.

The ML adapter must consume role semantics consistently; old G6 role manifests remain historical compatibility evidence, not a reason to rebuild current roles implicitly.

### Physical representation seam — D-010 / D-011

R4-GAP-008 exposed semantic defects in historical v9 call edges. The repaired path therefore versions graph semantics:

- **v9:** immutable historical/reproducibility and Run12 compatibility only for the new-full-training decision;
- **V10 V2.6:** exact D-011 accepted physical graph lineage under extractor `v2.6-r4-call-semantics-deterministic-cfg-mutators`.

D-011 also proves exact historical-control token equivalence for its bound token tensors. That makes D-011 a stable accepted rollback/control root; it does **not** mean the new guarded selector has already been applied.

### Fresh guarded-selector successor — D-012

D-012 promotes `target_aware_guarded_v1` only for a **new versioned candidate**. The correct seam is therefore:

```text
accepted logical V3 semantics/roles
+ accepted D-011 V10 graph lineage
+ fresh D-012 guarded token selection
→ new candidate artifact identity
→ binding/review/physical acceptance
→ only then later training/evaluation authority
```

The D-011 files must not be overwritten in place.

### Current ML consumer boundary

The repaired Phase-8 dataset/training code must preserve at least:

- target / state;
- strength (`STRONG`, `WEAK`, `NONE`);
- effective loss eligibility;
- model-selection/metric eligibility;
- accepted logical role/group identity;
- representation/token lineage identity;
- policy/class-order identity.

A valid target is not automatically training-authorized; a valid representation is not automatically model-selection or full-training authority.

## Interfaces, data shapes, and configuration

### Compatibility registry

| Artifact/interface | Current role |
|---|---|
| historical v1 binary export | Run12 reproduction only |
| G7 DATA vNext v2 export | repaired semantic compatibility/evidence layer |
| `data-vnext-policy-v1` | current semantic supervision policy |
| historical `r4-vnext-roles-v1` | frozen G6/G7 reproduction evidence |
| `r4-vnext-roles-v3` | current accepted logical role authority |
| v9 representation | historical/Run12/reproducibility; ineligible for new full run |
| D-011 V10 V2.6 representation | current exact accepted physical graph/control-token authority |
| D-012 guarded-selector successor | pending new candidate; separate acceptance required |
| class order/count | locked ten-class compatibility boundary |
| token tensor shape | `[4,512]` for the current historical-control / guarded-window contract |

Class order remains `CallToUnknown`, `DenialOfService`, `ExternalBug`, `GasException`, `IntegerUO`, `MishandledException`, `Reentrancy`, `Timestamp`, `TransactionOrderDependence`, `UnusedReturn`.

GasException and UnusedReturn remain output positions while supervision is disabled under policy v1.

## Failure modes and current limitations

- Loading a historical v1 export as if it carried repaired masks/strength is semantic corruption.
- Filling repaired `null`/unknown targets with zero recreates the original R4 defect.
- Treating historical `r4-vnext-roles-v1` as current logical V3 authority can reintroduce superseded grouping assumptions.
- Treating v9 as current future-training physical authority violates D-010.
- Treating D-011 as if it already contains guarded-selector tokens violates D-012.
- Overwriting D-011 to apply a new selector destroys the accepted rollback/control identity.
- Model-selection evidence remains positive-only limited; it cannot support ordinary binary false-positive/F1/AUC claims.
- threshold/calibration/untouched-acceptance manifests remain intentionally unsupported/empty.
- a fresh clone does not guarantee protected D-011 physical files.

## Common change recipe

For DATA/ML seam changes:

1. identify whether you are touching historical compatibility, semantic state, logical roles/groups, graph bytes, token selection, or trainer consumption;
2. preserve old artifact identity and introduce a versioned successor when meaning/bytes change;
3. carry target + state + strength + masks + role/group + lineage identities together;
4. reject missing repaired fields rather than silently falling back to historical binary semantics;
5. bind new candidates to policy, source/evidence, grouping/roles, representation extractor, selector, runtime, and exact artifact hashes;
6. require separate physical acceptance before downstream training consumes a successor;
7. update [Cross-module contracts](11_cross_module_contracts.md) and [Current status](16_current_status.md) when the accepted seam changes.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

The G6 validator remains a historical compatibility check. Current logical-V3 and D-011/D-012 claims are established by their later machine-readable evidence/ADR chain, not by reinterpreting the G6 validator.

## Optional deep references

- [Architecture](01_architecture.md)
- [DATA pipeline](03_data_pipeline.md)
- [ML training and quality](06_ml_training_quality.md)
- [Cross-module contracts](11_cross_module_contracts.md)
- [Current status](16_current_status.md)
- [R4 vNext policy specification](../plan/ml-R4/findings/07_data_vnext_policy_and_design_specification.md)

## Technical mastery layer

### Prerequisite knowledge

Know PyTorch/PyG interfaces, nullable semantic state, masks, strength weighting, leakage roles, graph/token representations, Parquet, content hashes, and versioned artifact promotion.

### Source map and reading order

Read historical v1 dataset/collate only for Run12 compatibility. For current repaired semantics, follow policy → D-008 → D-009 logical V3 → D-010/D-011 representation decisions → D-012 selector decision → current `VNextTrainingDataset`/Phase-8 consumer code.

### Execution trace and worked example

A SolidiFI class assertion can establish a strong positive while the other classes remain unknown. That semantic row can be assigned through accepted logical V3 without changing historical v1 artifacts. For a later repaired model, its graph must come from the exact accepted D-011 V10 lineage while a guarded token candidate must be generated under D-012 and accepted separately; neither change authorizes threshold fitting or full training by itself.

### Implementation practice

Test representation identity, semantic-policy validity, logical-role/group identity, and trainer masks as different failure classes. Never let one mismatch degrade into a warning/default that silently selects historical behavior.

### Review and ownership check

Can you trace one contract from historical v1 compatibility through DATA vNext v2 semantics, D-009 logical V3 role/group authority, D-011 V10 graph identity, pending D-012 token successor, and the repaired ML consumer without losing which artifacts are historical, accepted, or still pending?