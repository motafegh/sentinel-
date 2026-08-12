# 04 — DATA artifacts and the ML seam

**Read this when:** you need graph/token files, historical labels, DATA vNext semantic state, frozen roles, or the ML dataset boundary.

**Skip this if:** you only need runtime source inference.

**Estimated reading time:** 12 minutes.

## 30-second summary

SENTINEL now has two different DATA semantics that must not be mixed. Historical export v1 stores ten non-nullable binary class columns and fed Run12 through the legacy `SentinelDataset`/collate/loss path. R4 preserves that bundle for reproducibility but supersedes its label meaning for new training. DATA vNext v2 carries explicit contract×class outcome state, nullable target, training strength, loss/metric eligibility, provenance, and frozen role. The v9 graph/token representations remain the same physical representation layer.

## Just-enough mental model

```text
physical representation (unchanged by R4)
contract → graph x[N,12] + tokens[4,512]

historical v1 semantic seam
class_0..class_9 = non-nullable 0/1
→ y[10]
→ every 0 reached binary loss as negative

DATA vNext v2 semantic seam
contract×class state
→ target {1,0,null}
→ strength {STRONG,WEAK,NONE}
→ source-policy loss eligibility
→ frozen dataset role
→ effective training/metric masks
```

For policy v1, target `0` requires a real `CONFIRMED_NEGATIVE`. No recovered blanket negative source satisfies that rule.

## Actual runtime/source walkthrough

### Historical v1 compatibility

The existing legacy export/dataset stack remains source-compatible for reproducing Run12:

- representation schema v9;
- graph `[N,12]` and tokens `[4,512]`;
- historical ten binary label columns;
- `ml/src/datasets/sentinel_dataset.py` returns `y[10]`;
- historical collate/loss paths do not carry vNext masks/strength.

This is **historical compatibility**, not the target interface for repaired retraining.

### R4 semantic artifacts already canonical on main

- Phase-3 evidence ledger: 22,493 contracts × 10 classes = 224,930 rows;
- `data-vnext-policy-v1`: outcome/training/source authority contract;
- `data_vnext_label_state_v1.schema.json`: contract×class semantic schema;
- `r4-vnext-roles-v1`: one frozen role per leakage group/contract;
- support/unsupported/acceptance manifests.

Frozen role counts are recorded in [current status](16_current_status.md) and the R4 manifests, not duplicated as mutable split logic.

### Phase-7 v2 implementation

The active Phase-7 branch implements an **additive semantic overlay** rather than copying all graph/token tensors. It builds:

- canonical long-form contract×class label-state Parquet;
- derived per-contract ten-class ML projection carrying target/strength/masks/state/policy identity;
- manifest and validation/binding reports;
- explicit v2 loader that rejects silent fallback to historical v1 semantics.

Remote semantic generation is deterministic. Final G7 requires local binding to the existing 21,657 represented contracts before the v2 candidate can be promoted/merged.

## Interfaces, data shapes, and configuration

### Locked representation/class compatibility

| Contract | Current value |
|---|---:|
| graph schema | `v9` |
| node feature dim | 12 |
| node types | 14 |
| edge types | 12 |
| class count | 10 |
| token shape | `[4,512]` |

Class order remains `CallToUnknown`, `DenialOfService`, `ExternalBug`, `GasException`, `IntegerUO`, `MishandledException`, `Reentrancy`, `Timestamp`, `TransactionOrderDependence`, `UnusedReturn`.

### vNext semantic fields

The canonical semantic row includes at least:

- `contract_id`, class index/name;
- historical state and source claims;
- canonical outcome state;
- nullable target;
- training signal/strength;
- source-policy loss eligibility;
- outcome-metric eligibility;
- role eligibility / policy decision / evidence IDs / limitations.

The final effective training mask also requires a compatible Phase-6 role. A valid target is not automatically training- or metric-authorized.

### Current supervision state

Eight classes have approved strong-positive source support. GasException and UnusedReturn remain supervision-disabled pending evidence. DIVE Front Running→TOD is weak-positive only. No confirmed-negative rows are authorized in policy v1.

## Failure modes and current limitations

- Loading a historical v1 export as if it carried vNext masks is semantic corruption.
- Filling vNext `null`/unknown targets with zero recreates the original R4 defect.
- GasException/UnusedReturn output indices still exist; missing supervision must not be converted into zeros.
- model-selection support is positive-only limited; it is not a trustworthy full binary validation set.
- threshold/calibration/untouched-acceptance manifests are intentionally empty/unsupported.
- Phase-7 v2 is not canonical main until G7 local representation binding passes and the branch merges.

## Common change recipe

For DATA/ML seam changes:

1. identify v1 compatibility vs v2 repaired behavior explicitly;
2. never mutate historical export files in place;
3. preserve class order and graph schema unless separately versioned;
4. carry target + strength + masks + policy/role identities together;
5. fail if required v2 fields are absent rather than defaulting to v1;
6. bind generated semantic artifacts to policy, partition, ledger, code, and physical representations;
7. update trainer compatibility in Phase 8 rather than weakening v2 semantics.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

After G7 merges, use the committed vNext CLI/validator for v2 artifact verification. Historical `SentinelDataset` tests remain relevant only to the v1 compatibility seam until Phase-8 trainer/dataset compatibility is added.

## Optional deep references

- [DATA pipeline](03_data_pipeline.md)
- [ML training and quality](06_ml_training_quality.md)
- [Cross-module contracts](11_cross_module_contracts.md)
- [R4 vNext policy specification](../plan/ml-R4/findings/07_data_vnext_policy_and_design_specification.md)
- [R4 Phase-6 partition finding](../plan/ml-R4/findings/08_phase6_role_partition_and_acceptance_freeze.md)

## Technical mastery layer

### Prerequisite knowledge

Know PyTorch/PyG tensor shapes, nullable semantic state, masks, loss eligibility, leakage roles, Parquet, and content hashes.

### Source map and reading order

Read v9 representation constants/orchestrator first. Treat `ml/src/datasets/sentinel_dataset.py` and `collate.py` as historical v1 consumers today. For repaired semantics, read R4 policy/schema/partition artifacts; after G7, the additive `data_module/sentinel_data/vnext` package is the v2 implementation source.

### Execution trace and worked example

The same represented contract can have graph/token bytes unchanged while its training semantics change from historical `[0,1,...]` to explicit per-class states. A masked DIVE DoS claim contributes no DenialOfService target; the contract can still live in an unlabeled role. A SolidiFI injected class can contribute a strong positive target without turning the other nine classes into negatives.

### Implementation practice

Test three different failures separately: representation/schema mismatch, semantic-policy violation, and artifact/hash binding failure. Do not let any of them degrade into a warning/default label.

### Review and ownership check

Can you state which artifacts are physical representation, historical v1 semantic compatibility, R4 policy/roles, and v2 repaired semantic projection—and prove that a missing v2 field cannot silently become an old binary label?
