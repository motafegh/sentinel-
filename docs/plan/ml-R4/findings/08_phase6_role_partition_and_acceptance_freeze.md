# 08 — Phase 6 Role Partition and Acceptance Freeze

- **Phase:** R4 Phase 6 — Dataset Roles, Leakage-Safe Partitions, and Acceptance Freeze
- **Partition version:** `r4-vnext-roles-v1`
- **Ledger root:** `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`
- **Policy:** `data-vnext-policy-v1`
- **Status:** FROZEN CANDIDATE FOR G6

## 1. Population and leakage groups

The committed Phase-3 population contains 22,493 contracts / 224,930 contract×class rows.

Phase-6 grouping uses:

`project_group_id → dedup_group_id → contract_id`

The resulting population contains **13,509 groups**.

Inventory before representation filtering:

| Eligibility | Groups |
|---|---:|
| Strong eligible | 350 |
| Weak TOD eligible | 492 |
| Unlabeled | 12,667 |

No group spanned multiple historical train/val/test splits and no group spanned multiple sources in the recovered ledger. Historical split identity was nevertheless discarded as a role assignment; Phase 6 freezes new purpose-specific roles.

## 2. Representation eligibility

The active export contains 836 contracts without current representations. Instead of silently dropping them after role assignment, Phase 6 excludes the **entire leakage group** whenever any member lacks a representation.

Frozen exclusion:

- 836 contracts
- 835 groups
- role: `EXCLUDED`
- reason: `EXCLUDED_NO_COMPLETE_REPRESENTATION_GROUP`

This preserves the Phase-2/3 836-contract finding while preventing a mixed representation family from being partly trainable and partly excluded.

## 3. Frozen role counts

| Role | Groups | Contracts | Evidence meaning |
|---|---:|---:|---|
| TRAIN_STRONG | 238 | 275 | strong confirmed-positive source signal on at least one enabled class; other cells remain masked/unlabeled according to policy |
| MODEL_SELECTION | 51 | 56 | strong positive holdout; positive-only limited diagnostics |
| INTERNAL_AUDIT | 51 | 62 | strong positive holdout for internal audit |
| TRAIN_WEAK | 465 | 773 | DIVE Front Running→TOD weak-positive group; other cells masked/unlabeled |
| TRAIN_UNLABELED | 11,869 | 20,491 | structurally valid represented active contracts with no authorized supervised signal |
| EXCLUDED | 835 | 836 | incomplete representation group |

Totals:

- **13,509 groups exactly once**
- **22,493 contracts exactly once**

The three strong-positive roles preserve at least one represented group for each of the eight enabled supervised classes.

## 4. Strong-positive support after freeze

The represented strong-group inventory is:

| Class | Represented strong groups |
|---|---:|
| CallToUnknown | 70 |
| DenialOfService | 6 |
| ExternalBug | 51 |
| IntegerUO | 57 |
| MishandledException | 35 |
| Reentrancy | 47 |
| Timestamp | 35 |
| TransactionOrderDependence | 39 |
| GasException | 0 — supervision disabled |
| UnusedReturn | 0 — supervision disabled |

The rarest enabled class is DenialOfService with six groups. The deterministic class-coverage rule successfully preserves DoS support in TRAIN_STRONG, MODEL_SELECTION, and INTERNAL_AUDIT.

## 5. Weak support

After representation filtering, `TRAIN_WEAK` contains 465 groups / 773 contracts. Its only authorized weak-positive semantic signal is DIVE Front Running→TransactionOrderDependence.

No other DIVE historical positive is resurrected by the partition layer.

## 6. SmartBugs Timestamp ambiguity

The Phase-3 ledger cannot recover the original SmartBugs category for historical Timestamp positives. Historical Timestamp may represent direct `time_manipulation` or the superseded `bad_randomness→Timestamp` mapping.

Phase 6 identifies **13 SmartBugs Timestamp contracts** in this ambiguous state and withholds them from strong supervision. They remain unlabeled/masked rather than guessed.

This is a deliberate loss of potentially useful positives in exchange for semantic correctness.

## 7. Confirmed-negative support

**Zero policy-approved confirmed-negative rows exist in every frozen role.**

This is an expected consequence of repairing the historical zero corruption. It means the first repaired baseline currently supports positive/weak-positive/unlabeled learning, not ordinary trusted binary class discrimination.

The partition layer does not repair this by inventing negatives.

## 8. Model-selection limitation

MODEL_SELECTION contains strong confirmed positives but no confirmed negatives.

It may be used for:

- positive-only loss diagnostics;
- positive recall/sensitivity diagnostics;
- checkpoint regressions on supported positive classes.

It may **not** honestly support:

- F1 as a full binary discrimination objective;
- ROC-AUC/PR-AUC requiring trustworthy negatives;
- false-positive rate;
- threshold fitting;
- calibration fitting.

Phase 8 must choose checkpoint-selection logic compatible with this evidence structure rather than reusing historical validation metrics blindly.

## 9. Threshold and calibration roles

Both are frozen empty:

- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`

Reason: no trustworthy class-specific negative support exists under policy v1.

The empty roles are a controlled product limitation, not a missing implementation step.

## 10. Untouched acceptance audit

### Historical active train/val/test

Not eligible. These groups were part of prior model development/evaluation and were defined before repaired label semantics.

### `manual_hand_written_contracts`

Not untouched. The repository README explicitly states that the suite was created to validate ML predictions and AGENTS behavior. It is therefore exposed even where `// expect:` labels are explicit.

It may remain useful for qualitative internal audit/case studies, but Phase 6 does not import it as acceptance.

### quickstart Tier A

Not trustworthy negative evidence. The historical builder maps:

- SmartBugs `access_control` → `NonVulnerable`
- SolidiFI `tx.origin` → `NonVulnerable`

Both contradict the current canonical ExternalBug mapping. Therefore quickstart `NonVulnerable` cannot support threshold/calibration/acceptance.

### Tier E safe design

Not confirmed-negative evidence. The builder requires BCCC `NonVulnerable` plus no Slither/Aderyn high/medium findings. BCCC folder membership and tool silence do not establish class-specific absence. No committed Tier-E manifest exists in the current quickstart output.

### Unavailable/deferred sources

Web3Bugs is unavailable. BCCC/DeFiHackLabs are deferred and cannot be silently imported during partitioning.

### Acceptance decision

The frozen untouched-acceptance manifest is:

```text
status      = UNSUPPORTED_EMPTY_FROZEN
frozen      = true
contract_ids = []
group_ids    = []
```

This is the only defensible first-baseline decision from the recovered evidence.

## 11. Consequence for later gates

G6 may pass with an empty unsupported acceptance manifest because the Phase-6 contract explicitly permits that outcome.

However, later R4 work must preserve the limitation:

- Phase 7 may implement/retrain-ready data without inventing acceptance;
- Phase 8 may retrain, but checkpoint selection remains positive-only limited;
- Phase 9 cannot claim trustworthy threshold calibration unless new authorized negative evidence is introduced in a later versioned plan;
- Phase 10 cannot claim untouched-acceptance promotion for this baseline until a separately protected acceptance corpus exists.

The correct final result may therefore be a technically repaired/trained baseline that is **not promotion-eligible** under full untouched-acceptance criteria. That is preferable to manufacturing evidence.

## 12. Controlling artifacts

- `manifests/p6_role_support_inventory.json`
- `manifests/p6_group_eligibility_inventory.jsonl`
- `manifests/p6_role_group_manifest.jsonl`
- `manifests/p6_contract_role_manifest.jsonl`
- `manifests/p6_role_support_table.json`
- `manifests/p6_unsupported_roles.json`
- `manifests/p6_untouched_acceptance_manifest.json`
- `manifests/p6_partition_manifest.json`
- `ADR-R4-006-role-partition-and-acceptance-freeze.md`
