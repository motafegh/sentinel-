# 07 — DATA vNext Policy and Design Specification

- **Phase:** R4 Phase 5 — DATA vNext Policy and Design
- **Policy:** `data-vnext-policy-v1`
- **Export format target:** `v2`
- **Graph feature schema:** remains `v9`
- **Class vocabulary:** locked ten-class order, unchanged
- **Implementation status:** PROHIBITED until G6/Phase 7 authorization

## 1. Purpose

This specification converts the evidence recovered in R4 Phases 0–4 into an implementation-complete semantic contract.

The historical training target was a ten-cell binary vector. That representation was not capable of distinguishing a confirmed negative from unknown, unsupported, absent, dropped, out-of-taxonomy, or historically suppressed states. DATA vNext therefore makes **evidence state** and **training use** explicit instead of trying to repair binary zeros heuristically.

The controlling machine-readable policy is:

`docs/plan/ml-R4/specs/data_vnext_policy_v1.json`

The controlling contract-class row schema is:

`docs/plan/ml-R4/schemas/data_vnext_label_state_v1.schema.json`

## 2. Locked vocabulary and class enablement

The model output order remains:

| Index | Class | First-baseline supervision |
|---:|---|---|
| 0 | CallToUnknown | ENABLED |
| 1 | DenialOfService | ENABLED |
| 2 | ExternalBug | ENABLED |
| 3 | GasException | **DISABLED PENDING EVIDENCE** |
| 4 | IntegerUO | ENABLED |
| 5 | MishandledException | ENABLED |
| 6 | Reentrancy | ENABLED |
| 7 | Timestamp | ENABLED |
| 8 | TransactionOrderDependence | ENABLED |
| 9 | UnusedReturn | **DISABLED PENDING EVIDENCE** |

Disabled means the output index still exists but DATA vNext policy v1 provides no loss-eligible target for that class. Missing targets must never be replaced with zero.

GasException is disabled because no approved active first-baseline source has class-specific positive authority. UnusedReturn is disabled because DIVE was its only active positive source and the Phase-4 review did not justify retaining the blanket assertion even as weak training signal.

## 3. Three-layer truth model

### 3.1 Source claim

Source claims state what the source actually asserted, including claims that do not map to the canonical taxonomy.

Examples:

- `POSITIVE`
- `EXPLICIT_ZERO`
- `UNKNOWN`
- `UNSUPPORTED`
- `DROPPED_CATEGORY`
- `OUT_OF_TAXONOMY`
- `UNAVAILABLE`
- `NO_ASSERTION`

A source claim is never silently rewritten into a canonical negative.

### 3.2 Canonical outcome

Canonical outcome states are evidence conclusions:

- `CONFIRMED_POSITIVE`
- `CONFIRMED_NEGATIVE`
- `UNKNOWN`
- `NOT_APPLICABLE`
- `CONFLICTING_EVIDENCE`
- `NOT_REVIEWED`
- `INVALID_RECORD`

Historical binary state is retained separately.

### 3.3 Training signal

Training signal answers a different question: may this contract-class pair contribute to model optimization?

- signal: `POSITIVE`, `NEGATIVE`, or `NONE`
- strength: `STRONG`, `WEAK`, or `NONE`

Policy v1 authorizes no weak negative source.

A strong positive implies a confirmed positive outcome. A target zero requires a confirmed negative outcome. A weak positive can coexist with an unknown outcome and is always excluded from outcome metrics.

## 4. First-baseline source registry

### 4.1 SolidiFI — active strong-positive source

SolidiFI injection is accepted as strong positive evidence for the injected class only.

| Native injection | Canonical class |
|---|---|
| Unchecked-Send | CallToUnknown |
| tx.origin | ExternalBug |
| Overflow-Underflow | IntegerUO |
| Unhandled-Exceptions | MishandledException |
| Re-entrancy | Reentrancy |
| Timestamp-Dependency | Timestamp |
| TOD | TransactionOrderDependence |

For the injected class:

```text
outcome_state     = CONFIRMED_POSITIVE
training_signal   = POSITIVE
training_strength = STRONG
target_value      = 1
```

Every non-injected class on that same contract is unknown/no-target. SolidiFI does not create clean controls.

### 4.2 SmartBugs Curated — active strong-positive source for approved categories

Approved direct/in-taxonomy source categories:

| Native category | Canonical class |
|---|---|
| unchecked_low_level_calls | CallToUnknown |
| denial_of_service | DenialOfService |
| access_control | ExternalBug |
| arithmetic | IntegerUO |
| reentrancy | Reentrancy |
| time_manipulation | Timestamp |
| front_running | TransactionOrderDependence |

These are strong positive source claims. Non-target classes remain unknown.

Three historical mappings are superseded:

| Native category | Historical behavior | DATA vNext behavior |
|---|---|---|
| bad_randomness | mapped to Timestamp | preserve source claim, **no canonical target** |
| short_addresses | mapped to NonVulnerable/all-zero | preserve out-of-taxonomy claim, **no canonical target** |
| other | mapped to NonVulnerable/all-zero | preserve out-of-taxonomy claim, **no canonical target** |

This is deliberately asymmetric: a source can be trusted for the category it hand-labels without being trusted to prove the other nine classes absent.

### 4.3 DIVE — active unlabeled structure plus weak TOD

Recovered manual review and Phase-4 adjudication supersede the historical assumption that DIVE folder membership is a supervised-positive label.

| DIVE category | Canonical class | vNext training authority |
|---|---|---|
| Access Control | ExternalBug | NONE |
| Reentrancy | Reentrancy | NONE |
| DoS | DenialOfService | NONE |
| Arithmetic | IntegerUO | NONE |
| Time manipulation | Timestamp | NONE |
| Unchecked Return Values | UnusedReturn | NONE |
| Front Running | TransactionOrderDependence | **WEAK POSITIVE** |
| Bad Randomness | — | no canonical target |

DIVE contracts remain useful as structurally valid unlabeled data even when their source assertion is masked.

For authorized DIVE TOD weak positives:

```text
outcome_state           = UNKNOWN or NOT_REVIEWED
training_signal         = POSITIVE
training_strength       = WEAK
target_value            = 1
outcome_metric_eligible = false
```

DIVE zero/absence/unsupported states always yield no target.

### 4.4 Excluded/deferred sources

| Source | First-baseline status |
|---|---|
| Web3Bugs | EXCLUDED_UNAVAILABLE |
| DISL | EXCLUDED_UNAVAILABLE |
| BCCC | DEFERRED_NOT_IMPORTED |
| DeFiHackLabs | DEFERRED_NOT_IMPORTED |
| SmartBugs Wild | EXCLUDED_FROM_SUPERVISED_VNEXT |
| manual hand-written contracts | EVALUATION_CANDIDATE_NOT_IMPORTED |
| benchmark_v0.1_quickstart | EVALUATION_CANDIDATE_NOT_IMPORTED |

Phase 6 may consider evaluation candidates only after explicit exposure/leakage accounting. Phase 5 does not silently add them to DATA vNext.

## 5. Negative-evidence policy

The first baseline has **no blanket negative source**.

A target `0` can be created only when evidence shows the class was meaningfully assessed and absent for that contract-class pair.

The following never qualify:

- historical zero;
- source non-target cell;
- folder absence;
- parser default;
- unsupported class;
- dropped category;
- out-of-taxonomy category;
- all-zero vector;
- `NonVulnerable` synthesized by crosswalk;
- post-export DoS suppression;
- source not acquired;
- tool not firing.

This means vNext is intentionally positive/weak-positive/unlabeled heavy. Phase 8 must handle masks/weak roles correctly rather than recreating pseudo-negatives.

## 6. Crosswalk vNext

Crosswalk output is not only a class name; it is a semantic action.

Required action vocabulary:

- `DIRECT`
- `SEMANTIC_COMPRESSION`
- `LOSSY_NO_CANONICAL_TARGET`
- `OUT_OF_TAXONOMY_NO_CANONICAL_TARGET`
- `UNSUPPORTED`
- `DROPPED_CATEGORY`
- `NO_ASSERTION`

A no-target action preserves provenance and produces no canonical training target.

The historical SmartBugs `bad_randomness→Timestamp` and `short_addresses/other→NonVulnerable` transformations are specifically prohibited in vNext policy v1.

## 7. Aggregation vNext

Aggregation consumes source/evidence states, never already-collapsed binary cells.

### Confirmed outcomes

```text
confirmed positive only  -> CONFIRMED_POSITIVE
confirmed negative only  -> CONFIRMED_NEGATIVE
both                      -> CONFLICTING_EVIDENCE
neither                    -> UNKNOWN / NOT_REVIEWED
```

### Weak evidence

An authorized weak positive may create a weak training target if no confirmed contrary outcome exists. It does not create a confirmed positive.

### No implicit voting

The merger may not count Slither, Aderyn, source folders, or repeated derived artifacts as independent votes unless a future ADR validates an independence model.

### No global NonVulnerable synthesis

Absence of canonical positives means only that no canonical positive is currently established. It is not a ten-class safety claim.

## 8. Canonical semantic artifact

The semantic source of truth for DATA vNext is a long table with one row per contract×class conforming to `data_vnext_label_state_v1.schema.json`.

Key fields:

```text
contract_id
class_index
class_name
historical_state
source_claims[]
outcome_state
target_value nullable
training_signal
training_strength
loss_eligible
outcome_metric_eligible
role_eligibility[]
policy_decision_id
evidence_ids[]
limitations[]
```

`loss_eligible` at this layer means the source/evidence policy permits a training signal. The final training mask additionally requires Phase-6 assignment to a compatible training role.

## 9. Export format v2

Historical format v1 remains immutable.

The v2 export contains or binds:

1. the canonical long semantic label-state artifact;
2. Phase-6 partition/role manifest;
3. graph/token representation artifacts;
4. a derived ML projection for efficient `[10]` loading;
5. content hashes for policy/schema/partitions/generated artifacts.

The derived per-contract ML projection carries per class:

- nullable target;
- training strength;
- source-policy loss eligibility;
- outcome-metric eligibility;
- outcome state;
- policy-decision identity.

A v2 reader must hard-fail if those fields are absent. It must not substitute historical `class_i=0`.

## 10. ML compatibility boundary

Phase 5 does not modify the ML code, but Phase 8 is constrained to minimal compatibility changes:

- carry target/mask/strength tensors;
- apply loss only to eligible cells in training roles;
- treat weak and strong signals distinctly;
- mask unknown/disabled classes from metrics;
- bind any numeric weak-label optimizer weight to explicit training config;
- never fill unknown targets with zero.

The four-eye architecture and ten classifier outputs remain unchanged.

## 11. Phase-6 handoff

Phase 6 receives **role eligibility**, not preassigned partitions.

It must:

- build leakage groups across duplicates/project/template/injection families;
- assign groups to compatible roles;
- report per-class strong/weak/unlabeled/confirmed-negative support;
- freeze/declare acceptance support;
- ensure no group crosses incompatible roles.

If threshold-fit, calibration-fit, or untouched-acceptance support is insufficient, Phase 6 must say so explicitly rather than borrowing training data.

## 12. Historical compatibility and rollback

Lineage roots include:

- historical labels SHA-256 `26e739b5d82ba512e5a1830817d09609216e2184b79cf4ca7ec2d62ef34e32b5`;
- Phase-3 ledger SHA-256 `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`;
- Phase-4 sample/bundle/review identities;
- policy/schema/partition hashes;
- code commit.

Rollback selects a previously hashed compatible artifact bundle; nothing is reverse-edited in place.

## 13. Implementation mapping

Later implementation must change semantics at these seams, but Phase 5 does not modify them:

| Current seam | Historical behavior | Required vNext behavior |
|---|---|---|
| labeling parsers/crosswalk | fixed binary vector | preserve native state + semantic crosswalk action |
| merger | positive precedence over collapsed zeros | evidence-state aggregation |
| split records | binary classes/tier | semantic IDs + Phase-6 role/group identity |
| `label_writer.py` | non-nullable class_0..9 | v2 semantic artifact + derived masked projection |
| `SentinelDataset` | returns y[10] | return targets + strength + masks + lineage |
| `sentinel_collate_fn` | stacks y only | stack targets/strength/masks |
| `AsymmetricLoss`/trainer | every 0 is negative | operate only on authorized cells; weak handling explicit |
| metrics | all cells available | outcome/role masks mandatory |

No code author may make additional source/class semantic choices while implementing this table.

## 14. ADR authority

The following accepted ADRs are normative:

- `ADR-R4-001` — label state vs training signal;
- `ADR-R4-002` — source/class authority and class enablement;
- `ADR-R4-003` — crosswalk and aggregation;
- `ADR-R4-004` — export/consumer contract;
- `ADR-R4-005` — lineage/versioning/rollback.

The machine-readable policy controls where prose and implementation diverge accidentally.

## 15. G5 readiness checklist

G5 may pass when CI confirms:

- exact ten-class order;
- all ten classes have explicit enablement state;
- disabled GasException/UnusedReturn have no approved supervised-positive source;
- no blanket negative source exists;
- every DIVE Phase-4 constraint is preserved;
- SmartBugs lossy/out-of-taxonomy mappings produce no canonical target;
- unavailable/deferred sources cannot enter the first baseline;
- target zero requires confirmed negative;
- weak evidence cannot be metric-grade;
- historical v1 artifacts remain immutable and v2 is explicit;
- all five ADRs are Accepted;
- Phase 6, not implementation, owns partition assignment.
