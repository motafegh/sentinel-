# Phase 5 — DATA vNext Policy and Design

**Status:** PASSED — G5 PASS  
**Gate:** G5

## Objective

Approve the smallest versioned data repair that converts prior evidence and gap outcomes into a trustworthy training/evaluation contract.

## Entry evidence

G4 passed with the active DIVE source bounded conservatively:

- DoS, Arithmetic, Time manipulation, Unchecked Return Values, ExternalBug, and Reentrancy blanket DIVE assertions are masked for supervised use;
- DIVE Front Running→TransactionOrderDependence is weak-positive only;
- DIVE zeros remain unknown;
- absent/deferred sources are not silently imported.

## Accepted DATA vNext policy

The controlling machine-readable contract is:

`specs/data_vnext_policy_v1.json`

The controlling contract×class row schema is:

`schemas/data_vnext_label_state_v1.schema.json`

The implementation-facing narrative specification is:

`findings/07_data_vnext_policy_and_design_specification.md`

### Core state decision

DATA vNext separates:

- source-native claim;
- canonical outcome state;
- nullable training target;
- training signal;
- categorical training strength (`STRONG`, `WEAK`, `NONE`);
- source-policy loss eligibility;
- outcome-metric eligibility;
- role eligibility;
- provenance/evidence.

A numeric target `0` is permitted only for a class-specific `CONFIRMED_NEGATIVE`. Historical zero, source absence, unsupported class, dropped category, out-of-taxonomy category, parser default, all-zero vector, and historical post-export suppression do not qualify.

A weak positive may contribute to training while the canonical outcome remains `UNKNOWN`; it is never metric-grade evidence.

### First-baseline source authority

- **SolidiFI:** injected class only → strong confirmed positive; non-injected classes unknown.
- **SmartBugs Curated:** approved in-taxonomy hand-labeled category → strong confirmed positive; non-target classes unknown.
- **DIVE:** primarily unlabeled structure; only Front Running→TOD may be `TRAIN_WEAK`; every DIVE zero remains unknown.
- **Web3Bugs/DISL:** excluded/unavailable.
- **BCCC/DeFiHackLabs:** deferred/not imported.
- **SmartBugs Wild:** excluded from supervised vNext.
- manual/quickstart corpora remain evaluation candidates for Phase 6, not automatic vNext imports.

Historical SmartBugs mappings `bad_randomness→Timestamp` and `short_addresses/other→NonVulnerable` are superseded for vNext: these categories preserve source-native provenance but create no canonical target.

### Class enablement

The ten-output order remains locked.

Supervision enabled:

- CallToUnknown
- DenialOfService
- ExternalBug
- IntegerUO
- MishandledException
- Reentrancy
- Timestamp
- TransactionOrderDependence

Supervision disabled pending evidence:

- GasException
- UnusedReturn

Disabled classes remain model output positions but receive no loss-eligible vNext target until a later evidence-backed ADR re-enables them.

### Crosswalk/aggregation

- preserve source-native claims before canonical transformation;
- no-target mappings remain no-target;
- remove synthetic global NonVulnerable;
- remove positive-precedence-over-collapsed-zero semantics;
- aggregate confirmed outcomes explicitly;
- preserve conflicts;
- do not treat correlated tool/source votes as independent truth.

### Export/consumer contract

Historical format v1 remains immutable. DATA vNext uses export format v2.

The canonical semantic artifact is long-form contract×class state. A derived ML projection may pivot to ten class positions but must carry nullable target, training strength, loss mask eligibility, outcome metric eligibility, outcome state, and policy identity per class.

Phase 8, not DATA semantics, chooses any numeric weak-label optimizer weight in explicit checkpoint-bound training config.

## Accepted ADRs

- `ADR-R4-001` — label state vs training signal
- `ADR-R4-002` — source/class authority and class enablement
- `ADR-R4-003` — crosswalk and aggregation semantics
- `ADR-R4-004` — export and ML consumer contract
- `ADR-R4-005` — lineage, versioning, historical compatibility, rollback

All five are registered as `ACCEPTED` in `DECISION_REGISTER.md`.

## Validation

`p5_validate_data_vnext_policy.py` and `.github/workflows/r4-phase5-policy.yml` validate:

- locked ten-class order;
- explicit enablement for all ten classes;
- GasException/UnusedReturn disabled with no approved positive source;
- zero blanket-negative sources;
- DIVE weak-TOD-only rule;
- SmartBugs no-target mappings;
- unavailable/deferred source exclusion;
- schema rejection of `UNKNOWN→0`, weak metric eligibility, and masked cells with targets;
- v1 immutability and explicit v2 export boundary;
- all five ADRs Accepted;
- Phase-5 branch contains only R4 design/CI changes.

## G5 assessment

**G5 PASS.** The semantic contract is complete enough for later implementation without making new source/class/label-state decisions in code.

Implementation is still prohibited until Phase 6 completes role-isolated partitions and acceptance/support freeze at G6.

## Next permitted action

Begin Phase 6 — Dataset Roles, Leakage-Safe Partitions, and Acceptance Freeze.
