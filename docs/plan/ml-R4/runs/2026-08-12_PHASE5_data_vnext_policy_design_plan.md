# R4 Phase 5 — DATA vNext Policy and Design Plan

**Phase:** 5 — DATA vNext Policy and Design  
**Gate:** G5  
**Branch:** `r4/phase5-data-vnext-policy-design`  
**Entry condition:** G4 PASS on canonical `main` (`c6dde505...`)  
**Execution mode:** specification/ADR only; no DATA pipeline or ML training implementation

## 1. Objective

Produce the smallest complete, versioned semantic contract that can be implemented later without making new label, source-authority, crosswalk, aggregation, masking, or compatibility decisions in code.

The design must repair the historical defect where source absence, unsupported classes, unknown cells, dropped categories, synthetic NonVulnerable mappings, and post-export suppression all serialized as numeric zero and were then consumed as supervised negatives.

## 2. Frozen premises

- Canonical audit unit: `contract_id × vulnerability_class`.
- Locked class order remains the current ten-class v9 order; no class is added, removed, or reordered in Phase 5.
- Historical Run12 export, splits, representations, thresholds, checkpoints, and calibration artifacts remain immutable.
- Historical zero is never promoted to confirmed negative without class-specific negative evidence.
- Source-native claims and canonical outcomes are different layers and must remain distinguishable.
- DIVE Phase-4 decisions are controlling for the first baseline:
  - DoS/Arithmetic/Time manipulation/Unchecked Return Values blanket assertions are masked;
  - Access Control/ExternalBug and Reentrancy blanket assertions remain masked by recovered Phase-1 review evidence;
  - Front Running/TOD may be used only as weak positive training signal;
  - DIVE zeros remain unknown.
- Web3Bugs and DISL are unavailable for the first baseline; BCCC and DeFiHackLabs remain deferred/disabled unless separately reopened.
- Phase 6, not Phase 5, assigns leakage-safe dataset partitions and freezes acceptance.
- Phase 8, not Phase 5, chooses numeric optimization weight for weak labels; DATA vNext must expose categorical training strength so no hidden default is possible.

## 3. Current implementation seams to replace later

The current historical consumer path is binary:

1. `label_writer.py` emits non-nullable `class_0..class_9` integer cells and one contract-level `confidence_tier`;
2. `SentinelDataset` converts all ten cells to a single `y[10]` tensor;
3. `sentinel_collate_fn` carries no class mask or strength tensor;
4. `AsymmetricLoss` treats every `0` as a negative.

Phase 5 will design the compatibility contract that Phase 7/8 must implement; it will not modify those files now.

## 4. Design decisions to freeze

### WP1 — Label/outcome/training-signal schema

Separate:

- source-native claim state;
- canonical outcome state;
- optional training target;
- categorical training strength (`STRONG`, `WEAK`, `NONE`);
- metric eligibility;
- evidence/provenance;
- historical state.

A weak positive may exist while canonical outcome remains `UNKNOWN`; weak training evidence must not masquerade as a confirmed outcome.

### WP2 — First-baseline source/class authority matrix

Freeze positive and negative authority for every source used or explicitly excluded in the first baseline.

Core rules expected from Phase 0–4 evidence:

- SolidiFI injected class: strong confirmed positive; all non-injected cells unknown.
- SmartBugs Curated direct in-taxonomy category: strong confirmed positive; non-target cells unknown.
- SmartBugs `bad_randomness`, `short_addresses`, and `other`: preserve source-native category but do not synthesize a canonical target.
- DIVE: only Front Running→TOD remains weak positive; every other DIVE canonical positive assertion is masked in the first baseline; all DIVE zeros unknown.
- No active first-baseline source supplies blanket confirmed negatives.
- GasException and UnusedReturn have no approved first-baseline supervised-positive source after the Phase-4 repair and therefore must be explicitly supervision-disabled until evidence improves.

### WP3 — Crosswalk and aggregation policy

- Crosswalk transforms source-native claims but never manufactures negative evidence.
- `UNSUPPORTED`, `UNKNOWN`, `DROPPED`, `OUT_OF_TAXONOMY`, `UNAVAILABLE`, and `NO_ASSERTION` remain distinct provenance states.
- Aggregation operates on evidence-bearing states, not binary zero precedence.
- Confirmed positive/negative conflict becomes `CONFLICTING_EVIDENCE`.
- Weak source assertions cannot override a confirmed contrary outcome.
- No vote counting across correlated tools/sources without an explicit independence model.
- Synthetic global `NonVulnerable` is not a canonical source claim in DATA vNext.

### WP4 — Export/consumer compatibility contract

Design a new export format version rather than mutating historical `v1` semantics.

The canonical semantic table will remain one row per contract×class. A derived ML projection may pivot to ten columns but must carry, per class:

- nullable target value;
- training strength;
- training/loss eligibility mask;
- outcome-metric eligibility mask;
- canonical outcome state;
- lineage/policy decision identity.

Numeric weak-label optimization weight is not stored as semantic truth. Phase 8 must declare and bind any numeric mapping from `WEAK` to loss weight in training config.

### WP5 — Lineage and historical compatibility

- legacy artifacts are immutable and explicitly marked historical;
- new artifacts use new version identifiers and hashes;
- manifests bind source policy, label schema, class vocabulary, crosswalk policy, aggregation policy, partitions, and code commit;
- no reader may silently interpret a v1 historical label artifact as vNext;
- rollback is selection of historical artifacts, never in-place mutation.

## 5. Minimal ADR set

1. `ADR-R4-001` — label/outcome/training-signal state model.
2. `ADR-R4-002` — first-baseline source/class authority and class enablement.
3. `ADR-R4-003` — crosswalk and evidence aggregation semantics.
4. `ADR-R4-004` — DATA vNext export and ML consumer compatibility contract.
5. `ADR-R4-005` — lineage, versioning, historical compatibility, and rollback.

Partition assignment is intentionally deferred to Phase 6; the Phase-5 source policy only defines role eligibility/constraints.

## 6. Machine-readable control artifact

Create `specs/data_vnext_policy_v1.json` as the implementation-facing semantic contract. CI must fail if:

- the ten-class order changes;
- a source/class zero receives negative authority without explicit evidence;
- a DIVE stratum violates Phase-4 limits;
- SmartBugs lossy/out-of-taxonomy categories synthesize canonical negatives/positives;
- GasException or UnusedReturn is marked supervised-enabled without an approved positive source;
- any unavailable/deferred source becomes first-baseline active;
- weak labels are silently promoted to strong or metric-grade evidence;
- historical v1 artifacts are declared mutable/replaced.

## 7. G5 exit criteria

G5 passes only when:

- all five ADRs are accepted;
- every first-baseline source/class decision is explicit and machine-readable;
- the label-state/training-signal schema is complete;
- crosswalk and aggregation behavior is deterministic;
- export/consumer fields and version boundaries are explicit;
- class enable/disable status is explicit;
- Phase-6 responsibilities are clearly separated;
- CI demonstrates that implementation can consume the specification without inventing new semantic rules.

## 8. Non-goals

Do not in Phase 5:

- alter protected/historical data;
- rewrite parsers, merger, export, dataset loader, collate, loss, or trainer;
- create partitions or inspect/freeze acceptance;
- choose weak-label numeric optimization weights;
- tune thresholds/calibration;
- retrain the model;
- acquire Web3Bugs/BCCC/DeFiHackLabs or start new evidence review.
