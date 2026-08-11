# ADR-R4-002 — First-Baseline Source/Class Authority and Class Enablement

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 DATA vNext first-baseline source/class authority

## Context

Phase 1–4 established that source authority is class-specific:

- SolidiFI proves the injected class but not absence of other classes.
- SmartBugs Curated provides expert hand-labeled source categories, but historical expansion to nine cross-class zeros is unsupported.
- DIVE blanket positives are very noisy by class; Phase 4 retained only `Front Running→TransactionOrderDependence` as weak training signal.
- Web3Bugs and DISL are unavailable; BCCC and DeFiHackLabs are not active first-baseline sources.

The first DATA vNext baseline needs an explicit source/class matrix so implementation cannot infer authority from `enabled: true`, tier, folder presence, or historical numeric labels.

## Decision

### SolidiFI

For the one injected class only:

- canonical outcome: `CONFIRMED_POSITIVE`;
- training strength: `STRONG`;
- allowed first-baseline roles: `TRAIN_STRONG`, `INTERNAL_AUDIT`, `CASE_STUDY`;
- not eligible for untouched acceptance by default.

All non-injected classes on that contract are `UNKNOWN`; no negative target is created.

Approved injected mappings:

- `Unchecked-Send` → CallToUnknown
- `tx.origin` → ExternalBug
- `Overflow-Underflow` → IntegerUO
- `Unhandled-Exceptions` → MishandledException
- `Re-entrancy` → Reentrancy
- `Timestamp-Dependency` → Timestamp
- `TOD` → TransactionOrderDependence

### SmartBugs Curated

Approved in-taxonomy hand-labeled categories are strong confirmed positives:

- `unchecked_low_level_calls` → CallToUnknown
- `denial_of_service` → DenialOfService
- `access_control` → ExternalBug
- `arithmetic` → IntegerUO
- `reentrancy` → Reentrancy
- `time_manipulation` → Timestamp
- `front_running` → TransactionOrderDependence

Allowed roles: `TRAIN_STRONG`, `MODEL_SELECTION`, `INTERNAL_AUDIT`, `CASE_STUDY`, subject to Phase-6 group isolation.

Non-target classes are `UNKNOWN`, not negative.

The following historical mappings are **not** canonical targets in vNext:

- `bad_randomness` → no canonical target;
- `short_addresses` → no canonical target;
- `other` → no canonical target.

In particular, `short_addresses` and `other` must not synthesize global `NonVulnerable`, and generic `bad_randomness` must not be silently folded into Timestamp.

### DIVE

DIVE remains useful as structural/unlabeled corpus. Source-native claims remain preserved for provenance.

First-baseline training authority:

- Access Control→ExternalBug: `NONE`
- Reentrancy→Reentrancy: `NONE`
- DoS→DenialOfService: `NONE`
- Arithmetic→IntegerUO: `NONE`
- Time manipulation→Timestamp: `NONE`
- Unchecked Return Values→UnusedReturn: `NONE`
- Front Running→TransactionOrderDependence: `WEAK` positive only
- Bad Randomness: no canonical target

All DIVE zeros, absence states, unsupported classes, and dropped categories remain unknown/no-target.

Weak DIVE TOD is eligible only for `TRAIN_WEAK`, `CASE_STUDY`, and explicit outcome-metric exclusion. It is not eligible for model selection, threshold fit, calibration fit, or untouched acceptance.

### Other sources

- Web3Bugs: excluded/unavailable.
- DISL: excluded/unavailable; unlabeled is not NonVulnerable.
- BCCC: deferred/not imported into the first baseline.
- DeFiHackLabs: deferred/not imported.
- SmartBugs Wild: excluded from supervised vNext.
- manual hand-written and quickstart benchmark corpora: evaluation candidates only; Phase 6 may consider them after exposure/leakage accounting, but Phase 5 does not import them automatically.

## Class supervision status

The 10-output vocabulary remains locked. Supervision status is independent of output existence.

**Enabled:**

- CallToUnknown
- DenialOfService
- ExternalBug
- IntegerUO
- MishandledException
- Reentrancy
- Timestamp
- TransactionOrderDependence

**Supervision disabled pending evidence:**

- GasException
- UnusedReturn

GasException has no approved active positive source. UnusedReturn's only active blanket source is DIVE, which Phase 4 found too noisy for authority. Their output indices remain present but their vNext training targets are masked until a later evidence-backed ADR re-enables them.

## Negative authority

No current first-baseline parser/source receives blanket negative authority.

`target_value=0` requires direct class-specific confirmed-negative evidence. A source category's non-target cells, all-zero record, unsupported class, dropped category, or absence never qualifies.

## Consequences

- DATA vNext will be substantially smaller in supervised signal than the historical binary export.
- DIVE becomes primarily unlabeled structure plus one weak TOD source signal.
- Two classes are explicitly not supervised rather than trained on fabricated labels.
- Future evidence can re-enable classes without changing the model's positional class vocabulary.

## Implementation contract

The exhaustive source/class decisions are machine-readable in `specs/data_vnext_policy_v1.json`. Phase 7 may not infer additional authority from historical tier, config state, or current parser output.
