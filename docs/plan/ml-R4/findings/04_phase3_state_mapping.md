# 04 — Phase-2 Mechanism → Phase-3 Ledger State Mapping

- **Phase:** R4 Phase 3 — Contract-Class Evidence Ledger
- **Purpose:** deterministic initialization policy for the sidecar ledger
- **Rule:** this mapping preserves provenance. It does not turn historical binary labels into newly adjudicated truth.

## Core principle

The ledger separates three questions that the historical export collapsed:

1. **What numeric target was historically stored?** → `historical_state` / `historical_target`
2. **How did that target arise?** → source-native + transformation provenance fields
3. **What can R4 currently conclude?** → `outcome_state`, evidence references, masks and role eligibility

A row can therefore legitimately be:

```text
historical_state = HISTORICAL_ZERO
source_native_state = UNSUPPORTED
zero_origin_categories = [CLASS_UNSUPPORTED]
outcome_state = UNKNOWN
supervised_loss_masked = true
```

without contradiction.

## Default initialization mapping

| Phase-2 mechanism | Historical state | Source-native state | Transformation fields | Default R4 outcome | Masks | Default role posture |
|---|---|---|---|---|---|---|
| source positive with retained injection/exploit evidence | HISTORICAL_POSITIVE | EXPLICIT_POSITIVE | crosswalk DIRECT/LOSSY_MAP as applicable | `CONFIRMED_POSITIVE` only when the evidence item is contract-class scoped and resolves | unmasked only if confirmed | potentially TRAIN_STRONG/weak per later policy |
| source positive with only source assertion | HISTORICAL_POSITIVE | EXPLICIT_POSITIVE | source/parser/crosswalk provenance retained | `NOT_REVIEWED` by default | both true | TRAIN_UNLABELED or INTERNAL_AUDIT |
| explicit source negative with class-specific retained review evidence | HISTORICAL_ZERO | EXPLICIT_NEGATIVE | `EXPLICIT_SOURCE_ZERO` | `CONFIRMED_NEGATIVE` only when meaningful class-specific evidence resolves | unmasked only if confirmed | potentially supervised role later |
| historical source zero without sufficient negative authority | HISTORICAL_ZERO | EXPLICIT_NEGATIVE or ABSENT | `EXPLICIT_SOURCE_ZERO` or `SOURCE_ABSENCE` | `UNKNOWN` | both true | TRAIN_UNLABELED + EXCLUDE_OUTCOME_METRICS |
| source-native unknown erased by DIVE folderization | HISTORICAL_ZERO | UNKNOWN | `SOURCE_NATIVE_UNKNOWN`, parser/default provenance | `UNKNOWN` | both true | TRAIN_UNLABELED + EXCLUDE_OUTCOME_METRICS |
| source does not support class | HISTORICAL_ZERO | UNSUPPORTED | crosswalk `UNSUPPORTED`, zero origin `CLASS_UNSUPPORTED` | `UNKNOWN` — **not** NOT_APPLICABLE, because the vulnerability can still apply to the contract | both true | TRAIN_UNLABELED + EXCLUDE_OUTCOME_METRICS |
| source-native category dropped | HISTORICAL_ZERO or POSITIVE in another class | DROPPED_CATEGORY | crosswalk `DROP`, zero origin `DROPPED_CATEGORY` | `UNKNOWN` for the affected canonical class | both true | TRAIN_UNLABELED + EXCLUDE_OUTCOME_METRICS |
| source category mapped to NonVulnerable | HISTORICAL_ZERO | MAPPED_NONVULNERABLE | `MAP_NONVULNERABLE`, zero origin `MAPPED_NONVULNERABLE` | `UNKNOWN` unless independent class-specific safe evidence exists | both true | TRAIN_UNLABELED + EXCLUDE_OUTCOME_METRICS |
| parser/missing-key default | HISTORICAL_ZERO | ABSENT/NOT_RECONSTRUCTED | zero origin `PARSER_DEFAULT` | `UNKNOWN` | both true | TRAIN_UNLABELED + EXCLUDE_OUTCOME_METRICS |
| all-zero row classified as NonVulnerable by split | HISTORICAL_ZERO | source-dependent | add `SYNTHETIC_NONVULNERABLE` | preserve underlying outcome; if no evidence, `UNKNOWN` | both true unless independently confirmed | not strong supervised negative by default |
| merger positive precedence | HISTORICAL_POSITIVE | MIXED | merger `POSITIVE_PRECEDENCE` | depends on positive evidence; no automatic confirmation | evidence-dependent | evidence-dependent |
| merger all-zero selection | HISTORICAL_ZERO | MIXED/ABSENT | merger `ALL_ZERO_SELECTION`, usually `MERGER_PRESERVED_ZERO` | `UNKNOWN` unless negative evidence exists | both true | TRAIN_UNLABELED |
| credible contradictory evidence | historical state preserved | MIXED | merger `CONFLICT` or duplicate-family provenance | `CONFLICTING_EVIDENCE` | both true | INTERNAL_AUDIT / EXCLUDE_OUTCOME_METRICS |
| June DoS positive→zero Parquet suppression | HISTORICAL_ZERO in protected export | MIXED/NOT_RECONSTRUCTED | verification action `HISTORICAL_DIRECT_PATCH`; zero origin `HISTORICAL_POST_EXPORT_SUPPRESSION` | `CONFLICTING_EVIDENCE` by default until class-specific evidence establishes the correction | both true | INTERNAL_AUDIT / EXCLUDE_OUTCOME_METRICS |
| historical target missing entirely | HISTORICAL_MISSING | UNAVAILABLE/NOT_RECONSTRUCTED | zero origin `HISTORICAL_MISSING` | `NOT_REVIEWED` or `INVALID_RECORD` depending integrity | both true | excluded from supervised outcome use |
| row lacks representation | historical state unchanged | unchanged | `representation_available=false` | **outcome unchanged**; this is a population/role property, not label truth | outcome-dependent | no representation-dependent ML role until represented |
| graph/project duplicate family with contradictory labels | historical state per row | MIXED | leakage/dedup group + conflict provenance | `CONFLICTING_EVIDENCE` for affected class when contradiction is established | both true | INTERNAL_AUDIT / EXCLUDE_OUTCOME_METRICS |

## Important distinctions

### `UNSUPPORTED` does not mean `NOT_APPLICABLE`

A source's inability to label GasException, for example, says nothing about whether GasException can occur in the contract. Therefore the default outcome is `UNKNOWN`, not `NOT_APPLICABLE`.

### Historical positive does not automatically mean confirmed positive

The ledger preserves the historical positive cell even when Phase-1 evidence shows the source assertion is noisy. Confirmation requires a referenced evidence item whose scope and strength justify the class-specific conclusion.

### Historical zero does not automatically mean confirmed negative

This is enforced both by policy and validator. A zero-origin field is provenance, not a negative verdict.

### Exact per-row zero origin may remain unresolved

Phase 2 reconstructed the complete **category set**, but GitHub-only evidence does not always identify which mechanism generated a particular protected zero cell. Such rows use:

```text
zero_origin_categories = [UNRESOLVED_WITHIN_KNOWN_MECHANISMS]
source_native_state = NOT_RECONSTRUCTED or best-supported bounded state
outcome_state = UNKNOWN
```

This is preferable to inventing a source-native fact.

## Evidence resolution rule

A ledger row may reference evidence at several scopes:

- `CONTRACT_CLASS`: strongest direct row linkage;
- `CONTRACT`: applies to the contract but still requires class interpretation;
- `SOURCE_CLASS` / `CORPUS_CLASS`: supports source/class policy or uncertainty, not automatic per-row confirmation;
- `TRANSFORMATION_CATEGORY`: proves how a pipeline mechanism behaves, not whether one contract is vulnerable.

Only evidence whose scope and strength legitimately support a row may justify `CONFIRMED_POSITIVE` or `CONFIRMED_NEGATIVE`.

## Initialization versus later adjudication

Phase 3 should initialize the ledger conservatively from existing evidence. Phase 4 may later change `outcome_state` for authorized gap-review rows, but it must append new evidence/provenance rather than overwrite historical state.
