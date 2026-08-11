# Phase 5 — DATA vNext Policy and Design

**Status:** READY — G4 SATISFIED  
**Gate:** G5

## Objective

Approve the smallest versioned data repair that converts prior evidence and gap outcomes into a trustworthy training/evaluation contract.

## Entry evidence

G4 has passed. Phase 4 resolved the decision-critical DIVE gap and bounded first-baseline source roles:

- DIVE `DoS` → `DenialOfService`: source assertion masked/excluded;
- DIVE `Arithmetic` → `IntegerUO`: source assertion masked/excluded;
- DIVE `Time manipulation` → `Timestamp`: source assertion masked/excluded;
- DIVE `Front Running` → `TransactionOrderDependence`: at most `TRAIN_WEAK`, excluded from outcome metrics/high-authority roles;
- DIVE `Unchecked Return Values` → `UnusedReturn`: source assertion masked/excluded;
- absent Web3Bugs and provisional non-active BCCC strata remain excluded/deferred for the first baseline unless a new evidence-backed policy explicitly imports them.

These Phase-4 recommendations constrain Phase 5; implementation may not silently restore stronger authority.

## Decisions

Per source/class/stratum choose:

- retain as strong positive/negative;
- retain as weak training signal;
- convert to unknown/masked;
- exclude;
- retain only as unlabeled structure;
- reserve for case study;
- reserve for acceptance;
- disable class pending evidence.

## Design areas

- canonical source registry;
- source-native claim preservation;
- crosswalk vNext;
- explicit label-state schema;
- merger/aggregation policy;
- masks and weights;
- dedup/leakage grouping;
- export version;
- historical compatibility;
- artifact lineage.

## ADR requirements

Write ADRs for every mapping, schema, merger, role, or source/class policy change.

## G5 pass criteria

The DATA vNext specification is complete enough to implement without making new semantic decisions in code.
