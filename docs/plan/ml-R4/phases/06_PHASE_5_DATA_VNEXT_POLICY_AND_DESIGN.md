# Phase 5 — DATA vNext Policy and Design

**Status:** WAITING FOR G4  
**Gate:** G5

## Objective

Approve the smallest versioned data repair that converts prior evidence and gap outcomes into a trustworthy training/evaluation contract.

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
