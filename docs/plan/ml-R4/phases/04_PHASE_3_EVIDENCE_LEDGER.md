# Phase 3 — Contract-Class Evidence Ledger

**Status:** PASSED  
**Gate:** G3 — PASS

## Objective

Build a versioned sidecar ledger without modifying historical labels.

## Canonical key

`ledger_version × contract_id × class_index`

## Required field groups

- contract and source identity;
- dedup/project/leakage group;
- historical target and export identity;
- source-native claims;
- parser/crosswalk/merger decisions;
- evidence items and independence groups;
- prior review outcomes;
- final R4 outcome state;
- uncertainty/limitations;
- role eligibility;
- partition;
- artifact hashes;
- historical versus new provenance.

## Population

Every export-relevant contract-class pair receives a row, including no-evidence rows.

## Validation

Reject:

- duplicate keys;
- invalid class order;
- confirmed outcome without evidence reference;
- historical/new ambiguity;
- acceptance eligibility from tool-only evidence;
- masked outcome included in supervised metrics;
- incompatible role leakage;
- missing artifact identity.

## Outputs

- JSON Schema;
- Parquet ledger;
- JSONL evidence items;
- ledger manifest;
- validation scripts/tests;
- schema report.

## G3 pass criteria

The ledger can represent all required states without forcing unknowns into binary negatives.

## Closure evidence — 2026-08-11

The production ledger was materialized locally against the frozen protected Phase-0 population and then published to the Phase-3 branch.

- contracts: `22,493`;
- classes: `10`;
- canonical ledger rows / unique keys: `224,930`;
- represented contracts: `21,657`;
- historical positives preserved as `NOT_REVIEWED`: `51,546`;
- historical zeros preserved as `UNKNOWN`: `173,384`;
- supervised/outcome role initialization remains conservative; no historical zero was promoted to a confirmed negative;
- semantic validation: PASS, zero errors/warnings;
- strict schema-surface + semantic validation: PASS, zero errors/warnings;
- candidate artifact binding: PASS;
- canonical artifact binding: PASS;
- ledger SHA-256: `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`;
- generation commit recorded by the materializer: `b8911daed077db573a2c421fb5e21a9811b62526`;
- publication commit: `17fa204955e1228b1d2f691f2f7e3fe76875085a`.

**G3 PASS:** the complete protected contract×class population is represented without collapsing unknown evidence states into binary negatives. Phase 4 is authorized subject to its own gap-ID and adjudication controls.
