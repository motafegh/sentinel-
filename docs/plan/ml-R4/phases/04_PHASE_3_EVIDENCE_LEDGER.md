# Phase 3 — Contract-Class Evidence Ledger

**Status:** WAITING FOR G2  
**Gate:** G3

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
