# R4 Phase 2 — Label-Corruption Reconstruction Execution Plan

**Branch:** `r4/phase2-label-corruption-reconstruction`  
**Parent:** canonical `main` after R0 + R4 Phase 1 / G1 PASS  
**Phase:** 2 — Label-Corruption Mechanism Reconstruction  
**Gate:** G2

## Objective

Reconstruct, from executable source and retained artifacts, how source-native vulnerability assertions became the historical binary ML targets. The work must explain both historical positives and historical zeros at the category/mechanism level without changing active labels, exports, splits, thresholds, calibration, or model architecture.

## Execution order

1. **Trace active source acquisition and source identity**
   - enumerate active dataset sources and their configuration;
   - bind source-native label formats and source/class coverage;
   - distinguish declared sources from actually present/usable sources.

2. **Trace parser and normalization semantics**
   - inspect parser implementations for each active source;
   - identify default values, missing-field behavior, dropped records, and explicit/implicit negatives;
   - record any silent fallback or semantic compression.

3. **Trace crosswalk semantics**
   - map source-native categories into SENTINEL classes;
   - identify dropped categories, mappings to non-vulnerable/safe semantics, unsupported classes, and many-to-one mappings;
   - quantify crosswalk effects where retained artifacts permit it.

4. **Trace merger and verification semantics**
   - inspect source-tier logic, conflict resolution, verification overrides, and multi-source precedence;
   - identify where absence, conflict, or unsupported coverage can become a binary zero.

5. **Trace split/export semantics**
   - inspect deduplication, representation filtering, split membership, export construction, and all-zero behavior;
   - reconcile labels-without-representation and Run12/current-export population differences as far as retained evidence permits.

6. **Trace ML loading semantics**
   - inspect the current dataset loader/collation path that turns exported targets into training tensors;
   - identify whether masks exist historically and exactly how zeros enter supervised loss.

7. **Build representative end-to-end traces**
   Include at minimum:
   - direct positive;
   - true explicit negative where one exists;
   - dropped-class-only contract;
   - mapped-to-non-vulnerable contract;
   - source/class not covered;
   - multi-source conflict;
   - all-zero target;
   - duplicate/project family;
   - representation-filtered split row.

8. **Quantify mechanisms and reconcile populations**
   Produce source/class/category counts for each named corruption mechanism, while keeping unreconciled counts explicitly unresolved rather than forcing equality.

## Required Phase-2 outputs

Create under `docs/plan/ml-R4/`:

- `findings/03_source_authority_matrix.md`
- `findings/03_source_semantics_cards.md`
- `findings/03_crosswalk_effect_table.md`
- `findings/03_merger_sensitivity_table.md`
- `findings/03_all_zero_decomposition.md`
- `findings/03_population_reconciliation.md`
- `manifests/phase2_end_to_end_traces.jsonl`
- deterministic read-only helper scripts under `scripts/` only where needed for reproducible counts

Update only the R4 operational registers required to record Phase-2 findings and G2 status.

## Governing semantic categories

Every historical target origin must be classified as one of:

- explicit source positive;
- explicit source negative;
- source absence;
- class unsupported by source;
- dropped source-native category;
- mapped-to-NonVulnerable category;
- parser default;
- merger conflict resolution;
- verification override;
- export all-zero;
- missing representation;
- other (must be named precisely).

## Non-goals / prohibited work

Do not in Phase 2:

- perform new contract review without an APPROVED evidence-gap ID;
- change source labels or crosswalks;
- regenerate or overwrite historical exports;
- change dataset splits;
- change training targets or implement DATA vNext;
- retrain the model;
- change thresholds/calibration;
- redesign the model architecture;
- interpret source absence or historical zero as confirmed negative without evidence.

## Source-of-truth rule

Executable source is authoritative for current behavior. Documentation is historical evidence only and must be cross-checked against `.py`, `.sol`, `.ts`, or other executable/configured behavior before being used as a behavioral claim.

## Known Phase-2 risks to resolve or bound

- `R4-R002`: historical zeros may encode unknown/unsupported/default states rather than negatives.
- `R4-R010`: Run12 population differs from the current export by 2,635 contracts; exact historical split/export identity remains unresolved.
- `R4-R011`: BCCC v1.4 verified labels exist while configuration remains DEFERRED.
- `R4-R012`: DIVE ExternalBug TP tally has an off-by-one contradiction.

These are reconstruction targets, not permission to start broad new review.

## G2 acceptance rule

G2 passes only when every historical positive and historical zero is semantically explained at the **category/mechanism level** through a named transformation path. Individual contract-class outcomes may remain `UNKNOWN` or `NOT_REVIEWED`; Phase 2 does not require adjudicating them.

## First work package

Begin read-only source tracing from the active DATA configuration and source adapters/parsers, then follow the transformation path through crosswalk, merger, split/export, and finally ML loading. Record contradictions immediately and do not modify protected DATA/ML artifacts.
