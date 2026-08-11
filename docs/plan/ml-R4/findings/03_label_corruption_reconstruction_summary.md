# 03 — Label-Corruption Reconstruction Summary

- **Phase:** R4 Phase 2
- **Gate:** G2
- **Status:** PASS
- **Scope:** semantic reconstruction only; no label/export/split/model mutation

## Executive conclusion

The historical ten-class target format collapsed several distinct evidence states into the same binary `0`, and the ML training path supplied no class mask to recover that distinction. Historical positives and historical zeros are now explained at the required **category/mechanism level**.

The dominant corruption chain is:

```text
source-native positive / zero / unknown / unsupported / dropped category
→ source-specific folder/category representation
→ parser emits fixed ten-class binary vector
→ merger receives already-collapsed values
→ all-zero row becomes synthetic NonVulnerable role in splitting
→ export projects missing/non-positive classes to 0
→ ML loader returns only y[10] with no class mask
→ AsymmetricLoss treats every 0 as a negative loss cell
```

A separate historical mechanism directly changed 2,655 DoS positive cells to zero in `labels.parquet`.

## G2 semantic-category assessment

| Mandatory origin category | Reconstruction | Status |
|---|---|---|
| explicit source positive | SolidiFI injected positives; DIVE folder assertions; SmartBugs hand-labeled categories | EXPLAINED |
| explicit source negative | recovered SmartBugs NonVulnerable examples provide bounded row-level negative authority; DIVE CSV zero remains source-specific rather than universal negative | EXPLAINED WITH SCOPE |
| source absence | non-target cells from positive-only/single-category sources | EXPLAINED |
| class unsupported by source | DIVE CTU/Gas/Mishandled; SolidiFI DoS/Gas/UnusedReturn; unavailable sources | EXPLAINED |
| dropped source-native category | DIVE Bad Randomness | EXPLAINED |
| mapped-to-NonVulnerable category | SmartBugs short_addresses/other | EXPLAINED |
| parser default | no folder membership and `get(...,0)` projections | EXPLAINED |
| merger conflict resolution | any positive wins; best tier only among positives; all-zero provenance remains weak | EXPLAINED |
| verification override | **not implemented in current gate source**; current verification is report/gate-only | EXPLAINED AS ABSENT CURRENT MECHANISM |
| export all-zero | split/export preserve all-zero and label it NonVulnerable structurally | EXPLAINED |
| missing representation | 836 label rows omitted from ML shard-index population | EXPLAINED |
| other | June 13 direct DoS positive→zero Parquet patch | EXPLAINED |

## Principal findings

### F2.1 — DIVE destroys native unknown semantics

`label_folderize.py` explicitly documents empty CSV cells as unknown but creates symlinks only for positives. `dive.py` later sees absence of a folder and writes zero. Therefore:

```text
UNKNOWN → no symlink → no membership → 0
```

is an executable historical transformation.

### F2.2 — Unsupported and dropped categories become supervised negatives

Examples:

- DIVE has no CallToUnknown, GasException or MishandledException mapping; these become zero for DIVE rows.
- DIVE Bad Randomness is dropped; exclusive-only contracts can become all-zero.
- SolidiFI has no DoS, GasException or UnusedReturn injection but emits zeros for those cells.
- SmartBugs maps some out-of-taxonomy categories to NonVulnerable.

### F2.3 — The merger cannot repair information already erased

Per-source parsers emit only binary values. The merger has no state for unknown/unsupported/dropped. Any positive wins over zeros; when all entries are zero, negative provenance is not evidence-ranked in a meaningful way.

### F2.4 — Current verification is not a target-rewrite layer

The current gate computes class-level verdicts and hard-fail status but does not rewrite per-contract merged labels. Old audit prose describing in-place verification is historical/stale relative to current executable behavior. The known June DoS correction was an out-of-band direct Parquet mutation and is classified separately.

### F2.5 — Split/export amplify zero semantics

All-zero rows become `primary_class="NonVulnerable"`, enter the NonVulnerable cap, and later export as ten zeros. Missing dictionary keys are also defaulted to zero.

### F2.6 — ML has no class-target mask

`SentinelDataset` produces `y[10]`; collate stacks `y`; `AsymmetricLoss` computes negative loss wherever `label=0`. There is no target-state or class-mask channel. Therefore an unknown/unsupported historical zero is optimized as a negative.

### F2.7 — Run12's 2,635-row population gap is count-reconciled

The protected split has 20,579 train+val rows while Run12 loaded 19,858, a delta of 721. The export has 836 no-representation rows and the ML loader filters those IDs. The remaining 115 no-rep rows belong to test under the consistent protected layout. Run12 did not load the 1,914-row test set:

```text
721 train/val rows filtered for missing reps
+ 1,914 test rows not loaded
= 2,635
```

This exactly explains the count delta. Exact byte identity of the export at Run12 launch remains unproven, but the population arithmetic is no longer unexplained.

### F2.8 — Configured source is not evidence

Web3Bugs is enabled in config but absent as data/parser/crosswalk/connector. DISL is enabled for a NonVulnerable pool but the Etherscan connector is a stub. Neither can supply class evidence merely from configuration.

### F2.9 — The nominal label orchestration seam is not reproducible

`data_module/dvc.yaml` declares `sentinel-data label --config config.yaml`, while `_run_label()` currently prints `NOT IMPLEMENTED`. Lower-level parsers exist, but the current nominal DVC/CLI seam cannot reproduce the historical label build end-to-end.

## Quantitative anchors

Protected active corpus:

- DIVE: 22,073 generated rows;
- SolidiFI: 283 generated rows;
- SmartBugs Curated: 137 generated rows;
- merged/split/export labels: 22,493 rows;
- represented: 21,657;
- no representation: 836.

Exact active per-class positive counts are recorded in `03_quantification_matrix.md`. Mechanism counts include:

- ≥66,219 DIVE unsupported-class zero cells across three wholly unsupported canonical classes;
- 2,686 DIVE no-folder rows before Bad-Randomness-only increment;
- 634 DIVE Bad Randomness rows with exclusive subset unresolved;
- 2,547 SolidiFI non-target zero cells;
- 2,655 DoS positive cells directly zeroed by the June patch;
- 677 graph-duplicate groups with conflicting labels (~4,700 contracts historically reported).

## Evidence limitations that do not fail G2

The following exact sub-counts are unavailable from tracked GitHub-only artifacts:

- protected-v3 DIVE source-positive cross-tab by all seven mapped classes;
- protected-v3 SmartBugs 137-row class cross-tab;
- exclusive Bad-Randomness-only DIVE count;
- exact historical Run12-time export hash identity.

These limitations prevent some contract/cell-level quantification, but **do not leave a historical target origin category semantically unexplained**. Phase 2 explicitly permits individual outcomes to remain unknown.

## G2 decision

**G2 PASS.**

Every mandatory historical positive/zero origin category has a named source/transformation path, representative traces exist, current-vs-historical verification behavior is separated, and population effects are reconciled or explicitly bounded without converting missing evidence into negative evidence.

## Next permitted action

Proceed to **Phase 3 — Evidence Ledger**.

Phase 3 should materialize contract×class evidence states without yet changing the protected historical export. The Phase-2 categories should become provenance/state fields rather than being collapsed back into binary labels.
