# 03 — Quantification Matrix

This artifact satisfies Phase 2's quantitative requirement using the frozen Phase-0 counts and retained historical evidence. Counts that cannot be reconstructed from tracked remote evidence are explicitly marked `UNAVAILABLE` rather than inferred as fact.

## Active source-row population

The protected 22,493-row merged/split corpus is exactly partitioned by generated source labels:

| Source | Generated label rows | Share of 22,493 | Active-target role |
|---|---:|---:|---|
| DIVE | 22,073 | 98.13% | T2 multi-label folder-derived positives + absence/default zeros |
| SolidiFI | 283 | 1.26% | T0 one-injection positive + nine source-absence zeros |
| SmartBugs Curated | 137 | 0.61% | T1 one-category positive or mapped NonVulnerable + cross-class zeros |
| Web3Bugs | 0 | 0% | configured enabled but unavailable |
| DISL | 0 | 0% | configured negative-only but acquisition path unavailable/stub |
| **Total** | **22,493** | **100%** | — |

Because the three contributing row counts sum exactly to the merged-label total, there is no row-count evidence of SHA overlap in this protected population.

## Historical target counts by canonical class

These are the frozen v3 split totals from Phase 0. `Historical zero` is numeric serialization only and must not be interpreted as confirmed negative.

| Class | Historical positive | Historical zero (`22,493 - positive`) | Positive support status |
|---|---:|---:|---|
| CallToUnknown | 87 | 22,406 | sparse |
| DenialOfService | 1,101 | 21,392 | sparse / historically patched |
| ExternalBug | 16,638 | 5,855 | extremely dominant; DIVE-noise concern |
| GasException | 0 | 22,493 | unsupported in active split |
| IntegerUO | 9,452 | 13,041 | substantial |
| MishandledException | 39 | 22,454 | extremely sparse |
| Reentrancy | 11,399 | 11,094 | substantial but DIVE-noise concern |
| Timestamp | 6,324 | 16,169 | substantial / semantically mixed |
| TransactionOrderDependence | 647 | 21,846 | sparse |
| UnusedReturn | 5,859 | 16,634 | substantial |

Total positive cells exceed row count because DIVE is multi-label.

## Source/class quantification that is directly recoverable

### SolidiFI

The crosswalk documentation records 283 generated labels: 39 contracts for each of six injection folders and 49 Overflow-Underflow contracts.

| Canonical class | Positive rows | Other SolidiFI rows serialized 0 | Zero meaning |
|---|---:|---:|---|
| CallToUnknown | 39 | 244 | non-injected/source absence |
| DenialOfService | 0 | 283 | **unsupported by source** |
| ExternalBug | 39 | 244 | non-injected/source absence |
| GasException | 0 | 283 | **unsupported by source** |
| IntegerUO | 49 | 234 | non-injected/source absence |
| MishandledException | 39 | 244 | non-injected/source absence |
| Reentrancy | 39 | 244 | non-injected/source absence |
| Timestamp | 39 | 244 | non-injected/source absence |
| TransactionOrderDependence | 39 | 244 | non-injected/source absence |
| UnusedReturn | 0 | 283 | **unsupported by source** |
| **cell totals** | **283** | **2,547** | no ten-class negative-control authority |

### DIVE

Exact active row count: 22,073. Exact per-class source-positive counts are not retained in a tracked source-distribution table on remote; the protected split stores only all-source class totals. Phase 2 therefore quantifies DIVE mechanisms that are directly recoverable without pretending to reconstruct unavailable raw folder counts.

| Mechanism | Count | Scope |
|---|---:|---|
| generated DIVE rows | 22,073 | row population |
| multi-label rows (`n_pos > 1`) | 15,259 | historical June audit |
| source files with no vulnerability-folder membership | 2,686 | pre-crosswalk all-zero candidates |
| Bad Randomness folder files | 634 | dropped-category population; exclusive-only subset unavailable |
| CallToUnknown unsupported zeros | 22,073 | class cells before any cross-source override |
| GasException unsupported zeros | 22,073 | class cells before any cross-source override |
| MishandledException unsupported zeros | 22,073 | class cells before any cross-source override |
| unsupported-class zero cells, minimum | **66,219** | `22,073 × 3` |
| conflicting graph-duplicate groups | 677 | ~4,700 contracts; target inconsistency rather than row deletion |

Recovered manual precision evidence additionally shows that DIVE ExternalBug/Reentrancy **positive** authority is weak: raw-folder and tool-agreed samples had only low-single-digit TP rates. Those measurements are evidence-quality counts, not target-population counts.

### SmartBugs Curated

Tracked evidence establishes 137 generated active label rows out of 143 source contracts. Exact generated-row counts per folder/class are not retained in the remote Phase-0 manifests, so source/class positive cells are marked `UNAVAILABLE` rather than reconstructed from a different snapshot.

What is exact from the parser and recovered source evidence:

| Mechanism | Count |
|---|---:|
| source contracts recovered | 143 |
| active generated label rows | 137 |
| source rows not represented as generated active labels | 6 |
| explicit NonVulnerable examples in recovered 143-row evidence | 4 |
| zero-cell count on full 143-row source representation (single-category parser assumption documented by parser) | 1,291 |

The 1,291 figure describes source serialization across the full recovered source corpus (`139×9 + 4×10`), not the exact 137-row active-export zero-cell total.

## Quantification by corruption mechanism

| Corruption / transformation mechanism | Exact or bounded count | Confidence |
|---|---:|---|
| DIVE unsupported CallToUnknown/GasException/MishandledException zeros | ≥66,219 class cells | exact minimum from 22,073 rows × 3 unsupported classes |
| DIVE no-folder all-zero candidates | 2,686 rows | exact retained count before dropped BadRandomness-only increment |
| DIVE Bad Randomness affected | 634 rows | exact folder count; exclusive-only all-zero subset unavailable |
| SolidiFI non-target zero cells | 2,547 cells | exact |
| of which SolidiFI wholly unsupported-class zeros | 849 cells | exact |
| SmartBugs full recovered-source zero serialization | 1,291 cells | derived exactly from 143 rows, 4 NV, single-category parser; active 137-row class distribution unavailable |
| DoS positive→zero direct parquet patch | 2,655 cells | exact historical audit |
| labels/split rows omitted from ML for no representation | 836 rows | exact protected manifest |
| Run12 train+val no-rep filtering | 721 rows | exact arithmetic under protected layout; matches loader behavior |
| test rows not loaded by Run12 training | 1,914 rows | exact |
| previously described Run12 total load delta | 2,635 rows | exactly `721 + 1,914` |
| graph-duplicate groups with conflicting labels | 677 groups (~4,700 rows) | retained historical audit |
| Web3Bugs configured but contributes targets | 0 rows | exact active-source reality |
| DISL configured negative pool contributes targets | 0 rows | exact older export/repo reality; connector is stub |

## Missing exact cross-tabs

The following counts cannot be freshly recomputed from GitHub-only tracked material because the protected raw labels, split JSONL, Parquet and representation artifacts were stored outside ordinary Git:

- DIVE source-positive counts by each of its seven mapped classes in the protected v3 row population;
- SmartBugs active 137-row generated positive counts by canonical class;
- exact exclusive Bad-Randomness-only DIVE count;
- zero-origin decomposition at individual contract×class granularity.

These are **quantitative evidence limitations**, not semantic unknowns. Every mechanism by which those cells became positive or zero is reconstructed at category level in the other Phase-2 findings.

## Quantification conclusion

Phase 2 can make two defensible statements simultaneously:

1. the protected corpus's **historical numeric targets are quantitatively frozen** at 22,493 rows and the ten per-class positive/zero totals above;
2. the historical zeros do **not** have one negative meaning, and several zero-origin sub-counts cannot be recovered exactly from GitHub-only material.

Missing a source/class cross-tab is not a reason to reinterpret a zero as a confirmed negative.
