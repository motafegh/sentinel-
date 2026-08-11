# 03 — Population Reconciliation

## Purpose

Separate four populations that were previously discussed as if they were one:

1. labeled/split rows;
2. rows with graph/token representations;
3. rows loaded into Run12 train/validation datasets;
4. rows intentionally excluded because they belong to test.

## Recovered population checkpoints

### June 13 pre-SmartBugs audit state

The historical full-pipeline audit records:

- DIVE labeled rows: **22,073**;
- SolidiFI labeled rows: **283**;
- merged rows: **22,356**;
- SmartBugs Curated, Web3Bugs and DISL skipped at that audit point because their preprocessed directories were absent.

This is an earlier state than the protected Phase-0 export.

### Protected Phase-0 export

`sentinel-v3-smartbugs-2026-06-13`:

- total label/split rows: **22,493**;
- rows with representations: **21,657**;
- rows without representations: **836**;
- current protected split:
  - train: **18,596**;
  - val: **1,983**;
  - test: **1,914**.

The protected export is therefore later than the 22,356-row DIVE+SolidiFI audit state. Remote evidence establishes later SmartBugs participation but does not retain the raw Parquet/source artifacts needed to recompute every source-distribution count from scratch.

## Run12 loader counts

Recovered Run12 launch evidence records:

- train loaded: **18,027**;
- val loaded: **1,831**;
- test loaded: **0**;
- train+val loaded: **19,858**.

Phase 1 originally described `22,493 - 19,858 = 2,635` as a population discrepancy. Phase 2 can now decompose that number using executable loader behavior.

## Exact count decomposition

Current split train+val rows:

```text
18,596 + 1,983 = 20,579
```

Run12 train+val loaded:

```text
18,027 + 1,831 = 19,858
```

Difference:

```text
20,579 - 19,858 = 721
```

The current `SentinelDataset` explicitly filters each split to contract IDs present in the graph shard index. Therefore the 721-row train/val delta is exactly the kind of reduction caused by missing representations.

The protected export has 836 total rows without representations. If the Run12-named export and the protected export share this population layout, then:

```text
no-rep total       = 836
no-rep train+val   = 721
no-rep test        = 115
```

and the represented test population is:

```text
1,914 - 115 = 1,799
```

Check against manifest `n_contracts_with_reps`:

```text
18,027 train represented
+1,831 val represented
+1,799 test represented
=21,657 represented total
```

This matches the protected manifest **exactly**.

Now decompose the previously unexplained 2,635:

```text
2,635 = 721 train/val rows filtered for missing representations
      + 1,914 test rows intentionally not loaded during training
```

Therefore **the count discrepancy is fully explained by known loader behavior plus test exclusion**. No export-regeneration or alternate-split hypothesis is required to explain the numbers.

## What remains unresolved

The arithmetic reconciliation does **not** prove that the byte-identical protected Phase-0 export artifact was the artifact on disk at the exact instant Run12 began. Phase-1 evidence did not preserve a historical Run12-time manifest hash comparison sufficient to establish that identity.

So R4-R010 should be narrowed:

- **population-count discrepancy:** explained;
- **exact historical artifact identity/hash at Run12 start:** still unproven from tracked remote evidence.

This distinction matters. We should not keep an inflated “2,635 unexplained contracts” blocker after its count mechanism has been reconstructed, but we also should not claim byte-for-byte lineage proof we do not possess.

## Historical DoS patch population

The June audit records an out-of-band DoS patch:

- DoS before: **3,750** positive cells;
- DoS+Reentrancy rows zeroed for DoS: **2,655**;
- DoS after: **1,095**.

This changes class-cell population, not contract-row population. It must not be confused with the 836 missing-representation rows or the 2,635 Run12 load difference.

## Leakage/dedup population note

The older audit also records a separate representation-level duplication problem and later graph-hash split repair. Dedup enforcement reassigns families across splits; it does not itself change class-cell semantics. It can, however, change which split receives a corrupted target. Phase 2 therefore treats dedup as a **population placement** mechanism rather than a label-origin mechanism.

## Reconciliation status

| Population question | Status | Explanation |
|---|---|---|
| Why 22,493 labels but 21,657 represented? | EXPLAINED | 836 split/label rows lack representation shards |
| Why Run12 train 18,027 vs current train 18,596? | EXPLAINED BY LOADER MECHANISM | 569 train rows filtered for missing reps under the consistent protected-layout arithmetic |
| Why Run12 val 1,831 vs current val 1,983? | EXPLAINED BY LOADER MECHANISM | 152 val rows filtered for missing reps |
| Why Run12 loaded 0 test? | EXPLAINED | training launch did not instantiate test set |
| Why total difference is 2,635? | EXPLAINED | 721 train/val no-rep rows + 1,914 test rows |
| Exact byte identity of Run12-time export vs protected Phase-0 export | UNPROVEN | historical hash-at-launch evidence not retained remotely |
| June 22,356 rows vs later 22,493 rows | BOUNDED | earlier DIVE+SolidiFI snapshot vs later `smartbugs` export; exact source-level increment cannot be freshly recomputed from remote-only tracked data |

## Phase-2 conclusion

The previously alarming 2,635-row lineage gap is primarily a **population-view mismatch**, not evidence by itself of a mysterious dataset rewrite. The remaining lineage uncertainty is artifact identity, not row-count arithmetic.
