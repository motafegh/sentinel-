# WS-D: Cross-Corpus Overlap (BCCC vs SmartBugs-curated) — Report

**Date:** 2026-06-06
**Status:** Complete

## Summary

- **BCCC unique contracts:** 68,433
- **SmartBugs-curated contracts:** 143 (143 unique by sha256)
- **Byte-identical overlap:** 0 unique contracts (0.0% of SmartBugs)

## Method

1. Compute SHA-256 of every BCCC `.sol` file (111,897 files) and every SmartBugs-curated `.sol` file (143 files).
2. Set-intersect the SHA-256 hashes. A contract is in the overlap iff its byte content is identical in both corpora.
3. For each overlap, list both labels and compute a category mapping.

## Class Mapping (SmartBugs → Closest BCCC class)

| SmartBugs category | Closest BCCC class | Notes |
|---|---|---|
| `access_control` | `Class07:WeakAccessMod` | Best approximation
| `arithmetic` | `Class10:IntegerUO` | Best approximation
| `bad_randomness` | `Class04:Timestamp` | Best approximation
| `denial_of_service` | `Class09:DenialOfService` | Best approximation
| `front_running` | `Class05:TransactionOrderDependence` | Best approximation
| `other` | `` | **No good BCCC match** — would need a v2 schema
| `reentrancy` | `Class11:Reentrancy` | Exact match
| `short_addresses` | `` | **No good BCCC match** — would need a v2 schema
| `time_manipulation` | `Class04:Timestamp` | Best approximation
| `unchecked_low_level_calls` | `Class06:UnusedReturn` | Best approximation

## Overlap by SmartBugs Category

| SmartBugs category | n overlap | Closest BCCC class |
|---|---:|---|

## Decision Required: How to Handle Overlap

**0 contracts are in both BCCC and SmartBugs-curated.** This is a data leak risk if we use SmartBugs as an OOD test set (per ADR-0005).

Options:

1. **Drop overlap from SmartBugs OOD** — 0 contracts removed; remaining SmartBugs serves as a clean OOD test set.
2. **Relabel overlap to BCCC's labels** — use BCCC's 12-class label as canonical; SmartBugs label discarded. Risk: BCCC and SmartBugs may have different annotation conventions, so a 1:1 relabel is not always correct.
3. **Keep overlap in SmartBugs OOD, ignore in metrics** — measure test F1 on overlap contracts but don't count them in headline numbers.

**Recommendation: (1) Drop from SmartBugs OOD** — simplest, safest, and respects the OOD premise. After drop, SmartBugs has 143 contracts for OOD evaluation.

**No overlap detected.** SmartBugs-curated is a fully disjoint corpus from BCCC (good for OOD).


## Files

- `smartbugs_sha256.tsv` — 143 SmartBugs file hashes
- `bccc_vs_smartbugs_overlap.csv` — overlap detail (one row per SmartBugs file in overlap)
- `overlap_report.md` — this file

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/d_cross_corpus.py
```
