# WS-A: Integrity & Dedup — Manifest

**Date:** 2026-06-06
**Status:** Complete

## Source

- `BCCC-SCsVul-2024/Source Codes/` (12 folders, 111,897 .sol files, 1.6 GB)

## CSV Integrity

| Check | Result |
|---|---|
| MD5 of `BCCC-SCsVul-2024.csv` (actual) | `e38a2aa1c2b8a93c6cf8b23d2d7b870a` |
| MD5 (expected from `BCCC-SCsVul-2024.md5`) | `e38a2aa1c2b8a93c6cf8b23d2d7b870a` |
| Match | **YES** |

## Source-File Integrity

| Check | Result |
|---|---|
| Per-file MD5 list provided in dataset? | **NO** — `Sourcecodes.md5` validates a `SourceCodes.zip` that is NOT present in our extracted directory. |
| Per-file content verifiable against publisher? | **NO** — must trust extraction. |
| Per-file SHA-256 computed and persisted? | **YES** — `sha256_all_files.tsv` (this run). |
| Idempotent re-hash? | **YES** — same file content → same SHA-256. |

**Trust assumption:** the publisher's extraction from the original ZIP was correct. We have no way to validate this from disk. The CSV-level MD5 is verified, which is the strongest guarantee we have.

## Dedup Map

- **Unique content hashes:** 68,433
- **Total file copies:** 111,897
- **Duplicated files (in 2+ folders):** 43,464 (38.84%)
- **Distribution:**

| N folders | # unique contents | % |
|---:|---:|---:|
| 1 | 40,267 | 58.84% |
| 2 | 19,068 | 27.86% |
| 3 | 5,473 | 8.00% |
| 4 | 1,871 | 2.73% |
| 5 | 1,138 | 1.66% |
| 6 | 446 | 0.65% |
| 7 | 137 | 0.20% |
| 8 | 31 | 0.05% |
| 9 | 2 | 0.00% |

## Cross-Checks vs Phase 1

| Metric | Phase 1 (CSV-based) | WS-A (file-based) | Match |
|---|---:|---:|:---:|
| Unique contents | 68,433 | 68,433 | YES |
| Duplicated files | 43,464 | 43,464 | YES |

## Top 5 Most-Copied Contents

| Copies | N folders | Folders | Example file |
|---:|---:|---|---|
| 9 | 9 | CallToUnknown,ExternalBug,GasException,IntegerUO,MishandledException,Reentrancy,Timestamp,TransactionOrderDependence,UnusedReturn | `1cbf966046f79bfde8e008699e3e7ed0bbdaa9c0afc3c16a2dbd12f1db0234cc.sol` |
| 9 | 9 | CallToUnknown,ExternalBug,GasException,IntegerUO,MishandledException,Reentrancy,Timestamp,TransactionOrderDependence,UnusedReturn | `4b8e974db66e981a9cf3b08c875211d90e939f89c4f55274adc484956cc5cbe6.sol` |
| 8 | 8 | CallToUnknown,ExternalBug,GasException,IntegerUO,MishandledException,Reentrancy,Timestamp,UnusedReturn | `147725c17af042280cfb4a4b2b709e133fe0d38020057498052dc60bcf0edbb8.sol` |
| 8 | 8 | CallToUnknown,GasException,IntegerUO,MishandledException,Reentrancy,Timestamp,TransactionOrderDependence,UnusedReturn | `1749bc897ba729bec82ad4c476f3ea8a2b471cefff7ad935b90c4eb6856658a7.sol` |
| 8 | 8 | CallToUnknown,ExternalBug,GasException,IntegerUO,Reentrancy,Timestamp,TransactionOrderDependence,UnusedReturn | `490144f4d227eb863684b2839270fb362c8ea1c65ffd9f455fd0d4b326553ec8.sol` |

## Decision: How to Use the Dedup Map

**Recommendation:** for SENTINEL training, the 68,433 unique contracts form the training unit. The 12 folders are *candidate categories* (see Phase 1 §1.3), and a contract may appear in N folders (1 ≤ N ≤ 9). The dedup map provides the canonical content identity (sha256) and the canonical ID (filename in `NonVulnerable` folder if any, else first alphabetically).

For WS-G (stratified split), split by **canonical ID**, then map back to all (folder, file) tuples for that ID. This prevents the same contract from leaking across train/val/test via different folder copies.

## Files

- `sha256_all_files.tsv` — 111,897 rows × 4 columns (filename, folder, sha256, size_bytes)
- `dedup_map.csv` — 68,433 rows × 6 columns (content_sha256, canonical_id, n_files, n_folders, folders, sample_filename)
- `manifest.md` — this file

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/a_integrity_dedup.py
```

Run time: ~19s (1.6 GB of I/O on WSL filesystem).
