# `sentinel_data.preprocessing` — From Raw Source to Clean Contracts

## What This Module Does

The preprocessing module is Stage 2 of the SENTINEL data pipeline. It takes raw `.sol` files from the ingestion stage and runs them through a **5-step transformation pipeline** that makes them ready for representation extraction.

The pipeline is the quality gate of the entire system. A contract that fails to compile is dropped, not passed through with a warning. A near-duplicate is merged, not stored twice. A comment-heavy file is stripped to its essential structure. The output is a clean, deduplicated, normalized, version-bucketed contract with a `meta.json` sidecar recording every transformation.

## Why This Matters

The BCCC dataset (the predecessor) had a 38.8% duplication rate and no compile-time quality gate. Contracts that couldn't compile cleanly were silently reclassified as `NonVulnerable` — not because they weren't vulnerable, but because the tool couldn't verify them. This module prevents that class of failure by:

1. **Flattening imports** — resolving the dependency tree to a single self-contained file
2. **Two-pass compilation** — trying exact pragma first, then nearest available version
3. **Three-level deduplication** — catching exact copies, address-level duplicates, and near-duplicates
4. **Normalization** — stripping comments, SPDX headers, and non-semantic whitespace
5. **Version bucketing** — tagging each contract with its Solidity era (legacy/transitional/modern)

## The 5-Step Pipeline

```
Raw .sol file
    │
    ▼
┌─────────────┐
│ 1. Flatten   │  Resolve import chains → single file
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 2. Compile   │  Two-pass: exact pragma → nearest available
└──────┬──────┘  (drop on failure with reason recorded)
       │
       ▼
┌─────────────┐
│ 3. Dedup     │  SHA-256 → address-level → AST near-dup (0.85)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 4. Normalize │  Strip comments, SPDX, whitespace
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│ 5. Segment + Bucket  │  Version tag: legacy/transitional/modern
└──────────────────────┘  + has_unchecked_block detection
       │
       ▼
  Clean .sol + meta.json
```

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `pipeline.py` | Orchestrates the 5 steps; defines `ContractMeta` sidecar schema | 228 |
| `preprocess.py` | CLI service — runs pipeline for one or all sources | 256 |
| `flattener.py` | `solc --flatten` with fallback to recursive import stripping | 311 |
| `compiler.py` | Two-pass compilation with pragma tolerance | 183 |
| `deduplicator.py` | Three-level dedup (SHA-256 → address → AST near-dup) | 76 |
| `normalizer.py` | Strip comments, SPDX headers, normalize whitespace | 35 |
| `segmenter.py` | Version bucketing + `has_unchecked_block` detection | 64 |
| `parallel.py` | Multiprocessing wrapper for the pipeline | 156 |
| `_transitive_strip.py` | Helper for the flattener's import stripping | 104 |

## Step-by-Step Walkthrough

### Step 1: Flattening (`flattener.py`)

Solidity files often import from other files. The flattener resolves these import chains into a single self-contained file using `solc --flatten`.

```python
result = flatten_contract("path/to/Contract.sol")
# result.content       → flattened source code
# result.flatten_status → "flattened" | "skipped_no_imports" | "stripped_unresolved_imports"
```

**Fallback strategy:**
1. Try `solc --flatten` (preferred)
2. If that fails, recursively strip unresolvable imports
3. If the file has no imports, skip flattening entirely

The recursive stripping handles nested imports — if `A.sol` imports `B.sol` which imports `C.sol`, and `C.sol` can't be resolved, the flattener strips the import of `C.sol` from `B.sol`, then strips the import of `B.sol` from `A.sol`.

### Step 2: Compilation (`compiler.py`)

The compiler resolves the `pragma solidity` directive to a specific `solc` version and verifies the file compiles.

**Two-pass compilation:**
1. **Pass 1:** Try the exact version requested by the pragma
2. **Pass 2:** Try the nearest available version from `~/.solc-select/artifacts/`

This two-pass approach recovers contracts with unusual pragma syntax like `^ 0.4 .9` (with spaces) or exact-version pragmas like `0.4.25`.

```python
result = compile_contract("path/to/Contract.sol")
# result.success         → True/False
# result.solc_version    → "0.8.20" (the version that compiled)
# result.pragma_raw      → "^0.8.0"
# result.error           → error message if compilation failed
# result.attempted_versions → ["0.8.20", "0.8.19", "0.8.18"]
```

**Pragma tolerance:** The compiler strips whitespace before regex matching (`^ 0.4 .9` → `^0.4.9`), tries the requested version first, and falls back gracefully.

**Drop-not-fix policy:** Files that fail both compilation passes are dropped. The compile error and attempted versions are recorded in the `meta.json` sidecar and a `dropped.csv` file. This is a deliberate design choice — an unparseable contract cannot be reliably represented as a graph.

### Step 3: Deduplication (`deduplicator.py`)

Three levels of deduplication, in order of increasing sophistication:

| Level | Method | What It Catches |
|-------|--------|-----------------|
| 1 | Exact SHA-256 | Whitespace/comment-only differences across sources |
| 2 | Address-level | Same Ethereum address appearing in multiple sources |
| 3 | AST near-dup (0.85 threshold) | Copy-paste-with-minor-edits patterns |

The AST near-dup threshold is **0.85** (not 0.92). The BCCC duplication rate was 38.8% at 0.85–0.95 similarity; 0.92 misses the "minor edits" cases that caused the duplication.

```python
dedup = Deduplicator()
record = dedup.process(content, path)
# record.sha256           → "abc123..."
# record.dedup_group_id   → "group_042"
# record.is_duplicate     → True/False
# record.duplicate_of     → "path/to/original.sol" (if duplicate)
```

### Step 4: Normalization (`normalizer.py`)

Strips non-semantic content to produce clean, consistent source code:

- **SPDX license headers** — removed (they add noise to tokenization)
- **Block comments** — removed (`/* ... */`)
- **Line comments** — removed (`// ...`)
- **Trailing whitespace** — stripped
- **Multiple blank lines** — collapsed to single blank lines

```python
result = normalize(source_code)
# result.content           → cleaned source
# result.n_lines_before    → 450
# result.n_lines_after     → 320
```

**Important:** The normalizer preserves the `now` keyword (Solidity's alias for `block.timestamp`). The A9 fix in the graph extractor relies on detecting `now` in the source; stripping it would break that detection.

### Step 5: Segment + Version Bucket (`segmenter.py`)

Tags each contract with its Solidity era and detects the presence of `unchecked{}` blocks:

```python
segments = segment_and_bucket(source, pragma_raw)
# segments[0].version_bucket   → "modern" (≥0.8)
# segments[0].has_unchecked_block → True (if unchecked{} found)
# segments[0].contract_names   → ["MyContract", "Helper"]
```

**Version buckets:**
| Bucket | Solidity Version | Characteristics |
|--------|-----------------|-----------------|
| `legacy` | < 0.6 | Old-style, no `unchecked{}`, uses `now` keyword |
| `transitional` | 0.6–0.7 | Bridge era, limited `unchecked{}` support |
| `modern` | ≥ 0.8 | Full `unchecked{}` support, built-in overflow checks |

The `has_unchecked_block` field is recorded in the sidecar so Stage 4's semantic checker can use it for IntegerUO detection in 0.8.x-era contracts.

## The `ContractMeta` Sidecar

Every preprocessed file is accompanied by a `meta.json` with 21 fields:

```json
{
  "sha256": "abc123...",
  "source_name": "defihacklabs",
  "original_path": "src/test/2024-01/flashloan_exp.sol",
  "pragma": "^0.8.0",
  "solc_version": "0.8.20",
  "compile_status": "compiled",
  "compile_error": null,
  "flatten_status": "flattened",
  "dedup_group_id": "group_042",
  "is_duplicate": false,
  "version_bucket": "modern",
  "has_unchecked_block": true,
  "contract_names": ["FlashLoanExploit"],
  "n_raw_lines": 450,
  "n_normalized_lines": 320,
  "n_imports": 3,
  "inheritance_root": "Attack",
  "parent_sha256": null,
  "meta_schema_version": "v1"
}
```

This sidecar is the contract between preprocessing and every downstream stage. Representation reads the `sha256` for cache keys. Labeling reads the `source_name` for crosswalk lookup. Splitting reads the `version_bucket` for stratification.

## Parallel Processing

The `parallel.py` module wraps the pipeline with `multiprocessing.Pool`:

```python
run_preprocess_parallel(pipeline, sol_files, raw_base, n_workers=8)
```

The pool automatically tunes chunk size based on the number of files and workers. The performance budget is **30 files in < 5 minutes on 8 cores** (approximately 10 seconds per file for flatten + compile + dedup).

## How to Use

```bash
# Preprocess a single source
sentinel-data preprocess --source scabench

# Preprocess all enabled sources
sentinel-data preprocess

# With multiprocessing
sentinel-data preprocess --source dive --workers 4

# Sample mode (process first N files)
sentinel-data preprocess --source scabench --sample 30

# Retry previously failed files
sentinel-data preprocess --source scabench --retry-failed
```

## What Gets Dropped

| Drop Reason | When | Recorded In |
|-------------|------|-------------|
| `compile_failed` | Both compilation passes failed | `dropped.csv` + `meta.json` |
| `flatten_failed` | Flattening and import stripping both failed | `dropped.csv` |
| `no_sol_files` | No `.sol` files found after flattening | `dropped.csv` |

The `dropped.csv` includes the attempted solc versions and the specific error message, making it easy to diagnose and recover dropped contracts.

## Pipeline Position

```
Stage 1: Ingestion (pull raw .sol files)
    ↓
Stage 2: Preprocessing ← YOU ARE HERE (5-step pipeline)
    ↓
Stage 3: Representation (extract graphs + tokens)
```

## Design Decisions

1. **Drop-not-fix for compile failures** — an unparseable contract cannot be reliably represented. The BCCC dataset had 8,232 contracts that didn't compile cleanly; they were silently reclassified as `NonVulnerable`, which is not the same as clean.

2. **Dedup at 0.85 threshold** — catches the "copy-paste-with-minor-edits" pattern that caused BCCC's 38.8% duplication rate. 0.92 is too strict.

3. **`has_unchecked_block` as a sidecar field** — the v7 schema dropped `in_unchecked_block` (was dim 9) because 87.9% of BCCC is pre-0.8. But v2 sources will have more 0.8.x contracts, so the flag is preserved in the sidecar for Stage 4's semantic checker.

4. **Two-pass compilation** — recovers contracts with unusual pragma syntax that would otherwise be dropped. The Phase 5 retry script recovered 2,488 contracts this way.
