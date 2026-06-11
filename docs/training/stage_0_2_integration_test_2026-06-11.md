# Stage 0–2 Integration Test Report

**Date:** 2026-06-11  
**Tester:** SENTINEL data engineering  
**Purpose:** Confirm that the full Stage 0 → Stage 1 → Stage 2 pipeline produces correct, schema-compliant
outputs across all available preprocessed sources before proceeding to Stage 3.

---

## Test Scope

| Stage | Component | Status |
|-------|-----------|--------|
| Stage 0 | sentinel_data package skeleton, v9 schema stub | Verified (unit tests) |
| Stage 1 | Ingestion + preprocessing per source | Verified (unit tests) |
| Stage 2 | Representation extraction (graph + tokens + sidecar) | **Integration-tested here** |

### Versions asserted throughout

| Field | Value |
|-------|-------|
| `schema_version` | `v9` |
| `extractor_version` | `v2.1-windowed-gcb` |
| `NODE_FEATURE_DIM` | 12 |
| Token model | `microsoft/graphcodebert-base` |
| Token shape | `(4, 512)` — 4 windows × 512 tokens |

---

## Source Results

### 1. SolidiFI (full corpus)

| Metric | Value |
|--------|-------|
| Preprocessed contracts | 283 |
| Successfully extracted | 276 |
| **Success rate** | **97.5%** |
| Failures | 7 |
| Failure root cause | All `buggy_35.sol` — SolidiFI injection artifact (lone `/` on line 3) |
| Cache re-run (2nd pass) | 276 cache hits, 0 re-extracted, 0.1s |
| Duration (first pass) | 89.6s (~0.32s/contract) |

**Quality checks (276 graphs):**

| Check | Result |
|-------|--------|
| `x.shape[-1] == 12` | PASS — 0 violations |
| Edge attr in `[0, 11]` | PASS — 0 violations |
| Token shape `(4, 512)` | PASS — 0 violations |
| `tokenizer_name` = graphcodebert-base | PASS — 0 violations |
| `schema_version` = `v9` | PASS — all 276 |
| `extractor_version` = `v2.1-windowed-gcb` | PASS — all 276 |
| Edge types on disk | `[0,1,2,3,4,5,6,8,9,10,11]` (type 7 = REVERSE_CONTAINS built at runtime) |
| Node counts | min=35, max=974, median=336 |

**Failure detail:** The 7 failures are the same `buggy_35.sol` contract injected across 7 vulnerability
categories in the SolidiFI dataset. This file contains a lone `/` on line 3 as a SolidiFI injection marker
that Slither cannot parse. This is a known upstream dataset artifact — not a pipeline bug.

---

### 2. DIVE (500-contract sample, 486 total on disk)

| Metric | Value |
|--------|-------|
| Preprocessed contracts available | 22,073 |
| Tested (--limit 500) | 500 |
| Successfully extracted | 486 |
| **Success rate** | **97.2%** |
| Failures | 14 |
| Cache hit behavior | 195 hits correctly recognized from previous run |
| Duration | 149.4s (~0.30s/contract) |

**Quality checks (486 graphs):**

| Check | Result |
|-------|--------|
| `x.shape[-1] == 12` | PASS — 0 violations |
| Edge attr in `[0, 11]` | PASS — 0 violations |
| Token shape `(4, 512)` | PASS — 0 violations |
| `tokenizer_name` = graphcodebert-base | PASS — 0 violations |
| `schema_version` / `extractor_version` | PASS — all 486 |
| Edge types seen | `[0,1,2,3,4,5,6,8,9,10,11]` |
| Node counts | min=6, max=2730, median=258 |
| Window count distribution | 1 window: 19 (4%), 2: 12 (2%), 3: 13 (3%), 4: 442 (91%) |

**Failure breakdown (14):**

| Category | Count | Example |
|----------|-------|---------|
| URL strings in single quotes (preprocessing truncation) | ~8 | `string public baseURI = 'https:` |
| Interface-only files (no implementation to extract) | ~3 | `interface ERC721A__IERC721Receiver` |
| Library-only files | ~2 | `library Strings` |
| Other Slither parse errors | ~1 | — |

The URL truncation is a known Stage 1 preprocessing limitation: single-quoted strings containing `'https:`
are truncated at the `'` boundary during comment stripping. These are purely cosmetic strings (baseURI,
contact info) — not security-relevant contract logic. Affects ~2–3% of DIVE contracts; will be addressed
in Stage 1 bugfix pass before Run 11 if DIVE is included in the critical path.

---

### 3. DefiHackLabs (full corpus)

| Metric | Value |
|--------|-------|
| Preprocessed contracts | 23 |
| Successfully extracted | 14 |
| **Success rate** | **60.9%** |
| Failures | 9 |
| Duration | 3.5s |

**Quality checks (14 graphs):**

| Check | Result |
|-------|--------|
| `x.shape[-1] == 12` | PASS — 0 violations |
| Edge attr in `[0, 11]` | PASS — 0 violations |
| Token shape `(4, 512)` | PASS — 0 violations |
| Edge types seen | `[0,1,2,5,6,8,9,10,11]` (smaller corpus → fewer edge varieties) |
| Node counts | min=2, max=253, median=15 |

**Failure breakdown (9 — all missing imports):**

All 9 failures are contracts that import siblings or parent-directory files using relative paths
(`../IERC721.sol`, `./pool/IPoolActions.sol`, `./MathConstants.sol`). The Stage 1 preprocessor
copies each contract to a flat `preprocessed/defihacklabs/` directory, stripping the repo
directory structure. Slither cannot resolve the relative imports from this flat layout.

This is a **Stage 1 limitation** for multi-file DeFi contracts, not a Stage 2 bug. DefiHackLabs
contracts are complex multi-file DeFi protocol implementations — structurally different from
SolidiFI/DIVE single-file contracts. Stage 1 needs a "flatten before preprocess" step for this
source. Tracked for Stage 1 bugfix pass.

Additionally, 5 of the 14 successes are interface-only files (single function signatures, minimal
graphs with 2–15 nodes). These will produce near-zero features for most vulnerability classes —
acceptable for completeness but low training signal.

---

## Cache Behavior Verification

| Test | Result |
|------|--------|
| Second pass on SolidiFI (same versions) | 276/276 cache hits ✓ |
| Second pass on DIVE (195 prior + 291 new) | 195/195 cache hits ✓ |
| `extractor_version` bump (v2.0 → v2.1) | All old files invalidated — `--force` regenerated correctly |
| `schema_version` in all sidecars | `v9` consistent |

---

## Unit + Regression Test Suite

Run immediately before integration test:

```
pytest Data/tests/ -q
```

| Test class | Tests | Result |
|------------|-------|--------|
| test_representation/test_orchestrator.py | 5 | PASS |
| test_representation/test_13_issue_preservation.py | 13 | PASS |
| test_representation/test_solidifi_fixes.py | 5 | PASS |
| test_representation/test_byte_identical_regression.py | 2 | PASS |
| All Stage 0–2 tests | 45 | PASS |

---

## Known Issues (not blockers for Stage 3)

| ID | Source | Issue | Severity | Fix target |
|----|--------|-------|----------|------------|
| I-1 | DIVE | Single-quoted URL strings truncated during preprocessing (`'https:`) | Low | Stage 1 bugfix pass |
| I-2 | DefiHackLabs | Multi-file contracts with relative imports fail (flat preprocessing) | Medium | Stage 1 `--flatten` option |
| I-3 | SolidiFI | `buggy_35.sol` always fails (upstream dataset artifact) | Info | None — expected |
| I-4 | DefiHackLabs | Interface-only files produce minimal graphs (2–15 nodes) | Info | Filter in Stage 3/5 |

None of these issues affect Stage 3 (labeling). The contracts that fail at Stage 2 will simply
have no representation files and will be excluded from training splits at Stage 5.

---

## Verdict: PASS

All three sources pass schema, shape, and version-consistency checks on every successfully
extracted contract. The pipeline is ready for Stage 3 (labeling).

| Source | Success rate | Schema OK | Shape OK | Cache OK |
|--------|-------------|-----------|----------|----------|
| SolidiFI (full, 283) | **97.5%** | ✓ | ✓ | ✓ |
| DIVE (500 sample) | **97.2%** | ✓ | ✓ | ✓ |
| DefiHackLabs (full, 23) | **60.9%** | ✓ | ✓ | ✓ |

The 60.9% DefiHackLabs rate is a known Stage 1 structural limitation (multi-file imports),
not a Stage 2 bug. All extracted contracts pass quality gates.
