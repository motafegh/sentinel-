# Audit Report — Tests & Configuration

**Scope:** `tests/test_skeleton.py`, `tests/test_ingestion/`, `tests/test_preprocessing/`, `tests/test_representation/`, `config.yaml`, `pyproject.toml`, `pytest.ini`
**Plan Reference:** All 3 stage plans (exit criteria, task 1.8, task 2.10)

---

## 1. Test Completeness — Regression Tests for 8 Fixed Bugs

| Bug | Test Location | Status | Detail |
|-----|--------------|--------|--------|
| **A9** `now` keyword | `test_pipeline.py:131` + `test_13_issue_preservation.py:146` | **PASS** | Dual coverage (normalizer chain + live graph extraction) |
| **A15** def_use scope | `test_13_issue_preservation.py:173` | **PASS** | Asserts ≤ 8 DEF_USE edges for 2-function contract |
| **A20** label=0 hardcode | `test_13_issue_preservation.py:85` | **PASS** | Greps source for `label = 0` outside comments |
| **A34** prefix sort | `test_13_issue_preservation.py:97` | **PASS** | Greps for `raw_node_features[:, 10]` |
| **A38** NaN guard | `test_13_issue_preservation.py:105` | **PASS** | Greps for `torch.isfinite(loss)` |
| **resume** full is default | `test_13_issue_preservation.py:120` | **PASS** | Greps for `resume_model_only: bool = False` |
| **def_use** (A15) | `test_13_issue_preservation.py:173` | **PASS** | Same as A15 |
| **return_ignored** | `test_13_issue_preservation.py:303` | **PASS** | Live extraction asserts feat[7]=1.0 |

**All 8 regression tests present and correct.**

---

## 2. Specific Tests Requested by Plans

| Test | Plan Ref | Status | Location |
|------|----------|--------|----------|
| A20 label=0 hardcode regression | Task 1.8 | **PASS** | `test_13_issue_preservation.py:85-95` |
| A9 `now` keyword regression | Task 1.8 | **PASS** | `test_pipeline.py:131-142` + `test_13_issue_preservation.py:146-169` |
| Two-pass compile test | Task 1.8 / Exit #9 | **FAIL** | **NOT FOUND.** Docstring mentions "two-pass" but no test exercises two-pass compilation. |
| Pragma tolerance test | Task 1.8 / Exit #9 | **FAIL** | **NOT FOUND.** Same gap — no test verifies pragma version resolution. |
| has_unchecked_block test | Task 1.8 / Exit #6 | **PASS** | `test_pipeline.py:104-112` — two tests (positive + negative) |
| Dedup threshold test | Task 1.8 / Exit #5 | **WARN** | Tests dedup mechanism only (exact match). **No test verifies 0.85 AST similarity threshold.** |
| SolidiFI A-2 RETURN_TO edges | Task 2.6 | **PASS** | `test_solidifi_fixes.py` |
| SolidiFI A-3 interface injection | Task 2.6 | **PASS** | `test_solidifi_fixes.py` |
| SolidiFI A-1 comment stripping | Task 2.6 | **PASS** | `test_solidifi_fixes.py` |
| Byte-identical regression | Task 2.6 | **PASS** | `test_byte_identical_regression.py` |
| Thin-adapter identity | Task 2.6 | **PASS** | `test_thin_adapter.py:130-158` — uses `is` not `==` |
| Dict direction fix | Task 2.1 | **PASS** | `test_thin_adapter.py:100-128` |

---

## 3. Test Correctness

| File | Issue | Status | Detail |
|------|-------|--------|--------|
| `test_pipeline.py:56` | `n_lines_before` assertion | PASS | `"line1\nline2\nline3\n".count('\n') + 1 = 4`. Correct for trailing-newline behavior. |
| `test_label_folderize.py:42` | `symlinks_created == 6` | PASS | Contract 1→Re, 2→Re+DoS, 3→Ar, 5→Re+Ar = 6. Correct. |
| `test_label_folderize.py:59` | `multi_label == 2` | PASS | Contracts 2 and 5 are multi-label. Correct. |
| `test_byte_identical_regression.py:98-109` | `torch.equal` assertion | PASS | Thin adapter is same object, trivially true by construction. Correct. |
| `test_thin_adapter.py:130-158` | `test_missing_ml_raises_clear_error` | PASS | Uses monkeypatch for import simulation. Correct pattern. |
| `test_manifest.py:108` | `"CHANGED" in errors[0]` | PASS | Assumes correct `IngestionManifest.verify()` implementation. |
| `test_retry_failed.py:217-233` | `test_processed_without_meta_skipped_safely` | PASS | Orphan .sol stays in dropped.csv. Correct edge case. |

---

## 4. Test Isolation

| Check | Status | Detail |
|-------|--------|--------|
| No network calls | PASS | All tests use `tmp_path` fixtures or pre-existing local data |
| solc-dependent tests skip gracefully | PASS | `test_byte_identical_regression.py:54`, `test_solidifi_fixes.py:163`, `test_13_issue_preservation.py:136-139` use `pytest.skip` |
| No cross-test contamination | PASS | `test_label_folderize.py:79-98` tests idempotency; `test_retry_failed.py` uses isolated `tmp_path` |

### WARN: Hardcoded absolute paths in tests

| File | Line | Path |
|------|------|------|
| `test_orchestrator.py` | 15-16 | `/home/motafeq/projects/sentinel/Data/config.yaml` |
| `test_orchestrator.py` | 23 | `/home/motafeq/projects/sentinel/Data/data/preprocessed/solidifi` |
| `test_label_folderize.py` | 148 | `/home/motafeq/projects/sentinel/Data/data/raw_staging/dive_labels/DIVE_Labels.csv` |

These break on CI, other developers' machines, or Docker. Should use `REPO_ROOT` relative paths or `conftest.py` fixtures.

---

## 5. Mock Usage

| Check | Status | Detail |
|-------|--------|--------|
| External calls mocked | PASS | No external API/DB calls in tests |
| tmp_path for filesystem | PASS | All filesystem tests use pytest's `tmp_path` |
| monkeypatch for imports | PASS | `test_thin_adapter.py:120-158` uses `monkeypatch.setattr` |

---

## 6. Edge Case Coverage

| Edge Case | Test | Status |
|-----------|------|--------|
| Empty directory | `test_connector.py:74` | PASS |
| Deeply nested dirs | `test_connector.py:78` | PASS |
| Missing staging_path | `test_connector.py:152` | PASS |
| Nonexistent staging_path | `test_connector.py:235` | PASS |
| Bad materialize mode | `test_connector.py:245` | PASS |
| Glob staging paths | `test_connector.py:257` | PASS |
| Empty class_columns | `test_label_folderize.py:136` | PASS |
| Missing source file | `test_label_folderize.py:112` | PASS |
| Missing source dir | `test_label_folderize.py:125` | PASS |
| No positive labels | `test_label_folderize.py:100` | PASS |
| Manifest tamper | `test_manifest.py:94` | PASS |
| Manifest missing file | `test_manifest.py:110` | PASS |
| Empty dropped CSV | `test_retry_failed.py:91` | PASS |
| No dropped CSV | `test_retry_failed.py:100` | PASS |
| Orphan .sol (no meta.json) | `test_retry_failed.py:217` | PASS |
| Comments-only file | `test_solidifi_fixes.py:147` | PASS |
| Flat files → `__source__/` move | `test_label_folderize.py:180` | PASS |
| Straggler re-ingest | `test_label_folderize.py:263` | PASS |
| Relative imports kept | `test_pipeline.py:226-252` | PASS |
| Test inheritance stripped | `test_pipeline.py:277-314` | PASS |
| Unknown attr raises AttributeError | `test_thin_adapter.py:161` | PASS |

---

## 7. Configuration Verification

### config.yaml

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| 22 sources across tiers | PASS | — | 6 critical + 14 additive + 2 dropped = 22 |
| BCCC in deferred_sources | PASS | 351 | Present with reason and legacy_outputs |
| mlflow.uri == sqlite | PASS | 36 | `sqlite:///mlruns.db` |
| solc.baseline_versions has 6 | PASS | 39-44 | Exactly 6 entries |
| No sentinel-ml references | PASS | — | Verified |
| FORGE URL correct | PASS | 213 | `shenyimings/FORGE-Artifacts` |
| Tier-1 sources present | PASS | 162-181 | 8 tier-1 sources verified |
| scabench URL correct | PASS | 334 | `scabench-org/scabench` |
| Dedup threshold 0.85 | PASS | 46 | Correct |

### pyproject.toml

| Check | Status | Detail |
|-------|--------|--------|
| No sentinel-ml dep | PASS | `test_no_sentinel_dependency` verified |
| Python >=3.12,<3.13 | PASS | Line 11 |
| pytest in dev deps | PASS | Line 44 |
| CLI entry point | PASS | Line 51 |

### pytest.ini

| Check | Status | Detail |
|-------|--------|--------|
| testpaths = tests | PASS | Line 2 |
| addopts = -v --tb=short | PASS | Line 6 |

---

## 8. Missing Tests (Plan Exit Criteria Gaps)

| Missing Test | Plan Ref | Severity | Impact |
|-------------|----------|----------|--------|
| Two-pass compile test | Task 1.8 / Exit #9 | **HIGH** | Core preprocessing feature has zero test coverage |
| Pragma tolerance test | Task 1.8 / Exit #9 | **HIGH** | Pragma resolution is untested |
| AST dedup threshold (0.85) test | Task 1.8 / Exit #5 | **MEDIUM** | Dedup Level 3 mechanism untested |
| `ingest_source()` integration test | Task 1.8 | **HIGH** | Core ingestion function untested |
| `freshness.py` test | Task 1.7 | **MEDIUM** | Freshness checker untested |
| Pin validation (D-1.3) test | Task 1.8 | **HIGH** | Reproducibility guard untested |
| `test_representation/__init__.py` | — | **LOW** | Inconsistent with other test dirs |

---

## 9. Constructive Findings Summary

### FAIL (2)

1. **No two-pass compile test** — `test_pipeline.py` docstring promises "compiler (two-pass + pragma tolerance)" but no test exists. This is a **critical gap** — two-pass compilation is the mechanism that handles pragma `>=0.4.0` matching multiple solc versions.

2. **No pragma tolerance test** — Same gap. No test verifies that `^0.8.0` resolves to 0.8.20/0.8.24 or that `>=0.5.0` falls back correctly.

### WARN (3)

3. **Dedup threshold (0.85) untested** — `Deduplicator` is tested for exact-hash dedup only. The `ast_similarity_threshold: 0.85` from `config.yaml` is never exercised in tests.

4. **Hardcoded absolute paths** — `test_orchestrator.py:15-16`, `test_orchestrator.py:23`, and `test_label_folderize.py:148` use `/home/motafeq/projects/sentinel/Data/...`. These will fail on CI.

5. **Missing `test_representation/__init__.py`** — Inconsistent with `test_ingestion/__init__.py` and `test_preprocessing/__init__.py`.

### PASS (Summary)

- All 8 bug regressions covered
- A-1, A-2, A-3 SolidiFI fixes covered
- Thin adapter identity verified at object level (`is` not `==`)
- Dict direction fix verified (name→id, not id→name)
- CLI smoke tests comprehensive
- Config validation comprehensive
- Package boundary enforced
- Dockerfile validates bookworm (not slim)
- Schema registry JSON verified (active == v9)
- Idempotency tested
- Retry-failed merge logic well tested (5 scenarios)
- Edge cases well covered (22+ scenarios)

---

## Recommended Actions

| Priority | Action | Effort |
|----------|--------|--------|
| **P0** | Write `test_compiler_two_pass` — feed contract with `pragma solidity >=0.4.0`, verify compiler tries multiple versions | 30 lines |
| **P0** | Write `test_pragma_tolerance` — verify `^0.8.0` resolves to 0.8.20, `>=0.5.0` falls back | 20 lines |
| **P1** | Write `test_dedup_ast_threshold` — two near-identical contracts (similarity ~0.86) are deduped | 30 lines |
| **P1** | Write `test_ingest_source` — integration test for the core ingestion function | 50 lines |
| **P2** | Replace hardcoded paths with `REPO_ROOT / "Data" / ...` | 10 lines |
| **P2** | Add `test_representation/__init__.py` | 1 line |
| **P3** | Write `test_freshness` — unit test for freshness checker | 30 lines |
