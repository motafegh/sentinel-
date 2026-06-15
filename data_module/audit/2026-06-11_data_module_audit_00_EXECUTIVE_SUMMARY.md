# Data Module Audit — Executive Summary

**Audit Date:** 2026-06-11 (corrected from 2026-06-16 which was misleading)
**Auditor:** opencode (automated deep audit)
**Scope:** Stages 0–4 implementation + deep source code review of Stages 0–2
**Reference Plans:** `01_stage_0_skeleton.md`, `02_stage_1_ingest_preprocess.md`, `03_stage_2_representation.md`, `04_stage_3_labeling.md`, `05_stage_4_verification.md`

---

## Overall Status

| Module | PASS | WARN | FAIL | Verdict |
|--------|------|------|------|---------|
| Root files (cli, config, pyproject, etc.) | 8 | 3 | 2 | **WARN** |
| `ingestion/` (deep) | 14 | 6 | 4 | **WARN** |
| `preprocessing/` (deep) | 22 | 8 | 5 | **WARN** |
| `representation/` (deep) | 16 | 5 | 3 | **WARN** |
| `labeling/` (Stage 3) | 28 | 6 | 2 | **WARN** |
| `verification/` (Stage 4) | 18 | 9 | 5 | **WARN** |
| Tests & Config | ~30 | 5 | 2 | **WARN** |
| **TOTAL** | **~136** | **42** | **23** | **WARN** |

---

## Critical Findings (FAIL) — Must Fix Before Stage 3 Exit

| # | Module | Finding | Severity | Plan Ref |
|---|--------|---------|----------|----------|
| **F1** | config.yaml | Hardcoded absolute paths (`/home/motafeq/...`) break portability | HIGH | D-0.6 |
| **F2** | config.yaml | `defihacklabs` has `enabled: false` but is critical-path source | HIGH | Stage 1 |
| **F3** | ingestion | D-1.3 pin enforcement missing — empty pins accepted silently | HIGH | D-1.3 |
| **F4** | preprocessing | Dedup Level 3 (AST near-dup @ 0.85) entirely stubbed | HIGH | D-1.6 / 1-P4 |
| **F5** | preprocessing | `CompileResult` mutable default bug (`attempted_versions: list = None`) | MEDIUM | — |
| **F6** | preprocessing | Sidecar `meta.json` missing 3 plan fields (`inheritance_root`, `n_imports`, `contract_count`) | MEDIUM | D-1.7 |
| **F7** | representation | `cache_manager.py:95` — `stale_entries()` returns wrong sha256, stale cache never evicted | HIGH | D-2.5 |
| **F8** | tests | No two-pass compile test (plan exit criterion #9) | HIGH | Task 1.8 |
| **F9** | tests | No pragma tolerance test (plan exit criterion #9) | HIGH | Task 1.8 |
| **F-3.1** | labeling | `defihacklabs.yaml` missing `confidence_tier: T0` field | MEDIUM | D-3.1 |
| **F-3.2** | labeling | `smartbugs_curated.yaml` missing `confidence_tier: T2` field | MEDIUM | D-3.1 |
| **V-1** | verification | `test_patterns.py:7` relative path bug — fails when CWD is `Data/` | HIGH | — |
| **V-2** | verification | `gate.py:107` `flagged_classes` asymmetry — only `class_a` flagged, not `class_b` | HIGH | D-4.5 |
| **V-3** | verification | `gate.py:143-145` dead code in T0 branch (inner `if` unreachable) | MEDIUM | — |
| **REP-2** | representation | **CRITICAL: `graph_schema.py:73-84` `CLASS_NAMES` order MISMATCHES Stage 3 `taxonomy.yaml`** — different class indices will break training | CRITICAL | D-2.2 |
| **REP-7** | representation | `cache_manager.py:95` `stale_entries()` returns wrong sha256 — stale cache never evicted | HIGH | D-2.5 |
| **PRE-4** | preprocessing | Dedup Level 3 (AST near-dup @ 0.85) entirely stubbed | HIGH | D-1.6 |

---

## High-Priority Warnings (Should Fix)

| # | Module | Finding | Plan Ref |
|---|--------|---------|----------|
| W1 | cli.py | `_handle_run` drops `--source`, `--workers`, `--sample` args | D-0.7 |
| W2 | ingestion | `manifest.save()` overwrites — not truly append-only (D-1.2) | D-1.2 |
| W3 | ingestion | `datetime.utcnow()` deprecated in Python 3.12+ | — |
| W4 | ingestion | `freshness.py` pin comparison bug (`upstream.startswith(pin)`) | — |
| W5 | ingestion | No subprocess timeout on `git clone` | — |
| W6 | ingestion | `post_clone_cmd.split()` — should use `shlex.split()` | — |
| W7 | preprocessing | `_CONTRACT_RE` misses `abstract contract` keyword | — |
| W8 | preprocessing | `_IMPORT_LINE_RE` duplicated in flattener + _transitive_strip | — |
| W9 | preprocessing | `mp` fork safety — should use spawn context | — |
| W10 | representation | `cfg_builder.py:243` bypasses thin adapter, imports from `ml.src` directly | D-2.7 |
| W11 | representation | `orchestrator.py:251` dead `cfg: dict` parameter | — |
| W12 | tests | Hardcoded absolute paths in test_orchestrator.py | — |
| W13 | labeling | DeFiHackLabs parser not implemented (only crosswalk exists) | Task 3.7 |
| W14 | labeling | SmartBugs Curated parser not implemented (only crosswalk exists) | Task 3.8 |
| W15 | labeling | Relative paths in test_parser_solidifi.py | — |
| W16 | labeling | Missing conftest.py for shared test fixtures | — |
| W17 | verification | `tool_validator.py` not implemented (Slither integration) | Task 4.4 |
| W18 | verification | `fp_estimator.py` not implemented (sampling-based FP rate) | Task 4.5 |
| W19 | verification | `negative_checker.py` not implemented (5% threshold check) | Task 4.5 |
| W20 | verification | `probe_dataset.py` not implemented (40-per-class seed) | Task 4.6 |
| W21 | verification | CLI `sentinel-data verify` not wired | Task 4.8 |
| W22 | verification | CallToUnknown/ExternalBug indistinguishable in semantic_checker | D-4.2 |
| W23 | verification | UnusedReturn/MishandledException indistinguishable in semantic_checker | D-4.2 |
| W24 | cli | `_handle_run:385-390` hardcodes `workers=1, sample=None, retry_failed=False` | D-0.7 |
| W25 | representation | `orchestrator.py:162-169` 4-levels-up `allow_paths` heuristic is fragile | — |
| W26 | representation | `cfg_builder.py:265` mixes modifiers with functions in single CFG | — |

---

## Design Decision Compliance

| Decision | Status | Notes |
|----------|--------|-------|
| D-0.1 Module location | ✅ PASS | `Data/sentinel_data/` correct |
| D-0.2 Package boundary | ✅ PASS | No sentinel-ml references anywhere |
| D-0.3 Standalone venv | ✅ PASS | `Data/.venv/` independent |
| D-0.4 Stub vs real code | ✅ PASS | Thin adapters in representation, stubs for deferred builders |
| D-0.5 DVC orchestrator | ⚠️ WARN | `freshness` not in DVC DAG |
| D-0.6 Config-as-data | ⚠️ WARN | Absolute paths break portability |
| D-0.7 CLI surface | ⚠️ WARN | `_handle_run` drops stage-specific args |
| D-0.8 Docker | ✅ PASS | Dockerfile uses bookworm, 6 solc baselines |
| D-0.9 Documentation | ✅ PASS | README + architecture.md exist |
| D-0.10 ADR | ✅ PASS | ADR-0001 + ADR-0002 committed |
| D-1.1 Connector-per-family | ✅ PASS | 5 classes, registry correct |
| D-1.2 Manifest append-only | ⚠️ WARN | `save()` overwrites, no versioning |
| D-1.3 Pin enforcement | ❌ FAIL | Empty pins accepted silently |
| D-1.4 5-step pipeline | ✅ PASS | flatten→compile→dedup→normalize→segment correct |
| D-1.5 Drop-not-fix | ✅ PASS | Failed compiles → dropped.csv |
| D-1.6 Three-level dedup | ⚠️ WARN | Level 3 (AST @ 0.85) stubbed |
| D-1.7 Sidecar meta.json | ⚠️ WARN | 3 fields missing, 2 renamed |
| D-2.1 No extraction changes | ✅ PASS | Thin adapter preserves logic |
| D-2.2 Schema v9 | ✅ PASS | All constants correct |
| D-2.3 Tokenizer adapter | ✅ PASS | Clean re-export |
| D-2.4 CFG builder only | ✅ PASS | PDG/callgraph/opcode properly deferred |
| D-2.5 Content-addressed cache | ⚠️ WARN | Bug in stale_entries() |
| D-2.6 Sidecar rep.json | ✅ PASS | All fields present |
| D-2.7 Thin-adapter pattern | ✅ PASS | Correct implementation |
| D-2.8 SHA-256 from Stage 1 | ✅ PASS | No MD5 usage |
| D-3.1 10-class taxonomy locked | ✅ PASS | Verified against trainer.py |
| D-3.3 Conflict resolution T0 > T1 > T2 > T3 > T4 | ✅ PASS | Implemented in _tier_rank() |
| D-3.3 DoS+Reentrancy co-occurrence rule | ✅ PASS | Implemented in _check_co_occurrence_flag() |
| D-3.5 Merged labels = canonical record | ✅ PASS | merger.py writes to data/labels/merged/ |
| D-4.1 Per-class verification | ✅ PASS | Gate operates per-class, not per-source |
| D-4.2 Semantic checks | ⚠️ PARTIAL | Uses graph features, not AST patterns |
| D-4.3 Tool validation corroborative | ❌ FAIL | tool_validator.py not implemented |
| D-4.4 FP estimator sampling | ❌ FAIL | fp_estimator.py not implemented |
| D-4.5 Hard/soft gate | ⚠️ PARTIAL | Implemented but flagged_classes bug (V-2) |
| D-4.6 Negative checker 5% threshold | ❌ FAIL | negative_checker.py not implemented |
| D-4.7 Probe dataset | ❌ FAIL | probe_dataset.py not implemented |
| D-4.8 Phase 5 regression test | ❌ FAIL | Not implemented |
| D-4.9 SmartBugs Curated recall | ❌ FAIL | Not implemented |

---

## What's Working Well

1. **Thin-adapter pattern** — correctly implemented, byte-identical by construction
2. **v9 schema constants** — all correct, verified against live `ml/src/`
3. **Zip-slip protection** — manual connector properly defends path traversal
4. **Two-pass compilation** — correct implementation with pragma tolerance
5. **has_unchecked_block detection** — regex correct, populated in sidecar
6. **Drop-not-fix policy** — failed compiles never enter preprocessed output
7. **Package boundary** — zero sentinel-ml references in Data module
8. **8 bug regression tests** — A9, A15, A20, A34, A38, resume, def_use, return_ignored all covered
9. **SolidiFI fixes** — A-1 (comment stripping), A-2 (RETURN_TO), A-3 (interface injection) tested
10. **Parallel preprocessing** — multiprocessing with proper worker isolation
11. **Taxonomy lock** — class order verified against trainer.py and multilabel_index.csv
12. **Crosswalk documentation** — SolidiFI/DIVE crosswalks thoroughly documented with v2.1 notes
13. **Co-occurrence detection** — DoS+Reentrancy noise flagging implemented per plan D-3.3
14. **Gate implementation** — Go/No-Go gate correctly validates per-class thresholds
15. **Co-occurrence matrix** — class_auditor correctly computes 10×10 conditional probabilities
16. **BCCC pattern detection** — 50% threshold would catch the 99% DoS↔Reentrancy pattern
17. **Pattern YAMLs** — All 10 exist with positive/negative examples and BCCC FP rates
18. **v9 schema awareness** — semantic_checker correctly uses feat[2], feat[7], feat[11], edge type 11

---

## Recommended Fix Priority

### P0 — Fix Immediately (blocks Stage 3 exit)

1. Add pin validation in `ingest.py` (D-1.3) — 5 lines
2. Fix `cache_manager.py:95` stale_entries() sha256 extraction — 2 lines
3. Fix `compiler.py:34` mutable default — 1 line
4. Replace hardcoded paths in `config.yaml` — 2 lines
5. Enable `defihacklabs` in config.yaml — 1 line
6. Add `confidence_tier: T0` to `defihacklabs.yaml` — 1 line
7. Add `confidence_tier: T2` to `smartbugs_curated.yaml` — 1 line
8. Fix `test_patterns.py:7` relative path — 1 line
9. Fix `gate.py:107` flagged_classes asymmetry — 1 line
10. Remove dead code in `gate.py:143-145` — 3 lines

### P1 — Fix Before Stage 7 (seam swap)

8. Implement AST near-dup dedup (Level 3) or document the stub clearly
9. Add missing sidecar fields (`inheritance_root`, `n_imports`, `contract_count`)
10. Add manifest versioning (append-only)
11. Add two-pass compile test
12. Add pragma tolerance test
13. Implement DeFiHackLabs parser (critical-path source)
14. Implement SmartBugs Curated parser (Stage 4 recall gate)
15. Implement `tool_validator.py` (Slither integration)
16. Implement `fp_estimator.py` (sampling-based FP rate)
17. Implement `negative_checker.py` (5% threshold check)
18. Implement `probe_dataset.py` (40-per-class seed)
19. Wire CLI `sentinel-data verify` subcommand
20. Write BCCC regression test (Phase 5 reproduce ±0.5%)
21. Write SmartBugs Curated 143-contract recall test (≥90% threshold)

### P2 — Fix Before Run 11 Launch

15. Replace `datetime.utcnow()` with `datetime.now(timezone.utc)`
16. Fix `freshness.py` pin comparison logic
17. Add subprocess timeout to git connector
18. Use `shlex.split()` for `post_clone_cmd`
19. Fix `_CONTRACT_RE` to match `abstract contract`
20. Fix `_handle_run` to pass stage-specific args
21. Replace hardcoded paths in test files
22. Add conftest.py for test_labeling/ shared fixtures
23. Convert relative paths in test_parser_solidifi.py to absolute
24. Document CallToUnknown/ExternalBug indistinguishability in semantic_checker
25. Document UnusedReturn/MishandledException indistinguishability in semantic_checker
26. Remove unused `total` variable in `report_generator.py:114`
27. Use iterator instead of `list()` in `class_auditor.py:120`
28. Write ADR-0005 for verification design decisions

### P3 — Nice to Have

24. Remove unused imports in `manual_connector.py`
25. Extract shared `_IMPORT_LINE_RE` to common module
26. Add `__all__` to `ingestion/__init__.py`
27. Use `mp.get_context("spawn")` for parallel preprocessing
28. Add `test_representation/__init__.py`
29. Add merger edge case tests (empty inputs, all-Negative, tier precedence)
30. Add ExternalBug test to test_semantic_checker.py
31. Wire pattern YAMLs to semantic_checker (load from YAML, dispatch to checks)
32. Add `test_report_generator.py` integration tests that don't require data
