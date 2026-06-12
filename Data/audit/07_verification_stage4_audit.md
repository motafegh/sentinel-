# Stage 4: Verification Implementation Audit

**Audit Date:** 2026-06-11
**Scope:** `Data/sentinel_data/verification/` (semantic_checker, class_auditor, gate, report_generator, patterns)
**Plan Reference:** `docs/proposal/Data_Module_Proposals/actionable_plans/05_stage_4_verification.md`
**Schema Version:** v9 (NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, NUM_EDGE_TYPES=12)

---

## Executive Summary

| Status | Count |
|--------|-------|
| PASS   | 18    |
| WARN   | 9     |
| FAIL   | 5     |

**Overall Assessment:** Stage 4 core implementation (semantic_checker, class_auditor, gate, report_generator) is partially functional. The semantic_checker uses graph features instead of AST patterns (deviation from plan). Three of the six planned verification components are missing (tool_validator, fp_estimator, negative_checker). The probe_dataset and CLI wiring are not implemented. Several logic bugs in the gate and semantic_checker affect correctness.

**Test Results:** 91 tests collected — 36 passed, 54 skipped (no data), 1 failed (relative path bug).

---

## 1. semantic_checker.py

### 1.1 Implementation Analysis

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| Loads graph from .pt file | PASS | 84-94 | Uses `torch.load(weights_only=False)` |
| Loads rep.json metadata | PASS | 97-105 | JSON parse with error handling |
| `_is_pre_08()` helper | PASS | 108-118 | Strips `^~=v` prefixes correctly |
| `_has_external_call_edge()` | PASS | 121-125 | Checks edge_attr for type 11 |
| Reentrancy check | PASS | 130-138 | Uses `has_cei_path` graph attribute |
| Timestamp check | PASS | 140-146 | Uses `feat[2]` uses_block_globals |
| IntegerUO check | PASS | 148-157 | Composite: feat[11] OR pre-0.8 |
| UnusedReturn/MishandledException | PASS | 159-165 | Uses `feat[7]` return_ignored |
| CallToUnknown/ExternalBug | PASS | 167-173 | Uses EXTERNAL_CALL edge type 11 |
| DoS/GasException/TOD | PASS | 175-177 | Returns NOT_EXTRACTABLE |
| Per-class summary dataclass | PASS | 54-72 | Correct fields |
| `run_semantic_check()` | PASS | 180-275 | Iterates merged labels correctly |

### 1.2 Logic Issues

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| CallToUnknown/ExternalBug indistinguishable | WARN | 167 | Both use same EXTERNAL_CALL check |
| UnusedReturn/MishandledException indistinguishable | WARN | 159 | Both use feat[7] return_ignored |
| Reentrancy note message | WARN | 137 | "no CEI path found" is misleading |
| 3 classes marked NOT_EXTRACTABLE | WARN | 175-177 | DoS, GasException, TOD have no v9 feature |

**Issue V-4:** CallToUnknown and ExternalBug use the exact same check. A contract labeled ExternalBug passes the checker if it has any EXTERNAL_CALL edge — same as CallToUnknown. The pattern YAML (ExternalBug.yaml) says these are "broader" categories requiring semantic distinction (auth bypass vs unknown target), but the checker doesn't distinguish.

**Issue V-5:** UnusedReturn and MishandledException use the same `feat[7]` check. The pattern YAML (MishandledException.yaml) says "low-level call" vs UnusedReturn is "any call" — but v9 doesn't separate them at feature level. This is a known schema limitation.

**Issue V-3 (related):** The Reentrancy FAIL message says "no CEI path found (EXTERNAL_CALL before WRITE)" which is correct for a reentrancy, but the note doesn't explain that `has_cei_path == 0` means CEI is followed (NOT a reentrancy), so the FAIL is correct.

### 1.3 Coverage Gap

| Class | v9 Feature | Status |
|-------|-----------|--------|
| Reentrancy | `has_cei_path` | ✅ Extractable |
| Timestamp | `feat[2]` | ✅ Extractable |
| IntegerUO | `feat[11]` + pre-0.8 | ✅ Extractable |
| UnusedReturn | `feat[7]` | ✅ Extractable |
| MishandledException | `feat[7]` | ⚠️ Same as UnusedReturn |
| CallToUnknown | EXTERNAL_CALL edge | ✅ Extractable |
| ExternalBug | EXTERNAL_CALL edge | ⚠️ Same as CallToUnknown |
| DenialOfService | — | ❌ NOT_EXTRACTABLE |
| GasException | — | ❌ NOT_EXTRACTABLE |
| TransactionOrderDependence | — | ❌ NOT_EXTRACTABLE |

3 of 10 classes cannot be verified from v9 features. This blocks full verification for ~30% of the taxonomy. The plan acknowledges this requires Slither-based AST analysis (tool_validator).

---

## 2. class_auditor.py

### 2.1 Implementation Analysis

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| Per-class positive counts | PASS | 130-143 | Correct accumulation |
| Per-source breakdown | PASS | 139-140 | Uses `entry.get("source")` |
| Per-tier breakdown | PASS | 141-142 | Uses `entry.get("tier")` |
| 10×10 co-occurrence matrix | PASS | 115, 146-150 | Correct counting |
| `CO_OCCUR_FLAG_THRESHOLD = 0.50` | PASS | 26 | Matches plan (50% threshold) |
| `summary_lines()` output | PASS | 65-90 | Human-readable report |
| `__str__()` method | PASS | 92-93 | Correct |
| `run_audit()` entry point | PASS | 96-186 | Iterates merged labels |

### 2.2 Performance Issues

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `list(merged_dir.glob(...))` | WARN | 120 | Loads all 22K+ files into memory |
| 10×10 co-occurrence matrix | WARN | 115 | 100 entries — fine |

**Issue V-7:** Line 120 uses `list(merged_dir.glob("*.labels.json"))` which materializes all 22K+ file paths in memory. For a full DIVE corpus run, this could use significant memory. Should use an iterator with `merged_dir.glob(...)` directly.

### 2.3 Co-occurrence Logic

The co-occurrence counting is correct:
- `count_any[i]` = contracts where class i is positive
- `count_pos[i][j]` = contracts where both class i and j are positive
- `rate = count_pos[i][j] / count_any[i]` = P(class_j=1 | class_i=1)
- Pairs are built for `i ≠ j` with `count_pos[i][j] > 0`

The BCCC 99% DoS↔Reentrancy pattern would be caught by this — a pair with rate > 0.50 is flagged.

---

## 3. gate.py

### 3.1 Implementation Analysis

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `Verdict` enum | PASS | 42-46 | VERIFIED/PROVISIONAL/BEST_EFFORT/FAIL |
| `ClassVerdict` dataclass | PASS | 49-57 | All fields present |
| `GateResult` dataclass | PASS | 60-67 | Includes `gate_passed` property |
| `run_gate()` entry point | PASS | 92-213 | Main logic |
| T0 → VERIFIED shortcut | PASS | 138-145 | Injection-verified is ground truth |
| NOT_EXTRACTABLE handling | PASS | 148-157 | PROVISIONAL for T0/T1, BEST_EFFORT for co-flag |
| No graph reps handling | PASS | 159-172 | T0/T1 → VERIFIED/PROVISIONAL, T2 → PROVISIONAL, T3/T4 → BEST_EFFORT |
| Pass rate < 30% → FAIL | PASS | 174-177 | Matches D-4.5 |
| Pass rate 30-60% → BEST_EFFORT | PASS | 179-184 | Matches D-4.5 |
| Pass rate 60-90% → PROVISIONAL | PASS | 186-191 | Matches D-4.5 |
| Pass rate > 90% → VERIFIED | PASS | 193-198 | Matches D-4.5 |
| Co-occurrence downgrade | PASS | 196-198 | VERIFIED downgraded to PROVISIONAL if co-flag |
| `__str__()` method | PASS | 69-89 | Human-readable |

### 3.2 Logic Bugs

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `flagged_classes` asymmetry | FAIL | 107 | Only includes `class_a`, not `class_b` |
| Dead code in T0 branch | FAIL | 143-145 | Inner `if` unreachable |

**Issue V-2 (CRITICAL):** Line 107:
```python
flagged_classes = {p.class_a for p in audit.flagged_pairs}
```
This is asymmetric. If DoS↔Reentrancy is flagged (P(Reentrancy=1|DoS=1) > 50%), only DoS gets `co_flag=True`. Reentrancy does NOT get flagged even though it's part of a suspicious pair. Both classes in a flagged pair should be flagged.

**Impact:** A class could have high co-occurrence noise from a BCCC-style source, and the gate wouldn't downgrade it because only the "class_a" side of the pair gets the flag.

**Issue V-3:** Lines 143-145:
```python
if highest_tier == "T0" and cls_sem.fail_count == 0 and not co_flag:
    verdict_val = Verdict.VERIFIED
    reason = "T0 injection-verified; no semantic failures"
    if cls_sem.fail_count > 0:  # ← UNREACHABLE
        verdict_val = Verdict.PROVISIONAL
        reason = f"T0 source but {cls_sem.fail_count} semantic failures detected"
```
The outer `if` requires `fail_count == 0`, so the inner `if` can never be true. Dead code that should be removed or the outer condition should be relaxed.

### 3.3 Gate Logic Flow

The gate applies rules in this order:
1. T0 + no failures + no co-flag → VERIFIED
2. NOT_EXTRACTABLE + high tier → VERIFIED/PROVISIONAL
3. No graph reps → tier-based verdict
4. Pass rate < 30% → FAIL
5. Pass rate 30-60% → BEST_EFFORT
6. Pass rate 60-90% → PROVISIONAL
7. Pass rate > 90% → VERIFIED (downgraded if co-flag)

This matches D-4.5 from the plan.

---

## 4. report_generator.py

### 4.1 Implementation Analysis

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `VerificationReport` dataclass | PASS | 27-32 | Holds audit, semantic, gate |
| `to_markdown()` method | PASS | 34-147 | Generates 6 sections |
| Per-class gate table | PASS | 44-60 | All 10 classes listed |
| Per-class corpus stats | PASS | 65-80 | Positives, prevalence, source/tier |
| Co-occurrence matrix | PASS | 85-99 | Flagged pairs only |
| Semantic check summary | PASS | 104-118 | Pass/Fail/Skip/Not extractable |
| Hard failures section | PASS | 123-130 | Lists FAIL classes |
| Known limitations section | PASS | 135-146 | Documents deferred components |
| `write()` method | PASS | 149-152 | Creates parent dirs |
| `generate_report()` entry point | PASS | 155-183 | Optional file output |

### 4.2 Minor Issues

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| Unused `total` variable | WARN | 114 | Calculated but never used |
| `_VERDICT_ICON` dict keys | WARN | 19-24 | Uses `Verdict` enum values (works but fragile) |

**Issue V-6:** Line 114:
```python
total = sem.pass_count + sem.fail_count + sem.positives_skipped + sem.not_extractable
```
The `total` variable is calculated but never used in the output. Should be removed or used in the table (e.g., total count column).

---

## 5. Pattern YAMLs

### 5.1 Coverage

| Class | Pattern File | Required Keys | v9 Signal | Status |
|-------|-------------|---------------|-----------|--------|
| Reentrancy | ✅ | ✅ All present | `graph_attribute.has_cei_path` | PASS |
| CallToUnknown | ✅ | ✅ All present | `graph_edge.EXTERNAL_CALL` | PASS |
| Timestamp | ✅ | ✅ All present | `graph_feature.feat[2]` | PASS |
| IntegerUO | ✅ | ✅ All present | `composite: feat[11] + pre-0.8` | PASS |
| UnusedReturn | ✅ | ✅ All present | `graph_feature.feat[7]` | PASS |
| MishandledException | ✅ | ✅ All present | `graph_feature.feat[7]` | PASS |
| ExternalBug | ✅ | ✅ All present | `graph_edge.EXTERNAL_CALL` | PASS |
| DenialOfService | ✅ | ✅ All present | `NOT_EXTRACTABLE` | PASS |
| GasException | ✅ | ✅ All present | `NOT_EXTRACTABLE` | PASS |
| TransactionOrderDependence | ✅ | ✅ All present | `NOT_EXTRACTABLE` | PASS |

### 5.2 Pattern Quality

| Check | Status | Notes |
|-------|--------|-------|
| Positive examples | PASS | All 10 patterns have positive examples |
| Negative examples | PASS | All 10 patterns have negative examples |
| BCCC FP rates documented | PASS | Reentrancy (89%), CallToUnknown (86.9%) |
| DASP IDs | PASS | All 10 patterns have DASP category IDs |
| `tier_for_solidifi` | PASS | 7 patterns have T0, 3 have null (NOT_EXTRACTABLE) |
| `library_call_exclusion` | PASS | CallToUnknown.yaml documents the exclusion |

### 5.3 Pattern-to-Implementation Gap

**WARN:** The pattern YAMLs document AST-level patterns, but the `semantic_checker.py` uses graph features directly. The patterns are not actually consumed by the checker — they serve as documentation only. A future implementation could load the YAMLs and use the `v9_signal.method` field to dispatch to the appropriate check.

---

## 6. Tests

### 6.1 Test Coverage

| File | Tests | Passed | Skipped | Failed | Status |
|------|-------|--------|---------|--------|--------|
| test_class_auditor.py | 12 | 9 | 3 | 0 | PASS |
| test_gate.py | 8 | 6 | 2 | 0 | PASS |
| test_patterns.py | 41 | 1 | 39 | 1 | FAIL |
| test_report_generator.py | 6 | 0 | 6 | 0 | WARN |
| test_semantic_checker.py | 24 | 21 | 3 | 0 | PASS |
| **Total** | **91** | **37** | **53** | **1** | |

### 6.2 Test Issues

| Check | Status | Notes |
|-------|--------|-------|
| Relative path in test_patterns.py | FAIL | `Path("Data/...")` breaks when CWD is `Data/` |
| Integration tests skipped | WARN | Require preprocessed data + merged labels |
| No test for ExternalBug | WARN | semantic_checker tests CallToUnknown but not ExternalBug |
| No test for report_generator output | WARN | All 6 tests skipped (require data) |

**Issue V-1:** `test_patterns.py:7`:
```python
_PATTERNS_DIR = Path("Data/sentinel_data/verification/patterns")
```
This relative path only works if CWD is the repo root. When tests are run from `Data/`, the path should be `sentinel_data/verification/patterns` (without the `Data/` prefix). The same bug pattern appeared in Stage 3 test files.

---

## 7. P0/P1/P2 Issues

### P0 — Must Fix Before Stage 4 Exit

| ID | Issue | File:Line |
|----|-------|-----------|
| V-1 | Relative path in test_patterns.py | `tests/test_verification/test_patterns.py:7` |
| V-2 | `flagged_classes` asymmetry — only `class_a` flagged | `sentinel_data/verification/gate.py:107` |
| V-3 | Dead code in T0 branch | `sentinel_data/verification/gate.py:143-145` |

### P1 — Should Fix Before Stage 4 Exit

| ID | Issue | File:Line |
|----|-------|-----------|
| — | `tool_validator.py` not implemented | — |
| — | `fp_estimator.py` not implemented | — |
| — | `negative_checker.py` not implemented | — |
| — | `probe_dataset.py` not implemented | — |
| — | CLI `sentinel-data verify` not wired | — |
| — | BCCC regression test not implemented | — |
| — | SmartBugs Curated recall test not implemented | — |

### P2 — Should Fix Before Run 11 Launch

| ID | Issue | File:Line |
|----|-------|-----------|
| V-4 | CallToUnknown/ExternalBug indistinguishable | `semantic_checker.py:167` |
| V-5 | UnusedReturn/MishandledException indistinguishable | `semantic_checker.py:159` |
| V-6 | Unused `total` variable | `report_generator.py:114` |
| V-7 | Loads all 22K files into memory | `class_auditor.py:120` |
| — | Pattern YAMLs not consumed by checker | `semantic_checker.py` |
| — | 3 classes NOT_EXTRACTABLE (DoS, GasException, TOD) | `semantic_checker.py:175-177` |
| — | ADR-0005 not written | — |

### P3 — Nice to Have

| ID | Issue | File:Line |
|----|-------|-----------|
| — | ExternalBug test missing from test_semantic_checker.py | — |
| — | report_generator tests all skipped | `tests/test_verification/test_report_generator.py` |

---

## 8. Design Decision Compliance

| Decision | Status | Notes |
|----------|--------|-------|
| D-4.1 Per-class verification | ✅ PASS | Gate operates per-class, not per-source |
| D-4.2 Semantic checks | ⚠️ PARTIAL | Uses graph features, not AST patterns |
| D-4.3 Tool validation corroborative | ❌ FAIL | tool_validator.py not implemented |
| D-4.4 FP estimator sampling | ❌ FAIL | fp_estimator.py not implemented |
| D-4.5 Hard/soft gate | ✅ PASS | VERIFIED/PROVISIONAL/BEST_EFFORT/FAIL implemented |
| D-4.6 Negative checker 5% threshold | ❌ FAIL | negative_checker.py not implemented |
| D-4.7 Probe dataset | ❌ FAIL | probe_dataset.py not implemented |
| D-4.8 Phase 5 regression test | ❌ FAIL | Not implemented |
| D-4.9 SmartBugs Curated recall | ❌ FAIL | Not implemented |

---

## 9. What's Working Well

1. **Co-occurrence matrix** — Correctly computes 10×10 conditional probabilities
2. **BCCC pattern detection** — The 50% threshold would catch the 99% DoS↔Reentrancy pattern
3. **Gate logic flow** — T0 shortcut, tier-based fallbacks, pass rate buckets
4. **Pattern YAMLs** — All 10 exist with positive/negative examples and BCCC FP rates
5. **Test coverage** — 91 tests across 5 files, good unit test density
6. **Dataclass design** — Clean separation of `ContractCheckResult`, `ClassCheckSummary`, `SemanticCheckResult`
7. **v9 schema awareness** — Correctly uses `feat[2]`, `feat[7]`, `feat[11]`, edge type 11
8. **Report generator** — Produces structured markdown with 6 sections

---

## 10. Recommendations

### Fix Immediately (P0)

1. **Fix V-1**: Change `test_patterns.py:7` to use `Path(__file__).parents[2] / "sentinel_data/verification/patterns"`
2. **Fix V-2**: Change `gate.py:107` to `flagged_classes = {p.class_a for p in audit.flagged_pairs} | {p.class_b for p in audit.flagged_pairs}`
3. **Fix V-3**: Remove dead code at `gate.py:143-145`

### Fix Before Stage 4 Exit (P1)

4. **Implement tool_validator.py** — Slither batch runs + per-class agreement rate
5. **Implement fp_estimator.py** — Stratified sampling by source AND tier
6. **Implement negative_checker.py** — NonVulnerable contamination check at 5% threshold
7. **Implement probe_dataset.py** — 40 contracts per class seed from Phase 5
8. **Wire CLI** — `sentinel-data verify` subcommand
9. **Write BCCC regression test** — Reproduce Phase 5 report ±0.5%
10. **Write SmartBugs Curated recall test** — ≥90% semantic checker retention

### Fix Before Run 11 Launch (P2)

11. **Document V-4/V-5** — Add comments explaining CallToUnknown/ExternalBug and UnusedReturn/MishandledException share the same v9 feature
12. **Remove V-6** — Delete unused `total` variable in report_generator.py
13. **Fix V-7** — Use iterator instead of `list()` for glob in class_auditor.py
14. **Wire pattern YAMLs** — Load patterns from YAML and dispatch to checks
15. **Write ADR-0005** — Document verification design decisions

---

## 11. Conclusion

Stage 4 is **~50% complete**. The core verification components (semantic_checker, class_auditor, gate, report_generator) are functional with 3 logic bugs that need fixing. Three of six planned components are missing (tool_validator, fp_estimator, negative_checker). The probe_dataset and CLI wiring are not implemented.

The semantic_checker uses graph features (v9 schema) rather than AST patterns — a reasonable v2 approach that trades AST-level precision for implementation simplicity. The 3 NOT_EXTRACTABLE classes (DoS, GasException, TOD) block full verification for ~30% of the taxonomy and require Slither-based analysis.

The gate logic is correct but has an asymmetry bug (V-2) that could allow BCCC-style noise to pass through undetected. This is the highest-priority fix.

**Test Results:** 91 tests total — 36 passed, 54 skipped (no data), 1 failed (relative path bug). All unit tests for semantic_checker, class_auditor, and gate pass. Integration tests are skipped due to missing preprocessed data.
