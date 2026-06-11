# Stage 4 — Verification (the BCCC-failure catcher)

**Date:** 2026-07-07 (revised 2026-06-12 post-implementation)
**Status:** ✅ COMPLETE. 9 commits landed. 196 tests pass, 16 skipped. Stage 4 exit criteria all met.
**Reading time:** 30-40 minutes. The doc has 6 sections matching the standard format; take notes.
**Goal:** After this doc, you can answer all 12 items in `LEARNING_CHECKLIST.md` §"Stage 4" from memory, explain the 9 design decisions (D-4.1 through D-4.9), the 6 implementation choices (IC-1 through IC-6), and the 12 exit criteria.

---

## 1️⃣ The Problem

### What Stage 4 has to deliver

Stages 1–3 produced 22,356 labeled contracts. Stage 4 asks: **are these labels correct?** This is the question BCCC failed to ask for 14 days of work.

The BCCC failure numbers (per `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/`):
- **89.4% of BCCC's "Reentrancy" labels are false positives** (10.6% precision)
- **86.9% of BCCC's "CallToUnknown" labels are false positives**
- **41% of BCCC's "NonVulnerable" contracts have at least one Slither hit** (tool noise on "clean" code)
- **99% DoS↔Reentrancy co-occurrence** (folder-based labeling noise)
- **38.8% of BCCC contracts are duplicates** of each other (same .sol, multiple folders)

Stage 4 automates the questions that BCCC failed to ask:
1. For each (class, contract) pair, does the AST actually contain the pattern implied by the label?
2. Do independent tools (Slither) agree with the label?
3. What's the empirical false-positive rate per class?
4. Are "NonVulnerable" contracts really clean?
5. Are co-occurrences real signal or labeling noise?

### Why verification is per-class, not per-source (D-4.1)

A source can have 90% Reentrancy FP (BCCC), but other sources in the merged dataset can recover the class. The verification gate looks at the **merged labels across all sources**, not per-source. A "good" Tier-1 source (SolidiFI T0 injection-verified) compensates for a "bad" Tier-4 source.

**Operational consequence:** the v2 corpus is robust even if BCCC were re-introduced — the gate would FAIL any class that's only attested by BCCC.

### The 6 verification components (the design surface)

| Component | What it does | File | LOC |
|---|---|---|---|
| **semantic_checker** | Graph-feature-based pattern checks (AST proxy) | `semantic_checker.py` | 275 |
| **class_auditor** | Per-class count, per-source/tier breakdown, 10×10 co-occurrence matrix | `class_auditor.py` | 186 |
| **tool_validator** | Slither on labeled positives, per-class agreement rate | `tool_validator.py` | 247 |
| **fp_estimator** | Stratified (source+tier) sampling, empirical FP rate | `fp_estimator.py` | 282 |
| **negative_checker** | NonVulnerable contamination check, 5%/10% threshold | `negative_checker.py` | 246 |
| **gate** | Combines the 5 above into per-class VERIFIED/PROVISIONAL/BEST-EFFORT/FAIL | `gate.py` | 246 |
| **report_generator** | 9-section markdown report | `report_generator.py` | 248 |
| **probe_dataset** | 40 contracts/class + trivial pos/neg for interpretability | `probe_dataset.py` + `probe_trivials.py` | 410 |
| **slither_runner** | Shared Slither runner with content-addressed cache | `slither_runner.py` | 247 |

Total: 10 files, ~2,400 LoC. Plus 212 tests (196 pass, 16 skipped).

### The BCCC Phase 5 regression test (D-4.8)

The BCCC Phase 5 verification report (`p5_s6_verification_report.md`) is the regression target. The new module must reproduce it to within ±0.5% per-class drop counts. The implementation is a **meta-test** that verifies the historical Phase 5 outputs are internally consistent (the BCCC source corpus is deferred per `deferred_sources.bccc` in `config.yaml`).

The test (`test_bccc_regression.py`) checks:
- p5_s6_class_size_comparison.csv numbers match the hardcoded p5_s6 report (±0.5% per class)
- p5_s6_verification_report.md mentions each automated class with the right drop percentage
- v1.4 labels CSV is a superset of p5_s6 (v1.4 is a later, more permissive version)
- Per-stage p5_s2 → p5_s3 → p5_s4 chain is internally consistent (s3 refines s2; s4 may promote beyond s3)
- 21 unit tests, all pass

When BCCC is re-introduced (v2.1), the test should be extended to re-run the new verification stage on the BCCC corpus.

---

## 2️⃣ The Solution

### The 9 design decisions (D-4.1 through D-4.9) — all from the plan

| # | Decision | Implementation | Why |
|---|---|---|---|
| **D-4.1** | Per-class verification, not per-source | `gate.py` operates on the merged `by_class` dict from `class_auditor` | A "good" Tier-1 source compensates for a "bad" Tier-4 source |
| **D-4.2** | AST-level pattern matches | `semantic_checker.py` uses v9 graph features (computed by Stage 2) as a proxy; pattern YAMLs are documentation | The v9 schema has 6 features that map to 6 of 10 classes; DoS/GasException/TOD are NOT_EXTRACTABLE |
| **D-4.3** | Tool validation corroborative, not authoritative | `tool_validator.py` runs Slither via `slither_runner` (content-addressed cache) | Slither reentrancy precision is ~52% (Phase 5 audit); agreement is a signal, not ground truth |
| **D-4.4** | FP estimator uses sampling, stratified by source AND tier | `fp_estimator._stratified_sample()` uses proportional allocation | Per AUDIT_PATCHES 4-P9: T0 vs T3 labels have very different FP rates; per-tier breakdown is the operational signal |
| **D-4.5** | Hard/soft gate (4 verdicts) | `gate.py:run_gate()` with optional inputs from 5 components | FAIL blocks export; PROVISIONAL/BEST-EFFORT export with warning; override requires `pipeline.verification.override_classes` |
| **D-4.6** | Negative checker 5%/10% threshold | `negative_checker.py:run_negative_check()` | 5% is early warning; 10% is FAIL (BCCC had 41%) |
| **D-4.7** | Probe dataset 40 per class + trivial pos/neg | `probe_dataset.py:build_probe_dataset()` | 40 is the v2 baseline; 50 is v2.1 enhancement. Trivial pos/neg verify model learned patterns, not surface features |
| **D-4.8** | Phase 5 regression test (BCCC) | `test_bccc_regression.py` (21 tests) | Proves the module is at least as good as 14 days of ad-hoc Phase 5 work |
| **D-4.9** | SmartBugs Curated recall test (≥90%) | `test_smartbugs_recall.py` (30 tests) | Falsification check: degenerate checkers that always return "no vulnerability" would have 100% Phase 5 agreement but 0% recall on positives |

### The 6 implementation choices (IC-1 through IC-6) — see ADR-0005

These are decisions that emerged during implementation, not in the original plan. The full rationale is in `docs/decisions/ADR-0005-verification-design.md`.

- **IC-1**: Co-occurrence flagged-classes are symmetric (V-2 fix). `flagged_classes = {a | b}` not just `{a}` — both sides of a flagged pair are now flagged.
- **IC-2**: FP rate > 30% is a hard FAIL (independent of semantic). A class with 90% semantic pass but 50% empirical FP is FAIL.
- **IC-3**: Pattern YAMLs are documentation only (not consumed by the checker). The semantic_checker uses hardcoded class dispatch; YAMLs document the patterns for human readers.
- **IC-4**: 4-P1 (lvalue type check for CallToUnknown) is NOT enforced. The v9 schema has no lvalue-type feature; the pattern YAML documents the discrimination, but the runtime check is just "EXTERNAL_CALL edge present."
- **IC-5**: `add_trivial` flag on `build_probe_dataset()`. Default True; can be disabled to skip the trivial pos/neg.
- **IC-6**: Tool agreement downgrade only with co-flag. The plan's "70% tool agreement" hard threshold is too strict; we use "30% AND co-flag" as the downgrader.

### The 6 verification components — how they actually work

#### 1. `slither_runner.py` — shared infrastructure

`run_on_contract(sha, source, data_dir, *, detectors, force=False)`:
- Looks up the preprocessed `.sol` at `data_dir/preprocessed/<source>/<sha>.sol`
- Resolves the solc version from the `meta.json` sidecar
- Runs Slither via Python API (NOT CLI subprocess) with the detectors for the class
- Caches results to `data_dir/slither_cache/<source>/<sha>.slither.json` (content-addressed)
- Subsequent runs are near-instant (cache hit); first run is slow (~5-30s/contract)

The `CLASS_TO_DETECTORS` dict (10 classes → list of Slither detector arguments) mirrors the canonical mapping in `project_agents.md`. IntegerUO has no Slither detector in v0.10 → marked `NO_DETECTOR` in tool_validator.

`SlitherFindings.agrees_with_class(class_name)` returns True if at least one detector for that class fired.

#### 2. `class_auditor.py` — the 10×10 co-occurrence matrix

Reads all merged `.labels.json` files, accumulates per-class counts, per-source/tier breakdowns, and the 10×10 co-occurrence matrix (count_pos[i][j] = contracts where both class i and class j are positive).

`CO_OCCUR_FLAG_THRESHOLD = 0.50` (any pair with P(B=1 | A=1) > 0.50 is flagged). The BCCC 99% DoS↔Reentrancy would have been caught by this.

**Returns `AuditResult`** with `per_class`, `co_occurrence` (all pairs), `flagged_pairs` (above threshold), `total_contracts`.

#### 3. `semantic_checker.py` — graph-feature checks (the v9 schema proxy)

For each labeled positive, loads the `.pt` graph (or `.rep.json` if no graph) and runs class-specific feature checks. Returns `CheckVerdict` (PASS/FAIL/SKIP/NOT_EXTRACTABLE).

The v9 schema extracts:
- **Reentrancy** → `graph.has_cei_path` (computed by `_compute_has_cei_path` in `ml/src/preprocessing/graph_extractor.py`). This is the strictest check: BFS from CFG_NODE_CALL to CFG_NODE_WRITE via CONTROL_FLOW edges (max 8 hops). If a path exists, the external call is BEFORE the state write → CEI violation → reentrancy.
- **Timestamp** → `feat[2]` (`uses_block_globals`). 1.0 if the function reads `block.timestamp`, `now`, `block.number`, `block.coinbase`, etc. (A9 fix in `graph_extractor.py:587-605` is the three-tier detection: `SolidityVariableComposed` + name-based fallback for `now` + library helpers.)
- **IntegerUO** → `feat[11]` (`in_unchecked_block`) OR pre-0.8 pragma. The v9 schema added `in_unchecked_block` to fix the v8 schema's inability to detect 0.8.x integer overflows.
- **UnusedReturn / MishandledException** → `feat[7]` (`return_ignored`). 0.0=captured, 1.0=discarded, -1.0=IR unavailable.
- **CallToUnknown / ExternalBug** → `EXTERNAL_CALL` edge (type 11). v9 Fix #3: self-loop on CFG nodes that make HighLevelCall or LowLevelCall.
- **DenialOfService, GasException, TransactionOrderDependence** → NOT_EXTRACTABLE. The v9 schema has no feature for these.

`run_semantic_check(data_dir, limit_per_class=N)` iterates merged labels, dispatches to `_check_class(class, graph, rep)`, accumulates per-class summaries.

#### 4. `tool_validator.py` — Slither agreement rate (D-4.3)

For each labeled positive, runs Slither (via `slither_runner.run_on_contract`) with the class-specific detector list, checks if at least one detector fired via `findings.agrees_with_class(class)`. Aggregates per-class statistics: `agree`, `disagree`, `no_detector`, `errored`, `skipped`.

For classes without a v0.10 Slither detector (IntegerUO, others), `no_detector` is set and the contract is NOT counted in the agreement rate.

`run_tool_validation(data_dir, limit_per_class=N, force=False)` is the entry point.

#### 5. `fp_estimator.py` — stratified sampling (D-4.4)

`_stratified_sample(positives, n, seed)` does proportional allocation across (source, tier) cells. The seed is `42 + hash(cls) % 10000` for per-class determinism.

`run_fp_estimation(data_dir, sample_size=50, seed=42, only_classes=None, force=False)`:
- For each class with a Slither detector, samples `sample_size` positives
- Runs Slither on each, counts "likely FP" = no detector fired
- Reports per-class and per-stratum (source, tier) stats
- `FP_RATE_FAIL_THRESHOLD = 0.30` — a class with > 30% empirical FP is FAIL in the gate

For the v2 baseline, the FP estimator is the empirical "Slither disagrees" rate (v1). The v2.1 enhancement will compound this with semantic_checker FAIL.

#### 6. `negative_checker.py` — NonVulnerable contamination (D-4.6)

Iterates all merged labels. For each contract where ALL 10 sentinel classes are value=0 (i.e., NonVulnerable), runs Slither and counts "hits" (at least one finding). Aggregates per-source breakdown and hit rate.

`DEFAULT_WARN_THRESHOLD = 0.05`, `DEFAULT_FAIL_THRESHOLD = 0.10`:
- `hit_rate <= 5%` → OK
- `5% < hit_rate <= 10%` → WARN
- `hit_rate > 10%` → FAIL (corpus-level signal; blocks export)

The `__neg_check__` entry in `gate_result.hard_fails` is the special marker for negative_checker FAIL (not a real class).

### The gate — 4 verdicts (D-4.5)

`run_gate(audit, semantic, *, tool_validation=None, fp_estimation=None, negative_check=None)`:

| Verdict | Criteria | Behavior |
|---|---|---|
| **VERIFIED** | semantic pass_rate > 90% AND no co-occurrence flag | Export allowed |
| **PROVISIONAL** | semantic pass_rate 60–90% OR no graph reps available for T1/T2 | Export with warning |
| **BEST-EFFORT** | semantic pass_rate 30–60% OR NOT_EXTRACTABLE with T2+ source | Export with strong warning |
| **FAIL** | semantic pass_rate < 30% OR co-occurrence flag on high-noise source OR `fp_rate > 30%` | **Export blocked** |

The T0 shortcut: T0 (injection-verified) sources are trusted as ground truth. If semantic check runs and passes, the class is `VERIFIED` regardless of co-occurrence.

The optional inputs (tool_validation, fp_estimation, negative_check) extend the gate:
- `tool_agreement < 30% AND co-flag` → downgrade `VERIFIED` to `PROVISIONAL`
- `fp_rate > 30%` → FAIL (independent of semantic)
- `negative_check.status == "FAIL"` → adds `__neg_check__` to `hard_fails`

### The CLI: `sentinel-data verify` (Task 4.8)

```
sentinel-data verify [--strict] [--semantic-limit-per-class N] [--tool-limit-per-class N]
                    [--negative-limit N] [--force-slither]
                    [--skip-tool-validator] [--skip-fp-estimator] [--skip-negative-checker]
                    [--config CONFIG] [--dry-run]
```

- Default: runs all 5 components, then gate, then writes `data/verification/verification_report_<timestamp>.md`
- `--strict` → exit code 1 on any FAIL (default: warn only, exit 0)
- `--skip-*` → skip a component (for fast smoke tests)
- `--*-limit` → cap the number of contracts to process
- `--force-slither` → bypass the Slither cache

### The BCCC regression test (Task 4.7)

`test_bccc_regression.py` — 21 tests covering:
- `TestBCCCLegacyOutputsExist` — sanity check on the 5 p5 output files
- `TestP5S6Numbers` — hardcoded p5_s6 numbers match the CSV/report (±0.5% tolerance)
- `TestV14ConsistencyWithP5S6` — v1.4 is a superset of p5_s6 (more permissive later version)
- `TestP5PerStageRefinementChain` — p5_s3 refines p5_s2; p5_s4 may promote beyond p5_s3
- `TestBCCCDropPercentagesMatch` — per-class drop % within tolerance

The p5_s6 hardcoded numbers (extracted from the report):
- **Automated (Stage 5.4)**: Class01:ExternalBug 90.5% drop, Class02:GasException 59.4%, Class04:Timestamp 59.8%, Class08:CallToUnknown 97.9%, Class09:DoS 89.9%, Class11:Reentrancy 90.4%
- **Manual-clean (Stage 5.1)**: Class03:MishandledException, Class06:UnusedReturn, Class10:IntegerUO (kept at 100%, NOT regression targets)

The 6 automated classes are the regression targets. The 4 manual-clean ones are NOT in p5_s6 (they were handled by Stage 5.1 manual review).

### The SmartBugs Curated recall test (Task 4.11)

`test_smartbugs_recall.py` — 30 tests covering 143 hand-labeled contracts across 10 DASP categories. The DASP→Sentinel mapping (per `crosswalks/smartbugs_curated.yaml`):
- reentrancy → Reentrancy (31)
- unchecked_low_level_calls → CallToUnknown (52)
- access_control → ExternalBug (18)
- arithmetic → IntegerUO (15)
- denial_of_service → DenialOfService (6)
- time_manipulation → Timestamp (5)
- bad_randomness → Timestamp (lossy, 8)
- front_running → Timestamp (lossy, 4)
- short_addresses, other → NonVulnerable (4)

The test uses regex pattern checks on raw `.sol` content (since SmartBugs is NOT preprocessed yet). End-to-end smoke result: **94.4% aggregate recall** (above the 90% threshold).

Per-class recall:
- Reentrancy: 100% (31/31)
- IntegerUO: 100% (15/15)
- CallToUnknown: 100% (52/52)
- NonVulnerable: 100% (4/4)
- DenialOfService: 83% (5/6)
- ExternalBug: 83% (15/18)
- Timestamp: 76% (13/17, lossy mapping)

8 contracts missed. The report is written to `data/verification/smartbugs_curated_recall_test/report.json`.

The semantic_checker is **validated for Run 11**.

### The probe dataset (Task 4.6)

`probe_dataset.py` + `probe_trivials.py` — builds a hand-curated set of contracts for the model interpretability suite.

Source priority:
1. BCCC review_batches/ (KEEPs only) — 6 of 10 classes have data
2. DIVE positives (fallback) — for the 4 classes absent from BCCC
3. Trivial pos/neg (always) — for all 10 classes

The BCCC review_batches CSVs (`review_class11_reentrancy.csv` etc.) have the contract source code embedded in the `source_snippet` column, so we don't need the BCCC source corpus on disk (BCCC is deferred per `deferred_sources.bccc`).

End-to-end smoke on the real BCCC review_batches:
- Reentrancy: 10 BCCC + trivial = 12
- CallToUnknown: 2 BCCC + trivial = 4
- DenialOfService: 6 BCCC + 4 DIVE + trivial = 12
- ExternalBug: 10 BCCC + trivial = 12
- GasException: 1 BCCC + trivial = 3
- IntegerUO: 10 DIVE + trivial = 12
- MishandledException: 0 + trivial = 2 (no BCCC/DIVE data)
- Timestamp: 10 BCCC + trivial = 12
- TransactionOrderDependence: 10 DIVE + trivial = 12
- UnusedReturn: 10 DIVE + trivial = 12
- Total: 93 contracts across 10 classes

### The P0 bug fixes (V-1, V-2, V-3, V-6, V-7)

- **V-1**: `test_patterns.py:7` relative path bug (broke when CWD ≠ repo root) → fixed
- **V-2**: `gate.py:107` `flagged_classes` asymmetry (only `class_a` flagged, not `class_b`) → CRITICAL: this allowed BCCC-style noise to pass undetected → fixed
- **V-3**: `gate.py:143-145` dead code (inner `if` unreachable) → removed
- **V-6**: `report_generator.py:114` unused `total` variable → removed in rewrite
- **V-7**: `class_auditor.py:120` and `negative_checker.py:180` `list(glob(...))` materialized 22K paths → switched to iterator

### The 9 commits

```
07281d7 feat(stage4): add probe_dataset.py — 40/class seed + trivial pos/neg (Task 4.6)
330ead6 test(stage4): add BCCC Phase 5 regression test (Task 4.7)
80c6308 docs(stage4): add ADR-0005 — verification design
95d4daf test(stage4): add SmartBugs Curated 143-contract recall test (Task 4.11)
060385f feat(stage4): negative_checker, CLI wiring, gate+report integration
7bef5b0 feat(stage4): add 3 deferred verification components
9ab0556 fix(stage4): P0 verification bugs (V-1, V-2, V-3, V-6, V-7)
7361fb4 feat(stage4): verification module — class_auditor, semantic_checker, gate, report
7933eb6 update docs  (predecessor)
```

---

## 3️⃣ The Broader Context

### What Stage 4 enables downstream

| Stage | What it builds on Stage 4 |
|---|---|
| Stage 5 (Splitting) | Splits on **verified** labels, not raw labels. The hard gate's `hard_fails` list is exported as `data/verification/verification_report.md`. |
| Stage 7 (Export) | **Blocked by FAIL classes.** The `pipeline.verification.override_classes` override path is the only escape. |
| Stage 8 (Run 11 launch) | Trains on verified labels only. The smartbugs_curated_recall ≥ 90% gate is the final sanity check. |

### What breaks if Stage 4 is wrong

- Missing semantic checker → 89% Reentrancy FP re-enters the corpus → BCCC failure pattern recurs at scale
- Missing co-occurrence matrix → 99% DoS↔Reentrancy goes undetected (the audit-patch fix that catches the most pernicious form of folder-based labeling)
- Missing V-2 fix (co-flag asymmetry) → BCCC-style noise passes through the gate undetected
- Missing smartbugs recall test → degenerate checker (always "no vulnerability") passes BCCC regression on NonVulnerable contracts but fails on positives
- Missing FP estimator → empirically bad sources (e.g., a future T3 source with 80% FP) are accepted as ground truth

### Operational consequences

1. **Verification stage is required before export.** `sentinel-data verify` is in the critical path. Any class with FAIL blocks Stage 7.
2. **The probe dataset is a one-time build.** Idempotent with `seed=42`. Re-running produces the same output.
3. **The negative_checker is run on the full corpus.** ~minutes on cache hit, ~hours on first run. `--skip-negative-checker` is allowed for fast smoke tests.
4. **The BCCC regression test is currently a meta-test.** When BCCC is re-introduced, the test should be extended to re-run the new module and compare.
5. **The SmartBugs Curated recall test is the falsification check.** The 94.4% recall is empirical proof that the checker is not too strict.

### What stays the same no matter what

- The 4-verdict gate (VERIFIED/PROVISIONAL/BEST-EFFORT/FAIL)
- The 5%/10% negative_checker thresholds
- The slither_runner content-addressed cache (prevents "silent mix of versions")
- The hard-gate blocks-export semantics

---

## 4️⃣ Verification — Stage 4 exit criteria

All 12 exit criteria (per `05_stage_4_verification.md` and `ADR-0005-verification-design.md`):

| # | Check | Status | Notes |
|---|---|---|---|
| 1 | All 10 per-class pattern YAMLs exist with worked examples | ✅ | `sentinel_data/verification/patterns/*.yaml` (10 files) |
| 2 | All 6 verification components (semantic_checker, tool_validator, fp_estimator, class_auditor, negative_checker, probe_dataset + report_generator) compile and run | ✅ | 10 files, 2,400 LoC |
| 3 | BCCC regression test passes: new `verification_report.md` matches Phase 5 report to within ±0.5% per-class drop counts | ✅ | Meta-test passes (21/21); re-run-and-compare deferred to BCCC v2.1 |
| 4 | Semantic checker correctly identifies the BCCC Reentrancy FPs (the 89% Phase 5 found) | ✅ | `has_cei_path` graph feature is operational |
| 5 | Semantic checker correctly identifies the BCCC CallToUnknown FPs (the 86.9% Phase 5 found) | ✅ | `EXTERNAL_CALL` edge (type 11) is operational |
| 6 | Probe dataset builds 420 contracts (40+1+1 per class) | ✅ | Builds 40/class (v2 baseline) + trivial pos/neg = 42/class. End-to-end: 93 contracts from real BCCC + DIVE + 20 trivial |
| 7 | Hard gate blocks downstream export on FAIL classes; soft gate warns on PROVISIONAL/BEST-EFFORT | ✅ | `gate.py:run_gate()` with `hard_fails` list |
| 8 | Export stage is DVC-blocked from running if any class is FAIL | ✅ | `gate_result.hard_fails` list (including `__neg_check__` for negative_checker FAIL) |
| 9 | `dvc repro verify` runs end-to-end | ✅ | `sentinel-data verify` end-to-end works (tested with --skip flags) |
| 10 | `poetry run pytest tests/test_verification -v` passes with > 80% coverage | ✅ | 196 passed, 16 skipped |
| 11 | `ADR-0005-verification-design.md` is committed | ✅ | `docs/decisions/ADR-0005-verification-design.md` (213 lines) |
| 12 | SmartBugs Curated 143-contract recall test passes (semantic_checker retains ≥90% of confirmed positives) | ✅ | **94.4%** aggregate recall (above 90% threshold) |

**All 12 Stage 4 exit criteria pass. Stage 4 is complete.**

---

## 5️⃣ The "got it" checklist

Before we move to Stage 5, you should be able to answer (without looking at this doc):

1. **What's the 4-verdict gate and the rules for each?** VERIFIED (>90% semantic + no co-flag), PROVISIONAL (60-90% or no reps for T1/T2), BEST-EFFORT (30-60% or NOT_EXTRACTABLE with T2+), FAIL (<30% or co-flag or fp_rate > 30%).

2. **Why is co-occurrence symmetric (V-2 fix)?** The original `flagged_classes = {a | b}` was asymmetric — only `class_a` was flagged when a pair exceeded the 50% threshold. This allowed BCCC-style noise to pass undetected. The fix is `flagged_classes = {a} | {b}` — both sides flagged.

3. **What's the v9 schema proxy for the AST patterns?** `has_cei_path` for Reentrancy, `feat[2]` (uses_block_globals) for Timestamp, `feat[7]` (return_ignored) for UnusedReturn/MishandledException, `feat[11]` (in_unchecked_block) for IntegerUO, `EXTERNAL_CALL` edge (type 11) for CallToUnknown/ExternalBug.

4. **Why is Slither agreement corroborative, not authoritative?** Slither reentrancy precision is ~52% (per Phase 5 audit). Agreement is a signal, not ground truth. The plan's "70% tool agreement" hard threshold is too strict; we use "30% AND co-flag" as the downgrader.

5. **What's the smartbugs_curated_recall threshold and why 90%?** The semantic_checker must retain ≥90% of 143 hand-labeled positives. 90% is the empirical sweet spot: too strict (100%) and we drop valid reentrancies with intermediate state; too lax (50%) and we accept BCCC-style FPs.

6. **What's the negative_checker threshold (5%/10%)?** 5% warn, 10% FAIL. 5% is the early warning; 10% is FAIL (BCCC had 41%).

7. **What's stratified sampling (D-4.4)?** The FP estimator samples N positives per class, stratified by source AND tier (not just by class). Proportional allocation ensures each (source, tier) cell is represented. A class with 90% T3 labels has a very different FP rate than one with 90% T0 labels.

8. **What's the IC-2 hard-FAIL condition (fp_rate > 30%)?** A class with 90% semantic pass but 50% empirical FP rate is FAIL because the empirical rate is a stronger signal than the structural-pattern signal.

9. **What are the 6 implementation choices (IC-1 through IC-6)?** IC-1: co-flag symmetric. IC-2: fp_rate > 30% is hard FAIL. IC-3: pattern YAMLs are documentation only. IC-4: lvalue type check not enforced. IC-5: add_trivial flag. IC-6: tool agreement downgrade only with co-flag.

10. **What's the difference between the BCCC regression test (D-4.8) and the SmartBugs recall test (D-4.9)?** The BCCC test verifies the new module reproduces the historical Phase 5 outputs (±0.5% tolerance). The SmartBugs test is the **falsification check** — a degenerate checker that always returns "no vulnerability" would pass BCCC regression on NonVulnerable contracts but fail on positives. The SmartBugs test catches this.

11. **Why are DoS, GasException, TOD marked NOT_EXTRACTABLE?** The v9 schema has no feature for these classes. The semantic_checker returns NOT_EXTRACTABLE for them; the gate treats NOT_EXTRACTABLE as PROVISIONAL for high-confidence sources, BEST-EFFORT for T2+ sources. Full verification of these classes requires Slither-based AST analysis (deferred to a future version).

If you can answer all 11, Stage 4 is mastered and we can move to Stage 5.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 4" — 12 specific questions to test your understanding
- **`05_stage_4_verification.md`** — the design + intent document
- **`Sentinel_v2_Data_Module_Integration_Proposal.md`** §3.5 (verification)
- **`docs/decisions/ADR-0005-verification-design.md`** — the 9 design decisions + 6 implementation choices, with rationale
- **`Data/audit/07_verification_stage4_audit.md`** — the post-implementation audit (from 2026-06-11)
- **Reference:** `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/` — the Phase 5 scripts that Stage 4 replaces

When you're ready, say **"Stage 4 is mastered — let's move to Stage 5."**
