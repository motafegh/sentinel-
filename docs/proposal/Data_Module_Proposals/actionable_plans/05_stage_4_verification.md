# Actionable Plan — Stage 4: Verification (the BCCC-failure catcher)

**Date:** 2026-07-07 (revised 2026-06-09 post-friend-review)
**Stage:** 4 of 8 (Week 7: Jul 21–27)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.5, §5 (Week 7)
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §0 (F9, F25, F27, F28, F29, F30), §1 (4-P1 through 4-P10)
**Friend-review revisions (2026-06-09):**
- **NEW task 4.11: SmartBugs Curated recall test** — semantic_checker must retain ≥90% of the 143 hand-labeled positives; this is the independent falsification test the friend recommended (the BCCC Phase 5 regression test is necessary but not sufficient — it tests agreement with Phase 5 ad-hoc scripts, not the checker's own FN rate)
- The Phase 5 BCCC regression test is preserved (per-stage p5_s1 → p5_s6).
**Exit criteria:** re-running the new `verification/` stage on the legacy BCCC corpus produces a `verification_report.md` that matches `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/p5_s6_verification_report.md` to within ±0.5% per-class drop counts; the 6 verification components are independently testable; **probe dataset is 40 per class (matching BCCC seed, not 50)**; **negative-checker threshold is 5% (not 10%)**; **co-occurrence matrix is a primary class_auditor output (catches 99% DoS↔Reentrancy)**; **CEI ordering check is the Reentrancy pattern**; **semantic_checker retains ≥90% of SmartBugs Curated 143 hand-labeled positives (independent falsification test per friend review)**.

---

## Goal

Implement the **Verification** submodule: the AST-level semantic checker, the tool validator (Slither + optional Mythril + Semgrep), the FP estimator, the class auditor, the negative checker, the probe dataset builder, and the report generator. The critical regression test is that the new module reproduces the Phase 5 verification report on the legacy BCCC corpus — this is the proof that the module would have caught the BCCC failure (89% Reentrancy FP, 86.9% CallToUnknown FP) in hours rather than weeks.

After this stage, every labeled dataset run produces a `verification_report.md` with per-class gates (VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL), confidence histograms, drop counts, and a hard/soft gate that blocks downstream export unless explicitly overridden.

---

## Why this stage fifth

Stages 1–3 produced preprocessed, represented, labeled contracts. Stage 4 is the first stage that asks "are these labels *correct*?" — the question that BCCC failed to ask for 14 days of work.

Verification is the highest-leverage stage in the build. It is the single piece that, if it had existed before Run 9, would have prevented the silent model degradation that took 9 training runs to discover. Doing it as a dedicated stage (not folded into labeling) is what makes the verification logic reusable across all sources — the BCCC verification logic is the *template* for every future source's verification.

---

## Design decisions

### D-4.1 — Verification is per-class, not per-source

The verification gate is the property of a *class* across the merged dataset, not the property of a source. A source can have 90% Reentrancy FP (BCCC) but other sources in the merged dataset can recover the class — the per-class gate looks at the merged labels, not the per-source labels.

This decision means a "good" Tier-1 source can compensate for a "bad" Tier-4 source. The BCCC failure was that BCCC was the *only* source for many classes; with 12 sources, the merged class is much more robust.

### D-4.2 — Semantic checks are AST-level pattern matches (with per-class nuances)

The 6 component checks (semantic_checker, tool_validator, fp_estimator, class_auditor, negative_checker, probe_dataset) form a layered defense. The semantic_checker is the most important: it asks "for each (class, contract) pair, does the contract's AST actually contain the code pattern implied by the class label?" Per-class patterns (per AUDIT_PATCHES 4-P1, 4-P2, 4-P8, F25):

- **Reentrancy** → external call + state change **after** call, **with CEI ordering enforced** (per 4-P2). The BCCC Reentrancy detection was 89% FP partly because the pattern matched any external call + state write, even if the state write was BEFORE the call (which is not a reentrancy). The pattern must enforce the ordering: external call BEFORE state write.
- **CallToUnknown** → must have `.call{}` / `.delegatecall{}` / `.send()` / `.transfer()` (this single check would have caught 86.9% of BCCC's CallToUnknown FPs in minutes) **AND** the lvalue is `address` type, not `bool` **AND** the call is NOT in an OZ `SafeERC20.safeTransfer` wrapper (library call, not cross-contract, per 4-P1 and F25).
- **Timestamp** → must reference `block.timestamp` / `now` in a conditional. The `now` keyword is critical per F9 — `_compute_uses_block_globals` already detects it (per A9 fix), so the pattern can rely on `feat[2]=1.0`.
- **IntegerUO** → must have an arithmetic op in a pre-0.8 contract (or `unchecked{}` block in 0.8.x, per Stage 1's `has_unchecked_block` sidecar). The A9 / `_compute_uses_block_globals` fix is already in place; the pattern relies on `feat[11]=1.0` for 0.8.x.
- **ExternalBug** → must have a cross-contract call where the target is not a known interface (distinguished from CallToUnknown by the type of impact — financial loss vs unknown target).
- **GasException** → must have an unchecked `send()` / `transfer()` / low-level call.
- **MishandledException** → must have a call with unused return value (per F29, the `_compute_return_ignored` fix gives `feat[7]`).
- **UnusedReturn** → must have an internal function call with unused return (similar to MishandledException but for internal calls).
- **DoS** → must have a loop with external call or unbounded iteration (the gas-griefing pattern).
- **TOD** → must have `tx.origin` in a permission check.

**Library-call detection** (per AUDIT_PATCHES 4-P8, F25): the `SafeMath.add()` library call is classified as `HighLevelCall` by Slither but is NOT a cross-contract call. The pattern matcher must distinguish library calls (used internally) from cross-contract calls (CallToUnknown / ExternalBug). The check: if the call's destination is a `Contract` or `Library` declaration in the same source file, it's a library call, not cross-contract.

The pattern definitions live in `sentinel_data/verification/patterns/<class>.yaml` — one file per class, human-authored, reviewed. Patterns are reused by the merger (Stage 3) and the analysis (Stage 6) — single source of truth for "what does this class mean in code."

### D-4.3 — Tool validation is corroborative, not authoritative

Slither / Mythril / Semgrep output is used as a *corroboration* signal, not a ground truth. A labeled positive that is *also* flagged by Slither is higher confidence; one that is *not* flagged is not necessarily a false positive (Slither has FPs of its own). The tool_validator reports the per-class Slither agreement rate as one signal among many.

The default tool is Slither (already a dependency); Mythril and Semgrep are opt-in via config flag and require additional install steps.

### D-4.4 — FP estimator uses sampling, not exhaustive run

Running every tool on every labeled positive is expensive (Slither takes ~5–30s per contract; 7K positives × 5s = ~10 hours). The FP estimator samples N positives per class (default 50, configurable), runs all tools on the sample, and reports an empirical FP rate. The sample is stratified by source so each Tier-1 source is represented.

### D-4.5 — Hard gate vs soft gate

The verification produces a per-class gate:
- **VERIFIED** — semantic check passes for > 90% of positives; tool agreement > 70%; FP estimate < 15%
- **PROVISIONAL** — semantic check passes for 60–90%; tool agreement > 50%; FP estimate < 30%
- **BEST-EFFORT** — semantic check passes for 30–60%; structural patterns are documented but not enforced
- **FAIL** — semantic check passes for < 30% OR FP estimate > 30%

Hard gate: any class with `FAIL` blocks downstream export. Soft gate: `PROVISIONAL` and `BEST-EFFORT` export with a warning in the catalog. The override requires an explicit `pipeline.verification.override_classes: [<class_names>]` in `config.yaml` with a documented reason.

### D-4.6 — Negative checker prevents the "NonVulnerable has Slither hits" failure (5% threshold)

For every contract labeled `NonVulnerable`, run Slither and report what fraction has at least one tool hit. The threshold (**default 5%**, per AUDIT_PATCHES 4-P10) is in `config.yaml`; anything above is flagged. This was the BCCC "41% of NonVulnerable had Slither hits" pattern; the negative_checker would have caught it automatically.

**Why 5%, not 10%:** 10% is too lax — by the time 10% of NonVulnerable contracts have tool hits, the class is already heavily contaminated. 5% is the early warning; above 10% is FAIL.

**Why use the canonical CLASS_TO_DETECTORS list** (per AUDIT_PATCHES 4-P4): the `negative_checker` must use the detector list from `project_agents.md`, not a generic Slither run. The point is to catch OZ-flagged patterns that should be false positives on clean code (SafeMath, Reentrancy in `nonReentrant` modifier, etc.). A generic Slither run would flag too many false positives on clean code.

### D-4.7 — Probe dataset is the model interpretability input

The probe dataset is a hand-curated set of ~50 contracts per class where the vulnerability is *visually obvious* in the code (e.g. a known Simple Reentrancy contract, a known Timestamp Dependence contract). The model interpretability suite (existing in `ml/scripts/interpretability/`) uses this set to verify the model has learned the right patterns, not shortcuts. The probe dataset is built once, committed to `sentinel_data/verification/probe_dataset/`, and used by every future model run as a sanity check.

### D-4.8 — Regression test: reproduce the Phase 5 BCCC verification

The BCCC Phase 5 verification report (`p5_s6_verification_report.md`) is the regression test target. The new `sentinel_data verification` stage, when run on the legacy BCCC corpus, must produce a report that matches the Phase 5 report to within ±0.5% per-class drop counts. This is the proof that:
- The new module is *at least as good* as the ad-hoc Phase 5 scripts
- The 7 Phase 5 scripts (`p5_s1_evidence_integration.py`, `p5_s2_bulk_verification.py`, `p5_s3_discrepancy_resolution.py`, `p5_s4_manual_extrapolation.py`, `p5_s6_synthesis.py`, `p5_gap_fixes.py`, `p5_gap_full_slither.py`) can be deprecated

If the regression test fails, the Phase 5 logic is the source of truth and the new module is debugged until it matches.

---

## Tasks — ordered, each with verifiable exit condition

### 4.1 — Refactor the 7 Phase 5 scripts into the 6 verification components

The 7 ad-hoc scripts in `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/scripts/` are the reference implementation. The new module's 6 components (semantic_checker, tool_validator, fp_estimator, class_auditor, negative_checker, probe_dataset) are the structured refactor. The refactor is not a rewrite — the AST patterns, the Slither runs, the per-class statistics must all reproduce the same numbers.

**Why first:** the regression test (4.8) is the gate; the refactor must happen before the test can be written.

**Exit condition:** the 6 new modules compile and import; running each against the BCCC corpus produces intermediate outputs that match the Phase 5 intermediate outputs (file-by-file comparison).

**Commit:** `feat(data-verify): refactor Phase 5 logic into 6 verification components`

---

### 4.2 — Author the per-class pattern YAMLs

Author `sentinel_data/verification/patterns/<class>.yaml` for each of the 10 classes. Each pattern YAML defines the AST-level pattern that the semantic_checker looks for. The patterns are human-authored and reviewed.

The Reentrancy pattern is the most critical (it would have caught the 89% BCCC FP). The CallToUnknown pattern would have caught the 86.9% BCCC FP. The IntegerUO, Timestamp, and DoS patterns catch the medium-confidence classes. The remaining 5 patterns (ExternalBug, GasException, MishandledException, TOD, UnusedReturn) are simpler.

**Why after the refactor:** the patterns are used by the refactored semantic_checker; the refactor must exist first to test the patterns against.

**Exit condition:** all 10 pattern YAMLs exist; each has a worked example showing a positive match and a negative match; the semantic_checker correctly matches both.

**Commit:** 10 separate commits, one per pattern.

---

### 4.3 — Implement `semantic_checker.py`

Author `sentinel_data/verification/semantic_checker.py` that consumes the per-class patterns and runs them against every labeled (class, contract) pair. The output is a per-class report: total labeled positives, # passing semantic check, # failing semantic check, with the failing contracts listed for human review.

**Why:** this is the single most important verification component. The 86.9% CallToUnknown FP rate in BCCC would have been flagged in minutes.

**Exit condition:** the semantic_checker correctly identifies the BCCC Reentrancy FPs (the 89% that the Phase 5 report identified) when run against the legacy BCCC labels.

**Commit:** `feat(data-verify): add semantic_checker with AST pattern matching`

---

### 4.4 — Implement `tool_validator.py`

Author `sentinel_data/verification/tool_validator.py` that runs Slither (default) and optionally Mythril / Semgrep on every labeled positive. Reports the per-class agreement rate: of 100 Reentrancy positives, how many does Slither also flag for reentrancy?

**Why:** tool agreement is the corroboration signal. A class with low tool agreement is suspicious; a class with high agreement is reinforced.

**Exit condition:** tool_validator runs Slither on a 50-contract sample per class; the agreement rates match the Phase 5 report (within ±5% per class).

**Commit:** `feat(data-verify): add tool_validator with Slither integration`

---

### 4.5 — Implement `fp_estimator.py`, `class_auditor.py`, `negative_checker.py`

Author the remaining 3 verification components:
- `fp_estimator.py` — samples N positives per class, runs all tools, reports empirical FP rate per the proposal §3.5.3. **Per AUDIT_PATCHES 4-P9, sampling is stratified by source AND tier** (not just by class). A class with 90% T3 labels and 10% T0 labels has a very different FP rate by tier; the estimate must report per-tier per-class.
- `class_auditor.py` — per-class count, per-source breakdown, per-confidence-tier breakdown per the proposal §3.5.4. **Per AUDIT_PATCHES 4-P3, the per-class statistics must include the co-occurrence matrix as a primary output.** The BCCC 99% DoS↔Reentrancy co-occurrence would have been caught by an automated co-occurrence check.
- `negative_checker.py` — for `NonVulnerable` contracts, reports the fraction with tool hits per D-4.6. **Uses the canonical CLASS_TO_DETECTORS list from `project_agents.md`**, not a generic Slither run.

**Why batched:** the 3 components share infrastructure (per-class iteration, tool invocation) and are best authored together.

**Exit condition:** all 3 components run against the BCCC corpus and reproduce the corresponding Phase 5 numbers (FP estimate within ±5%, class counts within ±1%, negative-checker hit rate within ±5%); co-occurrence matrix is generated and the BCCC 99% DoS↔Reentrancy pattern is flagged automatically.

**Commit:** `feat(data-verify): add fp_estimator + class_auditor + negative_checker (co-occurrence matrix output)`

---

### 4.6 — Implement `probe_dataset.py` (40 per class, not 50) and `report_generator.py`

Author the remaining 2 verification components:
- `probe_dataset.py` — builds the **40-contracts-per-class** hand-curated set per AUDIT_PATCHES 4-P5, 4-P6, F28. The seed is the Phase 5 `review_batches/` (40 contracts per class) — we use this as-is for the v2 baseline. We **add a "trivial positive" + "trivial negative"** for each class (so the model interpretability suite can probe the simplest possible examples). The trivial negative is a clean OZ contract of similar size; the trivial positive is the simplest possible example (e.g. `function withdraw() public { msg.sender.call{value: balances[msg.sender]}(""); balances[msg.sender] = 0; }` for Reentrancy). **50 per class is a v2.1 enhancement, not v2.**
- `report_generator.py` — produces the human-readable `verification_report.md` using the Phase 5 report template. The template is the file `Data/docs/legacy/bccc_deep_dive/.../p5_s6_verification_report.md` (parse it, extract the structure, replicate it in the new module).

**Why 40, not 50:** the v2 baseline can launch with the 40-per-class seed; the trivial positive/negative additions are the only new work in Stage 4. Expanding to 50 is incremental v2.1 work that doesn't gate Run 11.

**Why the trivial positive/negative additions:** the existing 40 contracts per class from Phase 5 are real audit contracts (non-trivial). For the model interpretability suite, having a "the simplest possible example" is invaluable for verifying the model has learned the pattern, not the surface features. The OZ-clean contract as the trivial negative establishes the "what does the model say about clean code" baseline.

**Exit condition:** probe_dataset builds successfully and contains 40 contracts per class + 1 trivial positive + 1 trivial negative per class (420 total); report_generator produces a markdown report that matches the Phase 5 template structure.

**Commit:** `feat(data-verify): add probe_dataset + report_generator`

---

### 4.7 — Write the BCCC regression test (per-stage p5_s1 → p5_s6)

Author `Data/tests/test_verification/test_bccc_regression.py`. The test runs the full `sentinel-data verify` stage against the legacy BCCC corpus (using the v1.4 verified labels as ground truth). The output `verification_report.md` is compared to `Data/docs/legacy/bccc_deep_dive/.../p5_s6_verification_report.md` field by field. Per-class drop counts must match within ±0.5%. Per-class gate verdicts must match exactly.

**Per AUDIT_PATCHES 4-P7, the regression test must check each p5 stage's output individually, not just the final report.** Each stage script has a specific output:
- `p5_s1_evidence_integration.py` → `p5_s1_evidence_table.csv` + `p5_s1_coverage_report.md`
- `p5_s2_bulk_verification.py` → `p5_s2_automated_verdict.csv` + dispute CSVs per class
- `p5_s3_discrepancy_resolution.py` → `p5_s3_refined_verdict.csv` + residual CSVs
- `p5_s4_manual_extrapolation.py` → `p5_s4_final_verdict.csv` + `p5_s4_gate_results.csv`
- `p5_s6_synthesis.py` → `p5_s6_class_size_comparison.csv` + `p5_s6_verification_report.md`

The new module's verification stage must reproduce each stage's output to within tolerance. The 7 Phase 5 scripts are then deprecated in favor of the new module.

**Also per AUDIT_PATCHES N-7, the deprecation must be explicit:** the 7 ad-hoc scripts in `Phase5_LabelVerification_2026-06-08/scripts/` are replaced by the new module; the deprecation is committed (not silently abandoned).

**Exit condition:** all 5 per-stage output tests pass within tolerance; the 7 Phase 5 scripts are deprecated in favor of the new module; the BCCC failure pattern is now caught automatically by the semantic_checker.

**Commit:** `test(data-verify): add BCCC Phase 5 per-stage regression test (p5_s1 → p5_s6) + deprecate 7 ad-hoc scripts`

---

### 4.8 — Wire the `sentinel-data verify` CLI subcommand

Connect `cli.py` `verify` subcommand to the 6 verification components. The CLI iterates over the merged labels, runs each component, produces the per-class gate, and writes the `verification_report.md`. Add `--source <name>` to verify a single source; default is "all enabled sources." Add `--strict` to fail on PROVISIONAL gates (default is to pass with warning).

Update `dvc.yaml` stage `verify` to call `sentinel-data verify`. Add a downstream gate: the `export` stage declares a dependency on the verification report; if any class is `FAIL`, the export stage refuses to run.

**Exit condition:** `sentinel-data verify` runs all 6 components end-to-end; the verification_report.md matches the BCCC regression test; the export stage blocks on FAIL classes.

**Commit:** `feat(data-verify): wire CLI + DVC for the verify stage`

---

### 4.9 — Add tests for the verification stage

Author `Data/tests/test_verification/` with:
- **Pattern tests** — each of the 10 patterns matches a positive example and rejects a negative example
- **Component tests** — each of the 6 components runs against a small fixture
- **Regression test** (the 4.7 test, lives in this dir)
- **Gate tests** — VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL gates are assigned per the rules in D-4.5
- **Override tests** — explicit override in config.yaml bypasses the hard gate

**Exit condition:** `poetry run pytest tests/test_verification -v` passes; coverage > 80%.

**Commit:** `test(data-verify): add full test suite for verification stage`

---

### 4.10 — Author `ADR-0005-verification-design.md`

Document the key design decisions: per-class verification (D-4.1), AST patterns (D-4.2), tool corroboration (D-4.3), sampling-based FP estimation (D-4.4), hard/soft gate (D-4.5), negative checker (D-4.6), probe dataset (D-4.7), Phase 5 regression test (D-4.8), **SmartBugs Curated recall test (D-4.9, NEW 2026-06-09)**.

**Exit condition:** file exists; cites the BCCC failure as the motivation; references the regression test as the gate.

**Commit:** `docs(data): add ADR-0005 for verification design`

---

### 4.11 — NEW 2026-06-09 (friend review): SmartBugs Curated recall test (independent falsification)

**Friend's insight (paraphrased):** the existing Phase 5 BCCC regression test (4.7) is necessary but not sufficient. It tests that the new `semantic_checker` agrees with Phase 5's ad-hoc scripts to within ±0.5% per class. A degenerate checker that always returns "no vulnerability" would have 100% agreement with Phase 5 on NegativeVulnerable contracts but 0% recall on positives. The Phase 5 regression test doesn't catch this.

**Solution:** use the 143 SmartBugs Curated hand-labeled contracts as a **ground-truth probe** for the checker's independent false-negative rate. The 143 contracts are already on disk at `ml/data/smartbugs-curated/`; each has a known DASP category (the crosswalk maps DASP → Sentinel class). The test:

1. **Load** all 143 SmartBugs Curated contracts + their DASP labels.
2. **Run** the `semantic_checker` on each, asserting the positive classes from the crosswalk (per the SmartBugs Curated crosswalk from Stage 3.4).
3. **Compute** per-class recall: of N known positives for class X, how many does the semantic_checker correctly retain?
4. **Aggregate** to per-source recall: SmartBugs Curated as a whole should have ≥ 90% recall.
5. **Threshold (per config.yaml `pipeline.min_viable_corpus.smartbugs_curated_recall_min`):** if recall ≥ 90%, the semantic_checker is validated; if < 90%, **the checker pattern is too strict** (false-negatives are too high) and the Stage 3 minimum-viable-corpus gate §6.5 will defer Run 11 to v2.1.

**Why 90%, not 100%:** some valid reentrancies use intermediate state (not strict CEI ordering); the checker's CEI pattern is intentionally strict (per D-4.2) to drop BCCC-style FPs, but this may also drop some valid positives. 90% is the empirical sweet spot — high enough to retain most true positives, low enough to drop the noisy BCCC positives.

**Why SmartBugs Curated specifically:** it's the **only** Tier-1 / Tier-3 source with hand-labeled contracts that are also small enough (143) to be a tractable ground-truth probe. DeFiHackLabs exploits are too large and complex; DIVE is too large; Web3Bugs is contest-scale. SmartBugs Curated is the canonical 143-contract benchmark in the field.

**Why this is a Stage 4 task, not Stage 3:** the semantic_checker is built in Stage 4.1-4.3. The 143 SmartBugs Curated contracts are labeled in Stage 3.4 (one of the 5 critical-path crosswalks). Stage 4.11 uses both artifacts.

**Exit condition:** the test runs; per-class recall is reported in `data/verification/smartbugs_curated_recall_test/report.json`; aggregate recall is ≥ 90% (or Stage 3's minimum-viable-corpus gate §6.5 is documented to defer Run 11).

**Commit:** `test(data-verify): add SmartBugs Curated 143-contract recall test (≥90% threshold per friend review)`

---

---

## What NOT to fix (preservation list)

| Bug | Status | File:line | Stage 4 action |
|---|---|---|---|
| **A9** `now` keyword | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:587-605` | Do not re-fix. The Stage 2 36-issue test guards it; the Timestamp semantic pattern can rely on `feat[2]=1.0`. |
| **A15** def_map by name | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:1147-1179` | Do not re-fix. The semantic_checker for MishandledException / UnusedReturn can rely on the fixed `feat[7]`. |
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The Stage 2 36-issue test guards it. |
| `_compute_return_ignored` | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py` | Do not re-fix per AUDIT_PATCHES F29. The semantic_checker for MishandledException can use `feat[7]` directly. |
| 99% DoS↔Reentrancy co-occurrence in BCCC | Source: BCCC | (not in v2 corpus) | The Stage 3 merger de-duplicates; the Stage 4 class_auditor co-occurrence matrix flags it as a regression test. |
| 38.8% BCCC duplication | Source: BCCC | (not in v2 corpus) | The Stage 1 dedup at 0.85 handles it for v2 sources. |
| Phase 2 model-side root causes (7 issues) | N/A (model-side, not data-side) | `project_action_plan.md` "phase2_root_cause_analysis.md" | The data-side `complexity_proxy_risk.md` (Stage 6) is the proactive catcher. Stage 4 doesn't need to address the 7 model-side issues. |

## Final exit criteria check

| # | Check |
|---|---|
| 1 | All 10 per-class pattern YAMLs exist with worked examples |
| 2 | All 6 verification components (semantic_checker, tool_validator, fp_estimator, class_auditor, negative_checker, probe_dataset + report_generator) compile and run |
| 3 | The BCCC regression test passes: new `verification_report.md` matches Phase 5 report to within ±0.5% per-class drop counts |
| 4 | The semantic_checker correctly identifies the BCCC Reentrancy FPs (the 89% Phase 5 found) |
| 5 | The semantic_checker correctly identifies the BCCC CallToUnknown FPs (the 86.9% Phase 5 found) |
| 6 | The probe_dataset builds 500 contracts (50 per class × 10 classes) |
| 7 | The hard gate blocks downstream export on FAIL classes; soft gate warns on PROVISIONAL/BEST-EFFORT |
| 8 | The export stage is DVC-blocked from running if any class is FAIL |
| 9 | `dvc repro verify` runs end-to-end |
| 10 | `poetry run pytest tests/test_verification -v` passes with > 80% coverage |
| 11 | `ADR-0005-verification-design.md` is committed |
| 12 | **(NEW) SmartBugs Curated 143-contract recall test passes** (semantic_checker retains ≥90% of confirmed positives) |

All 12 pass → **Stage 4 complete**. Tag `data-stage-4`, proceed to Stage 5.

---

## Risk register

| Risk | Mitigation |
|---|---|
| The 7 Phase 5 scripts use specific Slither API calls that the new `slither-analyzer >= 0.10.0` version doesn't support | The refactor (4.1) ports the logic; if a Slither API has changed, the regression test catches it; the Slither pin in `pyproject.toml` matches the version used in Phase 5 |
| The 10 pattern YAMLs are not comprehensive — some classes have edge cases the patterns miss | The patterns are iterated on with the regression test; if a pattern is too strict, false-negative rate rises and the test catches it; the per-class `BEST-EFFORT` gate is the safety net |
| The semantic_checker is slow (every contract gets full AST analysis per class) | The checker caches AST parses; per-class runs are parallelized with multiprocessing; the worst case is 7K contracts × 10 classes = 70K AST walks, estimated ~2 hours on 8 cores |
| The BCCC regression test never matches the Phase 5 report exactly (small numerical differences) | The ±0.5% tolerance accounts for floating-point and rounding differences; if a field diverges by more, debug until it matches |
| The FP estimator samples are not representative of the class distribution | The sampling is stratified by source AND confidence tier; the test in 4.9 validates the sampling |
| The probe_dataset curation takes longer than estimated (need 50 contracts per class, hand-picked) | The seed is Phase 5's `review_batches/` (40 per class); expansion to 50 is incremental; the v2 baseline can run with the 40-per-class seed if 50 is not yet available |

---

**End of Stage 4 actionable plan. Total estimated time: 5 working days (Jul 7–11), with Jul 12–13 as buffer.**
