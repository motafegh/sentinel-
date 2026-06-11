# ADR-0005: Verification Design

**Date:** 2026-06-12
**Stage:** 4 of 8 (Week 7: Jul 21–27)
**Status:** Accepted (Stage 4 implementation complete)
**Author:** SENTINEL data engineering
**Plan reference:** [`docs/proposal/Data_Module_Proposals/actionable_plans/05_stage_4_verification.md`](../proposal/Data_Module_Proposals/actionable_plans/05_stage_4_verification.md)
**Audit reference:** [`docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md`](../proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md) §1 4-P1 through 4-P10

---

## Context

The BCCC failure (89% Reentrancy FP, 86.9% CallToUnknown FP, 41% NonVulnerable-with-Slither-hits) was caused by *no verification* between labeling and training. 14 days of debugging could have been avoided by an automated check that ran in minutes. The verification module is the BCCC-failure catcher.

The module's purpose is to answer: **"are these labels correct?"** — for every (class, contract) pair produced by Stage 3, the module must be able to say "yes, this is plausible" or "no, this looks like noise."

This ADR records the 9 design decisions that frame the module, the implementation choices made during Stage 4, and the operational consequences.

---

## Design Decisions

### D-4.1 — Verification is per-class, not per-source

The verification gate is the property of a *class* across the merged dataset, not the property of a source. A source can have 90% Reentrancy FP (BCCC) but other sources in the merged dataset can recover the class — the per-class gate looks at the merged labels.

**Operational consequence:** a "good" Tier-1 source (SolidiFI, T0 injection-verified) can compensate for a "bad" Tier-4 source (DISL, NonVulnerable-only).

### D-4.2 — Semantic checks are AST-level pattern matches (with per-class nuances)

The semantic_checker is the most important component. For each (class, contract) pair, it asks: "does the contract's AST contain the code pattern implied by the class label?"

**Implementation:** we use the v9 graph features (computed by Stage 2) as a proxy for AST patterns. The patterns are defined in `sentinel_data/verification/patterns/<class>.yaml` — one file per class, human-authored, reviewed. The patterns are documented but the checker currently uses hardcoded dispatch (not YAML consumption). This is a known deviation from the plan; the YAMLs serve as documentation. Wiring them is a v2.1 enhancement.

Per-class patterns:
- **Reentrancy** → `has_cei_path` (graph attribute, computed by `_compute_has_cei_path`)
- **Timestamp** → `feat[2]` `uses_block_globals`
- **IntegerUO** → `feat[11]` `in_unchecked_block` OR pre-0.8 pragma
- **UnusedReturn / MishandledException** → `feat[7]` `return_ignored` (v9 doesn't distinguish internal vs external)
- **CallToUnknown / ExternalBug** → EXTERNAL_CALL edge (type 11)
- **DenialOfService, GasException, TransactionOrderDependence** → NOT_EXTRACTABLE (no v9 feature)

3 of 10 classes (DoS, GasException, TOD) are NOT_EXTRACTABLE from v9 features alone. They require Slither-based AST analysis (the tool_validator) for full verification.

### D-4.3 — Tool validation is corroborative, not authoritative

Slither has known FPs and FNs (e.g. ~52% precision on reentrancy per Phase 5 audit). High agreement reinforces a class; low agreement is suspicious but not conclusive.

**Implementation:** `tool_validator.py` runs Slither on every labeled positive (using the slither_runner content-addressed cache) and reports the per-class agreement rate. The gate uses this signal in two ways:
1. If `tool_agreement < 30%` AND the class has a co-occurrence flag → downgrade `VERIFIED` to `PROVISIONAL`
2. The agreement rate is reported in the verification report (no other gate effect)

### D-4.4 — FP estimator uses sampling, not exhaustive run

Running Slither on every tool on every labeled positive is expensive (~10 hours first run). The FP estimator samples N positives per class (default 50, configurable).

**Implementation choice:** sampling is **stratified by source AND confidence tier** (per AUDIT_PATCHES 4-P9). A class with 90% T3 labels and 10% T0 labels has very different FP rates by tier; the per-tier per-class breakdown is the operational signal. Proportional allocation (n × cell_size / total) ensures each (source, tier) cell is represented.

**FP definition (v1):** a sampled positive is flagged as a "likely FP" if NO tool (currently only Slither) fires for that class on the contract. This is an upper-bound estimate (Slither has known FNs). v2.1 enhancement: compound Slither-disagreement with semantic-checker FAIL.

### D-4.5 — Hard gate vs soft gate

The gate has 4 verdicts: **VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL**.

| Verdict | Criteria | Behavior |
|---------|----------|----------|
| VERIFIED | semantic pass_rate > 90% AND no co-occurrence flag | Export allowed |
| PROVISIONAL | semantic pass_rate 60–90% OR no graph reps available | Export with warning |
| BEST-EFFORT | semantic pass_rate 30–60% OR NOT_EXTRACTABLE with T2+ source | Export with strong warning |
| FAIL | semantic pass_rate < 30% OR co-occurrence flag on high-noise source OR fp_rate > 30% | **Export blocked** |

**Hard gate:** any class with `FAIL` blocks downstream export. The override requires explicit `pipeline.verification.override_classes: [<class_names>]` in `config.yaml` with a documented reason.

**T0 shortcut:** T0 (injection-verified) sources are trusted as ground truth. If the semantic check runs and passes, the class is `VERIFIED` regardless of co-occurrence (per the plan's "T0 ground truth" design).

### D-4.6 — Negative checker prevents the "NonVulnerable has Slither hits" failure (5% threshold)

For every contract labeled `NonVulnerable` (i.e., none of the 10 sentinel classes is positive), run Slither and report what fraction has at least one tool hit. The threshold is **5% warn / 10% FAIL** (per AUDIT_PATCHES 4-P10).

**Why 5%, not 10%:** 10% is too lax — by the time 10% of NonVulnerable contracts have tool hits, the class is already heavily contaminated. 5% is the early warning; above 10% is FAIL.

**Why use the canonical CLASS_TO_DETECTORS list** (per AUDIT_PATCHES 4-P4): the negative_checker uses the union of detectors from the 10 classes — NOT a generic Slither run. The point is to catch OZ-flagged patterns that should be false positives on clean code (SafeMath, Reentrancy in `nonReentrant`, etc.). A generic Slither run would flag too many false positives on clean code.

**Corpus-level signal:** a `negative_checker.status == "FAIL"` adds a special `__neg_check__` entry to `hard_fails` (which blocks export) but does NOT affect per-class verdicts.

### D-4.7 — Probe dataset is the model interpretability input

The probe dataset is a hand-curated set of contracts where the vulnerability is *visually obvious* in the code. The model interpretability suite uses this set to verify the model has learned the right patterns, not shortcuts.

**Per-class cardinality (v2 baseline):** 40 real audit contracts (from BCCC review_batches KEEPs) + 1 trivial positive (the simplest example of the pattern) + 1 trivial negative (a clean OZ-style contract that exhibits NONE of the 10 patterns). Total per class: 42. Total: 420.

**Why 40, not 50:** the v2 baseline launches with the 40-per-class seed (the BCCC review_batches/ has 30–40 KEEPs per class). 50 is incremental v2.1 work.

**Source priority** in the probe builder: BCCC review_batches (6 of 10 classes) → DIVE positives (fallback for 4 classes) → trivial pos/neg only (for classes with no data).

**BCCC deferred:** the BCCC source corpus is deferred per `deferred_sources.bccc` in `config.yaml`. The probe dataset still builds because the BCCC review_batches CSVs have the contract source code embedded in the `source_snippet` column. When BCCC is re-introduced, the probe dataset is unchanged (it doesn't need the BCCC source corpus on disk).

### D-4.8 — Phase 5 regression test (BCCC)

The BCCC Phase 5 verification report (`p5_s6_verification_report.md`, 2026-06-08) is the regression test target. The new `sentinel_data verification` stage, when run on the legacy BCCC corpus, must produce a report that matches the Phase 5 report to within **±0.5% per-class drop counts**.

**Status:** the BCCC source corpus is not on disk (deferred). The regression test (`test_bccc_regression.py`) is a meta-test that verifies:
1. The v1.4 labels CSV is a superset of p5_s6 (v1.4 may have added KEEPs)
2. The per-stage p5_s2 → p5_s3 → p5_s4 → p5_s6 refinement chain is internally consistent (s3 refines s2, s4 may promote beyond s3)
3. The 6 automated classes' drop percentages are within ±0.5% of p5_s6's reported values
4. The 4 manual-clean classes (MishandledException, UnusedReturn, IntegerUO) are correctly identified as Stage 5.1 (not Stage 5.4)

**When BCCC is re-introduced (v2.1):** the new module's verification stage will be re-run on the BCCC corpus. The per-class drop counts must match p5_s6 ±0.5%. If they diverge by more, the new module is debugged until it matches (per the plan's D-4.8 directive).

### D-4.9 — SmartBugs Curated recall test (independent falsification)

**Friend's insight (paraphrased):** the BCCC regression test (D-4.8) is necessary but not sufficient. It tests that the new `semantic_checker` agrees with Phase 5's ad-hoc scripts to within ±0.5% per class. A degenerate checker that always returns "no vulnerability" would have 100% agreement with Phase 5 on NegativeVulnerable contracts but 0% recall on positives. The Phase 5 regression test doesn't catch this.

**Solution:** use the 143 SmartBugs Curated hand-labeled contracts as a **ground-truth probe** for the checker's independent false-negative rate. Each contract has a known DASP category (the crosswalk maps DASP → 10 Sentinel classes). The test:

1. Load all 143 SmartBugs Curated contracts + their DASP labels
2. Run the `semantic_checker` on each, asserting the positive classes from the crosswalk
3. Compute per-class recall: of N known positives for class X, how many does the semantic_checker correctly retain?
4. Aggregate to per-source recall: SmartBugs Curated as a whole should have ≥ 90% recall
5. **Threshold (per `config.yaml` `pipeline.min_viable_corpus.smartbugs_curated_recall_min`):** if recall ≥ 90%, the semantic_checker is validated; if < 90%, the checker pattern is too strict (false-negatives too high) and the Stage 3 minimum-viable-corpus gate defers Run 11 to v2.1.

**Why 90%, not 100%:** some valid reentrancies use intermediate state (not strict CEI ordering); the checker's CEI pattern is intentionally strict (per D-4.2) to drop BCCC-style FPs, but this may also drop some valid positives. 90% is the empirical sweet spot.

**Why SmartBugs Curated specifically:** it's the only Tier-1 / Tier-3 source with hand-labeled contracts that are also small enough (143) to be a tractable ground-truth probe. DeFiHackLabs exploits are too large and complex; DIVE is too large (22K); Web3Bugs is contest-scale.

---

## Implementation Choices Made During Stage 4

These are decisions that emerged during implementation, not in the original plan:

### IC-1 — Co-occurrence flagged-classes are symmetric (gate.py:107)

**Original plan (gate.py:107):** `flagged_classes = {p.class_a for p in audit.flagged_pairs}` — only the class_a side of a flagged pair is downgraded.

**Issue (V-2 in audit):** if DoS↔Reentrancy is flagged (P(Reentrancy=1|DoS=1) > 50%), only DoS gets the co-flag. Reentrancy is not downgraded even though it's part of a suspicious pair.

**Fix:** `flagged_classes = {p.class_a for p in audit.flagged_pairs} | {p.class_b for p in audit.flagged_pairs}`. Both sides of the pair are now flagged.

**Impact:** a class could have high co-occurrence noise from a BCCC-style source, and the gate now correctly downgrades it.

### IC-2 — FP rate > 30% is a hard FAIL (gate.py)

The plan's D-4.5 only lists semantic pass rate thresholds for FAIL. The implementation adds a third hard-FAIL condition: `fp_rate > 30%` (from `fp_estimator`). This is independent of semantic pass rate — a class with 90% semantic pass but 50% empirical FP rate is FAIL because the empirical rate is a stronger signal of label quality than the structural-pattern signal.

**Rationale:** Slither's disagreement rate is the upper bound on FP rate. A class where >30% of sampled positives are Slither-disagreed is almost certainly a contaminated source (BCCC had 89% FP on Reentrancy). This is a "if in doubt, fail" safety net.

### IC-3 — Pattern YAMLs are documentation only (semantic_checker.py)

The plan's D-4.2 implies that the semantic_checker should consume the pattern YAMLs (`sentinel_data/verification/patterns/<class>.yaml`) and dispatch to the appropriate check via the `v9_signal.method` field.

**Implementation reality:** the semantic_checker uses hardcoded class dispatch (a big if/elif chain in `_check_class`). The patterns are checked into the repo and serve as human-readable documentation, but the runtime check is in code.

**Why:** the YAML dispatch would require a runtime pattern interpreter; the hardcoded dispatch is faster and easier to debug. Wiring the YAMLs is a v2.1 enhancement.

### IC-4 — 4-P1 (lvalue type check) is NOT enforced in semantic_checker

The plan's D-4.2 / AUDIT_PATCHES 4-P1 says: "the pattern must check: (a) presence of `.call{}` etc.; (b) the lvalue is `address` type, not `bool`; (c) the call is not in an OZ `SafeERC20.safeTransfer` wrapper."

**Implementation reality:** the v9 schema has no `lvalue type` feature for CFG nodes. The graph edge `EXTERNAL_CALL` (type 11) fires for any cross-contract call, including SafeERC20 wrappers. The pattern YAML documents the discrimination, but the runtime check is "EXTERNAL_CALL edge present" (which produces FPs on SafeERC20 wrappers — but SafeERC20 contracts are clean OZ code, not in our 10 classes).

**Impact:** a SafeERC20 contract incorrectly labeled `CallToUnknown` would PASS the semantic check. The Stage 3 crosswalk should not label SafeERC20 contracts as `CallToUnknown` (the SolidiFI crosswalk already excludes them); if it does, the model learns the wrong pattern.

### IC-5 — `add_trivial` flag on probe_dataset.build_probe_dataset

The probe dataset always adds trivial positive + trivial negative for each class. Adding a flag `add_trivial=False` lets the user skip the trivials (useful for a "real contracts only" pass). Default is `add_trivial=True` per the plan.

### IC-6 — Tool agreement downgrade only with co-flag (gate.py)

The plan's D-4.5 says: "VERIFIED — semantic check > 90%, tool agreement > 70%, FP estimate < 15%". The implementation uses a *weaker* condition: `tool_agreement < 30% AND co-flag → downgrade to PROVISIONAL`.

**Why weaker:** Slither's per-class precision is too variable to use a 70% threshold as a hard gate (e.g. Reentrancy has 52% precision per Phase 5 audit). Demoting only when Slither DISAGREES strongly AND there's an independent co-occurrence flag is a more defensible signal.

---

## Operational Consequences

1. **The verification stage is required before export.** Any class with `FAIL` blocks Stage 7 export. The override requires `pipeline.verification.override_classes: [<class_names>]` in `config.yaml`.

2. **The probe dataset is a one-time build.** The trivial pos/neg are version-controlled; the real contracts are pulled from BCCC + DIVE at build time. Re-running the build is idempotent (deterministic with `seed=42`).

3. **The negative_checker is run on the full corpus.** A full 22K-contract SolidiFI+DIVE run takes minutes on cache hit, hours on first run. Skipping with `--skip-negative-checker` is allowed for fast smoke tests.

4. **The BCCC regression test is currently a meta-test.** When BCCC is re-introduced (v2.1), the test should be extended to re-run the new module on the BCCC corpus and compare per-class drop counts. Until then, the test verifies the historical outputs are internally consistent.

5. **The SmartBugs Curated recall test is the falsification check.** If recall drops below 90%, the semantic_checker is too strict and Run 11 is deferred to v2.1 (per the Stage 3 minimum-viable-corpus gate).

---

## References

- Plan: [`docs/proposal/Data_Module_Proposals/actionable_plans/05_stage_4_verification.md`](../proposal/Data_Module_Proposals/actionable_plans/05_stage_4_verification.md)
- Audit: [`docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md`](../proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md) §1 4-P1 through 4-P10
- Stage 4 audit: `Data/audit/07_verification_stage4_audit.md` (2026-06-11)
- Implementation:
  - `Data/sentinel_data/verification/slither_runner.py` (shared Slither runner with content-addressed cache)
  - `Data/sentinel_data/verification/tool_validator.py` (Task 4.4)
  - `Data/sentinel_data/verification/fp_estimator.py` (Task 4.5, stratified sampling)
  - `Data/sentinel_data/verification/negative_checker.py` (Task 4.5, 5%/10% thresholds)
  - `Data/sentinel_data/verification/probe_dataset.py` (Task 4.6, 40 per class)
  - `Data/sentinel_data/verification/probe_trivials.py` (hand-crafted Solidity)
  - `Data/sentinel_data/verification/semantic_checker.py` (graph-feature-based)
  - `Data/sentinel_data/verification/class_auditor.py` (per-class stats + co-occurrence matrix)
  - `Data/sentinel_data/verification/gate.py` (per-class verdicts)
  - `Data/sentinel_data/verification/report_generator.py` (markdown report)
  - `Data/sentinel_data/cli.py` (verify subcommand)
- Tests: `Data/tests/test_verification/` (166 passing, 16 skipped)
- BCCC Phase 5 outputs: `docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/`

---

**End of ADR-0005.**
