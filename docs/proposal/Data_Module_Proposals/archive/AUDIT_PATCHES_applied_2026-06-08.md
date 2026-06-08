# Memory & Code Audit — Patch List for the 8 Actionable Plans

**Date:** 2026-06-08
**Scope:** cross-check every plan in `actionable_plans/` against the memory files in `~/.claude/projects/-home-motafeq-projects-sentinel/memory/` + live code in `ml/src/`, and produce a concrete patch list of what to add, change, or remove from each plan so we don't redo past mistakes.

---

## 0. What the memory + code tell us — top findings

Verified against the live source on 2026-06-08. The 8 plans need to absorb the following facts:

| # | Fact | Source | Plan(s) affected |
|---|---|---|---|
| F1 | **Active schema is v9, not v8** — `graph_schema.py:161` has `FEATURE_SCHEMA_VERSION = "v9"`, `NODE_FEATURE_DIM=12` (line 175), `NUM_EDGE_TYPES=12` (line 218). The proposal §2 + Stage 0 plan say v8. | `ml/src/preprocessing/graph_schema.py` | Stage 0, 2, 7 |
| F2 | **A20 (label=0 hardcode) is FIXED** in `ast_extractor.py:290,342,395` — `label_map` is now built from a ground-truth CSV via `--label-csv`. Plans calling it "future bug" are wrong. | `ml/src/data_extraction/ast_extractor.py` | Stage 1, 2 |
| F3 | **A9 (`now` keyword) is FIXED** in `graph_extractor.py:587-605` — three-tier detection: SolidityVariableComposed + name-based fallback for `now` + library helpers. Plans should not say "fix the bug" — they should say "keep the fix working through the port." | `ml/src/preprocessing/graph_extractor.py` | Stage 2, 4 |
| F4 | **A15 (def_map by name) is FIXED** in `graph_extractor.py:1147-1179` — uses `scope_key` two-tier (function_id + var_name), not just var name. | `ml/src/preprocessing/graph_extractor.py` | Stage 2, 4 |
| F5 | **A34 (prefix sort) is FIXED** in `sentinel_model.py:356,367` — uses `raw_node_features[:, 10]`, not post-GAT dim. | `ml/src/models/sentinel_model.py` | Stage 2 |
| F6 | **Resume overwrite bug is FIXED** in `trainer.py:383, 1184, 1206, 1212` — `resume_model_only` default is now `False`; full resume is the default. `--no-resume-model-only` is the opt-in to model-only. Plans saying "fix the bug" are wrong. | `ml/src/training/trainer.py` | Stage 2, 8 |
| F7 | **CALL_ENTRY for external calls is PARTIALLY FIXED** — `graph_extractor.py:1001,1018` has `_add_icfg_edges` and EXTERNAL_CALL=11 is a self-loop on CFG nodes with HighLevelCall/LowLevelCall. But audit finding (per memory) says external calls still get no cross-function edges — self-loop ≠ cross-function. Stage 2 plan D-2.4 mentions this correctly. | `ml/src/preprocessing/graph_extractor.py` | Stage 2 |
| F8 | **Predictor tier bug is OPEN** in `predictor.py:150, 168` — `TIER_CONFIRMED_THRESHOLD = 0.55` still hardcoded; `self.thresholds` is loaded but `_format_result` doesn't consult it. The plans don't mention inference at all — this needs Stage 7 / 8 explicit awareness. | `ml/src/inference/predictor.py` | Stage 7, 8 |
| F9 | **97 BCCC contracts have `now` keyword audit still matters** — but the v1.4 verification report already dropped them. The BCCC `now` corpus is gone; v2 sources are modern. The pattern matcher in Stage 4 should be future-proofed for 0.4.x anyway. | Phase 5 verification + MEMORY | Stage 4 |
| F10 | **Per-class training thresholds from `tune_threshold.py` must be respected by `predictor.py`** — the current `_format_result` ignores them; Run 11 will hit the same bug. Stage 7's seam swap doesn't fix predictor.py; Stage 8 should add this fix to the launch checklist. | `ml/src/inference/predictor.py:150` + `ml/calibration/temperatures_run9.json` | Stage 7, 8 |
| F11 | **`graph_extractor.py:1147` `_add_def_use_edges` uses `scope_key` two-tier (function id + var name) — but the test in Stage 2 must assert the cross-function dedup, not just intra-function** | `ml/src/preprocessing/graph_extractor.py:1147-1179` | Stage 2 |
| F12 | **Run 7 had 30 audit fixes before it ran** — the v2 build's Stage 2 port must preserve all of them, not just the move. The byte-identical regression test (Stage 2 D-2.1) is the right gate. | `project_action_plan.md` "Phase 0" table | Stage 2 |
| F13 | **The `learning_preferences.md` "depth check" rule applies to all plans** — I should not over-explain the concepts (e.g. "what is a connector", "what is DVC"). The plans should be terse, design + intent, not tutorials. | CLAUDE.md global rules + `ml_learning_with_claude/learning_preferences.md` | All plans |
| F14 | **All 98 solc versions (0.4.0–0.8.35) are installed in `~/.solc-select/artifacts/` + 3 venv-local copies** — Stage 1's compiler step should use `solc-select use <version>` not download-on-demand. The Dockerfile's 6 solc versions are an under-coverage. | `reference_solc.md` | Stage 0, 1, 7 |
| F15 | **Per-class thresholds in `tune_threshold.py` and `calibrate_temperature.py` accept the model params** (gnn_phase2_edge_types, fusion_max_nodes, drop_complexity_feature, appnp_alpha) — Run 11's tuned thresholds must be re-derived because the new v2 corpus changes per-class base rates. Stage 8 must include `tune_threshold.py` as a post-training step, not assume Run 9's thresholds carry over. | `project_run7_training.md` "Pre-Run-8" table | Stage 8 |
| F16 | **Run 9 launched at `--gnn-prefix-warmup-epochs 5` (not 15)** — Run 7's `gnn-prefix-warmup-epochs=15` was the original default; Run 8/9 reduced it. Run 11 should use 5 to match active config. | `project_run7_training.md` launch command | Stage 8 |
| F17 | **All 25 interpretability scripts in `ml/scripts/interpretability/` exist and have valid outputs** — they were the source of the "complexity proxy" finding. Stage 6 (analysis) is the data-side complement; the model-side interpretability suite is separate. Plans should not duplicate model-side interpretability in `sentinel_data`. | `project_interpretability.md` + `ml/scripts/interpretability/` | Stage 6 |
| F18 | **The RAG multi-class query anchor fix is open in agents** — the agents layer is separate from the data layer. The plans should not touch agents code; the only cross-package impact is that the new `SentinelDataset` (Stage 7) changes the input format to the agents, but the agents already call the inference server, not the dataset directly. No change needed. | `project_agents.md` "Known Improvement" | All plans |
| F19 | **`wsl.exe` PowerShell host errors on inline commands** — Stage 0/1 docs should use `wsl -- bash -c '...'` for complex commands. | CLAUDE.md + MEMORY operational facts | All plans |
| F20 | **`mlruns.db` is the MLflow backend (`sqlite:///mlruns.db`), NOT `file:///`** — file backend has corrupt experiments 1,2,3. Stage 8's Run 11 launch must use sqlite. | `project_v4_analysis.md` "Known Code Issues" | Stage 8 |
| F21 | **No `--run-name` collision check exists** — Run 9's resume silently overwrote ep14 best (F1=0.2586) because `train.py:2000` saves unconditionally on any F1 > best_f1, and the model-only path resets best_f1. Default is now `--resume-model-only=False` (full resume), but the plan should add an explicit MLflow run-name collision check. | `project_run9_resume.md` | Stage 8 |
| F22 | **The Run 8 watcher exists at `ml/scripts/run8_watcher.sh` and emits Windows toast notifications via PowerShell** — Stage 8's "set up Run 11 watcher" should copy this rather than reinvent. | `project_run8_audit_findings.md` "Background Monitor" | Stage 8 |
| F23 | **The pre-Run-8 audit doc `docs/pre-run8-fixes/VALIDATED_AUDITION.md` lists 36 confirmed code issues (A1–A38) across 9 source files** — Stage 2's port must include the regression test for all 9 files, not just the 3 named in the plan. The regression test must verify the fix is preserved, not just the logic. | `project_action_plan.md` "validated_audition.md" reference | Stage 2 |
| F24 | **The graph_extractor schema constants have a `BCCC_SOLC_PRAGMA_SPACED` bug pattern (e.g. `^ 0.4 .9`)** — Stage 1's preprocessor must parse the pragma with whitespace tolerance, not just regex. The actual bug was already found in Phase 5 retry but the parser may not be robust. | MEMORY §"Session 3 additions" | Stage 1 |
| F25 | **`graph_extractor.py:1081` LibraryCall <: HighLevelCall class hierarchy** — `_compute_external_call_count` relies on isinstance; a contract that uses `SafeMath.add()` (library call) is counted as HighLevelCall, not as internal. Stage 4's CallToUnknown pattern must handle this. | `ml/src/preprocessing/graph_extractor.py:1081` | Stage 4 |
| F26 | **The `p5_compile_retry.py` recovered 2,488 contracts that failed initial compilation** — Stage 1's preprocessor must do a two-pass compile (initial + retry with relaxed pragma parsing) to handle BCCC-style oddities even if BCCC is no longer the gold source. | MEMORY §"Session 3" + Phase 5 scripts | Stage 1 |
| F27 | **BCCC's 99% DoS↔Reentrancy co-occurrence is the core noise problem** — Stage 3's crosswalk for the 4 Tier-1 sources (Bastet, ScaBench, Web3Bugs, DeFiHackLabs) must explicitly avoid the co-occurrence pattern by requiring per-source evidence, not class co-occurrence. | `project_data_pipeline_audit.md` "Co-occurrence" table | Stage 3 |
| F28 | **The probe dataset per-class cardinality from Phase 5 is 40 per class (5.5 → 50 in Stage 4) — but the original BCCC `review_batches/` is 40/class with raw contracts. v2 sources don't have human-reviewed probes yet. Stage 4's probe_dataset task is real work, not just expansion.** | MEMORY §"Stage 5.5" + Phase 5 outputs | Stage 4 |
| F29 | **The `_compute_return_ignored` bug is FIXED at `ml/src/preprocessing/graph_extractor.py` — checks `id()` in subsequent IR ops, not `op.lvalue is None`. Stage 4's semantic checker should NOT flag this as a known issue.** | MEMORY + `_compute_return_ignored` location | Stage 4 |
| F30 | **The audit doc `docs/pre-run-fixes/phase2_root_cause_analysis.md` lists 7 root causes for Phase 2 failure** — these are model-side issues, not data-side. Stage 4 verification doesn't need to address them. But Stage 6's `feature_dist.complexity_proxy_risk` IS the same diagnosis from the data side. | `project_action_plan.md` "Phase 2 root cause" | Stage 4, 6 |

---

## 1. Per-plan patch list

Each plan gets a section with: **additions**, **changes**, **deletions**. The numbering follows the original plan's task numbers where possible.

---

### INDEX.md — patches

| # | Type | Patch |
|---|---|---|
| I-1 | Change | The 6 critical tests table must include a 7th row: **"36-issue regression test (pre-Run-8 audit)"** — Stage 2 must verify all A1–A38 fixes survive the port. |
| I-2 | Add | New "Q5 urgent" banner: schema freeze is **v9**, not v8. The proposal and Stage 0 plan must use v9 constants. |
| I-3 | Change | Open questions table — Q5 row updated to v9 with corrected constants (NODE_FEATURE_DIM=12, NUM_EDGE_TYPES=12). |
| I-4 | Add | New "Operational facts" subsection listing the 5 critical env / config / runtime facts: TRANSFORMERS_OFFLINE=1, TRITON_CACHE_DIR, sqlite:// mlflow, wsl -- bash -c pattern, solc-select locations. |
| I-5 | Add | New "Code-bug state at build start" subsection listing the 8 already-fixed bugs (A9, A15, A20, A34, A38, resume, def_use, return_ignored) and 3 still-open (CALL_ENTRY cross-function, predictor tier threshold, BCCC-vs-version skew). The plans must not "re-fix" the already-fixed bugs. |

---

### Stage 0 — Skeleton + Data/ Restructure — patches

| # | Type | Patch |
|---|---|---|
| 0-P1 | **Delete** | 0.4 task: the `representation/graph_extractor.py` stub with `NotImplementedError` — it conflicts with the active schema constants. The stub must replicate the v9 schema (`FEATURE_SCHEMA_VERSION="v9"`, `NODE_FEATURE_DIM=12`, `NUM_EDGE_TYPES=12`). |
| 0-P2 | **Change** | 0.4 task: the `graph_schema.py` stub must include **all** the active constants: `NODE_FEATURE_DIM=12`, `NUM_NODE_TYPES=14` (per MEMORY v9), `NUM_EDGE_TYPES=12`, `FEATURE_SCHEMA_VERSION="v9"`, `_MAX_TYPE_ID=13.0`. **Delete the v8 assumption.** |
| 0-P3 | **Change** | 0.6 config.yaml — add a top-level `pipeline.solc:` section that references the 6 most common solc versions (0.4.26, 0.5.17, 0.6.12, 0.7.6, 0.8.20, 0.8.24) and notes that the 98 installed versions live in `~/.solc-select/artifacts/`. The Dockerfile in 0.9 should pre-install these 6 but Stage 1's compiler step should fall back to on-demand `solc-select install` for the rest. |
| 0-P4 | **Add** | 0.6 config.yaml — add `pipeline.mlflow.uri: sqlite:///mlruns.db` per F20. The MLflow backend is locked at sqlite. |
| 0-P5 | **Add** | 0.10 README — add "WSL2 caveats" section: use `wsl -- bash -c '...'` for complex commands; the global CLAUDE.md says inline wsl.exe errors. |
| 0-P6 | **Add** | 0.10 README — add "MLflow backend" note: the v2 build's MLflow uses `sqlite:///mlruns.db` only. The `file:///` backend is corrupt. |
| 0-P7 | **Change** | 0.9 Dockerfile — add a comment that the 6 pre-installed solc versions are a baseline; the rest are installed on-demand via `solc-select install` (per F14). The 98-version full install is ~1.2 GB and the Dockerfile base image would bloat unnecessarily. |
| 0-P8 | **Add** | New 0.15 task: add a "code-state audit" ADR that lists the 8 already-fixed bugs and 3 still-open ones (per I-5). This is the first design artifact that downstream stages reference. |

---

### Stage 1 — Ingestion + Preprocessing — patches

| # | Type | Patch |
|---|---|---|
| 1-P1 | **Change** | 1.4 task: the preprocessor's compile step must do a **two-pass** compile (initial + retry with relaxed pragma parsing) to handle BCCC-style oddities like `^ 0.4 .9` (whitespace in pragma) and `0.4.25` (exact version, not range). The Phase 5 retry script `p5_compile_retry.py` is the reference implementation. |
| 1-P2 | **Change** | 1.4 task: the `pick_solc_version()` function (mentioned in MEMORY §"Session 3 additions") must (a) strip whitespace before regex, (b) try the requested version, (c) fall back to a known-good version if not available. The bug was found in BCCC but the same parser will see odd pragmas in ScaBench / Web3Bugs audit reports. |
| 1-P3 | **Change** | 1.4 task: the "drop-not-fix for compile failures" decision (D-1.5) is correct but the drop log should include the *attempted* solc versions and the *reason for failure* — not just "compile_failed." This is what Phase 5's retry script did. |
| 1-P4 | **Add** | 1.4 task: the deduplicator's `ast_similarity_threshold` default should be **0.85** (not 0.92 as in the plan) — BCCC's near-dup was at 0.85–0.95 range and 0.85 caught the copy-paste-with-edits cases that bit Run 9. 0.92 may be too strict for v2 sources. |
| 1-P5 | **Add** | 1.4 task: the version_bucketer must distinguish **Solidity 0.8+ with `unchecked{}`** from **0.8+ without**. This is the dim-9 / `in_unchecked_block` signal that was rightly dropped from v7 (87.9% of BCCC is pre-0.8). v2 sources will have more 0.8.x contracts; the bucketer must record whether the file uses `unchecked{}` so Stage 4 can flag it as a 0.8+ era marker. |
| 1-P6 | **Change** | 1.7 task: `freshness.py` should also check the **Slither version**, not just the source pin. Slither API changes between versions can break the graph extractor. The `pyproject.toml` pins `slither-analyzer >= 0.10.0`; the freshness report should compare the pinned version to the latest release. |
| 1-P7 | **Add** | 1.8 task: the test suite must include a **regression test for the A20 fix** (the label=0 hardcode). The test asserts that a fixture contract with known labels in the label CSV produces those labels in the graph `.y` tensor, not a hardcoded 0. |

---

### Stage 2 — Representation (port from ml/) — patches

| # | Type | Patch |
|---|---|---|
| 2-P1 | **Change** | D-2.2 — the schema version is **v9, not v8**. The stub from Stage 0 must be updated to v9 constants before the port. The regression test (2.6) compares v9 output to v9 output. |
| 2-P2 | **Change** | 2.6 task: the byte-identical regression test must run against **all 9 source files** in `ml/src/`, not just the 4 named in the plan. The 36-issue pre-Run-8 audit (A1–A38) lists files: `ast_extractor.py`, `graph_extractor.py`, `graph_schema.py`, `tokenizer.py`, `sentinel_model.py`, `gnn_encoder.py`, `trainer.py`, `tune_threshold.py`, `losses.py`. The regression test must verify each fix (A1, A9, A15, A20, A34, A38, etc.) survives the port. |
| 2-P3 | **Add** | 2.6 task: the regression test must specifically assert the **A9 fix** (`now` keyword detection in `_compute_uses_block_globals`). The test fixture includes a 0.4.x contract using `now`; the test asserts that `feat[2]` is 1.0 for the function. |
| 2-P4 | **Add** | 2.6 task: the regression test must assert the **A15 fix** (two-tier scope_key in def_map). Test fixture: two functions with the same variable name in different scopes; test asserts no spurious cross-function DEF_USE edges. |
| 2-P5 | **Add** | 2.6 task: the regression test must assert the **A34 fix** (`select_prefix_nodes` uses raw `external_call_features[:, 10]`, not post-GAT dim). |
| 2-P6 | **Add** | 2.6 task: the regression test must assert the **A20 fix** (label is from CSV, not hardcoded 0). Test fixture: a CSV with known labels; graph `.y` must match. |
| 2-P7 | **Add** | 2.6 task: the regression test must assert the **CALL_ENTRY for external calls** (EXTERNAL_CALL=11 self-loop). The test fixture has a contract with a `HighLevelCall`; test asserts an EXTERNAL_CALL edge exists on the CFG node. **Note: the audit says this is "PARTIALLY FIXED" — self-loop exists, but no cross-function edge. Stage 2's port preserves the partial fix; the full fix is post-Run-11 work.** |
| 2-P8 | **Add** | 2.7 task: the 4 new builders (CFG, PDG, call graph, opcode) must be **gated by schema version**. The CFG builder uses Slither's IR; if Slither's IR API changes, the builder fails. The versioner must invalidate the cache when the Slither version changes. |
| 2-P9 | **Delete** | 2.7 task — the `pdg_builder.py` description says "uses a lightweight data-flow analysis." This is a v3+ feature, not v2. Stage 2 should ship **only the CFG builder** (which is straightforward — it's already in `_add_icfg_edges`). PDG/call-graph/opcode should be marked **DEFERRED to v3.1** because they're not needed for Run 11 and the time would be better spent on Stage 4 verification. |
| 2-P10 | **Change** | 2.11 ADR-0003 — must reference the 36-issue pre-Run-8 audit, not just the v9-schema additions. The ADR is the design record of "what the port preserved." |
| 2-P11 | **Add** | 2.6 task: include a **performance regression test** — extracting 100 contracts should take within ±10% of the time of the old `ml/src/preprocessing/` path. If slower, profile and fix. |
| 2-P12 | **Change** | 2.7 task — reword the design decision D-2.4 to clarify that "Run 11 trains on AST+CFG as v1 did" means the v1 in-graph CFG (CFG_NODE_* node types) plus the existing CONTROL_FLOW edges, NOT a new standalone CFG builder. The standalone CFG builder is for v3+. |

---

### Stage 3 — Labeling — patches

| # | Type | Patch |
|---|---|---|
| 3-P1 | **Change** | D-3.1 — the canonical 10-class taxonomy must use the **v1 checkpoint's class order**, which is the order in `ml/src/preprocessing/graph_schema.py` (CallToUnknown=0, DenialOfService=1, ExternalBug=2, GasException=3, IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7, TransactionOrderDependence=8, UnusedReturn=9). Run 7/8/9 used this order. **Any change breaks every existing checkpoint.** |
| 3-P2 | **Add** | 3.2 task: the ScaBench crosswalk must explicitly handle **free-text descriptions** via a hybrid approach: (a) keyword matching for common patterns (reentrancy, overflow, timestamp, etc.), (b) LLM-assist for the rest, (c) human review for every LLM-suggested mapping. The MEMORY says "0.5–2 days per crosswalk" — ScaBench is on the harder end. |
| 3-P3 | **Add** | 3.7 task: the **Web3Bugs parser** must handle the O/L/S severity labels from `results/bugs.csv` + the report text from `reports/`. The O/L/S → class mapping is a 3-tier (Optimistic, Low, Speculative) severity; only O and L should map to positive labels. S is too uncertain. |
| 3-P4 | **Add** | 3.7 task: the **DeFiHackLabs parser** must extract the exploit type from the folder name + the `_exp.sol` file comments. The MEMORY says these are "self-documenting" but the actual parsing is non-trivial (e.g. `2024-01-flashloan-attack/` vs `2024-01-reentrancy/`). A regex-based initial parse + LLM-assist fallback is the right approach. |
| 3-P5 | **Add** | D-3.3 — the merger precedence must explicitly handle **99% DoS↔Reentrancy co-occurrence** (the BCCC failure pattern). When a contract is labeled both DoS and Reentrancy by a single source, that's a near-certain sign of noise — the merger should de-duplicate co-occurring labels from the same source unless there's independent evidence. |
| 3-P6 | **Add** | 3.8 task: the `confidence.py` tier assignment must consider **per-class tier overrides** in the crosswalk. E.g. if a crosswalk maps a Web3Bugs O-severity bug to a class, the tier is T1 (expert-audited) regardless of the source. The crosswalk is the override. |
| 3-P7 | **Add** | 3.7 task: the **SmartBugs Curated parser** must map the 9 DASP categories to the 10 Sentinel classes. DASP has 10 categories, not 9 — and one (front_running) maps to TOD. The crosswalk must be explicit about which DASP category doesn't have a Sentinel class equivalent. |
| 3-P8 | **Add** | 3.7 task: the **OpenZeppelin Contracts parser** marks every contract as `T1_clean` for `NonVulnerable` and skips positive class assignment. This is the clean negative source; it should never produce a positive label even if Slither flags something (false positive on clean code is a known pattern per the audit). |
| 3-P9 | **Add** | 3.7 task: the **slither_audited parser** is the highest-risk parser because it derives labels from tool output, not human audit. The crosswalk must be **explicitly conservative** — only map to a class if Slither's confidence is >0.8 AND the detector is in the canonical CLASS_TO_DETECTORS list (per `project_agents.md`). The tier is T3 (tool-generated), not T1. |
| 3-P10 | **Change** | 3.10 task: the test suite must include a **regression test for BCCC 99% DoS↔Reentrancy co-occurrence**. A fixture contract with the BCCC-style co-occurrence pattern must be de-duplicated by the merger, not preserved. |

---

### Stage 4 — Verification — patches

| # | Type | Patch |
|---|---|---|
| 4-P1 | **Change** | D-4.2 — the CallToUnknown AST pattern is more nuanced than the plan says. The pattern must check: (a) presence of `.call{}` / `.delegatecall{}` / `.send()` / `.transfer()` (raw calls); (b) the lvalue is `address` type, not `bool` (e.g. `bool success = addr.call(...)` is a low-level call but not CallToUnknown; `addr.call{value: x}("")` is); (c) the call is not in an OZ `SafeERC20.safeTransfer` wrapper (which is a library call, not unknown). The plan's "must contain low-level call" is necessary but not sufficient. |
| 4-P2 | **Add** | 4.3 task: the Reentrancy pattern must include the **CEI (Checks-Effects-Interactions) violation check** — the BCCC Reentrancy detection was 89% FP, partly because the pattern matched any external call + state write, even if the state write was BEFORE the call (which is not a reentrancy). The pattern must enforce the ordering: external call BEFORE state write. |
| 4-P3 | **Change** | 4.5 task: the `class_auditor.py` per-class statistics must include the **co-occurrence matrix** as a primary output. The BCCC 99% DoS↔Reentrancy co-occurrence would have been caught by an automated co-occurrence check, not just a per-class count. |
| 4-P4 | **Add** | 4.5 task: the `negative_checker.py` must use the actual Slither detector list from `project_agents.md` (CLASS_TO_DETECTORS), not a generic Slither run. The point is to catch OZ-flagged patterns that should be false positives on clean code. |
| 4-P5 | **Change** | 4.6 task: the `probe_dataset.py` cardinality should be **40 per class** (matching the BCCC `review_batches/` seed), not 50. The plan's "expand to 50" is incremental work; the v2 baseline can launch with 40. 50 is a v2.1 enhancement. |
| 4-P6 | **Add** | 4.6 task: the `probe_dataset.py` should include a "trivial positive" + "trivial negative" per class. The trivial negative is a clean OZ contract of similar size; the trivial positive is the simplest possible example (e.g. `function withdraw() public { msg.sender.call{value: balances[msg.sender]}(""); balances[msg.sender] = 0; }` for Reentrancy). The model interpretability suite uses these to verify the model has learned the pattern, not the surface features. |
| 4-P7 | **Add** | 4.7 task: the BCCC regression test must check the **p5_s1 to p5_s6 stage outputs** individually, not just the final report. Each stage script has a specific output (s1_evidence_table, s2_automated_verdict, s3_refined_verdict, s4_final_verdict, s6_verification_report). The new module's verification stage must reproduce each stage's output to within tolerance. |
| 4-P8 | **Add** | 4.3 task: the semantic_checker must include a **library-call detection** check. Per F25, `SafeMath.add()` is a library call that Slither classifies as HighLevelCall but is NOT a cross-contract call. The pattern matcher must distinguish library calls (used internally) from cross-contract calls (CallToUnknown). |
| 4-P9 | **Add** | 4.5 task: the `fp_estimator.py` must use **stratified sampling by source AND tier**, not just by class. A class with 90% T3 labels and 10% T0 labels has a very different FP rate by tier. The estimate must report per-tier per-class. |
| 4-P10 | **Change** | D-4.6 — the negative_checker threshold (10%) is too lax. The BCCC failure was 41% of NonVulnerable had Slither hits; the threshold should be **5%** to be useful as a gate. Anything above 5% is suspicious; above 10% is FAIL. |

---

### Stage 5 — Splitting + Registry — patches

| # | Type | Patch |
|---|---|---|
| 5-P1 | **Change** | D-5.1 — the default for `tool-derived sources` (Slither-Audited, SmartBugs Wild, Messi-Q) should be **stratified with project-level fallback** (not just stratified). These sources are large and a pure stratified split may put 90% of one tool's contracts in one split. The fallback is to use the source's own project grouping if available. |
| 5-P2 | **Add** | 5.4 task: the SQLite catalog must include a **schema migrations table**. The catalog schema will evolve (new tables for the per-class tier overrides, per-source lineage extensions, etc.); migrations must be tracked. The first migration creates the 4 base tables; subsequent migrations add columns. |
| 5-P3 | **Add** | 5.5 task: the catalog must include a **dataset version retirement table**. Old dataset versions can be marked as "superseded" but not deleted (the audit trail is permanent). The v1.4 BCCC labels, the v8 BCCC graphs, the v9 graphs, the v10 graphs — all are preserved in the catalog with their `superseded_by` chain. |
| 5-P4 | **Add** | 5.5 task: the catalog's `verify_artifact_hash()` must be the **same algorithm** as `ml/src/inference/cache.py:InferenceCache` uses for content addressing. Both are SHA-256 of the file bytes; the key includes the schema version. The two functions should call the same shared `sentinel_data.registry.compute_hash()` to avoid drift. |
| 5-P5 | **Change** | 5.4 task: the `data/splits/v10_deduped/` (current location) and the new `Data/data/splits/v1/` (Stage 5 output) must **coexist** for one release. The old splits are used by Run 9 / any active training; the new splits are for Run 11. The seam swap in Stage 7 retires the old. |
| 5-P6 | **Add** | 5.6 task: the lineage_tracker must record **the training run that consumed each artifact**. A dataset version + the Run N that trained on it = a link. This is the input to "what data did Run 11 train on, and how did it differ from Run 10?" |
| 5-P7 | **Add** | 5.7 task: the `dataset_diff.py` must produce a **per-class metric projection** — for each class, show the new vs old count, the new vs old label distribution, the new vs old confidence tier breakdown. The model team uses this to predict "Run 11's per-class F1 will likely be X% better than Run 9's because the v2 corpus has 30% more Reentrancy positives." |

---

### Stage 6 — Analysis — patches

| # | Type | Patch |
|---|---|---|
| 6-P1 | **Change** | D-6.2 — the complexity_proxy_risk threshold (1.5σ) is the right default but the report should also include a **per-class rank correlation** between the feature (e.g. node count) and the per-class precision. If a class's precision correlates strongly with the feature, the model is using that feature as a proxy. This is the data-side complement to the L4 finding in MEMORY (complexity dominates all 10 classes at 34-36%). |
| 6-P2 | **Add** | 6.2 task: the `feature_dist.py` should also compute the **label-conditional feature distribution** — for each class, the per-feature mean/std/median broken down by positive (label=1) and negative (label=0) contracts. The "complexity proxy" finding would have been visible as a strong per-class difference between positive and negative feature distributions. |
| 6-P3 | **Add** | 6.5 task: the `drift_monitor.py` must include **KS test for label distribution drift**, not just feature drift. If the new version has a different positive/negative ratio per class, that's a label drift even if the feature distributions are stable. |
| 6-P4 | **Add** | 6.3 task: the `cooccurrence.py` should output a **directed co-occurrence matrix** (X→Y means "if class X is positive, class Y is also positive with probability p") and a **conditional probability matrix** (P(Y=1 | X=1)). The BCCC 99% DoS→Reentrancy co-occurrence is visible as a very high entry in both matrices. The conditional matrix is what the multi-label loss design consumes. |
| 6-P5 | **Add** | 6.4 task: the `overlap_detector.py` should distinguish **exact overlap** (same contract_id across sources) from **near overlap** (AST-similar but different contract_id). The exact overlap drives the merger (Stage 3); the near overlap drives the dedup enforcer (Stage 5). |
| 6-P6 | **Change** | D-6.1 — analysis is read-only, but the **outputs should be DVC-tracked**. Each `data/analysis/<run_id>/` output is an `out:` in DVC; rerunning analysis produces a new version. The Stage 7 export should depend on the analysis output (the model team reviews the complexity_proxy_risk report before launching). |
| 6-P7 | **Add** | 6.7 task: the `sentinel-data analyze` CLI must support a `--corpus <version>` flag to analyze a specific dataset version (not just the current build). The historical analysis of the BCCC v1.4 corpus is a useful baseline. |

---

### Stage 7 — Export + Seam Swap — patches

| # | Type | Patch |
|---|---|---|
| 7-P1 | **Change** | 7.4 task: the `SentinelDatasetExport` class must include the **graph schema version** in the manifest. The ML module's `SentinelDataset.__init__` checks that the schema version matches the expected version for the current model. A mismatch raises an error (per the proposal's "schema verification gate"). |
| 7-P2 | **Change** | 7.6 task: the new `sentinel-ml/src/datasets/sentinel_dataset.py` must accept a `--graph-schema-version` flag (default: the model's trained version). If the export's schema version doesn't match, the loader raises `ValueError`. This is the gate that prevents "trained on v9, evaluated on v10" silent failures. |
| 7-P3 | **Add** | 7.6 task: the new `SentinelDataset` must validate the **per-class thresholds** from the export (if present). Run 11 will have its own tuned thresholds; loading the v9 export's thresholds for a v9-trained model is correct, but loading v8 thresholds for a v9 model is a silent miscalibration. The loader checks the run name in the export manifest and warns on mismatch. |
| 7-P4 | **Add** | 7.6 task: the new `SentinelDataset` must support the **`confidence_tier` field** in the returned tuple. The current `dual_path_dataset.py` returns `(graph, tokens, y)`; the new one returns `(graph, tokens, y, contract_id, confidence_tier)`. The collate function in `collate.py` stacks these. |
| 7-P5 | **Change** | 7.7 task: the dual-path test must verify that **the 8 fixed code bugs (F2–F6, F8, F11)** survive the seam swap. The test loads a fixture contract and asserts: (a) label is from CSV not hardcoded 0; (b) `now` keyword is detected; (c) def_map is scope-keyed; (d) prefix sort uses raw features; (e) resume default is full; (f) call_entry for external is present; (g) LibraryCall not miscounted as HighLevelCall; (h) tier threshold uses per-class. |
| 7-P6 | **Add** | 7.7 task: the dual-path test must also verify the **EMITS edge type bug** (F23 — only 12 edges across 41K contracts, "extractor bug" tracked as Interp-6). The test fixture has a contract with an event emit; the new path must produce an EMITS edge. If it doesn't, the bug must be fixed as part of the port (not deferred to v2.1). |
| 7-P7 | **Add** | 7.8 task: the `ml/src/inference/predictor.py:150 TIER_CONFIRMED_THRESHOLD` bug must be **fixed as part of the seam swap** (F8, F10). The fix: `_format_result` consults `self.thresholds` per class; "confirmed" tier is per-class-tuned threshold, not the hardcoded 0.55. This is a small, surgical fix that prevents Run 11 from hitting the same display bug. |
| 7-P8 | **Change** | 7.8 task: the `ml/pyproject.toml` update must **keep** `slither-analyzer` as a runtime dep (not remove it) — the inference path uses Slither indirectly via the `ContractPreprocessor`. Only `solc-select`, `py-solc-ast`, `solc` can be removed (the inference path uses the system solc, not the version-pinned batch solc). |
| 7-P9 | **Add** | 7.8 task: the deleted `ml/scripts/{reextract_graphs,retokenize_windowed,build_multilabel_index,create_splits,create_cache,validate_graph_dataset,archive_v8_data}.py` must be **archived to `ml/scripts/_legacy_data_pipeline/`** with a deprecation comment. They are the historical record of the v1 pipeline; deleting them without archiving loses the audit trail. |
| 7-P10 | **Add** | 7.9 task: the Docker build must use **`python:3.12.1-bookworm`** (not `slim`) because `slither-analyzer` requires `build-essential` + `libpq-dev` for the `psycopg2-binary` transitive dep. The image size difference is ~200 MB and the install is more reliable. |
| 7-P11 | **Add** | 7.10 task: the 6 v2-readiness gates must include a **7th gate**: "no open code-bug in the 36-issue pre-Run-8 audit is regressed by the port." This is the regression-test-on-36-bugs gate from Stage 2. |

---

### Stage 8 — Run 11 Launch — patches

| # | Type | Patch |
|---|---|---|
| 8-P1 | **Change** | D-8.2 — the Run 11 launch command must include **`--gnn-prefix-warmup-epochs 5`** (not 15, which was the Run 7 default). Run 8/9 reduced this to 5 and the change was active in those runs. Using 15 reverts a working change. |
| 8-P2 | **Change** | D-8.2 — the Run 11 launch command must use **`sqlite:///mlruns.db`** as the MLflow backend (F20). The `file:///` backend has corrupt experiments 1,2,3. |
| 8-P3 | **Add** | 8.4 task: the launch command must include **`--run-name v2-baseline-$(date +%Y%m%d-%H%M%S)`** with a timestamp suffix to prevent the silent-overwrite issue (F21). If the same `--run-name` is used twice, MLflow appends; the `.pt` file would silently overwrite. The timestamp makes collisions impossible. |
| 8-P4 | **Add** | 8.4 task: the launch command must include **`--early-stop-patience 30`** (matching Run 7's patience) and **`--threshold-tune-interval 10`** (matching Run 7's interval). The Run 8/9 history is the operational reference, not Run 7's defaults. |
| 8-P5 | **Add** | 8.4 task: the launch command must include **`--jk-entropy-reg-lambda 0.005`** (the Run 7/8/9 value). The Run 9 typo incident (0.0075 vs 0.005) is the cautionary tale; the launch command must be programmatically extracted from the original config, not typed by hand (per `project_run9_resume.md` "Lesson"). |
| 8-P6 | **Add** | 8.5 task: the launch must include a **post-launch immediate sanity check**: at ep1, the per-class F1 distribution should be non-degenerate (no class with F1=0.0 across all 10 classes, no class with F1=1.0). The Run 8 audit found that the predictor's tier-threshold bug hid 14/19 hits; Run 11 must check the per-class F1 directly, not the tier display. |
| 8-P7 | **Add** | 8.5 task: the launch must run **`tune_threshold.py` after the first ep10 milestone** (not at the end). The per-class thresholds derived at ep10 are likely close to the final; running at ep10 + ep20 + ep30 + ep40 gives a 4-point trajectory that catches threshold drift. The current `threshold-tune-interval=10` already does this; the task is to verify it's active. |
| 8-P8 | **Add** | 8.6 task: the first-epoch results report must include **per-class precision, recall, F1** (not just F1-macro). The Run 8 audit found "Test-set precision degenerate: DoS predicts positive for 76.8% of all test rows" — that was a precision problem, not F1. Reporting precision/recall separately surfaces this. |
| 8-P9 | **Change** | 8.7 task: the Run 11 watcher should be a **copy of `ml/scripts/run8_watcher.sh`** with the run-name substituted. The watcher is the proven infrastructure; the toast notification pattern is already validated. Re-purposing it is faster than rewriting. |
| 8-P10 | **Add** | 8.7 task: the watcher must **NOT trigger on F1=0.0 saves** (the Run 9 silent-overwrite pattern). The watcher's trigger logic should require F1 > 0.1 before emitting "new best" — a tiny threshold that filters out the "best_f1 reset to 0.0" path. This is the post-F6 defensive measure. |
| 8-P11 | **Add** | 8.6 task: the first-epoch report must include a **comparison to the BCCC Run 9 ep1 F1** (F1=0.2395 from the typo'd lambda; F1=0.2586 from the original ep14 best). The "before" baseline is Run 9 ep1; the "after" target is Run 11 ep1 should be > 0.2395. If lower, debug before continuing. |
| 8-P12 | **Add** | 8.7 task: the watcher should also monitor for **per-class F1 collapse** (any class going from >0.1 to 0.0 in a single epoch). The Run 8 audit's "DoS predicts positive for 76.8% of all test rows" is the kind of sudden collapse the watcher should catch. |

---

## 2. Cross-cutting patches (apply to multiple plans)

### C-1: Use `wsl -- bash -c '...'` for all WSL commands

All plans that show WSL commands must use the wrapper. Per CLAUDE.md and MEMORY operational facts, `wsl.exe` PowerShell host errors on inline commands.

**Affected plans:** all (in verification, exit criteria, README examples).

### C-2: Document the active schema constants in a single shared table

The 8 plans reference the schema (v8 or v9) inconsistently. Create a single shared table at the top of `actionable_plans/00_INDEX.md` (or a new `actionable_plans/_schema_reference.md`) with:
- `FEATURE_SCHEMA_VERSION = "v9"` (NOT v8)
- `NODE_FEATURE_DIM = 12` (NOT 11)
- `NUM_EDGE_TYPES = 12` (NOT 11)
- `NUM_NODE_TYPES = 14` (NOT 13)
- `_MAX_TYPE_ID = 13.0` (NOT 12.0)
- `gnn_num_layers = 8` (unchanged)
- `NUM_CLASSES = 10` (unchanged, class order locked)

**Affected plans:** Stage 0, 2, 7, 8.

### C-3: Add the 36-issue pre-Run-8 audit as a reference doc

The plans mention "the audit" but don't link to it. Create a pointer at the top of `actionable_plans/00_INDEX.md`:
```
Reference audit: docs/pre-run-fixes/validated_audition.md (36 issues A1–A38, 9 source files)
```

**Affected plans:** Stage 2 (regression test scope), Stage 4 (which bugs are still relevant).

### C-4: Add a "what NOT to fix" callout to each plan

Each plan should have a "What NOT to fix" section listing the bugs that are already fixed (F2–F6, F11) and should not be re-fixed during the plan. The plan's job is to preserve, not rewrite.

**Affected plans:** Stage 0, 1, 2, 4, 7, 8.

### C-5: Add the MLflow backend constant to all plans that touch MLflow

`sqlite:///mlruns.db` is the only correct backend. Any plan that runs an experiment must use this.

**Affected plans:** Stage 8 (Run 11 launch).

### C-6: Add the solc-versions-available note to Stage 0 and Stage 7

98 solc versions are pre-installed. The Dockerfile's 6 are a baseline; the rest are on-demand.

**Affected plans:** Stage 0 (Dockerfile), Stage 7 (Docker verification).

### C-7: Add a "performance budget" to each CLI command

The plans specify what each command should *do* but not how fast it should be. A 30-file ScaBench extraction should take < 5 min; a 1k-contract verification should take < 10 min; a full 30K-contract export should take < 30 min. Add these as performance gates in the exit criteria.

**Affected plans:** Stage 1, 2, 4, 5, 7.

### C-8: Use the existing 25 interpretability scripts, don't re-implement

The data-side analysis (Stage 6) is a *complement* to the model-side interpretability suite (`ml/scripts/interpretability/`). The plan must not duplicate model-side analysis in `sentinel_data`. The hand-off is: data-side analysis identifies potential issues; model-side interpretability confirms whether the model is using the issue.

**Affected plans:** Stage 6 (analysis), and the INDEX (cross-reference).

### C-9: Add the "complexity proxy" finding as the motivation for Stage 6

The `complexity_proxy_risk.md` is the data-side complement of the L4 finding in `project_interpretability.md` ("complexity dominates all 10 classes at 34-36%"). The Stage 6 plan should cite this finding explicitly so the report is understood as the proactive catcher.

**Affected plans:** Stage 6 (analysis).

### C-10: Add the 99% DoS↔Reentrancy co-occurrence as the motivation for Stage 3's merger

The co-occurrence matrix is a Stage 6 output, but the merger (Stage 3) is what *prevents* it. The Stage 3 plan should reference the BCCC co-occurrence data (`project_data_pipeline_audit.md` "Co-occurrence" table) as the failure mode the merger is designed to prevent.

**Affected plans:** Stage 3 (labeling), Stage 6 (analysis).

---

## 3. Deletions (parts of the plans that should be removed)

| Plan | Section to remove | Reason |
|---|---|---|
| Stage 2 | 2.7 task: pdg_builder.py description as "lightweight data-flow analysis" | Defer to v3.1 (per 2-P9); v2 baseline doesn't need PDG |
| Stage 3 | 3.7 task: assumption that all 12 parsers can be authored in 5–7 days | Realistic estimate is 10–14 days; the plan should budget for this |
| Stage 4 | 4.6 task: "expand probe_dataset to 50 per class" | Use 40 per class (the BCCC seed); 50 is v2.1 |
| Stage 6 | 6.7 task: "drift monitor requires significant compute" | Drift is cheap (KS test on per-class distributions); reword to remove the FUD |
| Stage 7 | 7.10 task: 6 gates (proposal §9) | Add the 7th gate (36-issue regression) per 7-P11 |
| Stage 8 | 8.1 task: "verify Run 9 has completed" | Run 9 already completed; the task is outdated, replace with "confirm Run 9 best checkpoint is archived" |

---

## 4. New tasks to add (not in the current plans)

| # | Plan | Task | Why |
|---|---|---|---|
| N-1 | Stage 0 | Add a "schema constants" reference file at `Data/sentinel_data/representation/_schema_constants.md` | Single source of truth for the v9 constants; prevents drift across stages |
| N-2 | Stage 0 | Add `_schema_version_registry.json` to track schema versions across the build | Versioner needs a place to read/write the active schema |
| N-3 | Stage 1 | Add a "BCCC-specific retry heuristics" module — even though BCCC is deferred, the preprocessor must handle BCCC-style oddities (whitespace pragmas, exact-version pragmas) because ScaBench / Web3Bugs have similar patterns | Per F26, the Phase 5 retry logic is the reference implementation |
| N-4 | Stage 1 | Add a "data lineage manifest" for each preprocessed file — record which version of which tool produced it | The lineage graph in Stage 5 needs the per-file lineage data |
| N-5 | Stage 2 | Add a "schema regression test" that explicitly asserts the v9 constants match the active `ml/src/preprocessing/graph_schema.py` constants (NODE_FEATURE_DIM=12, NUM_EDGE_TYPES=12, etc.) | Per F1, the proposal's v8 is wrong; the test catches any future drift |
| N-6 | Stage 3 | Add a "crosswalk review workflow" doc that describes the human-review process for each crosswalk YAML | The crosswalk is the highest-leverage artifact; review discipline must be documented |
| N-7 | Stage 4 | Add a "BCCC pipeline script deprecation" task — the 7 ad-hoc scripts in `Phase5_LabelVerification_2026-06-08/scripts/` are replaced by the new module; the deprecation must be explicit | The BCCC scripts are the source of truth for the regression test; deprecating them must be a deliberate act |
| N-8 | Stage 5 | Add a "data catalog API client" — a Python client that wraps the catalog for use by the ML module and the inference server | The ML module and the inference server need a uniform way to query the catalog |
| N-9 | Stage 6 | Add a "data card" generator — a markdown document per dataset version summarizing the data provenance, per-class statistics, known issues, and intended use | The data card is the human-readable summary that complements the catalog |
| N-10 | Stage 7 | Add a "seam swap rehearsal" task — before the actual seam swap, run a dry-run on a 10-contract fixture to verify the byte-identical test passes | The dual-path test is the gate; the rehearsal catches any setup issues before the real swap |
| N-11 | Stage 8 | Add a "Run 11 launch checklist" doc that consolidates the 12 launch conditions (timestamps, sqlite, thresholds, etc.) into a single pre-launch gate | The launch has 12+ conditions; a checklist prevents missing one |
| N-12 | Stage 8 | Add a "Run 11 vs Run 9 comparison" template — what to compare at ep1, ep10, ep20, etc. | The first run on the v2 corpus needs a clear "before vs after" template |

---

## 5. Verification of the verification

After all patches are applied, the 8 plans + INDEX should pass these meta-checks:

1. **No mention of "v8" as the active schema** — all references should be v9 or "active schema (currently v9)"
2. **No re-fixing of A1–A38 bugs** — each plan's "What NOT to fix" section should list the relevant ones
3. **The 36-issue pre-Run-8 audit is referenced** in Stage 2's regression test scope
4. **The 8 fixed bugs are documented** in the INDEX "Code-bug state" subsection
5. **All WSL commands use the `wsl -- bash -c` wrapper**
6. **All MLflow commands use `sqlite:///` backend**
7. **All `solc-select` references are aware of the 98 pre-installed versions**
8. **All 12+ launch conditions for Run 11 are in the launch checklist**
9. **All 7 v2-readiness gates (including the 36-issue regression) are in Stage 7's final check**
10. **The complexity_proxy_risk is the headline Stage 6 output, motivated by the L4 finding**
11. **The 99% DoS↔Reentrancy co-occurrence is referenced in Stage 3 and Stage 6**
12. **The seam swap includes the predictor.py tier-threshold fix (F8, F10)**
13. **The dual-path test asserts the 8 fixed bugs survive the swap**

---

## 6. Plan-by-plan summary of net change

| Plan | Additions | Changes | Deletions | Net |
|---|---:|---:|---:|---:|
| INDEX | 2 (sections) | 5 | 0 | +7 |
| Stage 0 | 3 (tasks) | 7 (tasks) | 0 | +10 |
| Stage 1 | 3 (tasks) | 5 (tasks) | 0 | +8 |
| Stage 2 | 6 (tasks) | 3 (tasks) | 1 (task partial) | +8 |
| Stage 3 | 9 (tasks/notes) | 1 (taxonomy note) | 1 (estimate) | +9 |
| Stage 4 | 6 (tasks/notes) | 5 (tasks) | 1 (probe size) | +10 |
| Stage 5 | 4 (tasks) | 4 (tasks) | 0 | +8 |
| Stage 6 | 5 (tasks/notes) | 4 (tasks) | 1 (FUD) | +8 |
| Stage 7 | 5 (tasks) | 5 (tasks) | 0 | +10 |
| Stage 8 | 6 (tasks) | 4 (tasks) | 1 (outdated) | +9 |
| **Total** | **49** | **43** | **5** | **+87 net additions** |

Roughly: each plan grows by ~9 net additions. The build window stays at 8 weeks; the additions are mostly "do not forget" notes and regression test assertions, not new work.

---

**End of audit. Apply these patches to the 8 plans and INDEX before any code is written.**
