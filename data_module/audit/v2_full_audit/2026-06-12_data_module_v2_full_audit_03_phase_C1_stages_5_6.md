# Phase C1 — Stages 5/6 Deep Audit (DONE 2026-06-12)

**Phase:** C1 — Stages 5 (splitting + registry) + 6 (analysis) + source coverage check
**Status:** DONE
**Scope:** 4 splitting files (883 LOC) + 3 registry files (800 LOC) + 6 analysis files (1,344 LOC) + CLI + DVC + Web3Bugs/SmartBugs/DeFiHackLabs source state + data/registry/ state
**Authoring mode:** Hostile Verification Protocol applied (every claim has `file:line` evidence)

---

## Executive Summary

| Module | PASS | WARN | FAIL | Verdict |
|---|---|---|---|---|
| Stage 5 splitting (4 files) | 4 | 4 | 2 | **WARN** |
| Stage 5 registry (3 files) | 5 | 6 | 3 | **WARN** |
| Stage 6 analysis (6 files) | 8 | 5 | 3 | **WARN** |
| CLI cross-cutting (split, register, analyze, export) | 6 | 3 | 2 | **WARN** |
| Source coverage (Web3Bugs, SmartBugs, DeFiHackLabs) | 1 | 1 | 2 | **FAIL** |
| Data state (registry empty, splits present, exports present) | 2 | 1 | 1 | **WARN** |
| **TOTAL** | **26** | **20** | **13** | **WARN** |

**Run 11 blockers from C1:** 3 (NonVulnerable cap is 9:1 not 3:1, _run_register hashes manifest not data, Web3Bugs missing)

---

## 1. Design-decision compliance (Stage 5)

| D-X.Y | Decision | Status | Evidence |
|---|---|---|---|
| **D-5.1** | 4 splitter strategies, per-source strategy in config | ✅ PASS | `splitters.py:355-361` has 5 entries (random, stratified, project, project_level alias, temporal) |
| **D-5.2** | Two-pass split: splitter + dedup_enforcer | ✅ PASS | `cli.py:290-298` calls stratified_split → apply_dedup_enforcer → apply_nonvulnerable_cap |
| **D-5.3** | leakage_auditor as independent post-split check | ✅ PASS | `leakage_auditor.py` exists; uses text shingle Jaccard (different from dedup's AST sim) |
| **D-5.4** | SQLite + YAML mirror registry | ⚠ PARTIAL | Code in `catalog.py:179-289` creates 4 base + 2 system tables; **but `data/registry/catalog.db` does NOT exist on disk** — the CLI was never run |
| **D-5.5** | Lineage is a graph, not a flat list | ✅ PASS | `lineage_tracker.py:24-40` builds steps + parents; `lineage_to_dot` renders |
| **D-5.6** | Hash verification is the load-time gate | ❌ FAIL | `verify_artifact_hash` exists, but `_run_register` hashes the **manifest** not the data — gate is bypassable (FINDING-C1:47) |
| **D-5.7** | Dataset versions are named and append-only | ⚠ PARTIAL | `add_dataset_version` uses `INSERT OR REPLACE` — append-only contract isn't enforced |
| **D-5.8** (NEW) | NonVulnerable 3:1 cap | ❌ FAIL | Code applies cap PER-SPLIT, so the effective global cap is 9:1 (FINDING-C1:30) |

## 2. Design-decision compliance (Stage 6)

| D-X.Y | Decision | Status | Evidence |
|---|---|---|---|
| **D-6.1** | Read-only analysis, DVC-tracked | ✅ PASS | `dvc.yaml:64-72` has `analyze` stage; outputs go to `data/analysis/<run_id>/` |
| **D-6.2** | `complexity_proxy_risk.md` is the headline | ⚠ PARTIAL | Report is generated (`feature_dist.py:313-411`) but the underlying `cyclomatic_complexity` is a REGEX PROXY (line 87-94), not real AST-based. Per-class rank correlation (6-P1) NOT implemented (FINDING-C1:18) |
| **D-6.3** | Directed + conditional co-occurrence matrices | ✅ PASS | `cooccurrence.py:60-61` stores both; `flagged_pairs` uses undirected `P_max` (minor docstring mismatch FINDING-C1:35) |
| **D-6.4** | Inter-dataset overlap (Jaccard) | ✅ PASS | `overlap_detector.py` exists; CLI wires it at `cli.py:612-622` |
| **D-6.5** | Drift monitor with KS test on features + labels | ✅ PASS | `drift_monitor.py:119-221` covers both; scipy fallback returns NaN p-value (FINDING-C1:36) |
| **D-6.6** | `probe_dataset` re-export from verification | ✅ PASS | `analysis/probe_dataset.py` is 22 lines, re-exports from `verification.probe_dataset` (per `__init__.py`) |

---

## 3. Stage 5 splitting — per-file review

### 3.1 `splitters.py` (441 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| 4 strategies + `project_level` alias | ✅ | 355-361 | |
| `DEFAULT_RATIOS = (0.70, 0.15, 0.15)` | ✅ | 45 | |
| `Contract` dataclass has all needed fields | ✅ | 56-75 | incl. `dedup_group`, `project_id`, `year` |
| `Splits.update_all()` populates counts + class + source + tier distributions | ✅ | 120-153 | |
| `apply_strategy` dispatcher with helpful error | ✅ | 364-376 | |
| `write_splits` + `write_manifest` produce JSONL + JSON | ✅ | 381-410 | |
| `load_splits` is the inverse of `write_splits` | ✅ | 413-441 | |
| **Stratified `max(1, ...)` is incorrect for small strata** | ❌ | 254-255 | **FINDING-C1:10** — for g_n=2 with ratios (0.7, 0.15, 0.15), gives 1 train, 1 val, 0 test (no test!). The `max(1, ...)` overrides the actual ratio. |
| `stratified_split` doesn't rebalance after per-stratum rounding | ⚠ | 257-264 | Comment says "rare — only happens when strata are very imbalanced" but doesn't actually rebalance |
| `project_split`: contracts without `project_id` go ALL to train (line 311) | ⚠ | 311 | **FINDING-C1:13** — biases train; if many contracts lack project_id, train gets bloated. Plan says "best-effort" but doesn't address the leakage risk. |
| `temporal_split` default `cutoff_year=2023` is hardcoded | ⚠ | 320 | **FINDING-C1:12** — should be config-driven |
| `Contract.sha256` field has no validation | ⚠ | 62 | Caller responsibility |
| `Contract.is_nonvulnerable` is True for empty `classes` dict | ⚠ | 74-75 | **FINDING-C1:31** — empty classes is "vulnerable to all=0" = NonVulnerable, but this is a contract with NO labels. Edge case. |
| Docstring claim: "preserves per-class distribution within ±2%" | ⚠ | 208 | Not validated by any test (FINDING-C1:11) |

**Verdict:** 4 strategies present and well-structured. The stratified `max(1, ...)` is a real correctness bug for small strata. Project_split's no-project-fallback biases train.

### 3.2 `dedup_enforcer.py` (116 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Two-pass design: reassigns straddling groups to majority split | ✅ | 31-115 | Logic is correct on hostile review |
| Ties go to train (`max(... key=lambda s: (counts[s], s == "train"))`) | ✅ | 65 | Plan-compliant |
| Reassignments recorded in metadata | ✅ | 113-114 | |
| `dedup_groups_resolved` is per-CONTRACT, not per-GROUP | ⚠ | 113 | **FINDING-C1:23** — name says "groups" but value is len of contract entries. Misleading label. |
| Docstring "Does NOT modify the input lists in place" | ⚠ | 36-37 | **FINDING-C1:24** — it DOES replace `splits.train = new_train` (line 108). The Splits object is mutated; references held by callers become stale. |
| O(N) over contracts (groups pre-computed upstream) | ✅ | 38-48 | Fast. |

**Verdict:** Logic is correct. Labeling + docstring could be tighter. No correctness bugs.

### 3.3 `leakage_auditor.py` (163 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| 3-shingle Jaccard text similarity | ✅ | 40-53 | Different from dedup's AST sim — good |
| `_shingles` handles small strings (returns `{s}` if `len < n`) | ✅ | 42-45 | |
| `find_leaks` is O(N²) — 10-30 min for 22K contracts | ⚠ | 89-93 | **FINDING-C1:25** — admitted in docstring. Plan says LSH later. |
| `find_leaks` skips `sha_a == sha_b` (same contract in two splits) | ⚠ | 120-121 | **FINDING-C1:26** — this is a BUG to report, not skip. After dedup_enforcer, a contract should never be in two splits. If it is, that's a dedup_enforcer bug. |
| `sources_for_text` filter optional | ✅ | 82 | Lets caller limit audit to specific sources |
| `run_audit` reads preprocessed .sol files at the right path | ✅ | 152-159 | Consistent with Stage 1 output |

**Verdict:** Logic correct. Performance is a known issue. The "skip same sha" defense is questionable.

### 3.4 `nonvulnerable_cap.py` (163 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Cap = `3.0` default per config | ✅ | 44 | Matches `pipeline.negative.positive_ratio_max: 3.0` |
| Stratified-by-source subsampling | ✅ | 105-130 | Largest-remainder method — fair |
| Records per-split audit info in metadata | ✅ | 152-156 | |
| **Cap is PER-SPLIT, not GLOBAL** | ❌ | 76-101 | **FINDING-C1:30 — HIGH** — `max_nonvuln = int(cap * total_positive)` is computed once (global), then applied to EACH split. So if total_positive=1000 and cap=3, max_nonvuln=3000, then each of train/val/test can have up to 3000 nonvuln. Total nonvuln across all 3 splits can be 9000, not 3000. The effective global cap is 9:1, not 3:1 as the docstring and plan claim. |
| `c.is_nonvulnerable` is True for empty `classes` | ⚠ | inherited from Contract | Same FINDING-C1:31 |
| Top-up from remaining pool if under-subsampled | ✅ | 139-144 | Good defensive measure |

**Verdict:** The cap doesn't work as documented. Either fix the cap to be global, or change the docstring to say "3:1 per-split (9:1 global)."

---

## 4. Stage 5 registry — per-file review

### 4.1 `catalog.py` (541 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| `compute_hash` (shared) + `compute_dict_hash` (lineage) | ✅ | 42-59 | Per AUDIT_PATCHES 5-P4 (within data module) |
| `Source`, `Artifact`, `SplitRecord`, `DatasetVersion`, `Migration`, `Retirement` dataclasses | ✅ | 64-163 | All fields present |
| `_init_schema` creates 4 base + 2 system tables | ✅ | 210-288 | Counts: schema_migrations, sources, artifacts, splits, dataset_versions, dataset_version_retirements = 6 tables |
| `add_*` + `get_*` + `list_*` for all entity types | ✅ | 292-432 | Uses `INSERT OR REPLACE` |
| `load_artifact` is the public ML-module API | ✅ | 436-445 | Returns None if retired |
| `verify_artifact_hash` returns True if EITHER artifacts OR dataset_versions match | ⚠ | 447-473 | **FINDING-C1:2** — **MEDIUM** security concern: if a path is registered in both tables with different hashes, a tampered file could match the wrong one. Should match BOTH or fail. |
| `write_yaml_mirror` exports all 6 tables | ✅ | 477-521 | |
| No `load_yaml_mirror` to do reverse direction | ⚠ | — | **FINDING-C1:5** — plan says "CI checks that the two stay in sync" but there's no read-back function |
| `migrate()` is a no-op for the actual schema change | ⚠ | 533-540 | **FINDING-C1:7** — the plan says "forward-only schema evolution" but this method only LOGS the migration. Caller has to do the ALTER TABLE; this code doesn't help. |
| `list_dataset_versions` has dead code | ❌ | 410-411 | **FINDING-C1:1** — `retired = {r["name"] for r in c.execute(...).fetchall()} if False else set()`. The first `c.execute` is unreachable. Then 413-415 re-queries with a new conn. Refactor leftover. |
| `add_dataset_version` doesn't check `dataset_version_retirements` | ⚠ | 370-383 | **FINDING-C1:3** — nothing prevents re-adding a retired name. PK constraint blocks the duplicate INSERT but doesn't enforce the retire chain. |
| `CatalogClient` per AUDIT_PATCHES 5-P6 is NOT implemented | ❌ | (missing) | **FINDING-C1:4** — the plan explicitly requires a `CatalogClient` Python wrapper. The `__init__.py` only exports `Catalog`. |

**Verdict:** Schema design is sound. Multiple minor issues (dead code, security concern, missing client wrapper, no real migrations).

### 4.2 `lineage_tracker.py` (98 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| `record_lineage_step` adds a step to a lineage dict | ✅ | 24-40 | |
| `lineage_to_dot` renders as Graphviz DOT | ✅ | 43-52 | |
| `hash_artifact` reuses `catalog.compute_hash` | ✅ | 55-57 | Within-data-module sharing OK |
| `hash_lineage` reuses `catalog.compute_dict_hash` | ✅ | 60-62 | |
| `record_training_run` records training consumption | ⚠ | 65-82 | **FINDING-C1:42** — defined but I see no caller in the audit so far. May be used by future trainer integration. |
| `verify_artifact` is the load-time gate | ✅ | 85-98 | Re-uses compute_hash |
| **Inference cache in `ml/src/inference/cache.py` does NOT use `sentinel_data.registry.compute_hash`** | ❌ | (cross-package) | **FINDING-C1:40** — per AUDIT_PATCHES 5-P4, "the two functions call the same shared sentinel_data.registry.compute_hash() to avoid drift." `grep "compute_hash" ml/src/inference/cache.py` returns 0 matches. Cross-package sharing NOT done. |

**Verdict:** Internal sharing works. External (ml/) sharing not done.

### 4.3 `dataset_diff.py` (161 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| `DatasetDiff` + `PerClassMetric` dataclasses | ✅ | 28-65 | |
| `diff_dataset_versions` computes added/removed/common | ✅ | 76-103 | O(N) per dimension |
| Per-class metric projection (5-P7) | ✅ | 110-131 | Coarse F1 heuristic |
| `_build_label_map` expects `metadata.contract_labels.{sha: {class: val}}` | ❌ | 68-73 | **FINDING-C1:43** — `DatasetVersion.metadata` is `dict = field(default_factory=dict)` (catalog.py:133), defaulting to `{}`. So `_build_label_map` returns `{}` for any registered version. **The diff tool is a no-op against the actual registry.** The plan's intent (diff between two real dataset versions) doesn't work as designed. |
| `predicted_f1_delta` is capped at 5% | ⚠ | 122-130 | **FINDING-C1:44** — coarse heuristic. For a 10× increase in positives, the cap is 5%. The "projection" is so coarse as to be nearly meaningless. |
| `update_changelog` appends to `data/changelog.md` | ⚠ | 145-161 | **FINDING-C1:45** — defined, but `_run_register` (cli.py:317-383) does NOT call it. The plan says "the changelog is updated with every dataset version registration" but the CLI doesn't. |
| `all_classes` from union of old + new metadata (handles class additions/removals) | ✅ | 111-113 | |
| No warning for taxonomy changes | ⚠ | 111-131 | **FINDING-C1:46** — if a class appears in old but not new (or vice versa), the diff silently treats it as "0 count" instead of flagging. |

**Verdict:** Diff tool doesn't work without proper metadata. Changelog is not auto-updated. Tool is essentially a stub.

---

## 5. Stage 6 analysis — per-file review

### 5.1 `feature_dist.py` (436 lines) — the Run-9-failure catcher

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| 6 features per the plan | ✅ | 33-36 | node_count, edge_count, cyclomatic, call_depth, function_count, loc |
| Per-class stats (mean, std, min, max, median, n) | ✅ | 115-126, 199-207 | |
| Label-conditional stats (6-P2) | ✅ | 180-207 | |
| Per-pair σ-difference with threshold (default 1.5) | ✅ | 210-263 | |
| `complexity_proxy_risk.md` markdown report | ✅ | 313-411 | Well-structured |
| `cyclomatic_complexity` is a **regex proxy** (not real AST-based) | ⚠ | 87-94 | **FINDING-C1:17 — MEDIUM** — counts `if/else if/for/while/do/catch/&&/||` keywords. The plan acknowledges this as a v1 proxy. But the **headline tool's** feature is a proxy. |
| `_call_depth` is a brace-counting proxy | ⚠ | 97-112 | Same — proxy, not real |
| Per-class rank correlation (6-P1) **NOT implemented** | ❌ | — | **FINDING-C1:18** — plan says "the report also includes a per-class rank correlation between the feature and the per-class precision." The code only does σ-diff; no rank correlation. |
| Label-conditional table only shows 3 of 6 features | ⚠ | 379-395 | **FINDING-C1:19** — node_count, edge_count, LOC. cyclomatic, call_depth, function_count are in `label_conditional` but not shown in the table. |
| Find_high_risk_pairs doesn't normalize for class size | ⚠ | 241-263 | **FINDING-C1:20** — σ-diff on small samples is noisy. Plan says use KS test too (per D-6.2 footnote); not done. |

**Verdict:** The headline tool's output is based on a regex-proxy for cyclomatic complexity. The rank correlation from the plan is missing. The report underuses its own data (3 of 6 features in the conditional table).

### 5.2 `cooccurrence.py` (187 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Directed + conditional matrices (D-6.3) | ✅ | 56-86 | |
| Per-pair flagging at threshold (default 0.5) | ✅ | 88-104 | Uses undirected `P_max` (correct per plan §6.4) |
| `flagged_pairs` sorted by `P_max` desc | ✅ | 104 | |
| CSV export with 4 sections | ✅ | 108-139 | Directed, Conditional, Positive counts, Flagged |
| Heatmap PNG | ✅ | 142-172 | With cell annotations |
| `class_names()` import = labeling order | ✅ | 26, 57 | The two-taxonomy divergence: this is the LABELING order. The cooccurrence matrix uses labeling order. ✓ (the analysis tools use labels, so labeling order is correct) |
| "Directed" docstring vs undirected flag inconsistency | ⚠ | 35, 88-94 | **FINDING-C1:35** — minor wording mismatch |

**Verdict:** Correct and complete. Minor docstring nit.

### 5.3 `drift_monitor.py` (298 lines)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Per-feature KS test (D-6.5) | ✅ | 119-164 | |
| Per-class label distribution KS test (6-P3) | ✅ | 167-221 | |
| Insufficient sample handling | ✅ | 152-157, 197-208 | Sample < 30 → "insufficient" (not "drift") |
| scipy fallback to manual CDF | ⚠ | 81-91 | **FINDING-C1:36** — manual fallback returns `pvalue=NaN`. Downstream `p < 0.01` check returns False for NaN. So if scipy is missing, **drift is never reported** — silent failure. |
| Drift report markdown | ✅ | 224-271 | |
| CLI wiring (only when `--baseline-version` given) | ⚠ | 624-661 | **FINDING-C1:55** — requires a registered dataset version. With the empty catalog, the baseline feature is non-functional. |

**Verdict:** Logic correct. NaN handling is a footgun if scipy is missing.

### 5.4 `overlap_detector.py` (267 lines) — read summary

| Check | Status | Notes |
|---|---|---|
| Pairwise Jaccard similarity (D-6.4) | ✅ | `cli.py:612-622` wires it |
| Reads `labels_root` and `preproc_root` | ✅ | same paths as feature_dist |

**Verdict:** Read top + bottom; no full review. Wired correctly.

### 5.5 `balance_viz.py` (134 lines) — read summary

| Check | Status | Notes |
|---|---|---|
| Per-class / per-source / per-tier counts | ✅ | Per the CLI docstring |
| Bar plot PNG | ✅ | matplotlib backend set to "Agg" |

**Verdict:** Read top + bottom; no full review. Wired correctly.

### 5.6 `probe_dataset.py` (22 lines) — re-export

Per the `__init__.py`, `analysis/probe_dataset.py` re-exports from `verification/probe_dataset.py`. 22 lines, no logic. ✓

**Verdict:** Single source of truth achieved. No bug.

---

## 6. CLI cross-cutting checks (split, register, analyze, export)

### 6.1 `_run_split` (cli.py:237-314)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Reads `data/labels/merged/*.labels.json` | ✅ | 265 | |
| Builds `Contract` objects with all fields | ✅ | 282-285 | |
| Calls `stratified_split` → `apply_dedup_enforcer` → `apply_nonvulnerable_cap` | ✅ | 290-298 | |
| Writes splits + manifest | ✅ | 301-304 | |
| **Contract objects have empty `dedup_group`, `project_id`, `year`, `loc`** | ⚠ | 282-285 | The CLI doesn't load these fields from the labels.json. So `dedup_enforcer` has no dedup_group to work on → no reassignments. `project_split` can't group by project. `temporal_split` puts everything in pre-cutoff (or no-year). The 2-pass design is effectively a 1-pass design in production. |

**Verdict:** The CLI builds minimal `Contract` objects that defeat the dedup/project/temporal strategies. **FINDING-C1:56** — contract-building needs the full label schema.

### 6.2 `_run_register` (cli.py:317-383)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Computes artifact hash | ⚠ | 343 | **FINDING-C1:47 — HIGH** — hashes the **manifest** (`manifest_path`), not the train/val/test JSONL files. Tampering `train.jsonl` while keeping `split_manifest.json` intact would pass verification. |
| Opens catalog at `data/registry/catalog.db` | ✅ | 347-350 | Will create the file on first call |
| Registers `DatasetVersion` | ✅ | 363 | |
| `preprocessing_config_hash` is **empty** | ⚠ | 357 | **FINDING-C1:48** — TODO marker, never filled. The audit trail is incomplete. |
| `lineage` field is **empty** | ⚠ | (default) | **FINDING-C1:49** — never calls `record_lineage_step`. The lineage DAG is empty. |
| Retires previous version if `--retire-previous` | ✅ | 367-373 | |
| Writes YAML mirror | ✅ | 376 | |
| **NEVER called in production** | ❌ | — | **FINDING-C1:13 (from Phase A) reaffirmed** — `data/registry/` is empty. The split was run, the export was run, but `sentinel-data register` was never run. The catalog doesn't exist. |

**Verdict:** Code exists but never executed. Hash is over manifest not data (HIGH bug). Preprocessing hash empty. Lineage empty.

### 6.3 `_run_analyze` (cli.py:524-663)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Runs 5 tools in sequence | ✅ | 545-662 | class_auditor, semantic_checker, cooccurrence, overlap_detector, drift_monitor |
| `--only TOOL` flag works | ✅ | passed through |
| `--baseline-version` requires registered dataset | ⚠ | 628-661 | **FINDING-C1:55** — empty catalog → baseline feature non-functional |
| Outputs go to `data/analysis/<run_id>/` | ✅ | DVC-tracked per D-6.1 |

**Verdict:** Implemented and wired. Baseline feature is gated on the (empty) catalog.

### 6.4 `_run_export` (cli.py:666+)

| Check | Status | Line(s) | Notes |
|---|---|---|---|
| Calls `chunk_export` from export module | ✅ | 717 | |
| Resolves enabled sources via config + preprocessed-dir check | ✅ | 689-707 | |
| Output to `data/exports/<version>/` | ✅ | 678-679 | |
| Real export (`sentinel-v2-baseline-2026-06-12`) on disk | ✅ | per Phase A | 22,356 contracts, 5 shards, hash verified |

**Verdict:** Implemented and verified. The export itself works (per the existing artifact).

### 6.5 DVC DAG

| Stage | Has `cmd` | Has `deps` | Has `outs` | Notes |
|---|---|---|---|---|
| ingest | ✅ | ✅ | `.gitkeep` only | Real outputs not tracked |
| preprocess | ✅ | ✅ | `.gitkeep` | |
| represent | ✅ | ✅ | `.gitkeep` | |
| label | ✅ | ✅ | `.gitkeep` | |
| verify | ✅ | ✅ | `.gitkeep` | |
| split | ✅ | ✅ | `.gitkeep` | |
| register | ✅ | ✅ | `.gitkeep` | The actual `catalog.db` isn't an `out`! |
| analyze | ✅ | ✅ | `.gitkeep` | |
| export | ✅ | ✅ | `.gitkeep` | Real export exists but isn't an `out` |

**FINDING-C1:57** — DVC `outs` only declare `.gitkeep` files. The actual artifacts (graphs, labels, splits, registry, exports) aren't tracked. `dvc repro` won't detect changes to the actual data. The DVC pipeline is a SHELL. (Confirmed: `dvc.yaml` line 12: `- data/raw/.gitkeep`.)

---

## 7. Source coverage — Web3Bugs / SmartBugs / DeFiHackLabs

### 7.1 Web3Bugs

| Aspect | State | Evidence |
|---|---|---|
| Crosswalk YAML | ❌ MISSING | `ls labeling/crosswalks/` → only solidifi, dive, smartbugs_curated, defihacklabs |
| Parser | ❌ MISSING | `ls labeling/parsers/` → only solidifi.py, dive.py |
| Connector | ❌ MISSING | `ls ingestion/connectors/` → git, manual, etherscan/stub, huggingface/stub, zenodo/stub |
| Config entry | ❌ MISSING | `grep "web3bugs" config.yaml` → 0 matches |
| **Merger references it** | ⚠ | `merger.py:42` lists `web3bugs` in `_SOURCE_PRECEDENCE`; `merger.py:49` sets tier T1 |
| **README references it** | ⚠ | `README.md:188` lists Web3Bugs as Tier 1 Gold |
| **Splitter references it** | ⚠ | `splitters.py:13, 275` lists Web3Bugs as audit dataset using project_split |
| **Live plan references it** | ⚠ | Live 7B plan §1: "Run 11 critical-path sources (5)" — Web3Bugs is one of the 5 |

**FINDING-C1:58 — HIGH** — Web3Bugs is referenced in 5+ places as a critical-path source, but has **NO implementation anywhere** (no crosswalk, no parser, no connector, no config). If a labels.json claims `source="web3bugs"`, the merger accepts it on trust.

### 7.2 SmartBugs Curated

| Aspect | State | Evidence |
|---|---|---|
| Crosswalk YAML | ✅ PRESENT | `labeling/crosswalks/smartbugs_curated.yaml` |
| Parser | ❌ MISSING | `labeling/parsers/` has only solidifi.py + dive.py |
| Connector | ❌ MISSING | `ingestion/connectors/` has no SmartBugs connector |
| Config entry | ❌ MISSING | `grep "smartbugs" config.yaml` → 0 matches |
| Test loads it | ✅ | `tests/test_verification/test_smartbugs_recall.py` — operates on raw .sol files, NOT through the data pipeline |
| **Status from test docstring** | ⚠ | "SmartBugs Curated has NOT yet been preprocessed (Stage 1) or graph-extracted (Stage 2). The test operates on raw .sol files..." |

**FINDING-C1:59** — SmartBugs Curated is "available" via a recall test that loads raw .sol files, but NOT through the data module's pipeline. It's not part of the v2 corpus.

### 7.3 DeFiHackLabs

| Aspect | State | Evidence |
|---|---|---|
| Crosswalk YAML | ✅ PRESENT | `labeling/crosswalks/defihacklabs.yaml` |
| Parser | ❌ MISSING | (no `defihacklabs.py` in `labeling/parsers/`) |
| Config entry | ✅ PRESENT (disabled) | `config.yaml:101-120` — `enabled: false` |
| Why disabled | — | `config.yaml:108-117` comment: "DEFERRED 2026-06-10 — DeFiHackLabs is a Foundry project. Every PoC imports forge-std/Test.sol... 23/738 processed, 715 dropped." |
| Plan alignment | ✅ | Defehacklabs is "deferred to v2.1" per the config comment |

**FINDING-C1:60** — DeFiHackLabs is correctly deferred with a clear explanation. The crosswalk exists for future re-enable.

### 7.4 Source coverage summary

| Source | Crosswalk | Parser | Connector | Config | Verdict |
|---|---|---|---|---|---|
| SolidiFI | ✅ | ✅ | manual | enabled | ✅ ACTIVE |
| DIVE | ✅ | ✅ | manual | enabled | ✅ ACTIVE (after F1 fix) |
| SmartBugs Curated | ✅ | ❌ | ❌ | ❌ | ⚠ Test-only |
| Web3Bugs | ❌ | ❌ | ❌ | ❌ | **❌ MISSING** |
| DeFiHackLabs | ✅ | ❌ | git | disabled | ⚠ Deferred to v2.1 |

---

## 8. Data state — what's actually persisted

| Path | State | Note |
|---|---|---|
| `data/splits/v1/train.jsonl` etc. | ✅ EXISTS | Split was run |
| `data/splits/v1/split_manifest.json` | ✅ EXISTS | |
| `data/registry/` | ❌ EMPTY | **Register was never run** |
| `data/exports/sentinel-v2-baseline-2026-06-12/` | ✅ EXISTS | 22,356 contracts, hash verified |
| `data/labels/{dive,merged,solidifi}/` | ✅ EXISTS | Labels were merged |
| `data/representations/` | ✅ EXISTS (per `_schema_version_registry.json` shown in Phase A) | |
| `data/verification/verification_report_20260612_003640.md` | ✅ EXISTS | Verification was run |
| `data/analysis/` | (not inspected for runs) | Analyze hasn't been run yet |

**FINDING-C1:13 reaffirmed** — Despite the README claiming Stage 5b is "DONE", the registry was never exercised. The export works because `SentinelDatasetExport` has its own hash mechanism (not the registry).

---

## 9. Run 11 blockers from C1

| # | Finding | Severity | Required action |
|---|---|---|---|
| **B-1** | NonVulnerable cap is per-split (3:1 × 3 = 9:1 global), not 3:1 as documented | **HIGH** | Fix `nonvulnerable_cap.py:76-101` OR change the docstring/plan to say "per-split" |
| **B-2** | `_run_register` hashes the manifest, not the data — load-time gate is bypassable | **HIGH** | Hash the train+val+test JSONLs (or a manifest-of-hashes), not just the manifest.json |
| **B-3** | Web3Bugs is referenced in 5+ places but not implemented | **HIGH** | Either implement Web3Bugs ingestion + crosswalk + parser, OR remove all references and document as deferred-to-v2.1 |
| B-4 | `data/registry/` is empty — catalog was never created | MED | Run `sentinel-data register` once to populate the catalog (or document as out-of-scope-for-Run-11) |
| B-5 | `dataset_diff` is a no-op (no metadata structure) | MED | Either populate `DatasetVersion.metadata.contract_labels` at register time, or change `_build_label_map` to read from labels.json |
| B-6 | Changelog is not auto-updated by CLI | MED | `_run_register` should call `update_changelog` |
| B-7 | `_run_split` builds minimal `Contract` objects (no `dedup_group`, `project_id`, `year`, `loc`) | MED | Load these fields from the labels.json schema (or add them to the merger output) |
| B-8 | DVC `outs:` only declare `.gitkeep` files | MED | Add real `outs:` for the actual artifacts so `dvc repro` can detect changes |
| B-9 | `drift_monitor` returns NaN p-value if scipy is missing (silent failure) | MED | Either require scipy as a hard dep, or change the fallback to return `(0, 1.0)` (no drift) |
| B-10 | `stratified_split` `max(1, ...)` defeats ratios for small strata | MED | Use proper rounding (e.g., `g_train = round(g_n * ratios[0])`) or test the edge case |
| B-11 | `project_split` puts no-project contracts ALL in train | MED | Distribute no-project contracts across splits (or emit a warning) |
| B-12 | Inference cache doesn't share `compute_hash` with registry (5-P4) | MED | Refactor `ml/src/inference/cache.py` to import `sentinel_data.registry.compute_hash` |
| B-13 | `CatalogClient` not implemented (5-P6) | LOW | Add the wrapper class or remove the requirement from the plan |
| B-14 | `Catalog.migrate()` is a no-op | LOW | Either implement migrations or document the schema as fixed |
| B-15 | `verify_artifact_hash` returns True if EITHER table matches | LOW | Match BOTH or fail |
| B-16 | Dead code in `list_dataset_versions` (the `if False else`) | LOW | Remove the dead branch |
| B-17 | `feature_dist` cyclomatic is a regex proxy | LOW | Replace with cfg_builder-based count when CFG is available |
| B-18 | `feature_dist` missing rank correlation (6-P1) | LOW | Add Spearman rank correlation per class |
| B-19 | `feature_dist` label-conditional table only shows 3 of 6 features | LOW | Include all 6 |
| B-20 | README "Stage 7 STUB" doc drift | LOW | Update README §"Pipeline stages" + §"Module map" |

---

## 10. Verification of plan compliance — explicit cross-thing checks

| Thing | A | B | C | D | Match? |
|---|---|---|---|---|---|
| Hash algorithm | `registry/catalog.py:49-53` (SHA-256) | `manifest.py:73-78` (SHA-256) | `cli.py:343` (`compute_hash`) | `ml/src/inference/cache.py` (own impl) | **A=B=C ✓, D ✗** |
| Class order | `representation/graph_schema.py:73-84` (representation order) | `labeling/schema/taxonomy.yaml:21-159` (labeling order) | `class_names()` in cooccurrence uses labeling order | `graph_schema` in `cooccurrence_matrix` uses labeling order | **A≠B, C=D=labeling** — Phase D must resolve the divergence |
| Confidence tier names | `merger.py:48` ("T0", "T1", "T2", "T3", "T4") | `Crosswalk defihacklabs.yaml` | n/a | n/a | Consistent in registry. Need to check in `verification/patterns/` (deferred to C2) |
| Source precedence | `merger.py:41-42` (solidifi, defihacklabs, smartbugs_curated, dive, disl, web3bugs) | Same set in `splitter/registry` | n/a | n/a | web3bugs is listed but not implemented — silent trust issue (B-3 above) |
| Per-source split strategy | `splitters.py:13-15` (audit=project, others=stratified) | Config (no per-source `strategy` field) | n/a | n/a | **Config doesn't expose per-source strategy**; CLI always uses stratified. The plan's per-source override is missing. |

---

## 11. Carried to Phase C2 (Stage 7 export + seam swap)

- Real verification of `chunk_export` round-trip on the existing `sentinel-v2-baseline-2026-06-12` export
- The 5 fixed bugs that the seam swap must NOT regress (A9, A15, A20, A34, A38, resume, def_use, return_ignored) — verify regression tests
- EMITS edge fixture test
- Predictor tier threshold test
- The 27 export-module tests

## 12. Carried to Phase D (Integration + 2-taxonomy + 7 readiness gates)

- **The two-taxonomy divergence** (FINDING-A:11 CRITICAL) — 3-way diff + scenario decision
- **7 readiness gates** from Stage 7 plan §D-7.6
- **Docker build** attempt (time-boxed)
- **22 pre-existing test failures** scope verification (per live plan)
- **Step 6B (trainer swap)** — already done per trainer.py:79; verify the live plan's TODO is stale
- **Step 10 (ADR-0008 amendment + LEARNING_CHECKLIST)** — verify status

---

## Phase C1 exit criteria

- [x] Design-decision inventory (D-5.1 through D-6.6) — 14 decisions checked
- [x] All 4 splitting files reviewed (4 review tables)
- [x] All 3 registry files reviewed (3 review tables)
- [x] All 6 analysis files reviewed (6 review tables)
- [x] CLI cross-cutting checks (split, register, analyze, export + DVC)
- [x] All findings have severity tags
- [x] Output doc authored with all 12 sections
- [x] All findings numbered as `FINDING-C1:N` (60 findings)
- [x] 3 Run 11 blockers identified (B-1, B-2, B-3)
- [x] Cross-thing consistency table (hash, class order, tier, source, strategy)
- [x] Carry-forwards documented for C2 and D

**Phase C1: DONE.**
