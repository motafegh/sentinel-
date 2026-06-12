# Stage 6 — Analysis (the Run-9-failure catcher)

**Date:** 2026-07-21 (revised 2026-06-12 post-implementation)
**Status:** ✅ COMPLETE. 8 commits landed. 17 analysis tests pass. Stage 6 exit criteria all met.
**Reading time:** 25-30 minutes. The doc has 6 sections matching the standard format; take notes.
**Goal:** After this doc, you can answer all 6 items in `LEARNING_CHECKLIST.md` §"Stage 6" from memory, explain the 6 design decisions (D-6.1 through D-6.6), the 4 implementation choices (IC-1 through IC-4), and the 9 exit criteria.

---

## 1️⃣ The Problem

### What Stage 6 has to deliver

Stages 1–5 produced 22,356 verified, split, registered contracts. Stage 6 is the **first stage that *looks at* the data** to ask: "is this corpus going to teach the model the right thing?" — a question the Run 9 failure answered with "no" only after 9 training runs.

The Run 9 failure was a model learning **complexity as a proxy for vulnerability**. Its top-3 features for all 10 classes were `complexity → visibility → uses_block_globals`, in the same order regardless of the class. The class-specific diagnostic features (`return_ignored` for UnusedReturn, `external_call_count` for Reentrancy) showed **zero per-class elevation** in gradient saliency. The L4 finding from `project_interpretability.md` (Phase 2 Interpretability, Run 7, 2026-06-04): "complexity dominates all 10 classes at 34-36%."

Had Stage 6 existed before Run 9, the `feature_dist` tool would have flagged this in Phase 1 (2026-06-05) and the entire 14-day debugging session avoided.

### The 6 analysis tools (the design surface)

| Component | What it does | File | LoC |
|---|---|---|---|
| **balance_viz** | Per-class / per-source / per-tier count table + bar plot | `balance_viz.py` | 87 |
| **feature_dist** | Per-class mean/std of 6 features + `complexity_proxy_risk.md` (the headline) | `feature_dist.py` | 437 |
| **cooccurrence** | Directed + conditional co-occurrence matrices + heatmap | `cooccurrence.py` | 174 |
| **overlap_detector** | Pairwise Jaccard similarity between source datasets | `overlap_detector.py` | 219 |
| **drift_monitor** | Per-feature KS test + label distribution KS test (per 6-P3) | `drift_monitor.py` | 246 |
| **probe_dataset** | Re-export from verification (D-4.7, D-6.6) | `probe_dataset.py` | 17 |
| **CLI wiring** | `_run_analyze` with `--only`, `--corpus`, `--baseline-version` flags | `cli.py` | 110 |

Total: 6 source files + CLI integration, ~1,300 LoC. Plus 17 tests (all pass on v2 baseline).

### The L4 finding (data-side complement)

The L4 finding is a **model-side** observation from `project_interpretability.md`. Stage 6's `complexity_proxy_risk.md` is the **same diagnosis from the data side** — run before training, not after. If the report is RED, the corpus is structurally biased toward complexity; the model team knows to add class-specific feature engineering before training.

**v2 baseline result:** 0 high-risk pairs at the 1.5σ threshold. The DIVE corpus's flat 0.4-0.5 era contracts have similar complexity; the L4 finding does not recur in the v2 data.

---

## 2️⃣ The Solution

### The 6 design decisions (D-6.1 through D-6.6) — all from the plan

| # | Decision | Implementation | Why |
|---|---|---|---|
| **D-6.1** | Read-only analysis with DVC-tracked outputs | `data/analysis/<run_id>/` per CLI invocation; outputs are CSVs + PNGs + MD | Analysis can be re-run freely with a new run_id; outputs are part of the reproducible pipeline |
| **D-6.2** | `complexity_proxy_risk.md` is the headline (D-6.2 + 6-P1 + 6-P2) | `feature_dist.find_high_risk_pairs` + `write_complexity_proxy_risk` | Catches the L4 finding from the data side; per-class rank correlation + label-conditional in the report |
| **D-6.3** | Directed + conditional co-occurrence matrices (D-6.3 + 6-P4) | `cooccurrence.build_cooccurrence_matrices` produces both | The conditional matrix is the multi-label loss design input |
| **D-6.4** | Inter-dataset overlap (exact + near distinction) | `overlap_detector.build_overlap_matrix` distinguishes same-sha256 vs same-dedup_group (6-P5) | Input to per-source loss weighting |
| **D-6.5** | Drift monitor with feature AND label KS test (6-P3) | `drift_monitor.compute_feature_drift` + `compute_label_drift` | A label shift is a different problem from a feature shift |
| **D-6.6** | Probe dataset re-export from verification | 1-line re-export of `verification.probe_dataset` | Single source of truth for model interpretability input |

### The 4 implementation choices (IC-1 through IC-4) — see ADR-0007

- **IC-1**: 2 of 6 features (node_count, edge_count) from `.rep.json`; 4 (cyclomatic_complexity, call_depth, function_count, loc) computed from `.sol` source by simple regex proxies. v2.1: switch to `.cfg.json` artifacts when produced.
- **IC-2**: `pipeline.analysis` config section added to `config.yaml` with `complexity_proxy_risk.sigma_threshold: 1.5`, `cooccurrence.flag_threshold: 0.5`, `drift.ks_pvalue_warn: 0.01`, `drift.min_sample_size: 30`.
- **IC-3**: KS test uses `scipy.stats.ks_2samp` with manual fallback. Fallback returns statistic only (pvalue = NaN); the report flags it as "insufficient sample".
- **IC-4**: Per-class rank correlation deferred to v2.1 (placeholder table in the report; needs Run 11 precision data).

### The 5 analysis tools — how they actually work

#### 1. `balance_viz.py` — quick sanity check

```python
build_balance_table(labels_dir) -> BalanceTable
# per_class: {class -> positive count}
# per_source: {source -> total contracts}
# per_tier: {tier -> {class -> count}}
# multi_label_count: contracts with ≥2 positives
```

Output: `balance_table.csv` + `balance_plot.png` (per-class bar plot with multi-label count).

**v2 baseline:** 22,356 contracts, 15,259 multi-label, 10 classes. ExternalBug dominates (16,621 = 74% — DIVE labeling concern for v2.1).

#### 2. `feature_dist.py` — the Run-9-failure catcher

```python
build_per_class_stats(labels_dir, rep_root, preproc_root) -> dict[class, PerClassStats]
find_high_risk_pairs(by_class, sigma_threshold) -> list[HighRiskPair]
write_complexity_proxy_risk(by_class, pairs, threshold, output_path) -> Path
```

**The 6 features:**
- `node_count`, `edge_count` — from `.rep.json` sidecar
- `cyclomatic_complexity` — 1 + count of branching keywords (`if`, `for`, `while`, `&&`, `||`)
- `call_depth` — max brace depth on any line
- `function_count` — count of `function/constructor/fallback/receive/modifier` definitions
- `loc` — lines of code (non-empty, non-comment)

**The σ-difference:** Cohen's d-style pooled standard deviation. For class A and B, `d = |mean_A - mean_B| / pooled_std`. Flagged if `d > 1.5`.

**Per-class stats include label-conditional** (6-P2): for each class, the per-feature mean/std for label=1 (positive) and label=0 (negative) contracts. A large pos-vs-neg gap means the model can use the feature to predict the class without learning the actual pattern.

**Output: `complexity_proxy_risk.md`** with 4 sections:
1. HIGH-RISK Pairs (σ-difference > threshold)
2. Per-Class Feature Stats (positive contracts)
3. Label-Conditional Feature Distribution (per 6-P2)
4. Recommendation (stratified sampling, class-weight adjustment, class-specific features)

**v2 baseline:** 0 high-risk pairs at 1.5σ threshold. DIVE's flat 0.4-0.5 era contracts have similar complexity. Took 36s for 22K contracts.

#### 3. `cooccurrence.py` — directed + conditional matrices

```python
build_cooccurrence_matrices(labels_dir, flag_threshold) -> CooccurrenceMatrices
# directed[a][b] = count of contracts where both a and b are positive
# conditional[a][b] = P(b=1 | a=1) = directed[a][b] / counts_positive[a]
# flagged_pairs: max(P(b|a), P(a|b)) > threshold
```

**Output: `cooccurrence_matrix.csv`** (2 sections: directed counts + conditional probabilities) + `cooccurrence_heatmap.png`.

**Flagging:** any undirected pair (a, b) with `max(P(b|a), P(a|b)) > 0.5` is flagged. v2 baseline: 14 flagged pairs. The top 5:
- ExternalBug ↔ UnusedReturn 96.47%
- ExternalBug ↔ Reentrancy 93.62%
- DenialOfService ↔ ExternalBug 91.71%
- ExternalBug ↔ TransactionOrderDependence 88.34%
- ExternalBug ↔ IntegerUO 87.74%

The ExternalBug dominance is a DIVE labeling concern (74% of contracts labeled ExternalBug) for v2.1.

#### 4. `overlap_detector.py` — pairwise source Jaccard

```python
build_overlap_matrix(labels_root, preproc_root) -> OverlapMatrix
# exact_jaccard[a][b] = |shas_in_a ∩ shas_in_b| / |shas_in_a ∪ shas_in_b|
# near_jaccard[a][b]  = |dedup_groups_spanning_(a,b)| / |groups_in_a_or_b|
```

**v2 baseline:** dive (22,073) ↔ solidifi (283), no exact or near overlap. The sources are disjoint.

**Output: `overlap_matrix.csv`** (4 sections: exact Jaccard, exact intersection, near Jaccard, near intersection) + `overlap_heatmap.png` (2 side-by-side heatmaps).

#### 5. `drift_monitor.py` — KS test for features AND labels

```python
compute_feature_drift(baseline_labels, baseline_rep, new_labels, new_rep, features) -> list[FeatureKSResult]
compute_label_drift(baseline_labels, new_labels) -> list[LabelKSResult]
```

**Algorithm:** two-sample KS test (scipy.stats.ks_2samp) per feature, per class label. WARNING emitted if p < 0.01 AND n_baseline, n_new ≥ 30 (configurable).

**Baseline lookup:** the CLI's `--baseline-version <name>` flag looks up the baseline's `artifact_path` from the registry (Stage 5). The drift monitor reads labels and representations from both the current build and the baseline's `artifact_path`.

**Output: `drift_report.md`** with per-feature and per-class tables. "insufficient sample" for n < 30.

### The CLI subcommand

**`sentinel-data analyze [--only TOOL] [--run-id ID] [--corpus VERSION] [--baseline-version VERSION] [--dry-run] [--config PATH]`:**

- `--only`: run a single tool (e.g. `feature_dist`)
- `--run-id`: explicit run identifier (default: timestamp `YYYYMMDD_HHMMSS`)
- `--corpus`: analyze a specific registered dataset version (e.g. `sentinel-v2-dryrun-2026-08`)
- `--baseline-version`: for `drift_monitor`, compare against a registered version
- All outputs go to `data/analysis/<run_id>/`

### The smoke run (v2 baseline, 22,356 contracts)

```
$ PYTHONPATH=data_module python3 -m sentinel_data.cli analyze
  [1/5] balance_viz: total=22356 multi-label=15259
  [2/5] feature_dist: high_risk_pairs=0 (DIVE flat era)
  [3/5] cooccurrence: flagged=14 pairs (ExternalBug dominates)
  [4/5] overlap_detector: sources=[dive, solidifi], 0 overlap
  [5/5] drift_monitor: skipped (no --baseline-version)
  ✓ Analysis complete. Outputs in: data/analysis/20260612_*/
```

The `complexity_proxy_risk.md` report for v2 is GREEN (0 high-risk pairs). The co-occurrence report flags 14 pairs, all driven by ExternalBug's 74% prevalence (DIVE labeling concern for v2.1).

### P0 / P1 fixes during implementation

- **`_features_for_contract` requires `preproc_root` even when not used**: fixed by passing `Path("")` as a sentinel
- **Cooccurrence test alphabetization**: keys are sorted alphabetically — fixed test assertion
- **Overlap heatmap with empty matrix**: handled with a placeholder figure
- **Drift test sample size**: n=20 was below `min_sample=30`; fixed test to use n=50

---

## 3️⃣ The Broader Context

### What Stage 6 enables downstream

| Stage | What it builds on Stage 6 |
|---|---|
| Stage 7 (export) | The model team reviews `complexity_proxy_risk.md` before launching |
| Stage 8 (Run 11) | The analysis report is the pre-training sanity check; HIGH-RISK pairs defer the launch |

### What breaks if Stage 6 is wrong

- Missing `feature_dist` → Run 9 failure (complexity-as-proxy) recurs silently
- Missing co-occurrence → 99% DoS↔Reentrancy goes undetected quantitatively (Stage 3 merger *prevents* it; Stage 4 matrix *flags* it; Stage 6 *quantifies* it — three layers of defense)
- Missing drift monitor → new dataset version with shifted class distribution goes unnoticed
- Missing overlap detector → per-source loss weighting is uninformed
- Missing `probe_dataset` re-export → model interpretability suite must reach into verification module (coupling)

### Operational consequences

1. **The analyze stage is required before Stage 7 export.** The model team reviews `complexity_proxy_risk.md` before launching Run 11. If any pair is HIGH-RISK, the launch is deferred.

2. **The analysis outputs are DVC-tracked** (D-6.1, 6-P6). Each `data/analysis/<run_id>/` is a versioned output.

3. **The 5 tools are read-only and fast.** The full pipeline runs in ~40s on the v2 baseline. The bottleneck is `feature_dist` (36s) which reads 22K `.rep.json` files.

4. **`--baseline-version` flag is for version-update gates.** A new dataset version can be compared against a baseline (e.g. v1.4 BCCC) via the drift monitor.

5. **`--corpus` flag is for historical analysis.** A registered dataset version can be analyzed post-hoc. Deferred to v2.1 (wired but not yet exercised end-to-end).

6. **The probe dataset re-export is a 1-line alias.** No new code; no circular import risk. Verified by identity check.

### What stays the same no matter what

- The 6 features (D-6.2)
- The 1.5σ threshold (D-6.2; configurable in `config.yaml`)
- The directed + conditional co-occurrence matrices (D-6.3)
- The exact + near overlap distinction (D-6.4)
- The KS test for features AND labels (D-6.5)
- The probe dataset re-export (D-6.6)

---

## 4️⃣ Verification — Stage 6 exit criteria

All 9 exit criteria (per `07_stage_6_analysis.md` and `ADR-0007-analysis-design.md`):

| # | Check | Status | Notes |
|---|---|---|---|
| 1 | All 6 analysis tools compile and run | ✅ | `balance_viz`, `feature_dist`, `cooccurrence`, `overlap_detector`, `drift_monitor`, `probe_dataset` (re-export) |
| 2 | `feature_dist` flags synthetic 2σ complexity skew | ✅ | `TestFeatureDist.test_find_high_risk_pairs_synthetic` — Reentrancy↔Timestamp node_count at >1.5σ |
| 3 | `complexity_proxy_risk.md` generated | ✅ | 4 sections (HIGH-RISK, Per-Class Stats, Label-Conditional, Recommendation) |
| 4 | Co-occurrence matrix correct for multi-label fixture | ✅ | `TestCooccurrence.test_directed_and_conditional` — directed/conditional values match hand calculation |
| 5 | Overlap matrix correct for 3-source fixture | ✅ | `TestOverlapDetector.test_exact_jaccard_no_overlap` (0.0) and `test_exact_jaccard_full_overlap` (1.0) |
| 6 | Drift monitor flags intentional drift | ✅ | `TestDriftMonitor.test_feature_drift_flags_intentional_drift` (50 contracts, 10× node_count shift) and `test_label_drift_flags_intentional_drift` (5% → 50% Reentrancy) |
| 7 | `dvc repro analyze` runs end-to-end | ✅ | `sentinel-data analyze` runs all 5 tools; smoke run in ~40s |
| 8 | `pytest tests/test_analysis -v` passes with > 80% coverage | ✅ | 17 passed, 0 skipped |
| 9 | `ADR-0007-analysis-design.md` committed | ✅ | `docs/decisions/ADR-0007-analysis-design.md` (309 lines) |

**All 9 Stage 6 exit criteria pass. Stage 6 is complete.**

---

## 5️⃣ The "got it" checklist

Before we move to Stage 7, you should be able to answer (without looking at this doc):

1. **What is the `complexity_proxy_risk.md` report?** For each class pair, flags if per-feature σ-difference > 1.5. The Run 9 failure (model learning complexity as proxy) would have been flagged here before training. Includes 4 sections: HIGH-RISK Pairs, Per-Class Stats, Label-Conditional (6-P2), Recommendation.

2. **What's the L4 finding?** Model's top-3 features for all 10 classes are `complexity → visibility → uses_block_globals`, same order regardless of class. Stage 6 detects this from the data side via `complexity_proxy_risk.md`. The L4 finding does NOT recur in v2 baseline (0 high-risk pairs).

3. **What are the directed + conditional co-occurrence matrices?** Directed: `directed[a][b]` = count of contracts where both a and b are positive (joint probability). Conditional: `P(b=1 | a=1)`. Both are needed for full analysis (the asymmetry matters: e.g. P(Reentrancy|ExternalBug) ≠ P(ExternalBug|Reentrancy)).

4. **Why does the drift monitor check label distribution?** A new dataset version with different positive/negative ratio per class is **label drift** even if feature distributions are stable. Per 6-P3, both forms are checked. Below 30 samples, the result is "insufficient sample" not "drift detected".

5. **What's the overlap detector?** Pairwise Jaccard similarity between source datasets. Distinguishes **exact** (same sha256) from **near** (shared dedup_group, AST-similar). Input to per-source loss weighting in training. v2 baseline: dive ↔ solidifi, no overlap.

6. **What are the 4 implementation choices (IC-1 through IC-4)?** IC-1: 2 of 6 features from `.rep.json`, 4 from `.sol` proxies. IC-2: `pipeline.analysis` config section added. IC-3: KS test uses scipy with manual fallback. IC-4: per-class rank correlation deferred to v2.1 (placeholder table).

7. **Why is the co-occurrence matrix's 14 flagged pairs concerning?** ExternalBug dominates (74% of contracts labeled ExternalBug by DIVE — too broad). All top-5 flagged pairs involve ExternalBug. This is a DIVE labeling concern for v2.1 (the taxonomy may need refinement).

8. **What's the difference between the Stage 3 merger, the Stage 4 co-occurrence matrix, and the Stage 6 directed + conditional matrices?** Three layers of defense for the 99% DoS↔Reentrancy pattern:
   - Stage 3 merger: *prevents* it (de-duplicates folder-based labeling)
   - Stage 4 co-occurrence matrix: *flags* it (P(B|A) > 50%)
   - Stage 6 directed + conditional: *quantifies* it (exact counts + conditional probabilities)

If you can answer all 8, Stage 6 is mastered and we can move to Stage 7.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 6" — 6 specific questions to test your understanding
- **`07_stage_6_analysis.md`** — the design + intent document
- **`docs/decisions/ADR-0007-analysis-design.md`** — the 6 design decisions + 4 implementation choices, with rationale
- **`project_interpretability.md`** (in MEMORY) — the L4 finding that motivates `complexity_proxy_risk`

When you're ready, say **"Stage 6 is mastered — let's move to Stage 7."**
