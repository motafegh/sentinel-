# Actionable Plan — Stage 6: Analysis

**Date:** 2026-07-21
**Stage:** 6 of 8 (Week 7 part A: Jul 21–22)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.9, §5 (Week 7 part A)
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §0 (F27, F30), §1 (6-P1 through 6-P7), §2 (C-8, C-9, C-10)
**Exit criteria:** all 6 analysis tools run against the ScaBench fixture and produce sensible outputs; the `feature_dist` tool correctly flags the Phase 4 Reentrancy complexity skew on a synthetic test; the `complexity_proxy_risk.md` report is generated for the merged dataset; **the co-occurrence matrix is the directed + conditional form (catches 99% DoS↔Reentrancy)**; **label-conditional feature distribution is computed (catches the L4 complexity-proxy finding on the data side)**.

---

## Goal

Implement the **Analysis** submodule: the 6 exploratory tools that surface dataset properties before and after pipeline runs. After this stage, the `sentinel-data analyze` command produces plots, CSVs, and a `complexity_proxy_risk.md` report that catches the Run 9 failure mode (model learning complexity as a proxy for vulnerability) before training.

Stage 6 is short (2 days) because the analysis tools are mostly read-only operations over the artifacts produced by Stages 1–5. The hard work is the `feature_dist` tool, which is what would have caught the Run 9 issue automatically.

---

## Why this stage seventh

Stages 1–5 produced preprocessed, represented, labeled, verified, split, and registered data. Stage 6 is the first stage that *looks at* the data to ask "is this corpus going to teach the model the right thing?" — the question that Run 9 answered with "no" only after 9 training runs.

The `feature_dist` tool is what the BCCC failure needed. Had it existed, the per-class node-count / edge-count / cyclomatic complexity skew would have been flagged in Phase 1 (2026-06-05) and the entire 14-day debugging session avoided.

---

## Design decisions

### D-6.1 — Analysis is read-only BUT outputs are DVC-tracked

The 6 analysis tools do not modify any input artifact. They read from `data/representations/`, `data/labels/merged/`, `data/splits/`, and the verification reports, and write plots + CSVs + markdown reports to `data/analysis/<run_id>/`. **Per AUDIT_PATCHES 6-P6, the outputs are DVC-tracked** (each `data/analysis/<run_id>/` output is an `out:` in DVC; rerunning analysis produces a new version). The Stage 7 export depends on the analysis output (the model team reviews the `complexity_proxy_risk.md` report before launching).

This decision means analysis can be re-run freely (with a new run_id) without invalidating the pipeline cache, AND the analysis output is part of the reproducible pipeline.

### D-6.2 — The `complexity_proxy_risk` is the headline output (data-side complement of L4 finding)

The `feature_dist` tool produces a `complexity_proxy_risk.md` report that is the single most important analysis output. The report computes, for each pair of classes, the difference in mean (and std) of: node count, edge count, cyclomatic complexity, call depth, function count, LOC. If any pair differs by > 1.5σ, the pair is flagged as HIGH-RISK for the model to learn a complexity proxy instead of a class-specific pattern.

**Per AUDIT_PATCHES 6-P1, 6-P2:** the report also includes a **per-class rank correlation** between the feature (e.g. node count) and the per-class precision. If a class's precision correlates strongly with the feature, the model is using that feature as a proxy. The tool also computes the **label-conditional feature distribution** — for each class, the per-feature mean/std/median broken down by positive (label=1) and negative (label=0) contracts. The "complexity proxy" finding from `project_interpretability.md` ("complexity dominates all 10 classes at 34-36%") would have been visible as a strong per-class difference between positive and negative feature distributions.

**Why this is the data-side complement:** the L4 finding (model uses `complexity` as proxy) is a model-side observation from interpretability. The `feature_dist.complexity_proxy_risk.md` is the *same diagnosis from the data side* — run before training, not after. If the report is RED, the corpus is structurally biased toward complexity; the model team knows to add class-specific feature engineering before training.

The threshold (1.5σ) and the features to check are in `config.yaml`. The report is the gate that the model training pipeline can opt to check before launching (Run 11 will check it).

### D-6.3 — Class co-occurrence is the multi-label loss design input (directed + conditional form)

The `cooccurrence` tool produces **two matrices** (per AUDIT_PATCHES 6-P4, C-10):
- **Directed co-occurrence matrix** (X→Y means "if class X is positive, class Y is also positive with probability p")
- **Conditional probability matrix** (P(Y=1 | X=1))

The BCCC 99% DoS→Reentrancy co-occurrence is visible as a very high entry in both matrices. The conditional matrix is what the multi-label loss design consumes. The matrices are exported as both CSVs and heatmap PNGs.

This is the data-side detection of the 99% DoS↔Reentrancy co-occurrence that the Stage 3 merger *prevents*. The merger is the fix; the analysis is the verification.

### D-6.4 — Inter-dataset overlap is the source-weighting input

The `overlap_detector` computes pairwise overlap between source datasets (e.g. is SmartBugs Curated a subset of SmartBugs Wild? is OZ Contracts in any audit dataset?). The output is a Jaccard-similarity matrix per source pair. This is the input to the per-source loss weighting in the model training pipeline.

### D-6.5 — Drift monitor is the version-update gate (label distribution drift included)

The `drift_monitor` takes a new dataset version and a baseline, and reports whether the feature distributions have shifted significantly. The output is a `drift_report.md` with per-feature KS test results **AND label distribution KS test results** (per AUDIT_PATCHES 6-P3). If the new version has a different positive/negative ratio per class, that's a label drift even if the feature distributions are stable. A new dataset version with significant drift (feature or label) emits a WARNING to the catalog; the ML training pipeline can opt to require explicit acknowledgement of the warning before training.

### D-6.6 — Probe dataset is a re-export from verification

The `probe_dataset` in `analysis/` is an alias / re-export of `sentinel_data.verification.probe_dataset` (D-4.7). The probe dataset is the input to the model interpretability suite (existing in `ml/scripts/interpretability/`). Doing it as a re-export means there is one source of truth for the probe dataset; the model suite imports from `sentinel_data.analysis.probe_dataset` (or equivalently, `sentinel_data.verification.probe_dataset`).

---

## Tasks — ordered, each with verifiable exit condition

### 6.1 — Implement `balance_viz.py`

Author `sentinel_data/analysis/balance_viz.py` that reads the merged labels and produces a per-class / per-source / per-confidence-tier count table and bar plot. The output: `data/analysis/<run_id>/balance_table.csv` + `balance_plot.png`.

**Exit condition:** runs against the ScaBench fixture; produces correct counts; plot renders without error.

**Commit:** `feat(data-analysis): add balance_viz with per-class/per-source/per-tier counts`

---

### 6.2 — Implement `feature_dist.py` (the Run-9-failure catcher)

Author `sentinel_data/analysis/feature_dist.py` that computes per-class feature distributions (node count, edge count, cyclomatic complexity, call depth, function count, LOC) from the graph `.rep.json` sidecars. Produces:
- `feature_dist_table.csv` — per-class mean/std/min/max/median for each feature
- `feature_dist_plot.png` — boxplot per class per feature
- `complexity_proxy_risk.md` — the headline report with the per-class-pair risk assessment (D-6.2)

The risk assessment flags any pair where the per-feature σ-difference > threshold; the report explains what this means and recommends mitigation (e.g. "Reentrancy positives are 2.3σ more complex than Reentrancy negatives — consider stratified sampling or class-weight adjustment in the loss").

**Why this is the headline tool:** the Run 9 failure was a model learning complexity as a proxy. This tool would have caught it before training.

**Exit condition:** runs against the ScaBench fixture; produces all 3 outputs; correctly flags a synthetic test case where one class has 2σ higher node count than another.

**Commit:** `feat(data-analysis): add feature_dist with complexity_proxy_risk report`

---

### 6.3 — Implement `cooccurrence.py`

Author `sentinel_data/analysis/cooccurrence.py` that produces the per-class co-occurrence matrix for multi-label contracts. Output: `cooccurrence_matrix.csv` + `cooccurrence_heatmap.png`.

**Exit condition:** runs against a fixture with multi-label contracts; matrix is correct (verified by hand-counted reference).

**Commit:** `feat(data-analysis): add cooccurrence matrix + heatmap`

---

### 6.4 — Implement `overlap_detector.py`

Author `sentinel_data/analysis/overlap_detector.py` that computes pairwise Jaccard similarity between source datasets based on shared `contract_id`s. Output: `overlap_matrix.csv` + `overlap_heatmap.png`.

**Exit condition:** runs against a 3-source fixture with known overlaps; matrix is correct.

**Commit:** `feat(data-analysis): add overlap_detector for pairwise source similarity`

---

### 6.5 — Implement `drift_monitor.py`

Author `sentinel_data/analysis/drift_monitor.py` that takes a new dataset version + a baseline, and reports per-feature KS test results. Output: `drift_report.md` with WARNINGs for any feature with p < 0.01.

**Exit condition:** runs against two fixture versions (one with intentional drift); correctly flags the drifted features.

**Commit:** `feat(data-analysis): add drift_monitor with KS test per feature`

---

### 6.6 — Add `probe_dataset` re-export

Add `sentinel_data/analysis/probe_dataset.py` that re-exports from `sentinel_data.verification.probe_dataset`. This is the alias per D-6.6.

**Exit condition:** `from sentinel_data.analysis.probe_dataset import load_probe_dataset` works and returns the same data as `from sentinel_data.verification.probe_dataset import load_probe_dataset`.

**Commit:** `feat(data-analysis): re-export probe_dataset from verification`

---

### 6.7 — Wire the `sentinel-data analyze` CLI subcommand (with `--corpus` flag for historical analysis)

Connect `cli.py` `analyze` subcommand to the 6 tools. The CLI runs all 6 tools by default; flags allow individual runs (`--only feature_dist`). Add `--baseline-version <name>` to `drift_monitor` for the version-comparison case. **Add `--corpus <version>` flag (per AUDIT_PATCHES 6-P7) to analyze a specific dataset version, not just the current build.** The historical analysis of the BCCC v1.4 corpus is a useful baseline (compares Run 11's v2 corpus against the v1.4 BCCC baseline).

Update `dvc.yaml` stage `analyze` to call `sentinel-data analyze`. The `analyze` stage is a soft gate for Stage 7 export (the model team reviews the `complexity_proxy_risk.md` report before launching).

**Exit condition:** `sentinel-data analyze` runs all 6 tools and produces all outputs; `complexity_proxy_risk.md` is generated; `--corpus` flag works for historical analysis.

**Commit:** `feat(data-analysis): wire CLI + DVC for the analyze stage (with --corpus flag)`

---

### 6.8 — Add tests for the analysis stage

Author `Data/tests/test_analysis/` with:
- **balance_viz tests** — counts match the merged labels
- **feature_dist tests** — per-class statistics are correct; complexity_proxy_risk correctly flags the synthetic test
- **cooccurrence tests** — matrix is correct
- **overlap_detector tests** — Jaccard similarity is correct
- **drift_monitor tests** — KS test correctly flags intentional drift

**Exit condition:** `poetry run pytest tests/test_analysis -v` passes; coverage > 80%.

**Commit:** `test(data-analysis): add full test suite for analysis stage`

---

### 6.9 — Author `ADR-0007-analysis-design.md`

Document the key design decisions: read-only analysis with DVC-tracked outputs (D-6.1), complexity_proxy_risk as the headline (D-6.2), directed + conditional co-occurrence matrices (D-6.3), overlap as source-weighting input (D-6.4), drift with label distribution (D-6.5), probe dataset re-export (D-6.6). **Cites the L4 finding (complexity dominates all 10 classes) as the motivation for the data-side `complexity_proxy_risk` check** (per AUDIT_PATCHES C-9, F30).

**Exit condition:** file exists; cites the L4 finding from `project_interpretability.md` as motivation; references the 99% DoS↔Reentrancy co-occurrence as the motivation for the directed + conditional co-occurrence matrices.

**Commit:** `docs(data): add ADR-0007 for analysis design`

---

## What NOT to fix (preservation list)

| Bug / Decision | Status | File:line | Stage 6 action |
|---|---|---|---|
| 25 model-side interpretability scripts | ✅ EXIST | `ml/scripts/interpretability/` | Do not duplicate in `sentinel_data` per AUDIT_PATCHES C-8. The data-side analysis (Stage 6) is a *complement* to the model-side interpretability suite. The hand-off is: data-side analysis identifies potential issues; model-side interpretability confirms whether the model is using the issue. |
| L4 finding ("complexity dominates all 10 classes") | ✅ DOCUMENTED | `project_interpretability.md` + `ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04/` | The data-side `complexity_proxy_risk.md` is the *same diagnosis from the data side*. Stage 6 cites the L4 finding; does not re-derive it. |
| Per-class F1 ceilings (Run 9 best ep52) | ✅ DOCUMENTED | MEMORY.md "Training History" | The Stage 6 analysis predicts per-class F1 trajectory from the data; the model's actual F1 is the model team's concern. |
| 99% DoS↔Reentrancy co-occurrence in BCCC | Source: BCCC | (not in v2 corpus) | The Stage 3 merger de-duplicates; the Stage 4 co-occurrence matrix flags it; the Stage 6 directed + conditional co-occurrence matrices quantify it. Three layers of defense. |

## Final exit criteria check

| # | Check |
|---|---|
| 1 | All 6 analysis tools (balance_viz, feature_dist, cooccurrence, overlap_detector, drift_monitor, probe_dataset) compile and run |
| 2 | `feature_dist` correctly flags the synthetic test where one class has 2σ higher node count than another |
| 3 | `complexity_proxy_risk.md` is generated for the ScaBench fixture |
| 4 | The co-occurrence matrix is correct for a multi-label fixture |
| 5 | The overlap matrix is correct for a 3-source fixture |
| 6 | The drift monitor correctly flags intentional drift in a 2-version fixture |
| 7 | `dvc repro analyze` runs end-to-end |
| 8 | `poetry run pytest tests/test_analysis -v` passes with > 80% coverage |
| 9 | `ADR-0007-analysis-design.md` is committed |

All 9 pass → **Stage 6 complete**. Tag `data-stage-6`, proceed to Stage 7.

---

## Risk register

| Risk | Mitigation |
|---|---|
| The `feature_dist` tool's per-class σ-difference calculation is not statistically rigorous (e.g. assumes normality) | The report uses KS test in addition to σ-difference; the report explicitly notes the assumption and recommends non-parametric checks for skewed distributions |
| The complexity_proxy_risk threshold (1.5σ) is too strict or too lenient | The threshold is in `config.yaml`; the test in 6.8 validates the threshold against a synthetic case; the report recommends a follow-up review of any flagged pair |
| The drift monitor's KS test is sensitive to sample size (over-flags small datasets) | The drift monitor reports p-value AND effect size; small datasets with high p-values are flagged as "insufficient sample" rather than "drift detected" |
| The analysis tools are slow on large corpora (41K contracts) | The tools parallelize per-class analysis; the worst case is ~10 min on 8 cores for the full v2 corpus |
| The probe dataset re-export is a circular import risk | The re-export is a one-line import, not a re-implementation; no circular risk |

---

**End of Stage 6 actionable plan. Total estimated time: 2 working days (Jul 21–22).**
