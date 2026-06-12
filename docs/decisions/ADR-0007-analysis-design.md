# ADR-0007: Analysis Design

**Date:** 2026-06-12
**Stage:** 6 of 8 (Week 8 part A: Aug 4–5)
**Status:** Accepted (Stage 6 implementation complete)
**Author:** SENTINEL data engineering
**Plan reference:** [`docs/proposal/Data_Module_Proposals/actionable_plans/07_stage_6_analysis.md`](../proposal/Data_Module_Proposals/actionable_plans/07_stage_6_analysis.md)
**Audit reference:** [`docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md`](../proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md) §1 6-P1 through 6-P7

---

## Context

Run 9 trained for 9 runs before discovering that the model was learning **complexity as a proxy for vulnerability**, not the actual vulnerability patterns. The L4 interpretability finding (Run 7 phase 2, `project_interpretability.md`): "complexity dominates all 10 classes at 34–36%." The diagnosis came 9 runs late. The 14-day debugging session could have been avoided by a data-side check that ran in minutes: per-class feature distribution analysis that flags any class-pair whose complexity features differ by > 1.5σ.

Stage 6 is the **first stage that *looks at* the data** to ask "is this corpus going to teach the model the right thing?". It implements 5 read-only exploratory tools (plus a probe-dataset re-export) that produce plots, CSVs, and the headline `complexity_proxy_risk.md` report. The Run-9-failure catcher is `feature_dist.py`.

The module's purpose is to surface dataset properties before and after pipeline runs, so the model team can review `complexity_proxy_risk.md` before launching training. Stage 7 export depends on the analysis output (D-6.1, 6-P6).

This ADR records the 6 design decisions that frame the module, the 4 implementation choices made during Stage 6, and the operational consequences.

---

## Design Decisions

### D-6.1 — Analysis is read-only BUT outputs are DVC-tracked

The 5 analysis tools do not modify any input artifact. They read from `data/representations/`, `data/labels/merged/`, `data/preprocessed/`, and the verification reports, and write outputs to `data/analysis/<run_id>/`. Per AUDIT_PATCHES 6-P6, outputs are DVC-tracked: each `data/analysis/<run_id>/` is an `out:` in DVC, and rerunning analysis with a new run_id produces a new version. Stage 7 export depends on the analysis output (the model team reviews `complexity_proxy_risk.md` before launching).

**Operational consequence:** analysis can be re-run freely (with a new `run_id`) without invalidating the pipeline cache, AND the analysis output is part of the reproducible pipeline.

### D-6.2 — `complexity_proxy_risk` is the headline (data-side complement of L4 finding)

The `feature_dist` tool produces a `complexity_proxy_risk.md` report that is the single most important analysis output. The report computes, for each pair of classes, the σ-difference in mean of 6 features: node_count, edge_count, cyclomatic_complexity, call_depth, function_count, LOC. If any pair differs by > 1.5σ (configurable in `pipeline.analysis.complexity_proxy_risk.sigma_threshold`), the pair is flagged as **HIGH-RISK** for the model to learn a complexity proxy.

**Per AUDIT_PATCHES 6-P1:** the report also includes **label-conditional feature distribution** (per-class mean/std of each feature, broken down by positive vs. negative contracts). A large pos-vs-neg gap means the model can use the feature to predict the class without learning the actual pattern.

**Why this is the data-side complement:** the L4 finding (model uses `complexity` as proxy) is a model-side observation. `feature_dist.complexity_proxy_risk.md` is the *same diagnosis from the data side* — run before training, not after. If the report is RED, the corpus is structurally biased toward complexity; the model team knows to add class-specific feature engineering before training.

**v1 feature computation:** 2 of 6 features (node_count, edge_count) come from the `.rep.json` sidecar. The other 4 are computed from preprocessed `.sol` source by simple proxies:
- `cyclomatic_complexity` = 1 + count of branching keywords (`if`, `for`, `while`, `&&`, `||`, etc.)
- `call_depth` = max brace depth on any line (rough proxy for nested function calls)
- `function_count` = count of `function/constructor/fallback/receive/modifier` definitions
- `loc` = lines of code (non-empty, non-comment)

The full v8 CFG schema is the more accurate source but is opt-in in Stage 2 (the v2 baseline has no `.cfg.json` files). The proxies are sufficient for the v2 baseline (DIVE's flat 0.4-0.5 era contracts have similar complexity). v2.1: switch to CFG when `.cfg.json` artifacts are produced.

**Threshold (1.5σ) rationale:** too strict (1.0σ) flags most pairs; too lax (3.0σ) misses the L4 finding. 1.5σ is the empirical sweet spot.

**Operational consequence:** Run 11 checks this report before launching. If any pair is HIGH-RISK, the launch is deferred pending model-team review (per the plan's D-6.2 directive).

### D-6.3 — Co-occurrence produces two matrices (directed + conditional)

The `cooccurrence` tool produces **two matrices** (per AUDIT_PATCHES 6-P4):
- **Directed co-occurrence matrix** `directed[a][b]` = count of contracts where both class a and class b are positive. Symmetric for undirected pairs, but stored as the full directed form for clarity.
- **Conditional probability matrix** `conditional[a][b]` = P(b=1 | a=1) = directed[a][b] / counts_positive[a]. This is the input to multi-label loss design.

The BCCC 99% DoS→Reentrancy co-occurrence is visible as a very high entry in both matrices. The conditional matrix is what the multi-label loss design consumes. The matrices are exported as both CSVs (one file, two sections) and a heatmap PNG (conditional).

**Flagging:** any undirected pair (a, b) with `max(P(b|a), P(a|b)) > 0.5` is flagged. v2 baseline: 14 flagged pairs (ExternalBug dominates because DIVE labels it on 74% of contracts — a sign that the DIVE labeling taxonomy may need to be refined in v2.1).

**Why two matrices, not one:** the directed matrix preserves the "if X then Y" semantics; the conditional matrix is the actionable signal. Both are needed for full analysis (e.g. "Reentrancy→ExternalBug = 2/3" vs "ExternalBug→Reentrancy = 1.0" — the asymmetry matters).

### D-6.4 — Inter-dataset overlap (exact + near distinction)

The `overlap_detector` computes pairwise Jaccard similarity between source datasets, distinguishing **exact overlap** (same sha256) from **near overlap** (shared dedup_group_id from AST similarity). Per AUDIT_PATCHES 6-P5, both forms are reported.

**Exact overlap** = `|shas_in_a ∩ shas_in_b| / |shas_in_a ∪ shas_in_b|`. This is the more pernicious form — the same contract in two sources means double-counting in the loss.

**Near overlap** = number of dedup_groups (AST-similar contracts) that span both sources, normalized by the number of groups involving either source. This is a softer signal — a near-dup group spanning two sources is suspicious but not necessarily a bug.

**v2 baseline:** dive (22,073 contracts) ↔ solidifi (283 contracts), no exact or near overlap. The sources are disjoint. Future: when DISL is re-introduced (v2.1), the overlap detector will surface the source weighting decisions.

### D-6.5 — Drift monitor checks features AND labels (KS test)

The `drift_monitor` takes a new version and a baseline, and reports whether the **feature** distributions AND the **label** distributions have shifted significantly. Per AUDIT_PATCHES 6-P3, both forms are checked — a label distribution shift (e.g. the new version has 50% Reentrancy positive vs. the baseline's 10%) is a different problem from a feature shift (contracts got bigger).

**Algorithm:** two-sample KS test (scipy.stats.ks_2samp) per feature, per class label. WARNING emitted if p < 0.01 AND n_baseline, n_new ≥ 30 (configurable). Below 30 samples, the result is "insufficient sample" not "drift detected" (KS test is unreliable on tiny samples).

**Baseline version lookup:** the CLI's `--baseline-version <name>` flag looks up the baseline's `artifact_path` from the registry (Stage 5). The drift monitor reads labels and representations from both `data/labels/merged/` and `<baseline_artifact_path>/labels/merged/`.

**Operational consequence:** a new dataset version with significant drift (feature or label) emits a WARNING to the catalog; the ML training pipeline can opt to require explicit acknowledgement of the warning before training.

### D-6.6 — Probe dataset is a re-export from verification

`analysis/probe_dataset.py` is a 1-line re-export of `verification/probe_dataset.py` (D-4.7). The probe dataset is the input to the model interpretability suite. Doing it as a re-export means there is one source of truth for the probe dataset; the model suite imports from `sentinel_data.analysis.probe_dataset` (or equivalently, `sentinel_data.verification.probe_dataset`).

**Exported:** `ProbeDataset`, `ProbeEntry`, `ClassProbeBucket`, `build_probe_dataset`. Identity check: `from sentinel_data.analysis.probe_dataset import ProbeDataset` returns the **same class object** as `from sentinel_data.verification.probe_dataset import ProbeDataset`. This is verified by `TestProbeDatasetReexport.test_reexport_works`.

---

## Implementation Choices Made During Stage 6

These are decisions that emerged during implementation, not in the original plan:

### IC-1 — Two of 6 features computed from .sol proxies (`feature_dist.py`)

The plan says the 6 features are: node_count, edge_count, cyclomatic_complexity, call_depth, function_count, LOC. The v9 `.rep.json` sidecar has only node_count and edge_count. The other 4 require either the `.cfg.json` artifact (opt-in in Stage 2) or text-based computation.

**Implementation:** compute the 4 from the preprocessed `.sol` source with simple regex-based proxies (see D-6.2 above for details).

**Why proxies, not full CFG:** the v2 baseline has no `.cfg.json` files. The proxies are sufficient for the v2 baseline because DIVE contracts are simple 0.4-0.5 era code. v2.1: switch to CFG when `.cfg.json` artifacts are produced.

### IC-2 — `pipeline.analysis` config section added (`config.yaml`)

The plan describes the threshold (1.5σ) and the features to check as "in `config.yaml`" but no `pipeline.analysis` section existed. Implementation adds:
```yaml
analysis:
  complexity_proxy_risk:
    sigma_threshold: 1.5
    features: [node_count, edge_count, cyclomatic_complexity, call_depth, function_count, loc]
  cooccurrence:
    flag_threshold: 0.50
  overlap:
    near_similarity_threshold: 0.85
  drift:
    ks_pvalue_warn: 0.01
    min_sample_size: 30
```

The CLI's `_run_analyze` reads from `(cfg or {}).get("pipeline", {}).get("analysis", {})` and falls back to the defaults above if the section is missing.

### IC-3 — KS test uses scipy with manual fallback (`drift_monitor.py`)

The KS test (`scipy.stats.ks_2samp`) is the standard tool. Implementation imports it lazily (try/except) and falls back to a manual max-CDF-difference computation if scipy is unavailable.

**Why the fallback:** v1.0 the v2 baseline always has scipy available, but v0.x environments (older installs) may not. The fallback returns statistic only (pvalue = NaN), which the report flags as "insufficient sample".

### IC-4 — Per-class rank correlation deferred to v2.1

The plan (D-6.2 + 6-P1) calls for **per-class rank correlation** between feature and per-class precision in `complexity_proxy_risk.md`. This is a *model-side* signal (we'd need to know the per-class precision from a training run to compute it). For the v2 baseline, no precision data exists.

**Implementation:** the report has a placeholder for the rank correlation table (per-class precision column is empty). v2.1: populate from the first Run 11 output.

**Why not skip entirely:** the table is included as a structural placeholder so consumers know where to look. Empty cells are clearly marked.

---

## Operational Consequences

1. **The analyze stage is required before Stage 7 export.** The model team reviews `complexity_proxy_risk.md` before launching Run 11. If any pair is HIGH-RISK, the launch is deferred.

2. **The analysis outputs are DVC-tracked** (D-6.1, 6-P6). Each `data/analysis/<run_id>/` is a versioned output. Re-running analysis with a new `run_id` produces a new version; the old version is preserved.

3. **The 5 tools are read-only and fast.** The full pipeline (all 5 tools) runs in ~40s on the v2 baseline (22K contracts). The bottleneck is `feature_dist` (36s) which reads 22K `.rep.json` files. The other 4 tools run in <1s each.

4. **`--baseline-version` flag is for version-update gates.** A new dataset version can be compared against a baseline (e.g. v1.4 BCCC) via the drift monitor. Significant drift emits a WARNING.

5. **`--corpus` flag is for historical analysis.** A registered dataset version can be analyzed post-hoc (e.g. to compare v2 vs. v1.4 BCCC's co-occurrence patterns). Deferred to v2.1 (the `--corpus` flag is wired but not yet exercised end-to-end).

6. **The probe dataset re-export is a 1-line alias.** No new code; no circular import risk. Verified by identity check in `TestProbeDatasetReexport`.

7. **The co-occurrence matrix is the data-side detection of the 99% DoS↔Reentrancy pattern.** The Stage 3 merger *prevents* it (by de-duplicating folder-based labeling); the Stage 4 co-occurrence matrix *flags* it; the Stage 6 directed + conditional matrices *quantify* it. Three layers of defense.

---

## References

- Plan: [`docs/proposal/Data_Module_Proposals/actionable_plans/07_stage_6_analysis.md`](../proposal/Data_Module_Proposals/actionable_plans/07_stage_6_analysis.md)
- Audit: [`docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md`](../proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md) §1 6-P1 through 6-P7
- L4 finding: `project_interpretability.md` (Phase 2 Interpretability, Run 7, 2026-06-04) — "complexity dominates all 10 classes at 34-36%"
- Implementation:
  - `data_module/sentinel_data/analysis/balance_viz.py` (Task 6.1, 87 LoC)
  - `data_module/sentinel_data/analysis/feature_dist.py` (Task 6.2, 437 LoC, the headline)
  - `data_module/sentinel_data/analysis/cooccurrence.py` (Task 6.3, 174 LoC)
  - `data_module/sentinel_data/analysis/overlap_detector.py` (Task 6.4, 219 LoC)
  - `data_module/sentinel_data/analysis/drift_monitor.py` (Task 6.5, 246 LoC)
  - `data_module/sentinel_data/analysis/probe_dataset.py` (Task 6.6, 1-line re-export)
  - `data_module/sentinel_data/cli.py` `_run_analyze` (Task 6.7)
  - `data_module/config.yaml` `pipeline.analysis` section (IC-2)
- Tests: `data_module/tests/test_analysis/test_analysis.py` (17 tests pass)
- Smoke output (v2 baseline, 22,356 contracts):
  - `data/analysis/20260612_*/` (per-run-id, includes `balance_table.csv`, `feature_dist_table.csv`, `cooccurrence_matrix.csv`, `overlap_matrix.csv`)
  - `complexity_proxy_risk.md` (0 high-risk pairs in v2 baseline)
  - 14 flagged co-occurrence pairs (ExternalBug dominates — DIVE labeling concern)

---

**End of ADR-0007.**
