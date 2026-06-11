# Stage 6 — Analysis (complexity_proxy_risk)

**Date:** 2026-07-21
**Status:** NOT STARTED. Reading required before Stage 7.
**Reading time:** 15-20 minutes.
**Goal:** After this doc, you can answer all 6 items in `LEARNING_CHECKLIST.md` §"Stage 6" from memory.

---

## 1️⃣ The Problem

### What Stage 6 has to deliver

Stages 1–5 produced preprocessed, represented, labeled, verified, split, and registered data. Stage 6 **looks at** the data to ask: "is this corpus going to teach the model the right thing?"

The Run 9 failure was a model learning **complexity as a proxy for vulnerability**. Its top-3 features for all 10 classes were `complexity → visibility → uses_block_globals`, in the same order, regardless of the class. The class-specific diagnostic features (`return_ignored` for UnusedReturn, `external_call_count` for Reentrancy) showed **zero per-class elevation** in gradient saliency.

Had Stage 6 existed before Run 9, the `feature_dist` tool would have flagged this in Phase 1 (2026-06-05) and the entire 14-day debugging session avoided.

### The L4 finding (data-side complement)

The L4 finding from `project_interpretability.md` ("complexity dominates all 10 classes at 34-36%") is a **model-side** observation. Stage 6's `complexity_proxy_risk.md` is the **same diagnosis from the data side** — run before training, not after. If the report is RED, the corpus is structurally biased toward complexity; the model team knows to add class-specific feature engineering before training.

---

## 2️⃣ The Solution

### The 6 analysis tools

| Tool | What it produces | Why it matters |
|---|---|---|
| **balance_viz** | Per-class / per-source / per-tier count table + bar plot | Quick sanity check on class distribution |
| **feature_dist** | Per-class mean/std of node count, edge count, cyclomatic complexity, call depth, function count, LOC + `complexity_proxy_risk.md` | **The Run-9-failure catcher** |
| **cooccurrence** | Directed + conditional co-occurrence matrix + heatmap | Catches 99% DoS↔Reentrancy quantitatively |
| **overlap_detector** | Pairwise Jaccard similarity between sources | Source-weighting input for training |
| **drift_monitor** | Per-feature KS test + label distribution drift | Version-update gate |
| **probe_dataset** | Re-export from verification | Model interpretability input |

### The `complexity_proxy_risk.md` report (D-6.2)

For each pair of classes, computes the difference in mean (and std) of: node count, edge count, cyclomatic complexity, call depth, function count, LOC. If any pair differs by > 1.5σ, flagged as HIGH-RISK.

Also computes **per-class rank correlation** between features and per-class precision. If a class's precision correlates strongly with a feature, the model is using that feature as a proxy.

Also computes **label-conditional feature distribution** — for each class, per-feature mean/std/median broken down by positive and negative contracts.

### Directed + conditional co-occurrence (D-6.3)

Two matrices:
- **Directed**: X→Y means "if class X is positive, class Y is also positive with probability p"
- **Conditional**: P(Y=1 | X=1)

The BCCC 99% DoS→Reentrancy co-occurrence is visible as a very high entry. The conditional matrix is what the multi-label loss design consumes.

### Drift monitor with label distribution (D-6.5)

The drift monitor reports per-feature KS test **AND** label distribution KS test. A new dataset version with different positive/negative ratio per class is label drift even if feature distributions are stable.

---

## 3️⃣ The Broader Context

### What Stage 6 enables downstream

- **Stage 7 (export)** — model team reviews `complexity_proxy_risk.md` before launching
- **Stage 8 (Run 11)** — the analysis report is the pre-training sanity check

### What breaks if Stage 6 is wrong

- Missing `feature_dist` → Run 9 failure repeats silently
- Missing co-occurrence → 99% DoS↔Reentrancy goes undetected quantitatively
- Missing drift monitor → new dataset version with shifted class distribution goes unnoticed

---

## 4️⃣ Verification — Stage 6 exit criteria

| # | Check | Status |
|---|---|---|
| 1 | All 6 analysis tools compile and run | ⏳ |
| 2 | `feature_dist` flags synthetic 2σ complexity skew | ⏳ |
| 3 | `complexity_proxy_risk.md` generated for ScaBench fixture | ⏳ |
| 4 | Co-occurrence matrix correct for multi-label fixture | ⏳ |
| 5 | Overlap matrix correct for 3-source fixture | ⏳ |
| 6 | Drift monitor flags intentional drift | ⏳ |

---

## 5️⃣ The "got it" checklist

1. **What is the `complexity_proxy_risk.md` report?** For each class pair, flags if per-feature σ-difference > 1.5. The Run 9 failure (model learning complexity as proxy) would have been flagged here before training.

2. **What's the L4 finding?** Model's top-3 features for all 10 classes are `complexity → visibility → uses_block_globals`, same order regardless of class. Stage 6 detects this from the data side.

3. **What are the directed + conditional co-occurrence matrices?** Directed: X→Y = P(Y|X). Conditional: P(Y=1|X=1). BCCC 99% DoS→Reentrancy shows as very high entry.

4. **Why does the drift monitor check label distribution?** A new dataset version with different positive/negative ratio per class is label drift even if feature distributions are stable.

5. **What's the overlap detector?** Pairwise Jaccard similarity between sources. Input to per-source loss weighting in training.

6. **Why is Stage 6 only 2 days?** The analysis tools are read-only operations over artifacts from Stages 1–5. The hard work was in earlier stages.

If you can answer all 6, Stage 6 is mastered.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 6"
- **07_stage_6_analysis.md** — the design + intent document
- **project_interpretability.md** (in MEMORY) — the L4 finding that motivates `complexity_proxy_risk`

When you're ready, say **"Stage 6 is mastered — let's move to Stage 7."**
