# Plan: Doc 08 — Evaluation Framework: F1, Fbeta, Reliability Matrix, Bayesian Shrinkage

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/08_evaluation_framework.md`
**Session:** 4 of 5
**Prerequisite docs:** Doc 01 (Pipeline), Doc 02 (Evidence/Fuse), Doc 04 (Reproducibility)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that the pipeline produces verdicts for each vulnerability class. The eval framework scores these verdicts against ground-truth labels (the `// expect:` headers in contract files).

**From Doc 02 (Evidence/Fuse):** You learned that `fuse()` uses per-source reliability values to weight evidence. These reliability values started as hand-set constants (L0), were externalized to config (L1), and eventually were fitted from data (L3). The F1 jumped from 0.1998 to 0.3008 when we moved from L1 to L3.

**From Doc 04 (Reproducibility):** You learned that `SENTINEL_DETERMINISTIC=1` produces reproducible verdicts. The eval framework runs in this mode to ensure consistent measurements.

**Connection to this doc:** This doc explains the evaluation framework in detail — why we use Fbeta(β=2) instead of F1, how the reliability matrix is built, how Bayesian shrinkage handles small samples, and the L0→L3 maturity ladder for decision numbers.

**Key concepts carried forward:** `verdict_provable` / `verdict_full`, `Evidence.reliability` field, `get_reliability()` L3→L1 fallback, F1=0.3008 baseline.

---

## Step 1: Read source files

- [ ] `agents/src/eval/pipeline_metrics.py` — `PipelineMetrics` class, F1, Fbeta (β=2), precision, recall, per-class metrics
- [ ] `agents/src/eval/reliability_matrix.py` — `ReliabilityMatrix` builder, TP/FP/FN/TN counting, Rule 5C compliance (exclude didn't-run)
- [ ] `agents/src/eval/reliability_fit.py` — Bayesian shrinkage fitter, α=5, drop-gate ±5pp, justifications, output to YAML
- [ ] `agents/src/eval/run_benchmark.py` — CLI runner, `--name`, `--baseline`, `--reports`, `--corpus` args
- [ ] `agents/src/eval/gates.py` — eval gates (`macro_f1_vs_baseline` threshold check)
- [ ] `agents/src/eval/regression.py` — regression detection (compare across runs)
- [ ] `agents/src/eval/benchmarks.py` — benchmark definitions
- [ ] `agents/src/orchestration/verdict/reliability.py` — `get_reliability()` function, L3→L1 fallback chain

## Step 2: Read scratch files

- [ ] `~/.claude/scratch/system_finalization_statecheck_20260625.md` — reliability matrix design, Rule 5C findings (Aderyn silent-skip, ML-failure-as-pass), L0→L3 progression, the +50% F1 jump analysis

## Step 3: Read eval reports

- [ ] `agents/eval/runs/20260624T133420Z_p0_honest_baseline/eval_report.md` — P0 baseline (F1=0.1958, Fbeta=0.2515)
- [ ] `agents/eval/runs/20260624T231228Z_p2_calibrated/eval_report.md` — P2 calibrated (F1=0.1998, Fbeta=0.2246)
- [ ] `agents/eval/runs/20260626T123145Z_p3_rule5c_v3/eval_report.md` — P3 data-derived (F1=0.3008, Fbeta=0.3821)

## Step 4: Read config files

- [ ] `agents/configs/verdicts_default.yaml` — L1 config (thresholds, weights, reliability defaults)
- [ ] `agents/configs/reliability_v3.yaml` — L3 fitted config (data-derived reliability values, schema_version=1)

## Step 5: Read CLAUDE.md

- [ ] `~/projects/sentinel/CLAUDE.md` §B — Rule 5B: "No decision-number changes without a measured delta"

## Step 6: Write sections

- [ ] **TL;DR:** Fbeta(β=2) for FN/FP asymmetry (recall weighted 2× over precision), reliability matrix per-tool per-class TP/FP/FN/TN, Bayesian shrinkage (α=5) for small samples, L0→L3 maturity ladder, +50% F1 from L3 alone
- [ ] **The Problem:** How do you measure if your security analysis system is actually good, not just if it runs? Standard F1 doesn't weight recall for security (a missed vulnerability costs millions). Hand-set reliability weights are biased by human intuition
- [ ] **How We Arrived at This Design:** invariant (FN costs more than FP → need asymmetric metric) → constraint (need per-tool reliability, not global) → simplest measurement (confusion matrix + Bayesian shrinkage) → stress-test (61 contracts, rare classes have few samples) → measure (F1 0.20→0.30 from L3 reliability)
- [ ] **The Solution:** Metric progression diagram:
  ```
  P0 baseline:  F1=0.1958, Fbeta=0.2515  (first honest measurement)
  P2 calibrated: F1=0.1998, Fbeta=0.2246  (config externalized, L1)
  P3 data-derived: F1=0.3008, Fbeta=0.3821  (reliability fitted from data, L3)
  ```
  Fbeta formula: `Fβ = (1+β²) · (P·R) / (β²·P + R)` with β=2
  Reliability matrix: per-tool × per-class TP/FP/FN/TN table. Rule 5C: exclude contracts where tool didn't run
  Bayesian shrinkage: `reliability = (TP + α·prior) / (TP + FP + α)` with α=5
  L0→L3 ladder diagram
- [ ] **Key Code:**
  - `PipelineMetrics` (pipeline_metrics.py) — computes F1, Fbeta(β=2), precision, recall, per-class and macro
  - `ReliabilityMatrix` (reliability_matrix.py) — builds TP/FP/FN/TN per tool per class. Reads `tool_status` to exclude didn't-run (Rule 5C)
  - `fit_reliability()` (reliability_fit.py) — Bayesian shrinkage with α=5, prior=global average, drop-gate ±5pp
  - `get_reliability()` (reliability.py) — L3 (reliability_v3.yaml) → L1 (verdicts_default.yaml) → hardcoded fallback
  - `run_benchmark.py` CLI — `--name`, `--baseline` (compares against previous run), `--reports` (directory of audit reports)
- [ ] **Design Decision:** Bayesian shrinkage vs MLE vs bootstrapping (tradeoff table: small sample handling, bias, variance, complexity, interpretability)
- [ ] **Technology Choice:** Fbeta(β=2) vs F1 vs precision/recall (5-question framework: category, alternatives, why β=2, when β=1 is fine, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ Count silent-skips as TN — "if the tool didn't run, it didn't find anything." Breaks: inflates reliability (tool looks good because it didn't run on hard contracts). Right: Rule 5C — exclude didn't-run from counts
  - ❌ Use F1 (β=1) — "standard metric." Breaks: doesn't weight recall for security. A missed vulnerability (FN) costs millions; a wasted review (FP) costs minutes. Right: Fbeta with β=2 (recall weighted 2×)
- [ ] **Mistakes & Fixes:**
  - Aderyn silent-skip: `FileNotFoundError` caught, logged at DEBUG, `[]` returned. 83 contracts had zero Aderyn findings — indistinguishable from "ran clean." Biased the reliability matrix (Aderyn looked reliable because it "found nothing" on hard contracts). Fix: Rule 5C — `tool_status["aderyn"] = {"ran": False}`. Exclude didn't-run from TP+FP+FN+TN
  - ML-failure-as-pass: ML server down → `ml_result = {}` → treated as "ML says safe." False negative factory. Fix: `ml_result = {"ran": False, ...}`. Synthesizer treats absent `ran=True` as failure
  - P0 baseline F1=0.1958 was the first honest number. Before P0, there was no eval framework at all — just tests proving code runs. Lesson: "tests prove the code runs; evals prove the system is good"
  - L3 reliability caused +50% F1 improvement (0.1998→0.3008) — not from model changes, not from new tools, just from replacing hand-set weights with data-derived ones. Lesson: measurement > intuition
- [ ] **What Would Break Without This:** Remove Fbeta → use F1, recall underweighted, FN not penalized enough. Remove reliability matrix → hand-set weights (L0), biased. Remove shrinkage → noisy estimates on rare classes (e.g., Halmos ran on 3 Reentrancy contracts — MLE would be unreliable). Remove L3→L1 fallback → config failure crashes the pipeline
- [ ] **At Scale:** 61 contracts (current, rare classes noisy) / 610 (better) / 6,100 (most classes have 50+ samples) / 61,000 (MLE viable, shrinkage unnecessary)
- [ ] **Try It Yourself:**
  ```
  cd agents && source .venv/bin/activate
  python -m src.eval.run_benchmark --name test_eval --reports test_audit_reports_post_quarantine_no_llm --corpus ../manual_hand_written_contracts
  cat configs/reliability_v3.yaml | head -30   # see the L3 fitted values
  cat configs/verdicts_default.yaml | head -30  # see the L1 config
  ```
- [ ] **Limitations:** 61 contracts is small (rare classes noisy even with shrinkage). No cross-validation. No statistical significance tests (just point estimates). No automatic refit trigger (manual `build_reliability_matrix.py` run). Drop-gate is ±5pp heuristic, not learned
- [ ] **Transferable Patterns:** (1) FN/FP asymmetry in metrics — choose β based on cost asymmetry (2) Bayesian shrinkage for small samples — pull toward prior with strength proportional to sample size (3) Maturity ladder for decision numbers — L0→L1→L2→L3, each level is an upgrade. Each with interview story + when wrong.

## Step 7: Verify

- [ ] Open `reliability_fit.py` and verify α=5 and drop-gate=±5pp
- [ ] Open `pipeline_metrics.py` and verify β=2
- [ ] Confirm eval numbers (0.1958 → 0.1998 → 0.3008) match the 3 eval reports
- [ ] Open `reliability_v3.yaml` and verify `schema_version=1`
- [ ] Open `verdicts_default.yaml` and verify it has `ml_positive_threshold`, `confirmed_threshold`, etc.
- [ ] Open `reliability.py` and verify `get_reliability()` falls back from L3 to L1
