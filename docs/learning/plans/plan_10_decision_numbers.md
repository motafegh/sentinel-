# Plan: Doc 10 — Decision Numbers: From Hand-Set Constants to Data-Derived Reliability

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/10_decision_numbers.md`
**Session:** 5 of 5
**Prerequisite docs:** Doc 01 (Pipeline), Doc 02 (Evidence/Fuse), Doc 08 (Evaluation Framework)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that the pipeline uses thresholds to decide which path to take (fast vs deep). `compute_active_tools()` checks ML probabilities against `DEEP_THRESHOLDS`. These thresholds are decision numbers — they control system behavior.

**From Doc 02 (Evidence/Fuse):** You learned that `fuse()` uses `reliability` values to weight each evidence source. These reliability values started as hand-set constants (L0), were externalized to config (L1), and eventually were fitted from data (L3). The `get_reliability()` function falls back from L3 to L1.

**From Doc 08 (Evaluation Framework):** You learned the eval progression: P0 (F1=0.1958) → P2 (F1=0.1998) → P3 (F1=0.3008). The +50% jump came from L3 reliability — replacing hand-set weights with data-derived ones. You also learned about Bayesian shrinkage (α=5) and the drop-gate (±5pp).

**Connection to this doc:** This doc is the capstone — it explains the philosophy behind ALL decision numbers in the system. Every threshold, weight, and confidence value is POLICY. The maturity ladder (L0→L3) is the framework for how each number matures from a guess to a measured, data-derived value.

**Key concepts carried forward:** `get_reliability()` L3→L1 fallback, `verdicts_default.yaml` (L1), `reliability_v3.yaml` (L3), Rule 5B ("no decision-number changes without measurement"), F1=0.3008.

---

## Step 1: Read source files

- [ ] `agents/src/config/schema.py` — Pydantic schema for config validation (all config fields defined here)
- [ ] `agents/src/config/loader.py` — `get_config()` function, config loading from YAML, caching
- [ ] `agents/src/orchestration/verdict/reliability.py` — `get_reliability(source, vuln_class)` function, L3→L1 fallback chain
- [ ] `agents/src/eval/reliability_fit.py` — Bayesian shrinkage fitter, α=5, drop-gate ±5pp, justifications, YAML output
- [ ] `agents/src/eval/reliability_matrix.py` — `ReliabilityMatrix` builder, TP/FP/FN/TN per tool per class, Rule 5C compliance
- [ ] `agents/src/orchestration/routing.py` — `compute_active_tools()`, `DEEP_THRESHOLDS`, `CLASS_TO_DETECTORS`

## Step 2: Read config files

- [ ] `agents/configs/verdicts_default.yaml` — L1 config: `ml_positive_threshold`, `confirmed_threshold`, `suspicious_threshold`, reliability defaults, all decision numbers
- [ ] `agents/configs/reliability_v3.yaml` — L3 fitted config: per-tool per-class reliability values, `schema_version=1`, justifications

## Step 3: Read scratch files

- [ ] `~/.claude/scratch/system_finalization_statecheck_20260625.md` — L0→L3 progression, Rule 5C impact on reliability (Aderyn inflated), the +50% F1 analysis

## Step 4: Read eval reports

- [ ] `agents/eval/runs/20260624T133420Z_p0_honest_baseline/eval_report.md` — P0 baseline (L0/L1 weights)
- [ ] `agents/eval/runs/20260626T123145Z_p3_rule5c_v3/eval_report.md` — P3 (L3 weights, +50% F1)

## Step 5: Read CLAUDE.md

- [ ] `~/projects/sentinel/CLAUDE.md` §B — Rule 5B: "No decision-number changes without a measured delta." The maturity ladder definition. "I think 0.35 feels right" is not sufficient.

## Step 6: Write sections

- [ ] **TL;DR:** Every threshold/weight/confidence is POLICY. Maturity ladder: L0 constant → L1 config → L2 measured → L3 learned. The +50% F1 jump (0.1998→0.3008) came from L3 reliability alone — not from model changes, not from new tools. Rule 5B: no number changes without measured delta
- [ ] **The Problem:** "I think 0.35 feels right" is not engineering. Decision numbers control system behavior — thresholds decide routing, weights decide verdicts, reliability decides trust. If these are guessed, the system is biased by human intuition. Must be measured and versioned
- [ ] **How We Arrived at This Design:** invariant (every number is policy → must be measurable and versioned) → constraint (can't measure without a baseline → need eval framework first) → simplest progression (L0→L1→L2→L3, each level is an upgrade) → stress-test (what if data is wrong? → drop-gate ±5pp) → measure (F1 0.20→0.30 from L3 alone)
- [ ] **The Solution:** Maturity ladder diagram:
  ```
  L0: threshold = 0.5                    # hardcoded in .py (prototype only)
       ↓ externalize
  L1: threshold in verdicts_default.yaml  # versioned, changeable (P1)
       ↓ measure baseline
  L2: threshold measured vs P0 baseline    # "before: 0.20, after: 0.22" (P0.1)
       ↓ fit from data
  L3: reliability in reliability_v3.yaml   # data-derived, Bayesian shrinkage (P3)
  ```
  Config hierarchy: L3 (`reliability_v3.yaml`) > L1 (`verdicts_default.yaml`) > hardcoded fallback. `get_reliability()` tries L3 first, falls back to L1 if missing/malformed/wrong schema. Bayesian shrinkage formula: `reliability = (TP + α·prior) / (TP + FP + α)` with α=5. Drop-gate: if fitted value differs from L1 default by >±5pp, flagged for review
- [ ] **Key Code:**
  - `get_config()` (loader.py) — loads config from YAML, caches instance
  - `get_reliability(source, vuln_class)` (reliability.py) — L3→L1→hardcoded fallback chain
  - `fit_reliability()` (reliability_fit.py) — Bayesian shrinkage, α=5, prior=global average, drop-gate ±5pp
  - `ReliabilityMatrix` (reliability_matrix.py) — builds TP/FP/FN/TN, reads `tool_status` for Rule 5C
  - `compute_active_tools()` (routing.py) — uses `DEEP_THRESHOLDS` (L1 config) to decide routing
  - Config schema (schema.py) — Pydantic models for all config fields, validation
- [ ] **Design Decision:** L3 learned vs L2 measured vs L1 config vs L0 constant (tradeoff table: accuracy, bias, data needed, complexity, when appropriate)
- [ ] **Technology Choice:** Bayesian shrinkage for fitting (5-question framework: category, alternatives, why shrinkage, when MLE is fine, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ Hand-set constants forever — "I know the domain, 0.35 is right." Breaks: unmeasured, unversioned, biased by intuition, can't track changes. Right: L1 config (at minimum), L3 data-derived (ideal)
  - ❌ Change number without eval — "let me try 0.35, it feels better." Breaks: no measured delta, can't tell if better or worse. Right: Rule 5B — "show me the before/after eval result"
- [ ] **Mistakes & Fixes:**
  - Aderyn reliability inflated because silent-skips counted as TN. Tool didn't run on 83 contracts, but `[]` return was treated as "Aderyn found nothing" (TN). Fix: Rule 5C — `tool_status["aderyn"]["ran": False]`. Exclude didn't-run from TP+FP+FN+TN counts. This was the origin of Rule 5C
  - P0 baseline F1=0.1958 was the first honest number. Before P0, there was no eval framework at all — just tests proving code runs. Lesson: "tests prove the code runs; evals prove the system is good"
  - L3 reliability caused +50% F1 improvement (0.1998→0.3008) — not from model changes, not from new tools, not from better prompts. Just from replacing hand-set weights with data-derived ones. Lesson: measurement > intuition. The single biggest improvement in the entire project came from data, not code
  - Drop-gate caught a fitting artifact: one tool/class had a fitted reliability of 0.95 (vs L1 default of 0.50) on only 3 samples. The ±5pp drop-gate flagged it. Fix: kept L1 default until more samples available
- [ ] **What Would Break Without This:** Remove L3 → hand-set weights (L0), biased by human intuition, F1 drops back to ~0.20. Remove config → constants in `.py` files, unversioned, can't change without code edit. Remove drop-gate → fitting artifacts on small samples go unnoticed, unreliable values enter fuse(). Remove Rule 5B → "I think 0.35 feels right" returns, no measured deltas
- [ ] **At Scale:** 61 contracts (current, rare classes noisy even with shrinkage) / 610 (better estimates) / 6,100 (most classes have 50+ samples, MLE viable) / 61,000 (shrinkage unnecessary, could learn fusion weights too)
- [ ] **Try It Yourself:**
  ```
  cd agents && source .venv/bin/activate
  cat configs/reliability_v3.yaml | head -30    # see the L3 fitted values
  cat configs/verdicts_default.yaml | head -30   # see the L1 config
  python3 -c "
  from src.orchestration.verdict.reliability import get_reliability
  print('L3 ml Reentrancy:', get_reliability('ml', 'Reentrancy'))
  print('L3 halmos Reentrancy:', get_reliability('halmos', 'Reentrancy'))
  "
  ```
- [ ] **Limitations:** 61 contracts is small (rare classes noisy even with shrinkage). No A/B testing framework (can't compare two configs simultaneously). No automatic refit trigger (manual `build_reliability_matrix.py` run). Drop-gate is ±5pp heuristic, not learned. Only reliability is L3 — thresholds and weights are still L1. No uncertainty quantification on fitted values (point estimates only)
- [ ] **Transferable Patterns:** (1) Maturity ladder for configuration — L0→L1→L2→L3, each level is an upgrade, never skip levels (2) Bayesian shrinkage for small samples — pull toward prior with strength proportional to sample size (3) Measured deltas over intuition — "show me the before/after eval" not "I think this is better" (4) Drop-gates for fitting artifacts — flag values that deviate too far from prior. Each with interview story + when wrong.

## Step 7: Verify

- [ ] Open `reliability_v3.yaml` and verify `schema_version=1`
- [ ] Open `verdicts_default.yaml` and verify it has `ml_positive_threshold`, `confirmed_threshold`, `suspicious_threshold`
- [ ] Open `reliability.py` and verify `get_reliability()` falls back from L3 to L1 to hardcoded
- [ ] Open `reliability_fit.py` and verify α=5 and drop-gate=±5pp
- [ ] Confirm the F1 progression: 0.1958 (P0) → 0.1998 (P2) → 0.3008 (P3)
- [ ] Open `CLAUDE.md` §B and verify Rule 5B text
