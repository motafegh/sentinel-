> **Superseded v1 plan:** retained for history. Use [D1 v2](../D1_developer_handbook.md) and [evaluation](../../../handbook/13_evaluation.md).

# D1.4b — Evaluation Doc

**Doc target:** `docs/handbook/09_evaluation.md`
**Estimated time:** 0.75h
**Rule:** Every claim verified against source code.

---

## Source files to read before writing (7 files)

1. `agents/src/eval/pipeline_metrics.py` — PipelineMetrics, Fbeta
   - Extract: Fbeta formula (β=2), how it's computed
   - Why β=2: recall weighted higher than precision for security (missing a vulnerability is worse than a false alarm)
   - Other metrics: macro_F1, per-class precision/recall

2. `agents/src/eval/gates.py` — 9 gate assertions
   - Extract: all 9 gate names and what each checks
   - Which gates block a release vs warn
   - How gates are run (in run_benchmark.py)

3. `agents/src/eval/run_benchmark.py` — benchmark runner
   - How to run a benchmark
   - Output format: eval_metrics.json + eval_report.md
   - Where results are stored: `agents/eval/runs/{timestamp}_{name}/`

4. `agents/src/eval/reliability_matrix.py` — per-tool confusion matrix
   - Data model: per-tool TP/FP/FN/TN counts
   - Rule 5C integration: excludes contracts where tool didn't run (reads tool_status)
   - Output: confusion_matrix_v{N}.json

5. `agents/src/eval/reliability_fit.py` — Bayesian shrinkage fitter
   - α=5 (prior strength)
   - Drop-gate: ±5pp — tools with reliability outside this range of their fitted weight are flagged
   - Output: reliability_v{N}.yaml with justifications

6. `agents/configs/reliability_v3.yaml` — L3 fitted weights
   - Verify: schema_version=1
   - Structure: per-tool, per-class accuracy weights
   - How it's loaded: `reliability.py:_load_l3_table()`

7. `agents/configs/verdicts_default.yaml` — L1 hand-set weights
   - consensus.accuracy_weights (per-class)
   - consensus.default_weights (per-source)
   - Band thresholds: confirmed_band, likely_band, disputed_band
   - Other decision numbers: thresholds, confidence weights

8. `agents/src/orchestration/verdict/reliability.py:73-140` — L3→L1→L0 fallback
   - L3: reads SENTINEL_RELIABILITY_CONFIG env var → reliability_v3.yaml
   - L1: reads verdicts_default.yaml consensus.accuracy_weights
   - L0: hardcoded _source_defaults (rag=0.50, debate=0.55, quick_screen=0.40)
   - Fallback triggers: missing file, malformed YAML, wrong schema_version

---

## Sections to write

**1. TL;DR** (4 lines)
```
What: Fbeta(β=2) metric, 9 gate assertions, Bayesian shrinkage (α=5), L0→L3 maturity ladder
Key insight: decision numbers (thresholds, weights) have maturity levels — hand-set → measured → learned
Config: agents/configs/verdicts_default.yaml (L1) + reliability_v3.yaml (L3)
```

**2. Fbeta metric** (~0.5 page)
- Formula (verify from `pipeline_metrics.py`):
  - Fβ = (1 + β²) × (precision × recall) / (β² × precision + recall)
  - β=2: recall weighted 4× more than precision
- Why β=2 for security: missing a real vulnerability (false negative) is more costly than investigating a false alarm (false positive)
- macro_F1: unweighted mean of per-class F1
- macro_Fbeta: unweighted mean of per-class Fβ (the primary metric)

**3. Gate system** (~0.5 page)
- 9 gates in `gates.py` (list all 9 with one-line description):
  - Verify exact gate names and assertions from source
  - Which gates block a release (hard fail) vs warn (soft)
  - Example: `macro_f1_vs_baseline` — asserts current run F1 >= baseline F1
- How to run: `run_benchmark.py` calls gates after computing metrics
- Output: pass/fail per gate in eval_report.md

**4. Reliability matrix** (~1 page)
- Per-tool confusion matrix (verify from `reliability_matrix.py`):
  - For each tool (slither, aderyn, ml, etc.): TP, FP, FN, TN counts
  - Rule 5C integration: contracts where tool_status[tool].ran == False are EXCLUDED from that tool's counts (not counted as TN)
  - Why: counting "tool didn't run" as "tool found nothing" biases the matrix
- Output: `confusion_matrix_v{N}.json` with per-tool, per-class counts
- How it's built: `scripts/build_reliability_matrix.py` reads all audit reports + tool_status

**5. Bayesian shrinkage** (~1 page)
- Purpose: per-tool reliability rates are unreliable with small samples → shrink toward prior
- Formula (verify from `reliability_fit.py`):
  - Fitted rate = (TP + α) / (TP + FP + α + β)
  - α=5 (prior strength — higher = more shrinkage toward 0.5)
  - This prevents a tool that ran on 3 contracts and got all 3 right from being rated 100% reliable
- Drop-gate: ±5pp — if a tool's observed reliability differs from its fitted weight by more than 5 percentage points, it's flagged for review
- Output: `reliability_v{N}.yaml` with fitted weights + justifications (reliability_justifications.yaml)
- Maturity: this is Level 3 (learned from data) on the L0→L3 ladder

**6. L0→L3 maturity ladder** (~0.5 page)
- Level 0: hand-set constant buried in .py code (prototype only)
- Level 1: externalized versioned config (verdicts_default.yaml) — hand-set but versioned
- Level 2: measured against baseline before every change
- Level 3: learned from data (reliability_v3.yaml — fitted from confusion matrix)
- The fallback chain (verify from `reliability.py:73-140`):
  - L3 first (reliability_v3.yaml, schema_version=1)
  - Falls back to L1 (verdicts_default.yaml) if L3 missing/malformed/wrong schema
  - Falls back to L0 (hardcoded defaults) if L1 also missing
- Current state: ML= L3, Slither/Aderyn = L3, RAG/debate/quick_screen = L1 (not enough data for L3)

**7. Deep reference**
- → `docs/learning/08_evaluation_framework.md` (deep dive on Fbeta, confusion matrix, gates)
- → `docs/learning/10_decision_numbers.md` (deep dive on L0→L3 ladder, Rule 5B)
- → source: `pipeline_metrics.py`, `gates.py`, `reliability_matrix.py`, `reliability_fit.py`
- → config: `verdicts_default.yaml`, `reliability_v3.yaml`, `reliability_justifications.yaml`

---

## Verification checklist
- [ ] Fbeta formula in `pipeline_metrics.py` uses β=2
- [ ] Gate count = 9 — list all 9 names from `gates.py` source
- [ ] `reliability_v3.yaml` has schema_version=1
- [ ] L3→L1→L0 fallback chain matches `reliability.py:73-140` logic
- [ ] α=5 in `reliability_fit.py`
- [ ] Drop-gate is ±5pp
- [ ] `verdicts_default.yaml` has consensus.accuracy_weights and band thresholds
- [ ] Rule 5C integration in reliability_matrix.py: excludes contracts where ran=False
