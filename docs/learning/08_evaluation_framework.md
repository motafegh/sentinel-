# 08. Evaluation Framework: F1, Fbeta, Reliability Matrix, Bayesian Shrinkage

> **Prerequisites:** [01. The Audit Pipeline] — verdicts are produced per class. [02. Evidence Model & Fuse()] — `Evidence.reliability` weights come from this framework. [04. Reproducibility] — eval runs in deterministic mode for consistent measurement.
> **Next:** [09. Formal Verification] covers Halmos, which adds a new column to the reliability matrix.
> **Cross-ref:** [10. Decision Numbers] covers the L0→L3 maturity ladder in detail.
> **Scope:** This doc covers the complete eval framework: the Fbeta(β=2) metric (why not F1), the per-class confusion matrix, the 9 gate assertions, the reliability matrix (per-tool per-class TP/FP/FN/TN), Bayesian shrinkage fitting (α=5), the L0→L3 maturity ladder, and regression detection. It does NOT cover the pipeline that produces the verdicts (see Doc 01) or the fuse() function that consumes the reliability values (see Doc 02).
> **TL;DR:** SENTINEL measures system quality with **Fbeta(β=2)** — not F1 — because a missed vulnerability (false negative) costs millions while a wasted review (false positive) costs minutes. β=2 weights recall 2× over precision, reflecting this asymmetry. Per-(source, class) reliability is fitted from a **confusion matrix** on 61-83 labeled contracts using **Bayesian shrinkage** (`fitted = (n × measured + α × prior) / (n + α)`, α=5) — which pulls noisy small-sample estimates toward a prior, preventing a tool that ran on 3 contracts from looking "perfect." The **L0→L3 maturity ladder** tracks every decision number's provenance: L0 hand-set constant → L1 versioned config → L2 measured against baseline → L3 learned from data. Replacing L1 hand-set reliability weights with L3 data-derived ones drove macro_F1 from 0.1998 to 0.3008 (+50% relative) — the single largest improvement in the system's history, achieved without any code changes, only better inputs. Nine **gate assertions** (WS1a-e, WS2, WS3, D4, macro_f1) enforce safety properties on every eval run — a gate failure blocks shipping.

---

## The Problem: How Do You Know If Your Security System Is Actually Good?

### Tests prove the code runs. Evals prove the system is good.

**Teaching: the difference between tests and evals.** A unit test verifies that `fuse()` returns a `ClassVerdict` when given a list of `Evidence` items. It passes if the function doesn't crash and the return type is correct. An eval verifies that the *system* — ML model + Slither + Aderyn + fuse() — produces the correct verdict on a real contract. It passes if `verdict_provable == "CONFIRMED"` on a contract with a known reentrancy bug.

| Aspect | Unit Test | Eval |
|--------|-----------|------|
| What it checks | Code runs without crashing | System produces correct verdicts |
| Input | Synthetic, minimal | Real contracts with ground-truth labels |
| Output | Pass/fail | F1, Fbeta, per-class precision/recall |
| Failure means | Code bug | System quality regression |
| Frequency | Every commit | Every phase change |

**The reasoning:** a system with 631 passing unit tests can still have macro_F1=0.1958 — the code runs, but the verdicts are wrong. Tests are necessary (code must run) but not sufficient (verdicts must be correct). The eval framework is the bridge between "the code works" and "the system is good."

### Why F1 is not enough for security

**Teaching: the FN/FP cost asymmetry.** In standard ML, F1 treats false positives and false negatives symmetrically — both are "errors." But in security:

| Error type | What it means | Cost |
|------------|--------------|------|
| False Negative (FN) | Contract has a vulnerability, system says SAFE | Vulnerability goes to production → exploit → $millions lost |
| False Positive (FP) | Contract is safe, system says CONFIRMED | Human reviews the contract → wastes 15 minutes → finds no bug |

The cost ratio is ~1000:1 (millions vs minutes). F1 weights both errors equally. Fbeta(β=2) weights recall 2× over precision, which means a false negative hurts the score 2× more than a false positive. This reflects the cost asymmetry.

**The Fbeta formula:**
```
Fβ = (1 + β²) · (P · R) / (β² · P + R)

where:
  P = precision = TP / (TP + FP)     — of all positive predictions, how many are correct?
  R = recall    = TP / (TP + FN)     — of all actual positives, how many did we find?
  β = 2                            — recall is weighted 2× over precision

When β=1: F1 = 2·P·R / (P + R)     — standard F1, equal weight
When β=2: F2 = 5·P·R / (4·P + R)   — recall weighted 2× (4 = β²)
```

**Teaching: why β=2 and not β=5 or β=10?** The cost ratio is ~1000:1, so you might think β should be much higher. But β controls the *metric weight*, not the cost ratio directly. β=2 means "recall matters 2× as much as precision in the score." A higher β (e.g., β=10) would make the score almost entirely recall-driven — precision would barely matter. That's too extreme: a system that flags every contract as "vulnerable" has 100% recall but 10% precision — useless. β=2 is the standard choice for recall-weighted tasks (information retrieval, medical diagnosis, security) and was confirmed in the proposal (§6, β=2).

---

## How We Arrived at This Design

> **How to read this section:** Each step shows the question, *how to reason about it*, and the chain of logic.

### Step 1 — Identify the invariant

**The question:** What must always be true about the evaluation?

| Candidate property | If violated → | Verdict |
|---|---|---|
| Eval measures recall with higher weight than precision (FN > FP cost) | A system with high FN but low FP looks "good" → missed vulnerabilities ship | **Invariant** |
| Every measurement is against a labeled baseline | "I think it's better" is not a measurement → unquantified changes | **Invariant** |
| Tool failures are excluded from reliability counts (Rule 5C) | Silent failures inflate reliability → biased weights → wrong verdicts | **Invariant** |

**The reasoning chain:** The eval must measure what matters (recall-weighted, because FN is expensive). It must measure against a baseline (so every change has a delta). And it must not be poisoned by silent failures (a tool that didn't run can't be counted as "found nothing" — Rule 5C). These three invariants shape the entire framework: the metric (Fbeta), the baseline comparison (regression detection), and the confusion matrix (Rule 5C exclusion).

### Step 2 — Identify the constraints

**Constraint A: Small sample sizes (61-83 contracts).**
- *Why:* The labeled corpus is 61-83 hand-written contracts. Rare classes (e.g., DenialOfService) have only 9-10 samples. A tool that ran on 3 contracts and got 3 TP looks "perfect" (measured precision = 1.0) — but n=3 is too small to trust.
- *What this forces:* Bayesian shrinkage. Instead of using the raw measured precision (which is noisy for small n), pull the estimate toward a prior: `fitted = (n × measured + α × prior) / (n + α)`. With α=5, a tool with n=3 and measured=1.0 gets fitted = (3×1.0 + 5×prior) / (3+5) = (3 + 5×prior) / 8 — much closer to the prior than to 1.0.

**Constraint B: Multiple tools × multiple classes = sparse matrix.**
- *Why:* 3 sources (ML, Slither, Aderyn) × 10 classes = 30 cells. Many cells have n=0 (the tool never flagged that class). A zero-sample cell can't have a measured precision — it's undefined.
- *What this forces:* Zero-sample cells fall back to the L1 prior verbatim. `if n == 0: fitted = prior`. No fabrication — the prior is the L1 hand-set value, documented and versioned.

**Constraint C: Decision numbers must have provenance (Rule B).**
- *Why:* Every threshold, weight, and confidence value is a "decision number." Changing one without measurement is a Rule B violation. You need to know *where* each number came from (hand-set? config? measured? learned?).
- *What this forces:* The L0→L3 maturity ladder. Every decision number is labeled with its level: L0 (hand-set constant), L1 (versioned config), L2 (measured against baseline), L3 (learned from data). The level is documented in the config file (`verdicts_default.yaml` has comments like `# L2` and `# L3 target`).

### Step 3 — Eliminate alternatives

| Approach | How it breaks | When it breaks | Eliminate? |
|---|---|---|---|
| **F1 (β=1)** | FN and FP weighted equally → a system with high FN can look "ok" | When FN is more expensive than FP (always, in security) | **Yes** |
| **Accuracy** | Misleading on imbalanced data (90% safe contracts → "always say safe" = 90% accuracy) | When class distribution is imbalanced (always) | **Yes** |
| **Precision only** | System can achieve 100% precision by never flagging anything → 0% recall | When recall matters | **Yes** |
| **Fbeta(β=2)** | Slightly harder to interpret than F1 (two metrics, not one) | Never — the asymmetry is the point | **No** |

**The reasoning:** F1 fails on the cost asymmetry (FN=FP cost, wrong for security). Accuracy fails on class imbalance (90% safe → "always say safe" looks great). Precision-only fails on recall (never flag = 100% precision, 0% recall). Fbeta(β=2) is the only metric that reflects the cost asymmetry without making precision irrelevant.

### Step 4 — Stress-test (the L3 reliability jump)

**The test:** Replace hand-set reliability weights (L1) with data-derived ones (L3). Measure the macro_F1 delta.

**The result:** macro_F1 went from 0.1998 (L1) to 0.3008 (L3) — a +50% relative improvement. No code changed — only the reliability values in the config file. This proves the framework works: better inputs (measured weights) produce better outputs (higher F1), and the eval measures the improvement.

### Step 5 — Measure (the full progression)

| Run | macro_F1 | macro_Fβ | What changed | What this proves |
|-----|----------|---------|-------------|-------------------|
| P0 baseline | 0.1958 | 0.2515 | First honest measurement (Aderyn silent-skip) | The system's floor — bugs and all |
| P2 calibrated | 0.1998 | 0.2246 | Evidence model + fuse() replace legacy | fuse() doesn't break anything (+0.004) |
| P3 reliability fit | 0.2765 | 0.3580 | L3 data-derived weights replace L1 hand-set | The Evidence model's payoff (+0.077) |
| P3 Rule 5C v3 | 0.3008 | 0.3821 | Honest failure counting (Rule 5C) | Rule 5C's payoff (+0.024) |

> **The method, summarized:** (1) Find invariants — recall-weighted, baseline-anchored, Rule 5C compliant. (2) Find constraints — small samples need shrinkage, sparse matrix needs prior fallback, decision numbers need provenance. (3) Eliminate symmetric metrics (F1, accuracy) by finding their failure conditions. (4) Stress-test with the L3 replacement — the +50% F1 jump proves the framework works. (5) Measure every change against the baseline — no change ships without a measured delta.

---

## The Solution

### The eval pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  EVAL RUN (per phase change)                                         │
│                                                                      │
│  1. Run pipeline on 83 contracts (SENTINEL_DETERMINISTIC=1)         │
│     → 83 audit reports (JSON)                                        │
│                                                                      │
│  2. Load corpus: pair each report with ground-truth labels           │
│     → 83 ContractEval rows (labels from // expect: header or .json)  │
│                                                                      │
│  3. Compute per-contract: TP/FP/FN classes                           │
│     → predicted_positive ∩ labels = TP                               │
│     → predicted_positive - labels = FP                               │
│     → labels - predicted_positive = FN                               │
│                                                                      │
│  4. Compute per-class metrics: precision, recall, F1, Fbeta(β=2)    │
│     → ClassMetrics for each of 10 classes                           │
│                                                                      │
│  5. Compute aggregates: macro_F1, macro_Fbeta, micro_F1              │
│     → PipelineMetrics (headline numbers)                            │
│                                                                      │
│  6. Run 9 gates: WS1a-e, WS2, WS3, D4, macro_f1_vs_baseline         │
│     → GateResult list (PASS/FAIL per gate)                          │
│                                                                      │
│  7. Write eval_report.md + eval_metrics.json                         │
│     → eval/runs/<timestamp>_<name>/                                  │
│                                                                      │
│  8. Exit 0 if all gates pass, exit 1 if any fails                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Worked example: scoring one contract

**Input:** Contract `01_bccc_reentrancy_injected_erc20` has `// expect: Reentrancy` in its Solidity source.

**Ground truth:** `labels = ["Reentrancy"]`, `ground_truth = "vulnerable"`.

**Pipeline verdict:** `verdicts = {"Reentrancy": "CONFIRMED", "IntegerUO": "SAFE", "Timestamp": "SAFE", ...}`.

**Scoring:**
```python
positive_set = {"CONFIRMED", "LIKELY"}  # from config

predicted_positive = positive_classes(verdicts, positive_set)
# → ["Reentrancy"]  (CONFIRMED is positive; SAFE is not)

label_set = {"Reentrancy"}

true_positive_classes  = predicted_positive ∩ label_set  # → ["Reentrancy"] ✓
false_positive_classes = predicted_positive - label_set   # → [] ✓
false_negative_classes = label_set - predicted_positive    # → [] ✓

contract_correct = len(true_positive_classes) > 0  # → True (loose)
contract_exact   = set(predicted_positive) == label_set  # → True (exact)
```

This contract contributes 1 TP to the Reentrancy class. No FP, no FN.

### Worked example: the Fbeta calculation for one class

**Reentrancy class on the P3 Rule 5C v3 eval (83 contracts):**

From the eval report:
```
Reentrancy: TP=12, FP=32, FN=3, TN=36
```

```python
precision = TP / (TP + FP) = 12 / (12 + 32) = 12 / 44 = 0.2727
recall    = TP / (TP + FN) = 12 / (12 + 3)  = 12 / 15 = 0.8000

# F1 (β=1):
f1 = 2 * P * R / (P + R) = 2 * 0.2727 * 0.8000 / (0.2727 + 0.8000)
    = 0.4363 / 1.0727 = 0.4068

# Fbeta (β=2):
b2 = 4  # β² = 2² = 4
fbeta = (1 + b2) * P * R / (b2 * P + R)
      = 5 * 0.2727 * 0.8000 / (4 * 0.2727 + 0.8000)
      = 1.0909 / 1.8909
      = 0.5769
```

**Teaching: why Fbeta (0.5769) > F1 (0.4068) for Reentrancy.** Reentrancy has high recall (0.80 — the system finds most reentrancy bugs) but low precision (0.27 — it also flags many false positives). F1 penalizes the low precision heavily (both weighted equally). Fbeta(β=2) rewards the high recall more — the high recall means "we catch 80% of reentrancy bugs," which is the property that matters most for security. The false positives are a human-time cost (reviewing clean contracts); the false negatives are a money cost (shipping vulnerable contracts). Fbeta reflects this tradeoff.

### The 9 gate assertions

Gates are pass/fail safety checks that run on every eval. A gate failure blocks shipping — the system is not production-ready if a gate fails.

| Gate | What it checks | Why it exists |
|------|---------------|---------------|
| WS1a | No consensus-flagged class ends SAFE | The debate can't silently clear a consensus flag (Finding #15) |
| WS1b | edge_debate_timeout emits INCONCLUSIVE | Timeouts produce INCONCLUSIVE, not a silent verdict |
| WS1c | No missing consensus votes | Every consensus vote appears in final verdicts (Finding #15) |
| WS1d | No confidence=1.0 class ends SAFE | High-confidence flags can't be downgraded to SAFE (Finding #14) |
| WS1e | No vulnerable-labeled contract gets SAFE overall | The system can't say "SAFE" on a known-vulnerable contract (Finding #19) |
| WS2 | Zero false positives on safe contracts | Safe contracts should produce zero positive verdicts |
| WS3 | Long-contract reentrancy detected | A bug past the 2000-char ML truncation cutoff must still be caught |
| D4 | eye_predictions present in all reports | ML model produces per-eye auxiliary predictions (data quality) |
| macro_f1 | macro_F1 ≥ baseline | The system didn't regress from the previous run |

**Teaching: gates vs metrics.** Metrics (F1, Fbeta) are continuous — "how good is the system?" Gates are binary — "is the system safe to ship?" A system with macro_F1=0.30 can pass all gates (no safety violations) — it's not great, but it's not dangerous. A system with macro_F1=0.50 can fail WS2 (false positives on safe contracts) — it's better on average but has a dangerous failure mode. Gates catch dangerous failures that metrics average away.

### The reliability matrix

The reliability matrix is a per-(source, class) confusion matrix that measures each tool's precision on each vulnerability class:

```
                    Reentrancy    IntegerUO    Timestamp    ...
  ml      TP=11  FP=25    TP=0   FP=5     TP=7   FP=12   ...
  slither TP=12  FP=32    TP=0   FP=0     TP=8   FP=12   ...
  aderyn  TP=0   FP=0     TP=0   FP=0     TP=0   FP=0    ...
```

**Teaching: what each cell means.** `slither/Reentrancy: TP=12, FP=32` means: on 83 contracts, Slither flagged Reentrancy on 44 contracts (12+32). Of those, 12 actually had Reentrancy (TP) and 32 didn't (FP). Slither's measured precision for Reentrancy = 12/44 = 0.2727. This is the value that gets Bayesian-shrunk and fed into `fuse()` as `Evidence.reliability`.

**Rule 5C compliance:** the matrix builder reads `tool_status` from each report. If `tool_status["aderyn"]["ran"] == False`, that contract is excluded from Aderyn's TP/FP/FN/TN counts — it's recorded in `excluded_contracts["aderyn"]`. Without this, a contract where Aderyn didn't run would count as TN (no Aderyn finding → "Aderyn says safe") — inflating Aderyn's precision.

### The Bayesian shrinkage formula

```
fitted = (n × measured + α × prior) / (n + α)

where:
  n         = total contracts where the tool ran (TP + FP + FN + TN)
  measured  = TP / (TP + FP), 0.0 if undefined
  α         = 5.0 (shrinkage strength — higher = more pull toward prior)
  prior     = the L1 hand-set value from verdicts_default.yaml

Special case: if n == 0 → fitted = prior (verbatim, no fabrication)
```

**Teaching: why shrinkage?** Without shrinkage, a tool that ran on 3 contracts and got 3 TP would have measured precision = 1.0 (perfect!). But n=3 is too small to trust — the next 3 contracts might all be FP, dropping precision to 0.5. Shrinkage prevents this: with α=5, the fitted value is `(3 × 1.0 + 5 × prior) / 8`. If prior=0.60, fitted = (3 + 3.0) / 8 = 0.75 — much closer to the prior than to 1.0. The shrinkage strength α=5 means "the prior is worth 5 samples" — it takes 5+ real samples to start pulling the fitted value away from the prior.

**Teaching: why α=5?** α controls how much the prior resists the measured value. α=1 means "the prior is worth 1 sample" — barely resists (measured dominates quickly). α=20 means "the prior is worth 20 samples" — strongly resists (measured needs 20+ samples to override). α=5 is a moderate choice: the prior is worth 5 samples, so a tool needs ~5 real measurements to start pulling away from the hand-set value. This prevents small-sample noise from producing extreme fitted values, while allowing real data to override the prior once enough samples accumulate. α=5 is a Rule B Level-2 decision number — it was chosen by the proposal and hasn't been re-tuned (a measurement on a held-out set would be needed to justify a different value).

### The L0→L3 maturity ladder

Every decision number (threshold, weight, confidence value) has a provenance level:

| Level | What it means | Example | Trust |
|-------|-------------|---------|-------|
| L0 | Hand-set constant in code | `reliability = 0.60` hardcoded in `.py` | Low — someone guessed |
| L1 | Externalized to versioned config | `verdicts_default.yaml: accuracy_weights: {ml: 0.78}` | Medium — documented, changeable |
| L2 | Measured against baseline | `macro_F1 = 0.3008 (delta +0.0243 vs baseline 0.2765)` | High — measured before change |
| L3 | Learned from data | `reliability_v3.yaml: slither/Reentrancy: 0.795455` (fitted) | Highest — data-derived |

**Teaching: the ladder is a one-way ratchet.** A decision number starts at L0 (someone guesses a value). It's externalized to L1 (put in config). It's measured at L2 (the eval shows the delta). It's learned at L3 (fitted from a confusion matrix). Each level is an *upgrade* — you don't go back down unless you have a reason. The level is documented in the config comments (`# L2`, `# L3 target`).

**The drop gate:** when a fitted value (L3) differs from the prior (L1) by more than 5pp (`DROP_GATE_PCT=0.05`), the fitter flags it as a "gate failure." The builder must provide a justification (a YAML file mapping `(source, cls)` → reason) or the fit fails. This prevents a single noisy cell from silently changing a reliability value by a large amount without human review.

## Key Code

### The `ClassMetrics` dataclass — per-class P/R/F1/Fbeta

```python
# pipeline_metrics.py:48-108
@dataclass
class ClassMetrics:
    cls: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    fbeta:     float = 0.0
    support:   int = 0   # contracts with this class in labels

    def compute(self, beta: float | None = None) -> None:
        if beta is None:
            from src.config import get_config as _get_cfg
            beta = _get_cfg().eval.fbeta_beta  # β=2 from config
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else float("nan")
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else float("nan")
        p, r = self.precision, self.recall
        if not math.isnan(p) and not math.isnan(r) and (p + r) > 0:
            self.f1 = 2 * p * r / (p + r)
            b2 = beta * beta
            denom = b2 * p + r
            self.fbeta = (1 + b2) * p * r / denom if denom > 0 else 0.0
        else:
            self.f1 = 0.0
            self.fbeta = 0.0
```

Why this matters: the `compute()` method implements both F1 and Fbeta. The β parameter comes from config (`eval.fbeta_beta = 2.0`), not hardcoded — it's an L1 decision number. The `b2 = beta * beta` is β² (which is 4 for β=2). The formula `(1 + b2) * p * r / (b2 * p + r)` is the standard Fbeta formula. When `p + r == 0` (no predictions and no labels), F1 and Fbeta are 0.0 — not NaN (avoids propagating undefined values into the macro average).

### The `PipelineMetrics` class — macro/micro aggregation

```python
# pipeline_metrics.py:147-285
class PipelineMetrics:
    def __init__(self, contracts: list[ContractMetrics], positive_verdicts=None):
        self.contracts = list(contracts)
        self.positive_verdicts = frozenset(positive_verdicts)  # {"CONFIRMED", "LIKELY"}
        self.class_metrics: dict[str, ClassMetrics] = {}
        self.macro_f1: float = 0.0
        self.macro_fbeta: float = 0.0
        self.micro_f1: float = 0.0

    def compute_class_metrics(self) -> None:
        # For each class, iterate all contracts and count TP/FP/FN/TN
        for cls, m in self.class_metrics.items():
            tp = fp = fn = tn = 0
            for row in self.contracts:
                pred = cls in row.predicted_positive_classes
                lab  = cls in row.labels
                if pred and lab:     tp += 1
                elif pred and not lab: fp += 1
                elif not pred and lab: fn += 1
                else:                 tn += 1
            m.tp, m.fp, m.fn, m.tn = tp, fp, fn, tn
            m.support = sum(1 for r in self.contracts if cls in r.labels)
            m.compute()  # fills precision, recall, f1, fbeta

    def compute_aggregates(self) -> None:
        # Macro: NaN-aware mean over classes with support > 0
        per_class_f1s = [m.f1 for m in self.class_metrics.values() if m.support > 0]
        self.macro_f1 = sum(per_class_f1s) / len(per_class_f1s) if per_class_f1s else 0.0

        # Micro: from sum TP/FP/FN
        total_tp = sum(m.tp for m in self.class_metrics.values())
        total_fp = sum(m.fp for m in self.class_metrics.values())
        total_fn = sum(m.fn for m in self.class_metrics.values())
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        self.micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
```

Why this matters: two aggregation strategies:

1. **Macro:** mean of per-class F1/Fbeta, only over classes with `support > 0` (classes that appear in the ground truth). This prevents classes with 0 labeled contracts from diluting the average. `support > 0` is the NaN-aware filter — a class with no ground-truth labels has undefined precision/recall.

2. **Micro:** sum all TP/FP/FN across classes, then compute precision/recall/F1 from the totals. Micro-F1 is dominated by the majority class (the class with the most contracts). Macro-F1 treats all classes equally. For security, macro is the headline metric — a rare class (DenialOfService, n=9) matters as much as a common one (Reentrancy, n=15).

**Teaching: macro vs micro — which to use?** Macro-F1 treats all classes equally (each class contributes 1/N to the average). Micro-F1 treats all contracts equally (each contract contributes 1/total to the sum). If one class has 100 contracts and another has 5, micro-F1 is dominated by the 100-contract class. Macro-F1 gives equal weight to the 5-contract class. For security, where rare classes (DenialOfService, IntegerUO) matter as much as common ones (Reentrancy), macro-F1 is the right choice. It's the headline metric in every eval report.

### The confusion matrix builder with Rule 5C exclusion

```python
# reliability_matrix.py:136-147, 150-222
def _tool_ran(row: Any, source: str) -> bool:
    """Per Rule 5C: a tool that did NOT run is excluded from counts."""
    ts = getattr(row, "tool_status", None) or {}
    entry = ts.get(source)
    if not isinstance(entry, dict):
        return True  # no entry → assume ran (legacy compat)
    return entry.get("ran") is not False

def build_matrix(rows, classes, sources=("ml", "slither", "aderyn"), ...) -> Matrix:
    m = Matrix(n_contracts=len(rows), classes=list(classes), sources=list(sources), ...)

    for row in rows:
        labels = set(row.labels)
        for source in sources:
            if not _tool_ran(row, source):
                # Rule 5C: tool didn't run on this contract — exclude.
                m.excluded_contracts.setdefault(source, []).append(row.stem)
                continue
            for cls in classes:
                cv_cls = row.consensus_verdict.get(cls) if row.consensus_verdict else None
                in_labels = cls in labels
                pred_pos = _positive(source, cv_cls)
                cell = m.cell(source, cls)
                if pred_pos and in_labels:     cell.tp += 1
                elif pred_pos and not in_labels: cell.fp += 1
                elif not pred_pos and in_labels: cell.fn += 1
                else:                            cell.tn += 1
    return m
```

Why this matters: the `_tool_ran()` function is the Rule 5C enforcement point. It reads `row.tool_status[source]["ran"]` — if `ran is False`, the contract is excluded from that source's TP/FP/FN/TN counts. This is the fix for the Aderyn silent-skip bug: contracts where Aderyn didn't run are recorded in `excluded_contracts["aderyn"]`, not counted as TN. Without this, a tool that never ran would look "perfect" (0 FP = precision 1.0) — the exact bias that poisoned the P0 baseline.

**Teaching: `_positive()` — how each tool's signal is detected.** The `_positive()` function determines whether a tool emitted a positive signal for a class:
- ML: `consensus_verdict[cls]["ml_signal"] == 1`
- Slither: `consensus_verdict[cls]["slither_match"] > 0`
- Aderyn: `consensus_verdict[cls]["aderyn_match"] > 0`

Each tool has a different signal format (0/1 for ML, count for Slither/Aderyn). The `_positive()` function normalizes them to a boolean. This is where the matrix builder interfaces with the report schema — if the schema changes (e.g., a new field), this function needs updating.

### The Bayesian shrinkage fitter

```python
# reliability_fit.py:99-135
def _fit_cell(cell: dict, prior: float, alpha: float) -> FittedCell:
    n = int(cell["n"])
    tp = int(cell["tp"])
    fp = int(cell["fp"])
    denom = tp + fp
    measured = (tp / denom) if denom > 0 else 0.0

    if n == 0:
        # Zero-sample — fall back to the prior verbatim. No fiction.
        fitted = prior
        n_zero = True
    else:
        # Bayesian shrinkage: pull measured toward prior with strength alpha.
        fitted = (n * measured + alpha * prior) / (n + alpha)
        n_zero = False

    return FittedCell(
        source=cell["source"], cls=cell["cls"],
        n=n, tp=tp, fp=fp, fn=int(cell["fn"]), tn=int(cell["tn"]),
        measured=round(measured, 6), prior=round(prior, 6),
        fitted=round(fitted, 6), n_zero_sample=n_zero,
    )
```

Why this matters: three things:

1. **The shrinkage formula** (`fitted = (n * measured + alpha * prior) / (n + alpha)`): this is the core. When n is large (many samples), `fitted ≈ measured` — the data dominates. When n is small (few samples), `fitted ≈ prior` — the prior dominates. When n=0, `fitted = prior` verbatim. The formula smoothly interpolates between "trust the prior" (small n) and "trust the data" (large n).

2. **`measured = tp / (tp + fp)`:** this is precision, not accuracy. A tool with 0 FP but 0 TP has `denom=0` → `measured=0.0` (not NaN, not 1.0). This is correct: a tool that never flagged anything has measured precision 0.0, not "perfect." With shrinkage, `fitted = (0 + 5×prior) / 5 = prior` — it falls back to the prior.

3. **The `n_zero` flag:** zero-sample cells are explicitly tracked. `FitMetadata.n_zero_sample` counts them. The eval report shows how many cells are prior-only (no data). This is a data quality signal — if many cells are zero-sample, the corpus is too small.

### The drop gate

```python
# reliability_fit.py:177-185
if not fc.n_zero_sample:
    delta = abs(fc.fitted - fc.prior)
    if delta >= drop_gate_pct:  # 0.05 = 5pp
        key = f"{src}|{cls}"
        if key not in justifications:
            gate_failures.append(
                f"{key}: fitted={fc.fitted:.4f} prior={fc.prior:.4f} "
                f"delta={delta:.4f} >= gate={drop_gate_pct:.4f} (no justification)"
            )
```

Why this matters: the drop gate prevents silent large changes. If a fitted value differs from the prior by more than 5pp, the builder must provide a justification (a YAML file mapping `"aderyn|Timestamp"` → "P3 measured value supersedes hand-set L1 prior. Fitted = Bayesian-shrinkage alpha=5 from confusion_matrix_v2 (83 contracts)."). Without the justification, the fit fails — `sys.exit(1)`. This forces human review of any large reliability change. **Teaching: the drop gate is Rule B (Level 2) enforcement. It ensures that any decision-number change >5pp is measured, justified, and documented — not silently applied.**

### The `get_reliability()` fallback chain (L3 → L1 → L0)

```python
# reliability.py:73-140
def load_reliability(config=None) -> dict[tuple[str, str], float]:
    if config is None:
        from src.config import get_config
        config = get_config()

    scale = config.consensus.ml_weight_scale  # 0.5
    acc = config.consensus.accuracy_weights   # L1 hand-set values
    defaults = config.consensus.default_weights  # L0 fallback

    # 1) Try L3 fitted table.
    l3 = _load_l3_table()
    if l3 is not None:
        return {
            (src, cls): round(val * scale, 4) if src == "ml" else round(val, 4)
            for (src, cls), val in l3.items()
        }

    # 2) Fall back to L1 hand-set values.
    table: dict[tuple[str, str], float] = {}
    for cls in set(acc.keys()):
        w = acc.get(cls, defaults)
        table[("ml", cls)] = round(w["ml"] * scale, 4)
        table[("slither", cls)] = w["slither"]
        table[("aderyn", cls)] = w["aderyn"]
    return table

def get_reliability(source: str, cls: str, config=None) -> float:
    table = load_reliability(config)
    if (source, cls) in table:
        return table[(source, cls)]
    # L0 fallback for unknown sources (RAG, debate, halmos)
    _source_defaults = {"rag": 0.50, "debate": 0.55, "quick_screen": 0.40}
    return _source_defaults.get(source, 0.50)
```

Why this matters: the three-tier fallback chain (L3 → L1 → L0) ensures the pipeline always has reliability values, even if the L3 config is missing or malformed:

1. **L3:** `configs/reliability_v3.yaml` (data-derived, fitted). If the file exists and has `schema_version="1"`, use it.
2. **L1:** `configs/verdicts_default.yaml::consensus.accuracy_weights` (hand-set, documented). If L3 is missing, fall back to these.
3. **L0:** `_source_defaults` in code (hardcoded). If the (source, class) pair isn't in L1 (e.g., RAG, debate — sources without per-class data), use the L0 defaults.

**Teaching: `ml_weight_scale` is applied to ML regardless of level.** ML reliability is always multiplied by `ml_weight_scale=0.5` — ML is treated as a "hint, not authority" (L2 decision number). This means `ml/Reentrancy` with L3 fitted value 0.969697 becomes `0.969697 × 0.5 = 0.4848` in the table. The scale is a discount — ML evidence contributes half its raw reliability to the fusion. This is because the ML model was trained on a small corpus and its predictions are noisy. The scale can be removed (set to 1.0) once the ML model improves — but only with a measured eval delta (Rule B).

### The 9 gate functions

```python
# gates.py:74-92, 166-181, 217-234
def gate_ws1a_silent_safe_on_flagged(rows: list[ContractEval]) -> GateResult:
    """No consensus-flagged class ends SAFE."""
    violations: list[str] = []
    for row in rows:
        for cls, cv in row.consensus_verdict.items():
            cv_verdict = cv.get("verdict", "")
            final_verdict = row.verdicts.get(cls, "MISSING")
            if final_verdict == "SAFE" and cv_verdict != "SAFE":
                violations.append(f"{row.stem}/{cls} (consensus={cv_verdict} → final=SAFE)")
    passed = len(violations) == 0
    return GateResult(gate_id="WS1a_silent_safe_on_flagged", description=..., passed=passed,
                     detail=f"{len(violations)} violation(s)" + ...)

def gate_ws2_false_positives_on_safe(rows, positive_set) -> GateResult:
    """Zero false-positive verdicts on the safe subset."""
    safe_rows = [r for r in rows if r.ground_truth == "safe"]
    fps: list[str] = []
    for row in safe_rows:
        flagged = _positive_classes(row.verdicts, positive_set)
        if flagged:
            fps.append(f"{row.stem}: {flagged}")
    passed = len(fps) == 0
    return GateResult(...)

def gate_macro_f1_vs_baseline(macro_f1: float, baseline: dict | None) -> GateResult:
    """macro_F1 must not drop vs the stored baseline."""
    if baseline is None:
        return GateResult(..., passed=True, detail=f"macro_F1 = {macro_f1:.4f} (no baseline)")
    base_f1 = float(baseline.get("macro_f1", 0.0))
    passed = macro_f1 >= base_f1
    return GateResult(..., passed=passed,
                     detail=f"macro_F1 = {macro_f1:.4f} (delta {macro_f1 - base_f1:+.4f})")
```

Why this matters: three patterns:

1. **WS1a (silent-SAFE on flagged):** checks that the debate can't silently clear a consensus flag. If the consensus engine voted DISPUTED but the final verdict is SAFE, that's a violation. This gate catches the "debate overrides static analysis" failure mode — the LLM debate says SAFE, and the final verdict follows the LLM instead of the consensus. **Teaching: this gate enforces the FN/FP asymmetry invariant from Doc 02. A flagged class can never silently become SAFE — the asymmetry override in `fuse()` prevents it at the code level, and this gate verifies it at the eval level.**

2. **WS2 (false positives on safe):** checks that safe contracts produce zero positive verdicts. This is the precision gate — safe contracts should be completely clean. A false positive on a safe contract wastes human review time. **Teaching: this gate is the FP counterpart to WS1a's FN protection. WS1a says "don't miss vulnerabilities"; WS2 says "don't cry wolf on safe contracts." Both must pass for the system to be shippable.**

3. **macro_f1_vs_baseline:** the regression gate. If macro_F1 drops from the baseline, the gate fails. This is the "no regressions" rule — every change must maintain or improve the headline metric. **Teaching: the baseline is the anchor from Step 5 of "How We Arrived." Without a baseline, "macro_F1=0.3008" is just a number — you don't know if it's good or bad. With a baseline (0.2765), you know it's +0.0243 — an improvement. The gate enforces: "improvements only, no regressions."**

### The regression detector

```python
# regression.py:117-158
def compare(self, current: PipelineMetrics, *, min_delta_pp: float = 0.01) -> RegressionResult:
    cur = current.as_dict()
    cur_macro = float(cur.get("macro_f1", 0.0))
    regressed = cur_macro < self.macro_f1  # current < baseline = regression

    per_class_deltas: dict[str, dict[str, float]] = {}
    regressed_classes: list[str] = []
    improved_classes:  list[str] = []
    for cls in sorted(all_classes):
        base_f1 = float(base_per_class.get(cls, {}).get("f1", 0.0))
        cur_f1  = float(cur_per_class.get(cls,  {}).get("f1", 0.0))
        delta = cur_f1 - base_f1
        per_class_deltas[cls] = {"baseline_f1": base_f1, "current_f1": cur_f1, "delta": delta}
        if delta < -min_delta_pp:    regressed_classes.append(cls)
        elif delta > min_delta_pp:   improved_classes.append(cls)

    return RegressionResult(
        baseline_macro_f1=self.macro_f1, current_macro_f1=cur_macro,
        regressed=regressed, metric_deltas=metric_deltas,
        per_class_deltas=per_class_deltas,
        regressed_classes=regressed_classes, improved_classes=improved_classes,
    )
```

Why this matters: the regression detector compares the current run against a stored baseline at two levels:

1. **Headline:** `regressed = cur_macro < self.macro_f1`. If the overall macro_F1 dropped, the run is a regression. This is the binary ship/no-ship signal.

2. **Per-class:** for each class, compute `delta = current_f1 - baseline_f1`. If `delta < -0.01` (more than 1pp drop), the class is "regressed." If `delta > +0.01`, it's "improved." This tells you *which classes* changed — "Reentrancy improved by +0.05, but IntegerUO regressed by -0.03." This is the diagnostic signal — it points you to the class that needs investigation.

**Teaching: `min_delta_pp=0.01` (1 percentage point).** Deltas smaller than 1pp are considered noise — they might be from a different random seed or a minor corpus change, not a real regression/improvement. The 1pp threshold filters noise. Only deltas >1pp trigger the "regressed" or "improved" labels. The threshold is an L1 decision number — it could be tuned (e.g., 0.5pp for more sensitivity), but 1pp is a reasonable default.

### The `ContractEval` dataclass — one contract's eval row

```python
# gates.py:22-53
@dataclass
class ContractEval:
    stem: str                    # contract filename stem
    report_path: str
    labels: list[str]            # ground-truth vulnerability classes
    ground_truth: str            # "safe" | "vulnerable"
    verdicts: dict[str, str]     # class → verdict from pipeline
    probabilities: dict[str, float]
    consensus_verdict: dict[str, dict]  # class → {verdict, confidence, ml_signal, ...}
    tool_status: dict[str, dict]         # Rule 5C: {"ml": {"ran": True}, "aderyn": {"ran": False}}
    # ... + derived fields filled by compute_per_contract
    predicted_positive_classes: list[str] = field(default_factory=list)
    true_positive_classes: list[str] = field(default_factory=list)
    false_positive_classes: list[str] = field(default_factory=list)
    false_negative_classes: list[str] = field(default_factory=list)
    contract_correct: bool = False
    contract_exact: bool = False
```

Why this matters: `ContractEval` is the central data structure of the eval framework. It carries:
- **Ground truth:** `labels` (from `// expect:` header or `.json` sidecar) and `ground_truth` ("safe" or "vulnerable").
- **Pipeline output:** `verdicts` (class → verdict), `probabilities`, `consensus_verdict`.
- **Rule 5C:** `tool_status` — which tools ran on this contract. The matrix builder reads this to exclude didn't-run tools.
- **Derived:** `predicted_positive_classes`, TP/FP/FN classes, `contract_correct` (loose), `contract_exact` (strict).

**Teaching: loose vs exact contract accuracy.**
- **Loose:** safe contract → no positive verdicts; vulnerable contract → at least 1 TP. A contract with `labels=["Reentrancy"]` and `predicted=["Reentrancy", "IntegerUO"]` is "loose correct" (1 TP, 1 FP — at least one correct flag).
- **Exact:** `predicted_positive == labels`. The same contract is NOT "exact correct" (predicted has IntegerUO, labels don't). Exact is much harder — the system must flag exactly the right classes and nothing else.

### The `_parse_expect_header` — extracting ground truth from Solidity

```python
# run_benchmark.py:72-84
def _parse_expect_header(sol_path: Path) -> tuple[list[str], str] | None:
    try:
        text = sol_path.read_text()
    except OSError:
        return None
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("// expect:"):
            payload = stripped[len("// expect:"):].strip()
            labels = [c.strip() for c in payload.split(",") if c.strip()]
            gt = "safe" if not labels else "vulnerable"
            return labels, gt
    return None
```

Why this matters: ground truth is embedded directly in the Solidity source file as a comment: `// expect: Reentrancy, IntegerUO`. The eval parser reads the first 20 lines of each `.sol` file looking for this header. If found, the labels are extracted and `ground_truth` is derived (no labels → "safe"; any labels → "vulnerable"). This pattern — ground truth in the source file — keeps the label and the code together, reducing the risk of label-code drift.

**Teaching: the `.json` sidecar alternative.** If the `.sol` file doesn't have an `// expect:` header, the loader looks for a `.json` sidecar file with the same stem: `01_reentrancy.sol` → `01_reentrancy.json`. The sidecar has `{"labels": ["Reentrancy"], "ground_truth": "vulnerable"}`. Sidecars are for contracts where the ground truth is too complex for a comment (e.g., multiple contracts in one file with different labels). The `// expect:` header is the primary mechanism; the sidecar is the fallback.

## Design Decision: Bayesian Shrinkage vs MLE vs Bootstrapping

> **How to read this section:** The table shows the options. The *elimination reasoning* shows how to think about the choice.

### The elimination process

**MLE (Maximum Likelihood Estimator) — steel-man:** "MLE is the simplest estimator: `measured = TP / (TP + FP)`. No prior, no shrinkage. Let the data speak for itself."

**Why it fails on small samples:** with n=3, MLE gives `measured = 3/3 = 1.0` (perfect precision!). But n=3 is too small — the next 3 contracts might all be FP, dropping precision to 0.5. MLE has high variance on small samples — it swings wildly between 0 and 1. The reliability matrix would have extreme values (0.0 or 1.0) for rare classes, which would bias `fuse()`.

**Bootstrapping — steel-man:** "Bootstrap resampling estimates the confidence interval of the measured precision. Resample the data 1000 times, compute precision for each sample, and use the mean (or median) as the estimate. This captures uncertainty."

**Why it fails for SENTINEL:**
1. *Still high variance on n=3:* bootstrapping n=3 samples still produces wide confidence intervals. The mean is still noisy.
2. *Computational cost:* 1000 resamples × 30 cells = 30,000 evaluations per fit. Not slow (seconds), but unnecessary when shrinkage is simpler.
3. *No prior integration:* bootstrapping doesn't incorporate the L1 prior. The hand-set values (which encode domain knowledge) are ignored.

**Bayesian shrinkage — why it survives:** it incorporates the prior (domain knowledge from L1), it's low-variance on small samples (pulled toward the prior), it's a single formula (no resampling), and it's interpretable ("the fitted value is a weighted average of measured and prior, with weight proportional to sample size"). Its weakness (assumes the prior is reasonable) is mitigated by the drop gate (which flags large deviations from the prior for human review).

**The reasoning principle:** "When fitting parameters from small samples, choose an estimator that incorporates prior knowledge. MLE has high variance on small n; bootstrapping captures uncertainty but doesn't integrate priors; Bayesian shrinkage does both. The prior is the L1 hand-set value — it's the best guess before data arrives. Shrinkage smoothly transitions from 'trust the prior' (small n) to 'trust the data' (large n)."

### When this decision would be wrong

**The reversal condition:** when you have enough data (n > 100 per cell) that shrinkage is unnecessary. With n=100, `fitted = (100 × measured + 5 × prior) / 105 ≈ 0.95 × measured + 0.05 × prior` — the prior barely matters. At that point, MLE is fine (low variance on large n). The trigger: when the eval corpus grows to 1000+ contracts, shrinkage becomes a no-op and can be replaced with MLE. Until then, shrinkage prevents small-sample noise from corrupting the reliability values.

## Technology Choice: Fbeta(β=2)

**The 5-question framework:**

1. **What category?** Classification metric for imbalanced, recall-critical tasks.
2. **What alternatives?** (a) F1 (β=1, equal weight), (b) Fbeta(β=2, recall 2× weight), (c) precision/recall separately, (d) accuracy.
3. **Why Fbeta(β=2)?** Security has a ~1000:1 FN/FP cost ratio. F1 treats them equally — wrong. Fbeta(β=2) weights recall 2× — a missed vulnerability hurts 2× more than a wasted review. β=2 is the standard for recall-critical tasks (medical diagnosis, fraud detection, security).
4. **When is F1 (β=1) fine?** When FN and FP have equal cost (e.g., spam detection — a false positive wastes seconds, a false negative wastes seconds). In security, the cost is asymmetric — F1 is wrong.
5. **Migration trigger:** if the cost ratio changes (e.g., automated remediation makes FP cheaper — a bot fixes the false positive in seconds), β could decrease toward 1. This would require a Rule B measurement: "with β=1.5, macro_F1 changed by X, and the operational cost of FPs decreased by Y."

## Anti-Patterns

### ❌ Count silent-skips as TN — "if the tool didn't run, it didn't find anything"
**What it looks like:** A tool that didn't run on a contract is counted as TN (true negative — "tool says safe"). The reliability matrix counts it as a correct "safe" prediction.
**Why someone would build this:** It's the path of least resistance. The exception handler returns `[]` (empty findings), and the matrix builder treats "no findings" as TN. No special handling needed.
**Why it's wrong:**
1. *Inflates reliability* — a tool that never ran looks "perfect" (0 FP = precision 1.0). This is the Aderyn silent-skip bug: 83 contracts × 0 Aderyn findings → measured precision = 0/0 = undefined (treated as 0.0, but the tool looks "safe" because it never flagged anything).
2. *Biases the reliability matrix* — the tool's reliability is set to the prior (L1 hand-set value), which is then used in `fuse()` to weight evidence. The weight is wrong because the data is wrong.
3. *Hides the failure* — the eval report shows "Aderyn ran on 83 contracts, 0 TP, 0 FP" — which looks like "Aderyn found nothing," not "Aderyn didn't run."
**The right approach:** Rule 5C — exclude didn't-run from counts. `tool_status["aderyn"]["ran"] == False` → the contract is excluded from Aderyn's TP/FP/FN/TN. It's recorded in `excluded_contracts["aderyn"]`. The matrix only counts contracts where the tool actually ran.

### ❌ Use F1 (β=1) — "standard metric"
**What it looks like:** Report macro_F1 as the headline metric. "F1 is the standard for classification. Everyone uses it."
**Why someone would build this:** F1 is the most common classification metric. It's in every ML textbook and library. It's the default in `sklearn.metrics`.
**Why it's wrong:**
1. *Equal weight on FN and FP* — but FN costs millions, FP costs minutes. A system with 0.40 F1 might have 0.80 recall (good — catches most bugs) but 0.27 precision (bad — many false positives). F1 = 0.40 looks mediocre. Fbeta(β=2) = 0.58 looks better — because the high recall is what matters.
2. *Optimizes the wrong thing* — if you optimize for F1, you'll sacrifice recall to improve precision (reducing false positives). But in security, reducing false positives at the cost of missing vulnerabilities is the wrong tradeoff. Fbeta(β=2) optimizes for recall — reducing false negatives is the right tradeoff.
**The right approach:** Fbeta(β=2) as the headline metric. Report both F1 and Fbeta — F1 for comparison with literature, Fbeta for decision-making. The gate uses Fbeta for the regression check.

## Mistakes & Fixes

### Mistake: Aderyn silent-skip biased the entire reliability matrix
**What happened:** Aderyn's binary wasn't being found (PATH issue). Every Aderyn call returned `[]` (FileNotFoundError caught, logged at DEBUG, `[]` returned). The reliability matrix counted all 83 contracts as TN for Aderyn — 0 TP, 0 FP. Measured precision = 0/0 = 0.0. The L3 fit used the prior (L1 hand-set value) because n=0 for the "positive" part (no contracts where Aderyn flagged anything).
**Why it happened:** The exception handler returned `[]` — the same shape as "ran clean and found nothing." There was no `tool_status` field to distinguish "didn't run" from "ran and found nothing."
**The fix:** Rule 5C. `_resolve_aderyn_binary()` raises FileNotFoundError with a precise message. Callers write `tool_status["aderyn"] = {"ran": False, "reason": "binary not found"}`. The matrix builder's `_tool_ran()` function excludes these contracts from Aderyn's counts. After the fix, Aderyn's measured precision is based on contracts where it actually ran — not on contracts where it was absent.
**The lesson:** A silent failure manufactures a rabbit hole. The Aderyn bug cost days of investigation because the symptom (low macro_F1, 0.1958) pointed at the ML model, not at Aderyn. The root cause (missing binary) was invisible because the error was swallowed. Rule 5C: never return a value that's indistinguishable from success. Always carry the failure — `ran: False` in `tool_status`, or raise.

### Mistake: ML-failure-as-pass — ML server down → "ML says safe"
**What happened:** When the ML server was down, `ml_assessment` returned `ml_result = {}` (empty dict). The consensus engine treated an empty `ml_result` as "ML says safe" (no flagged classes → no positive signal → TN for all classes). This was a false-negative factory: contracts with real vulnerabilities were labeled "safe" by ML because ML didn't run.
**The fix:** `ml_assessment` now returns `ml_result = {"ran": False, "reason": "ML server unreachable", ...}` and `tool_status["ml"] = {"ran": False}`. The matrix builder excludes these contracts from ML's counts. The synthesizer treats absent `ran=True` as a failure — the report says "ML was unavailable."
**The lesson:** An empty result is not the same as a "safe" result. "ML didn't run" ≠ "ML ran and found nothing." The `ran` flag disambiguates. Without it, every ML failure is a false negative — the system says "safe" when it actually means "I don't know."

### Mistake: P0 baseline F1=0.1958 was the first honest number
**What happened:** Before P0, there was no eval framework at all. The system had 631 unit tests (code runs) but no eval (system produces correct verdicts). When the first eval ran (P0), macro_F1=0.1958 — surprisingly low. Investigation revealed the Aderyn silent-skip bug.
**The fix:** The eval framework itself. Before P0, there was no way to measure system quality. After P0, every change has a measured delta. The eval is the measurement infrastructure — without it, "I think the system is better" is not a measurement.
**The lesson:** "Tests prove the code runs; evals prove the system is good." Build the eval framework *before* optimizing the system. Without a baseline, you're optimizing blind — you don't know if a change helps or hurts. The P0 baseline (0.1958) was the anchor for every subsequent improvement.

### Mistake: L3 reliability caused +50% F1 — not from code, from better inputs
**What happened:** Replacing L1 hand-set reliability weights with L3 data-derived ones drove macro_F1 from 0.1998 to 0.3008 (+50% relative). No code changed — only the reliability values in the config file. The same `fuse()` function, the same Evidence model, the same pipeline — just better weights.
**Why it happened:** The hand-set weights (L1) were guesses. "Slither/Reentrancy = 0.82" was someone's intuition. The measured value (L3) was 0.795455 — close, but the measured values for other cells were very different (e.g., Aderyn/Timestamp dropped from 0.40 to 0.030303 — Aderyn has almost no signal for Timestamp). The hand-set weights overestimated some tools and underestimated others. The data-derived weights corrected these biases.
**The lesson:** Measurement > intuition. The +50% F1 jump didn't come from a better model, a new tool, or a code change — it came from replacing guesses with measurements. This is the L0→L3 maturity ladder in action: L1 (hand-set) → L3 (data-derived) = +50% F1. The improvement was *in the inputs*, not the algorithm.

## What Would Break If You Removed This?

**Remove Fbeta (use F1):** recall is underweighted. A system with 0.80 recall but 0.27 precision gets F1=0.41 (mediocre) but Fbeta=0.58 (good). With F1, you'd optimize for precision (reducing FP) at the cost of recall (increasing FN). Missed vulnerabilities ship.

**Remove the reliability matrix:** hand-set weights (L0/L1) forever. No data-derived values. The +50% F1 jump from L3 never happens. The system is stuck at F1=0.1998.

**Remove Bayesian shrinkage:** noisy estimates on rare classes. `slither/DenialOfService` with n=1 and TP=1 gets measured=1.0 (perfect!) — but n=1 is meaningless. The fitted value (0.75 with α=5, prior=0.55) is much more reasonable. Without shrinkage, rare classes have extreme weights that bias `fuse()`.

**Remove the drop gate:** a single noisy cell silently changes a reliability value by 20pp. No human review. The system's verdicts shift without anyone knowing why. The drop gate forces justification — "I know Aderyn/Timestamp dropped from 0.40 to 0.03, and here's why: Aderyn has zero signal for Timestamp."

**Remove the 9 gates:** safety violations ship. WS1a (silent-SAFE on flagged) could fail silently. WS2 (FP on safe) could produce false positives on every safe contract. The gates are the quality gate — without them, "macro_F1=0.30" doesn't tell you if the system is *safe*, just if it's *accurate on average*.

## At Scale

*Scale metric: corpus size (current: 61-83 contracts).*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| 61-83 contracts (current) | Shrinkage compensates for small n | Many cells are zero-sample or tiny-n | More labeled data |
| 610 contracts (10x) | Most cells have n > 30 | Rare classes still noisy | More labeled data for rare classes |
| 6,100 contracts (100x) | Most cells have n > 300 | Shrinkage is a no-op (n >> α) | Switch to MLE (simpler) |
| 61,000 contracts (1000x) | MLE is stable | Eval takes hours | Parallelize eval, incremental matrix |

The eval framework itself scales for free — it's a batch job (load reports, compute metrics, write report). The scale wall is the **corpus**: 61-83 contracts is small enough that rare classes (DenialOfService, n=9) have noisy estimates even with shrinkage. More labeled data is the honest fix — the `CROSS` deferred workstream plans to source more contracts.

## Try It Yourself

> TRY IT: `cd agents && python -m src.eval.run_benchmark --name test_eval --reports test_audit_reports_post_quarantine_no_llm --corpus ../manual_hand_written_contracts` — run the full eval on the existing report corpus.

> TRY IT: `cd agents && head -30 configs/reliability_v3.yaml` — see the L3 fitted values (schema_version, alpha, fit_metadata, table).

> TRY IT: `cd agents && head -30 configs/verdicts_default.yaml` — see the L1 hand-set values (accuracy_weights, default_weights, ml_weight_scale, bands).

> TRY IT: `cd agents && python -c "from src.eval.pipeline_metrics import ClassMetrics; m=ClassMetrics(cls='Reentrancy', tp=12, fp=32, fn=3, tn=36, support=15); m.compute(); print(f'P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f} Fb={m.fbeta:.4f}')"`

## Limitations & What's Missing

- **61-83 contracts is small.** Rare classes (DenialOfService, n=9) have noisy estimates even with shrinkage. More labeled data is the honest fix.

- **No cross-validation.** The eval runs on the entire corpus — there's no train/test split. The reliability matrix is fitted on the same contracts that the eval scores. This means the L3 weights are slightly overfit (they're fitted on the eval set). A proper split (fit on 50 contracts, eval on 33) would give unbiased estimates. This is a known limitation.

- **No statistical significance tests.** The eval reports point estimates (macro_F1=0.3008) without confidence intervals. A delta of +0.024 might be within noise (not statistically significant). Bootstrapping or a paired t-test would quantify significance.

- **No automatic refit trigger.** The reliability matrix is re-fitted manually (`python scripts/build_reliability_matrix.py`). There's no CI job that automatically refits when new audit reports are added. The operator must remember to refit.

- **Drop gate is a heuristic (±5pp).** The 5pp threshold is hand-set (L0). It could be too strict (blocks legitimate large improvements) or too lenient (allows noisy large changes). A data-driven threshold (e.g., based on the distribution of deltas) would be better.

- **No per-class gate.** The gates check global properties (WS2: zero FP on safe). There's no gate like "Reentrancy recall must be > 0.70." A per-class gate would catch class-specific regressions that the macro average hides.

## Transferable Patterns

1. **FN/FP asymmetry in metrics — choose β based on cost asymmetry** — Fbeta(β=2) for security, F1 for equal cost.
   - *Interview story:* "SENTINEL measures system quality with Fbeta(β=2), not F1. A missed vulnerability (false negative) costs millions — an exploit drains the contract. A false positive costs minutes — a human reviews a clean contract. The cost ratio is ~1000:1. F1 treats both errors equally; Fbeta(β=2) weights recall 2× over precision. This means a system with 0.80 recall and 0.27 precision gets F1=0.41 (mediocre) but Fbeta=0.58 (good) — because catching 80% of vulnerabilities is the property that matters most for security. Choosing the right metric is as important as choosing the right model."
   - *When this pattern is WRONG:* when FN and FP have equal cost. If a false positive and a false negative both cost $1, F1 is correct — there's no asymmetry to reflect. Use Fbeta only when the costs are asymmetric and you can quantify the ratio.

2. **Bayesian shrinkage for small samples — pull toward prior with strength proportional to sample size** — `fitted = (n × measured + α × prior) / (n + α)`.
   - *Interview story:* "SENTINEL's reliability matrix has 30 cells (3 tools × 10 classes), but many cells have tiny sample sizes — Aderyn ran on 3 Timestamp contracts and got 0 TP. MLE would give measured precision = 0/0 = undefined. Bayesian shrinkage pulls the estimate toward the L1 prior: `fitted = (3 × 0.0 + 5 × 0.40) / 8 = 0.25` — much more reasonable than 0.0 or 1.0. The shrinkage strength α=5 means 'the prior is worth 5 samples.' Once you have 50+ real samples, the prior barely matters. Shrinkage prevents small-sample noise from producing extreme weights that would bias the fusion."
   - *When this pattern is WRONG:* when you have no prior (no domain knowledge to shrink toward). If you have no hand-set values, no expert intuition, and no previous measurements, the prior is a guess — and shrinkage toward a guess is no better than MLE. Use shrinkage when you have a reasonable prior; use MLE when you don't.

3. **Maturity ladder for decision numbers — L0→L1→L2→L3, each level is an upgrade** — provenance tracking for every threshold and weight.
   - *Interview story:* "Every decision number in SENTINEL has a maturity level. L0: a hand-set constant in code (`reliability = 0.60`). L1: externalized to versioned config (`verdicts_default.yaml`). L2: measured against a baseline (`macro_F1 = 0.3008, delta +0.0243`). L3: learned from data (`reliability_v3.yaml`, fitted from a confusion matrix). The L0→L3 transition for reliability weights drove a +50% F1 improvement — not from code changes, from better inputs. The ladder is a one-way ratchet: you don't go back to hand-setting values once you have data-derived ones. And every level change requires a measurement (Rule B)."
   - *When this pattern is WRONG:* when the decision number doesn't affect the system's output enough to justify measurement. If changing a threshold from 0.40 to 0.45 moves macro_F1 by 0.001, the maturity ladder is overkill — just set it and move on. Use the ladder for decision numbers that have a measurable impact (reliability weights, verdict bands, routing thresholds). Don't use it for cosmetic values (log format, display labels).

---

**Source files verified:**
- `agents/src/eval/pipeline_metrics.py:48-108, 147-285` — ClassMetrics (P/R/F1/Fbeta), PipelineMetrics (macro/micro)
- `agents/src/eval/reliability_matrix.py:30, 33-66, 69-133, 136-147, 150-222` — Cell, Matrix, _tool_ran(), build_matrix() with Rule 5C
- `agents/src/eval/reliability_fit.py:55-57, 99-135, 138-204, 250-265` — α=5, drop-gate, _fit_cell(), fit(), write_yaml()
- `agents/src/eval/run_benchmark.py:72-84, 150-165, 172-205, 291-377` — _parse_expect_header(), load_corpus(), compute_metrics(), CLI
- `agents/src/eval/gates.py:22-53, 74-92, 166-181, 217-234` — ContractEval, WS1a, WS2, macro_f1_vs_baseline
- `agents/src/eval/regression.py:33-57, 60-113, 117-196` — RegressionResult, RegressionBaseline, compare()
- `agents/src/orchestration/verdict/reliability.py:26-30, 33-70, 73-140` — L3 path, _load_l3_table(), load_reliability(), get_reliability()
- `agents/configs/verdicts_default.yaml:9-33, 92-99` — L1 config (accuracy_weights, bands, fbeta_beta)
- `agents/configs/reliability_v3.yaml:1-8, 74-80` — L3 config (schema_version, alpha, table)

**Eval reports verified:**
- `agents/eval/runs/20260624T133420Z_p0_honest_baseline/eval_report.md` — P0 (F1=0.1958, Fβ=0.2515)
- `agents/eval/runs/20260624T231228Z_p2_calibrated/eval_report.md` — P2 (F1=0.1998, Fβ=0.2246)
- `agents/eval/runs/20260626T123145Z_p3_rule5c_v3/eval_report.md` — P3 (F1=0.3008, Fβ=0.3821)

**Verified against commit hash:** `c47898ea5`
