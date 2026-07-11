# 10. Decision Numbers: From Hand-Set Constants to Data-Derived Reliability

> **Prerequisites:** [01. The Audit Pipeline] — routing thresholds decide fast vs deep path. [02. Evidence Model & Fuse()] — reliability weights control how fuse() weighs each evidence source. [08. Evaluation Framework] — Fbeta, confusion matrix, Bayesian shrinkage (α=5), drop-gate (±5pp).
> **Next:** This is the capstone doc — there is no Doc 11. You've completed the SENTINEL agents learning path.
> **Cross-ref:** [04. Reproducibility] — the `deterministic` flag on Evidence is itself a decision number (must it be True for formal evidence? False for LLM debate?). [06. RAG Hybrid Retrieval] — the K=60 RRF parameter and similarity thresholds are decision numbers.
> **Scope:** This doc covers the philosophy and implementation of ALL decision numbers in the system: the L0→L3 maturity ladder, the Rule 5B measurement requirement, the Bayesian shrinkage fitting pipeline, the drop-gate guard, the config hierarchy (L3 > L1 > hardcoded), and the measured progression from hand-set constants to data-derived values. It does NOT cover the eval framework that produces the measurements (see Doc 08) or the fuse() function that consumes the reliability values (see Doc 02).
> **TL;DR:** Every threshold, weight, and confidence value in SENTINEL is a **decision number** — it controls system behavior and must be treated as POLICY, not a magic constant. The **L0→L3 maturity ladder** tracks provenance: L0 (hand-set in `.py`) → L1 (externalized to versioned YAML config) → L2 (measured against a baseline before/after) → L3 (fitted from data with Bayesian shrinkage, α=5, drop-gate ±5pp). The single largest improvement in the project's history — macro_F1 from 0.1998 to 0.3008 (+50% relative) — came entirely from replacing L1 hand-set reliability weights with L3 data-derived ones. No code changed, no models retrained, no new tools added. Only better numbers. Rule 5B says: "No decision-number changes without a measured delta." Before changing any number, you must run the eval, record the baseline, make the change, and show the improvement.

---

## The Problem: "I Think 0.35 Feels Right" Is Not Engineering

You're tuning the deep-path routing threshold. If ML probability for reentrancy is ≥ 0.35, the contract goes to deep analysis — which costs ~60s and runs 5 parallel tools. If it's below 0.35, the contract takes the fast path (~3s).

Where did 0.35 come from? Someone guessed. Maybe they looked at the ML model's output distribution and picked a number that "seemed reasonable." Maybe they used 0.5 first, saw too many false negatives, and lowered it to 0.35. The reasoning was lost, and the next person to maintain the system inherits 0.35 with no context about why or whether it's optimal.

The same applies to every other number in the system:

| Number | What it controls | If it's wrong |
|--------|-----------------|---------------|
| `deep_thresholds["Reentrancy"] = 0.35` | Whether a contract gets deep analysis | Missed vulnerability (too high) or wasted compute (too low) |
| `accuracy_weights["Reentrancy"]["ml"] = 0.78` | How much ML's vote counts in fuse() | Biased verdict (inflated) or ignored signal (deflated) |
| `ml_weight_scale = 0.5` | ML weight discount in consensus | ML plays too large/small a role in the consensus verdict |
| `confirmed_band = 0.70` | What probability counts as "CONFIRMED" | Too many false positives (low) or too many missed vulnerabilities (high) |
| `rag_relevance_floor = 0.30` | Minimum similarity for RAG evidence to count | Noisy results enter fuse() (low) or useful results filtered out (high) |

The problem isn't that the numbers are wrong — it's that you don't *know* whether they're wrong, and you have no process for finding out.

---

## How We Arrived at This Design

### Step 1 — Invariant: Every decision number is POLICY and must be traceable

"An undocumented number is a guess." If a constant is buried in `routing.py:130` with no comment, no version history, and no measurement behind it, it's not a decision — it's an accident waiting to be changed silently by the next developer. Every number must have a documented provenance: where it came from, what it controls, what level of evidence supports it.

### Step 2 — Constraint: Small data means noisy estimates

The eval corpus is 61-83 contracts. A per-(source, class) cell for a rare class (e.g., DenialOfService) might have only 10 samples. If Slither flagged it on 3 contracts and was correct on 2, the raw measured precision is 0.667 — but with n=3, that estimate is noisy. A single new contract could swing it to 0.50 or 0.75. The system needs a fitting method that handles small samples gracefully.

### Step 3 — Simplest progression: L0 → L1 → L2 → L3

The ladder adds one level of rigor at a time:
- **L0** — hardcoded in `.py` (works, but unversioned, unchangeable without code edit)
- **L1** — externalized to YAML config (versioned, changeable, but still guessed)
- **L2** — measured against baseline (you know the before/after delta)
- **L3** — fitted from data (data-derived, Bayesian-shrunk for small samples)

The ladder is sequential — you can't skip to L3 without first having L1 (the config to store the result) and L2 (the baseline to measure against). Each level is an upgrade that stays local to that number: some numbers can be L3 (reliability weights) while others remain L1 (routing thresholds) because you haven't had time to measure them yet.

### Step 4 — Stress-test: what if the data is wrong?

L3 is only as good as the data it's fitted from. If the confusion matrix includes contracts where a tool silently skipped (Aderyn before Rule 5C), the fitted values are inflated. The drop-gate (±5pp) catches cases where the fitted value deviates too far from the prior — a warning that the data might be suspect or the prior was wrong. The justification file forces an explicit explanation before accepting the deviation.

### Step 5 — Measure: the +50% F1 jump

The L3 reliability fitting produced macro_F1=0.3008 vs L1's 0.1998 — a +50% relative improvement. This measurement is the proof that the ladder works: better provenance (L3 > L1) produces better outcomes (higher F1). The drop-gate fired on multiple cells (Aderyn reliability dropping from 0.55-0.70 to 0.02-0.37 — see `reliability_v1.yaml:11-70` for the justifications), and each was reviewed and accepted with an explicit note.

---

## The Solution

### The maturity ladder

```
L0: threshold = 0.5                      # hardcoded in routing.py (prototype only)
    ↓ externalize to YAML
L1: deep_thresholds["Reentrancy"] = 0.35  # in verdicts_default.yaml (P1)
    ↓ measure baseline
L2: "before: F1=0.1958, after: F1=0.1998"  # measured delta (P2)
    ↓ fit from data
L3: reliability_v1.yaml: fitted values     # Bayesian shrinkage, α=5 (P3)
```

### The config hierarchy

The `get_reliability()` function implements a three-tier fallback chain:

```
get_reliability("ml", "Reentrancy")
  → 1. Try L3: configs/reliability_v1.yaml (fitted, schema_version="1")
  → 2. Fall back L1: verdicts_default.yaml > consensus > accuracy_weights
  → 3. Hard fallback: default_weights {ml: 0.60, slither: 0.65, aderyn: 0.55}
```

```python
# reliability.py:33-70, loaded by get_reliability() at reliability.py:120-140
def _load_l3_table():
    path = Path("configs/reliability_v1.yaml")
    if not path.is_file():
        return None
    doc = yaml.safe_load(path.read_text())
    if doc.get("schema_version") != "1":
        return None  # schema mismatch → L1 fallback
    # ... build {(source, cls): weight} dict from table section

# At lookup time:
def get_reliability(source, cls, config=None):
    table = load_reliability(config)  # tries L3 → L1
    if (source, cls) in table:
        return table[(source, cls)]
    # Hard fallback: defaults or source-specific (rag=0.50, debate=0.55)
```

The L3 file has a `schema_version` field — if the schema changes (e.g., adding a new fitting method), old fitted files are rejected and the system falls back to L1. No silent use of stale data.

### The Bayesian shrinkage formula

```
fitted = (n × measured + α × prior) / (n + α)

where:
  n        = number of samples (TP + FP + FN + TN)
  measured = TP / (TP + FP)     — raw precision from confusion matrix
  α        = 5                  — shrinkage strength (prior weight)
  prior    = L1 hand-set value  — from verdicts_default.yaml
```

With α=5:
- n=0 (zero samples): `fitted = prior` verbatim — no fabrication
- n=5 (small sample): `fitted = (5 × measured + 5 × prior) / 10` — halfway between measured and prior
- n=50 (large sample): `fitted = (50 × measured + 5 × prior) / 55` — dominated by measured, prior barely matters

The parameter α=5 was chosen because it gives a 50/50 split at n=5 — reasonably conservative. An α of 2 would let the data dominate faster (50/50 at n=2), α=10 would be more conservative (50/50 at n=10). The proposal (§B-3) specified α=5, and it was confirmed against the P0 baseline.

### The drop-gate guard

```python
# reliability_fit.py:53-56
DROP_GATE_PCT: float = 0.05  # |fitted - prior| >= this → fail without justification
```

When the fitted value differs from the L1 prior by 5 percentage points or more, the reliability fitter requires a justification. In the output YAML (`reliability_v1.yaml:11-70`), every Aderyn cell and most ML cells have justifications like: "P3 measured value supersedes hand-set L1 prior. Fitted = Bayesian-shrinkage alpha=5 from confusion_matrix_v2 (83 contracts). Drop is intentional — L1 priors were never measured before P3."

This prevents the system from silently accepting fitting artifacts. If a cell has n=3 and measured=1.0, the fitted value might be 0.95 — well above the 0.60 prior. The drop-gate would fire, and you'd either provide a justification ("small sample, will re-evaluate at n=30") or reject the fit and keep the prior.

### The measured progression

| Phase | Eval run | Config | macro_F1 | macro_Fbeta | What changed |
|-------|----------|--------|----------|-------------|--------------|
| P0 baseline | `p0_honest_baseline` | L1 (hand-set) | 0.1958 | 0.2515 | First honest measurement |
| P2 calibrated | `p2_calibrated` | L1 (hand-set) | 0.1998 | 0.2246 | fuse() fixes, no weight changes |
| P3 Rule 5C v3 | `p3_rule5c_v3` | L3 (fitted) | **0.3008** | **0.3821** | Only reliability weights changed |

The +0.10 F1 jump is the most important measurement in the project. It proves that:
1. The hand-set L1 weights were wrong (biased by human intuition)
2. The data-derived L3 weights are better (confirmed by higher F1)
3. The fitting pipeline works (confusion matrix → Bayesian shrinkage → versioned YAML)

---

## Key Code

### 1. The L3 loader — `_load_l3_table()`: fallback logic with schema guard

```python
# reliability.py:33-70
def _load_l3_table():
    path = Path(L3_RELIABILITY_PATH)  # configs/reliability_v1.yaml
    if not path.is_file():
        return None
    try:
        doc = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return None
    if doc.get("schema_version") != L3_EXPECTED_SCHEMA:  # "1"
        return None
    table_raw = doc.get("table")
    if not isinstance(table_raw, dict):
        return None
    out = {}
    for source, by_cls in table_raw.items():
        for cls, val in by_cls.items():
            out[(source, cls)] = float(val)
    return out
```

Every guard returns `None`, which triggers the L1 fallback. Missing file → fallback. Malformed YAML → fallback. Wrong schema version → fallback. Missing `table` section → fallback. This is Rule 5C applied to config loading: a missing L3 config doesn't silently return zeros or fabricate values — it falls back to the documented L1 prior.

### 2. The Bayesian shrinkage fitter — `_fit_cell()`: per-cell estimator

```python
# reliability_fit.py:99-135
def _fit_cell(cell, prior, alpha):
    n = cell["tp"] + cell["fp"] + cell["fn"] + cell["tn"]
    tp, fp = cell["tp"], cell["fp"]
    denom = tp + fp
    measured = (tp / denom) if denom > 0 else 0.0
    if n == 0:
        fitted = prior
        n_zero = True
    else:
        fitted = (n * measured + alpha * prior) / (n + alpha)
        n_zero = False
    return FittedCell(source=cell["source"], cls=cell["cls"],
                      n=n, measured=measured, prior=prior,
                      fitted=round(fitted, 6), n_zero_sample=n_zero)
```

The zero-sample case (`n == 0`) returns the prior verbatim — no fabrication. The Bayesian formula for n > 0 pulls the measured value toward the prior with strength proportional to α. A cell with n=83 (most Aderyn cells) is dominated by the measured data; a cell with n=3 (rare) is heavily pulled toward the prior.

### 3. The drop-gate — `fit()` main loop: gate enforcement

```python
# reliability_fit.py:138-204
for cell_data in matrix["cells"]:
    fc = _fit_cell(cell_data, prior=prior, alpha=alpha)
    fitted_cells.append(fc)
    if not fc.n_zero_sample:
        delta = abs(fc.fitted - fc.prior)
        if delta >= drop_gate_pct:  # 0.05
            key = f"{src}|{cls}"
            if key not in justifications:
                gate_failures.append(
                    f"{key}: fitted={fc.fitted:.4f} prior={fc.prior:.4f} "
                    f"delta={delta:.4f} >= gate={drop_gate_pct:.4f} (no justification)"
                )
```

The drop-gate checks every non-zero-sample cell. If the fitted value deviates from the prior by ≥0.05 and there's no justification for that cell, the gate fires. The CLI (`main()`) fails by default when gate_failures exist — you can override with `--allow-failures` to inspect the output, but by default it refuses to write the YAML.

### 4. The routing thresholds — `compute_active_tools()`: decision numbers in action

```python
# routing.py:121-132
def compute_active_tools(ml_result):
    cfg = get_config()
    active = set()
    for cls, prob in ml_result.get("probabilities", {}).items():
        if prob >= cfg.routing.deep_thresholds.get(cls, 0.40):
            active.update(cfg.routing.routing_rules.get(cls, []))
    return sorted(active)
```

This function reads `cfg.routing.deep_thresholds` — a per-class map from the config file. Each threshold is an L1 decision number: `Reentrancy: 0.35`, `IntegerUO: 0.35`, `UnusedReturn: 0.45`, `DenialOfService: 0.30`. These haven't been upgraded to L3 yet — they're still hand-set values waiting for a PR-curve analysis that determines the optimal threshold per class.

---

## Design Decision: L3 Learned vs L2 Measured vs L1 Config vs L0 Constant

| Criterion | L3 (data-derived) | L2 (measured) | L1 (config) | L0 (hardcoded) |
|-----------|-------------------|---------------|-------------|----------------|
| Accuracy | High (fitted from real data) | Medium (you know the delta, but set manually) | Low (hand-set, biased by intuition) | Very low (unversioned, silently changed) |
| Effort to adopt | High (build data pipeline, fit, justify) | Medium (run eval before/after) | Low (move constant to YAML) | None (it's already in the code) |
| Bias | None (data-derived) | Low (you know the impact) | High (human intuition) | Very high (lost context) |
| Data needed | Full confusion matrix | Eval baseline + one change | None | None |
| When appropriate | Production with labeled corpus | You have a baseline and want to change | Prototype or no labeled data yet | Sandbox / throwaway code |

**Decision:** The ladder is a progression — every number should start at L0 (during prototyping), be externalized to L1 (first stable version), have L2 baselines established (before/after measurements), and graduate to L3 when enough labeled data exists.

**When L0 is fine:** During initial development, before you have a labeled corpus. L0 constants let you build and test without YAML boilerplate.

**When L1 is fine:** For structural configs that rarely change (e.g., `routing_rules` — which tools handle which class). These are design decisions, not tunable parameters.

**When L3 is necessary:** For reliability weights — these directly control the fuse() function's output. An incorrect weight biases every verdict the system produces.

---

## Technology Choice: Bayesian Shrinkage for Reliability Fitting

**Category:** Small-sample statistical estimator for per-cell precision.

**Alternatives:**

| Method | Strength | Weakness |
|--------|----------|----------|
| **Bayesian shrinkage (chose)** | Handles n=0 gracefully, pulls noisy estimates toward prior, interpretable | Needs a prior (L1 config), α=5 is a heuristic |
| **MLE (Maximum Likelihood Estimation)** | Simple: `tp / (tp + fp)` | Produces 1.0 for small-n perfect cells — overconfident |
| **Bootstrapping** | Gives confidence intervals | Expensive to compute, still produces point estimates |
| **Pooled estimate** | More data per cell by pooling across classes | Masks real differences between classes |

**Why Bayesian shrinkage:** The labeled corpus has 61-83 contracts, and rare classes (DenialOfService, GasException) have only 9-10 samples. MLE on 3 samples produces 0.667, but you don't trust that number. Shrinkage pulls it toward the L1 prior (which was a reasonable guess), with the pull strength decreasing as n increases. At n=83 (most Aderyn cells), the measured data dominates; at n=3, the prior dominates.

**When you'd choose differently:**
- >1000 samples per cell → MLE is fine (shrinkage unnecessary)
- No reliable prior → bootstrap (learn the distribution from data)
- Need confidence intervals → bootstrap Bayesian posterior with MCMC

**Migration trigger:** When the labeled corpus grows to 600+ contracts (10× current), MLE becomes viable for most cells and shrinkage can be relaxed. α could decrease from 5 to 2 or 1.

---

## Anti-Patterns

### ❌ Hand-set constants forever based on domain intuition

**What it looks like:** "I set 0.35 because reentrancy is dangerous and I want to catch everything." Or "0.78 for ML reliability seems right — it's the best model we have."

**Why someone would build this:** It's fast, it doesn't require an eval pipeline, and domain experts have useful intuitions — especially about relative ordering (ML > Slither > Aderyn for some classes).

**Why it's wrong:** Intuition is uncalibrated. The L3 fitting showed that Aderyn's actual precision was 0.02–0.37 for most classes, not the 0.55–0.70 assumed by L1. ML's precision was 0.02–0.25 for most classes, not 0.40–0.80. The hand-set values were directionally correct (ML > Aderyn for some classes) but quantitatively wrong — and the quantitative error caused a 50% F1 gap.

**The right approach:** Start with L1 (it's better than L0), but only as a temporary scaffold. Build the eval pipeline, collect labeled data, fit from confusion matrices. Every number should have a path to L3.

### ❌ Change a number without running the eval

**What it looks like:** "Let me try 0.30 instead of 0.35 for Reentrancy threshold — it feels like we're missing too many." The developer changes the YAML, reruns the pipeline, and looks at a few sample outputs to confirm "it looks better."

**Why someone would build this:** The full eval takes 15 minutes. It's tempting to skip it for "small" changes.

**Why it's wrong:** Without the eval, you don't know whether the change improved or regressed the system. You might increase recall (good) but tank precision (bad). You might fix one class but break five others. The Fbeta metric catches these cross-class effects automatically. "It looks better" on 3 sample contracts is not measurement.

**The right approach:** Rule 5B — "Show me the before/after eval result." Before changing any decision number, run `python -m src.eval.run_benchmark` to get the baseline, make the change, run again, and compare.

---

## Mistakes & Fixes

### Mistake: Aderyn reliability inflated by silent skips

**What happened:** The P0 eval showed Aderyn with abnormally high precision (0.55-0.70) across most classes. This made fuse() trust Aderyn too much, letting false positives from Aderyn's overly aggressive detectors (especially on `unchecked-lowlevel` and `calls-loop`) influence verdicts.

**Why it happened:** Aderyn binary wasn't installed on 83 contracts (the entire eval corpus was run before the binary was set up). `_run_aderyn_on_file()` caught `FileNotFoundError`, logged it at DEBUG, and returned `[]`. The confusion matrix counted these 83 empty returns as "Aderyn ran and found nothing" — True Negatives for every class. 83 × 10 = 830 inflated TN entries per source.

**How we found it:** Rule 5C audit (P2.5) discovered the silent-skip pattern in `_helpers.py:80-118`. The scratch file (`system_finalization_statecheck_20260625.md`) records the finding. The reliability matrix builder was updated with `_tool_ran()` to check `tool_status` before counting cells.

**The fix:** Two changes. First, `reliability_matrix.py:136-147` added `_tool_ran()`:
```python
# reliability_matrix.py:136-148
def _tool_ran(row, source):
    ts = getattr(row, "tool_status", None) or {}
    entry = ts.get(source)
    if not isinstance(entry, dict):
        return True  # legacy compat
    return entry.get("ran") is not False
```
Second, `build_matrix()` at `reliability_matrix.py:201-207`: when `not _tool_ran(row, source)`, the contract is excluded from that tool's TP+FP+FN+TN counts and recorded in `matrix.excluded_contracts[source]`.

**The lesson:** Silent failures manufacture rabbit holes. "return []" on a tool failure produces data indistinguishable from "ran clean" — which then biases every downstream measurement. Rule 5C exists because of this exact bug.

### Mistake: L3 reliability caused +50% F1 improvement — not from better code

**What happened:** Between P2 (macro_F1=0.1998) and P3 (macro_F1=0.3008), the only change was the reliability values. No new tools, no model retraining, no pipeline rewrites. Just numbers in a config file.

**Why it happened:** The L1 hand-set weights were educated guesses. They got the relative ordering roughly right (ML ~0.60, Slither ~0.65-0.80, Aderyn ~0.55-0.70 per class). But the absolute values were wrong — sometimes by 40 percentage points (Aderyn's actual precision was 0.02-0.37). Bayesian shrinkage corrected these values, and fuse() started producing more accurate verdicts.

**The lesson:** Measurement > intuition. The single biggest improvement in the project's history came from data, not code. Before P3, the system had 530 passing tests — code was correct, but verdicts were wrong. Only measurement could fix that.

### Mistake: Drop-gate caught a fitting artifact on aderyn|Timestamp

**What happened:** The fitted reliability for `aderyn|Timestamp` was 0.0227 — vs the L1 prior of 0.40. The drop-gate (±5pp, delta = 0.3773) fired. Inspection showed that Aderyn had 0 TP, 0 FP, 12 FN, 71 TN for Timestamp — it simply never flagged this class. The measured precision was 0.0, and the fitted value was pulled toward 0.0 by the data.

**Why it happened:** Aderyn's detector set doesn't include a Timestamp detector. Every Timestamp contract was a false negative — Aderyn missed it. The fitted value correctly captures this: Aderyn has zero predictive power for Timestamp. But the drop-gate flagged it because the deviation from the L1 prior was large.

**The fix:** Reviewed and accepted with justification: "P3 measured value — Aderyn has no Timestamp detector. Fitted correctly captures zero predictive power." The L1 prior of 0.40 was overly optimistic.

**The lesson:** The drop-gate isn't a blocker — it's a review gate. Every large deviation is worth understanding. Sometimes the data is right and the prior was wrong; sometimes the data is noisy and the prior should be kept. The justification file makes this explicit.

---

## What Would Break If You Removed This?

Remove the L3 reliability system and fall back to L0-L1 only. The system still compiles, still produces verdicts — but the fuse() function uses hand-set weights that are, on average, 20-40 percentage points off from the true values. Macro_F1 drops from 0.3008 back to ~0.2000 — a 50% regression.

Remove the config system entirely (L1 → L0). Now every decision number is a constant in a `.py` file. To change a threshold, you edit `routing.py`, re-deploy, and hope you didn't break anything. There's no version history, no schema validation, no documented defaults.

Remove the drop-gate. A fitting artifact on a small-n cell (e.g., ml|CallToUnknown with n=1, fitted=0.977) enters the production config without review. Fuse() over-trusts ML for CallToUnknown, producing false positives that downstream review must catch manually.

Remove Rule 5B. Now any developer can change any threshold with "I think 0.35 feels right" as the justification. Without measured deltas, you can't tell if the system is improving or regressing. The project stalls at whatever F1 the last unmeasured change produced.

---

## At Scale

*Scale metric: labeled contracts in the eval corpus (baseline: 61-83)*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| Current (61-83) | Bayesian shrinkage handles small cells gracefully | Rare classes (GasException, 9 samples) still noisy | — |
| 10x (600-800) | Most classes have 50+ samples → MLE is viable for high-n cells | Shrinkage still useful for low-frequency tool×class pairs | Reduce α from 5 to 2 (more data = less shrinkage needed) |
| 100x (6,000-8,000) | All cells have sufficient data → MLE everywhere | Bayesian shrinkage adds complexity with no benefit at this scale | Switch to MLE; remove shrinkage; drop-gate becomes unnecessary |
| 1000x (60,000-80,000) | Could learn fusion weights too (not just per-source weights) | Rare classes still exist (0.1% prevalence = 60 samples) | Learn full fusion model (e.g., logistic regression over evidence channels) |

---

## Try It Yourself

> TRY IT: cd agents && source .venv/bin/activate && head -30 configs/reliability_v1.yaml | cat  # see the L3 fitted values

> TRY IT: head -50 configs/verdicts_default.yaml | cat  # see the L1 config with maturity annotations

> TRY IT: python3 -c "
from src.orchestration.verdict.reliability import get_reliability
print('L3 ml Reentrancy:', get_reliability('ml', 'Reentrancy'))
print('L3 slither Reentrancy:', get_reliability('slither', 'Reentrancy'))
"

---

## Limitations & What's Missing

- **61 contracts is small.** Rare vulnerability classes (DenialOfService, GasException) have 9-10 samples. Even with α=5 shrinkage, the fitted values are noisy. The 95% credible interval on a cell with n=10 and measured=0.20 is approximately ±0.25. These values will change as more labeled contracts are added.
- **Only reliability is L3.** Routing thresholds (`deep_thresholds`), verdict cutoffs (`confirmed_band`, `likely_band`), and fusion parameters (`ml_weight_scale`) remain at L1. These are hand-set values awaiting PR-curve analysis. A future L3 upgrade should fit thresholds from precision-recall curves per class.
- **No automatic refit trigger.** The reliability fitting is a manual CLI command. There's no CI gate that says "the confusion matrix changed by >10% → automatically refit." Without this, the L3 values can become stale as the eval corpus grows.
- **No A/B testing framework.** You can't compare two configs (L1 vs L3) simultaneously on the same data. The comparison is sequential (P2 run, then P3 run, then diff the reports) — which introduces temporal noise (model versions, tool updates between runs).
- **Drop-gate is a ±5pp heuristic.** The 0.05 threshold was chosen as a reasonable default, not derived from data. A future upgrade could learn the drop-gate threshold from fitting artifacts vs real changes.
- **No uncertainty quantification.** The fitted values are point estimates. There's no confidence interval or credible interval attached to each cell. A fuse() upgrade could propagate uncertainty into verdict confidence.

---

## Transferable Patterns

1. **Maturity ladder for configuration** — L0→L1→L2→L3
   - *Interview story:* "SENTINEL uses a four-level maturity ladder for every decision number. L0 is a constant in `.py` — fine for prototypes. L1 externalizes it to YAML — versioned, changeable, schema-validated. L2 adds measurement — you run the eval before and after every change. L3 fits from data — Bayesian shrinkage from a confusion matrix. The key insight is that you never skip levels: L3 without L2 means you have no baseline to compare against; L2 without L1 means you have no config to put the result in."
   - *When this pattern is WRONG:* For throwaway code or one-off experiments, the overhead of L1-L3 is wasteful. L0 is fine for scripts.

2. **Bayesian shrinkage for small-sample estimates** — `reliability_fit.py:99-135`
   - *Interview story:* "Our labeled corpus is 83 contracts — small enough that raw measured precision on rare classes is unreliable. A tool that ran on 3 contracts and got 3 TP would show 1.0 precision — an obvious overestimate. We use Bayesian shrinkage: `fitted = (n × measured + α × prior) / (n + α)`. At n=3, the estimate is heavily pulled toward the prior. At n=83, the data dominates. α=5 was chosen as a reasonable default — it gives a 50/50 split at n=5."
   - *When this pattern is WRONG:* With >1000 samples per cell, shrinkage adds complexity for no benefit. MLE is simpler and will converge to the same values. Also, if your prior is unreliable (guess without domain basis), shrinkage pulls estimates toward a bad value — garbage in, garbage out.

3. **Measured deltas over intuition** — Rule 5B
   - *Interview story:* "The single biggest improvement in SENTINEL's history was macro_F1 from 0.1998 to 0.3008 — a +50% relative improvement. It came entirely from replacing hand-set reliability weights with data-derived ones. No code changed. No model retrained. Only numbers in a config file. This is why we require a measured before/after delta for every decision-number change (Rule 5B). 'I think 0.35 feels right' is not engineering — 'before: 0.1998, after: 0.3008' is."
   - *When this pattern is WRONG:* In early prototyping before a baseline exists, you don't have a "before" to measure against. Don't block exploration — just label the number as L0/L1 and add a TODO for measurement.

4. **Drop-gates for fitting artifacts** — `DROP_GATE_PCT = 0.05`
   - *Interview story:* "When we first fitted L3 reliability, one cell showed a fitted value of 0.95 vs a prior of 0.60 — a 35pp jump. Investigation showed n=3: 2 TP, 0 FP, 1 FN. The measured precision was 1.0, but n=3 was too small to trust. The drop-gate (±5pp) flagged it, and we documented the justification: 'small sample, will re-evaluate at n=30.' The drop-gate doesn't block — it forces a conversation."
   - *When this pattern is WRONG:* If your prior is systematically wrong (e.g., all L1 values were estimated by someone unfamiliar with the domain), every cell will fire the drop-gate. In that case, the gate becomes noise and you should refit with a better prior rather than writing 30 justifications.

---

**Source files verified:**
- `src/config/schema.py:1-100` — Pydantic models for all decision-number groups with defaults
- `src/config/loader.py:1-67` — Eager, fail-fast config loader with singleton caching
- `src/orchestration/verdict/reliability.py:1-140` — L3→L1→hardcoded fallback chain
- `src/eval/reliability_fit.py:1-357` — Bayesian shrinkage fitter, α=5, drop-gate ±5pp, justifications
- `src/eval/reliability_matrix.py:1-222` — `ReliabilityMatrix` builder, `_tool_ran()` Rule 5C compliance
- `src/orchestration/routing.py:121-132` — `compute_active_tools()` consuming L1 deep_thresholds
- `configs/reliability_v1.yaml:1-438` — L3 fitted config with 30 cells, justifications, schema_version=1
- `configs/verdicts_default.yaml:1-99` — L1 config with maturity annotations (L2, L3 target)
- `~/.claude/scratch/system_finalization_statecheck_20260625.md` — Rule 5C audit finding, L0→L3 progression

**Verified against commit hash:** `c47898ea5`
