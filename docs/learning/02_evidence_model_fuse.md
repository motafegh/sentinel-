# 02. Evidence Model & Fuse(): How 6 Channels Become One Verdict

> **Prerequisites:** [01. The Audit Pipeline] — you need to know what `AuditState`, nodes, and the append-reducer on `evidence_list` are.
> **Next:** [03. Prompt Injection Defense] covers how untrusted contract source is sanitized before it reaches the LLM debate (which produces `Evidence` with `deterministic=False`).
> **Cross-ref:** [08. Evaluation Framework] covers the measured reliability values cited here. [09. Formal Verification] covers Halmos, which emits `Evidence` with `kind=FORMAL`.
> **Scope:** This doc covers the `Evidence` dataclass, the `fuse()` function, the per-source emit helpers, and the reliability lookup chain. It does NOT cover prompt injection (see Doc 03), the eval framework that produces the reliability matrix (see Doc 08), or the Halmos integration itself (see Doc 09).
> **TL;DR:** Every analysis channel — ML, Slither, Aderyn, RAG, LLM debate, Halmos — emits zero or more `Evidence` items into one append-only list. A single function, `fuse()`, consumes that list and produces two verdicts per class: `verdict_provable` (deterministic evidence only → ZK-anchored) and `verdict_full` (all evidence → human report). The old system had 8 hand-coded pairwise reconciliation cases for 3 sources; adding 6 more channels would have grown it to 28. The uniform Evidence model makes each new channel an append — Halmos was added in P8a with a 40-line emit function and zero changes to `fuse()`. Replacing hand-set reliability weights with data-derived ones (L3) drove macro_F1 from 0.1998 to 0.2765 (+38% relative) — the single largest improvement in the system's history.

---

## The Problem

The old `consensus_engine` + `_reconcile_verdicts` was correct but pairwise and hand-cased. For 3 sources (ML, Slither, Aderyn), it had 8 cases: "ML says X and Slither says Y and Aderyn says Z → verdict is V." This worked. But the architecture proposal (D-A) identified a scaling cliff: Phase B/D adds ~6 more channels (Halmos, Z3, Gigahorse, taint, access-control, economic). An 8-case pairwise reconciler for 3 sources becomes a 28-case combinatorial explosion for 8 sources (C(8,2) = 28). Each case is a hand-coded branch with its own thresholds, its own edge cases, and its own bugs.

The question: do you keep adding cases, or do you generalize *before* the channels multiply?

## How We Arrived at This Design

> **How to read this section:** Each step shows the question, *how to reason about it*, and the chain of logic that connects the answer to the next step. The method is reusable; the answers are specific to SENTINEL.

### Step 1 — Identify the invariant (the "must always be true" test)

**The question:** What must always be true about the verdict, even if every tool disagrees?

**How to identify it:** Apply the test — *"If this property is violated, does the system become useless or dangerous?"*

**Applying the test:**

The FN/FP asymmetry is the invariant: *a flagged class may never silently become SAFE*. A missed vulnerability (false negative) can cost millions — a DeFi reentrancy exploit drains the pool. A wasted review (false positive) costs minutes — a human looks at the contract, sees no bug, moves on. The asymmetry is ~1000:1 in cost. So the fusion function must never clear a strong vulnerability signal to SAFE, even if another source disagrees.

**The reasoning chain:** If we allow a REFUTES (e.g., the LLM debate says SAFE) to clear a SUPPORTS (e.g., Slither found reentrancy), then a single non-deterministic LLM call can override a deterministic static analysis finding. That means: (a) the ZK-provable tier (deterministic evidence only) would still flag the vulnerability, but (b) the human report (`verdict_full`) would say SAFE — and humans trust the report, not the ZK proof. The human reviewer sees SAFE and approves the contract. The exploit fires. This is the exact failure mode the system exists to prevent.

**What this forces:** The fusion function needs a "strong SUPPORTS" concept — evidence strong enough that no amount of REFUTES can clear it to SAFE. This is the asymmetry override in `fuse.py:136-137`.

### Step 2 — Identify the constraint (what forces a specific shape)

**The question:** What external requirements narrow the design space before we start choosing?

**Constraint A: ZK provability requires a deterministic boundary.**
- *Why:* ZKML can prove a fixed, deterministic function. It cannot prove an LLM debate (non-deterministic across runs). So the Evidence model must carry a `deterministic` flag, and `fuse()` must emit two tiers.
- *What this eliminates:* Any design where all evidence is treated equally — there's no way to separate what to prove from what to report.

**Constraint B: The fusion function must be extensible without rewriting.**
- *Why:* Phase B/D adds ~6 more channels (Halmos, Z3, Gigahorse, taint, access-control, economic). If the fusion function is pairwise (N² cases), adding channels means rewriting the function each time.
- *What this eliminates:* The existing 8-case `_reconcile_verdicts` — it works for 3 sources but can't scale to 8 without becoming a 28-case monster.

**Constraint C: No training data for fusion.**
- *Why:* We have 61-83 labeled contracts — enough to fit per-source reliability (a 1-D problem), but not enough to train a fusion classifier (an N-D problem that needs thousands of labeled fusion examples).
- *What this eliminates:* An ML classifier for fusion — there's no data to train it, and even if there were, it would be non-deterministic (breaking the ZK boundary) and uninterpretable (you can't explain why the classifier said CONFIRMED).

### Step 3 — Eliminate alternatives (find what breaks under *current* conditions)

**The three approaches for evidence fusion:**

| Approach | How it breaks | When it breaks | Eliminate? |
|---|---|---|---|
| **Pairwise rules** (8-case reconciler) | O(n²) growth: 3 sources → 8 cases, 8 sources → 28 cases. Each case needs its own thresholds. No de-correlation (can't tell Slither + Aderyn they're correlated). | When you add the 4th source. We have Halmos coming (P8a). | **Yes** — breaks now. |
| **ML classifier** for fusion | Needs labeled fusion training data (doesn't exist). Non-deterministic (breaks ZK). Black box (can't explain verdict). | When you try to ZK-prove the verdict or explain it to a reviewer. | **Yes** — breaks on ZK + interpretability. |
| **Weighted Bayesian** (signed sum with family discount) | Linear assumption (no synergy between sources). Family assignments are hardcoded. No uncertainty quantification. | When you have enough data to learn non-linear interactions (thousands of labeled examples). | **No** — breaks in the future, not today. |

**The reasoning:** Pairwise breaks *now* — we have 3 sources and Halmos is the 4th. The ML classifier breaks on the ZK constraint (non-deterministic) and the interpretability constraint (black box). The weighted Bayesian breaks only when we have enough data for non-linear interactions — that's a future problem (the `CROSS` deferred workstream: clean-data retrain). The weighted Bayesian is the *only* approach that doesn't break under *current* conditions.

**Steel-manning the rejected ML classifier:** "But an ML classifier could learn that Slither + Aderyn together are worth more than either alone — synergy!" True. But: (a) you need ~1000+ labeled fusion examples to train this without overfitting (you have 61), (b) the classifier's output is non-deterministic (model weights change across training runs → can't ZK-prove), (c) "why CONFIRMED?" → "the classifier said so" is not an answer a security auditor can use. The synergy benefit is real but the cost (losing ZK + interpretability + needing 1000x more data) is too high.

### Step 4 — Stress-test against future growth (the "add a channel" test)

**The test:** "What happens when we add Halmos in P8a?"

**Tracing through the design:**
1. `evidence.py`: `Evidence.formal()` already exists (evidence.py:159-184) — the constructor was designed for this.
2. `emit.py`: `emit_halmos_evidence()` (emit.py:205-243) — 1 new function, ~40 lines.
3. `fuse.py`: `FAMILIES["halmos"] = "FORMAL"` (fuse.py:31) — 1 line.
4. `fuse()` function itself: **0 changes** — it's the same group→de-correlate→aggregate→band pipeline.

**Total: 1 new function + 1 line. Zero changes to fusion logic.** The channel is an append, not a rewrite. The test passes.

**Counter-argument:** "But what if Halmos needs *different* fusion logic — e.g., a formal proof should override everything?" This is already handled: `_is_strong_supports()` (fuse.py:87-88) treats any `kind=FORMAL` evidence as unconditionally "strong." A formal proof of invariant violation (`SUPPORTS`, strength=1.0) can never be cleared to SAFE — no matter what the LLM debate says. The override is built into the existing design.

### Step 5 — Measure, don't guess (the baseline anchor)

**The measured progression — this is where the Evidence model's value becomes visible:**

| Run | macro_F1 | What changed | What this proves |
|-----|----------|-------------|-------------------|
| P0 baseline | 0.1958 | Legacy reconciler, Aderyn silent | The system's floor |
| P2 (fuse() active) | 0.1998 | Evidence model + fuse() replace legacy | fuse() doesn't break anything (+0.004) |
| P3 (L3 reliability) | 0.2765 | Data-derived weights replace hand-set | The Evidence model's payoff (+0.077) |
| P3 Rule 5C v3 | 0.3008 | Honest failure counting | Rule 5C's payoff (+0.024) |

**The reasoning:** The P2→P3 jump (+0.077 F1, +38% relative) is the *same `fuse()` function* — only the reliability weights changed (L1 hand-set → L3 data-derived). The Evidence model was the *prerequisite* for this: you can't fit per-(source, class) precision from a confusion matrix if every source speaks a different data shape. The generalization paid off not in P2 (when it was introduced, +0.004) but in P3 (when it enabled measured improvement, +0.077).

**The insight:** Generalizing the data model is an *investment* that pays off later. In P2, the Evidence model looked like a wash (+0.004 F1 — barely moved). In P3, it was the foundation that enabled the biggest single improvement in the system's history (+0.077 F1). If we had judged the Evidence model by its P2 delta alone, we would have abandoned it as "not worth the complexity." The measurement discipline says: judge a generalization by what it *enables*, not just by what it *immediately produces*.

> **The method, summarized:** (1) Find the invariant by asking "if violated, is the system dangerous?" (2) Find constraints from external requirements (ZK needs determinism, no training data for fusion). (3) Eliminate alternatives by finding *current* failure conditions — pairwise breaks on the 4th source, ML classifier breaks on ZK. (4) Stress-test by tracing the next addition — append is good, rewrite is bad. (5) Judge a generalization by what it *enables* in future phases, not just what it *produces* today.

---

## The Solution

### The Evidence dataclass

Every channel emits zero or more `Evidence` items — the same 8-field frozen record regardless of source:

```
┌───────────────────────────────────────────────────────────────────────┐
│  Evidence (frozen dataclass, evidence.py:29-38)                       │
├───────────────────┬───────────────────────────────────────────────────┤
│  source           │  "ml" | "slither" | "aderyn" | "rag" |            │
│                   │  "debate" | "quick_screen" | "halmos" | "z3" |     │
│                   │  "gigahorse" | "taint" | "consensus"              │
│  vuln_class       │  "Reentrancy" | "IntegerUO" | "Timestamp" | ...   │
│  polarity         │  SUPPORTS (vulnerable) | REFUTES (safe) |          │
│                   │  NEUTRAL (ran, nothing dispositive)               │
│  strength         │  [0, 1] — how strongly this observation points    │
│  reliability      │  [0, 1] — this source's per-class trustworthiness │
│                   │         (L3 data-derived, not hand-set)           │
│  kind             │  STATISTICAL | SYNTACTIC | SEMANTIC |             │
│                   │  FORMAL | ECONOMIC                                │
│  deterministic    │  True → ZK-provable | False → advisory only       │
│  detail           │  {detector, lines, counterexample, ...}          │
└───────────────────┴───────────────────────────────────────────────────┘
```

Seven helper constructors populate it, one per channel. Each maps the channel's native output format to the uniform shape:

| Constructor | Source | Polarity | Strength from | Deterministic |
|-------------|--------|----------|---------------|---------------|
| `Evidence.ml()` | ML model | SUPPORTS | class probability | True |
| `Evidence.slither()` | Slither | SUPPORTS | impact map (High=1.0, Med=0.6, Low=0.3) | True |
| `Evidence.aderyn()` | Aderyn | SUPPORTS (High/Med) or NEUTRAL (Low) | impact map | True |
| `Evidence.rag()` | RAG | SUPPORTS | similarity score | True |
| `Evidence.debate()` | LLM debate | SUPPORTS/REFUTES/NEUTRAL | judge confidence | **False** |
| `Evidence.quick_screen()` | Quick screen | SUPPORTS | impact map | True |
| `Evidence.formal()` | Halmos/Z3 | SUPPORTS (violation) or REFUTES (proven) | 1.0 / 0.9 | True |

### The fuse() function

`fuse()` is the sole verdict producer (since P2 T2.7 flip). It takes a flat list of Evidence, groups by class, and for each class runs the same 6-step pipeline:

```
  evidence_list (flat, all sources)
       │
       ▼
  Step 1: Group by vuln_class
       │
       ├── Reentrancy: [ml(0.87), slither(High), aderyn(High), debate(SAFE), halmos(violated)]
       ├── IntegerUO:  [ml(0.42), slither(none)]
       └── Timestamp:  [ml(0.05)]
       │
       ▼
  Step 2: De-correlate by witness family
       │
       │  ML family:     {ml}           → discount 1/1
       │  STATIC_SYNTAX: {slither, aderyn, quick_screen} → each 1/N
       │  RAG:           {rag}           → 1/1
       │  LLM_DEBATE:    {debate}        → 1/1
       │  FORMAL:        {halmos}        → 1/1
       │
       ▼
  Step 3: Aggregate signed (discounted_rel × strength)
       │
       │  positive_mass = Σ SUPPORTS(rel×str/N) - Σ REFUTES(rel×str/N)
       │  confidence = clamp(positive_mass, [0, 1])
       │
       ▼
  Step 4: FN/FP asymmetry override
       │
       │  if strong_SUPPORTS AND verdict == "SAFE":
       │      verdict = "DISPUTED"   ← a flagged class never silently clears
       │
       ▼
  Step 5: Map confidence to verdict band
       │
       │  confidence ≥ 0.70 → CONFIRMED
       │  confidence ≥ 0.50 → LIKELY
       │  confidence ≥ 0.30 → DISPUTED
       │  else              → SAFE
       │
       ▼
  Step 6: Dual emit — run Steps 2-5 TWICE per class
       │
       ├── verdict_provable: deterministic=True evidence only → ZK-anchored
       └── verdict_full:      all evidence → human report
```

The dual emit (Step 6) is the ZK boundary (D-B). `verdict_provable` is computed over `deterministic=True` evidence only — ML, Slither, Aderyn, Halmos. This is the tier ZKML proves. `verdict_full` includes the LLM debate — richer but non-reproducible. We anchor the former, report the latter.

### The reliability lookup chain

`reliability` is where the L3 magic lives. The lookup has a three-tier fallback:

1. **L3 (data-derived):** `configs/reliability_v3.yaml` — Bayesian-shrunk per-(source, class) precision from a confusion matrix on 61-83 labeled contracts. α=5 shrinkage toward the L1 prior.
2. **L1 (hand-set):** `verdicts_default.yaml::consensus.accuracy_weights` — the pre-P3 defaults, kept as the shrinkage prior.
3. **L0 (hardcoded):** `_source_defaults` in `reliability.py:139` — `{"rag": 0.50, "debate": 0.55, "quick_screen": 0.40}`.

This is the L0→L1→L2→L3 maturity ladder (Principle 5) in action. The system runs at L3 for the three measured sources (ml, slither, aderyn) and at L0 for sources without enough data (rag, debate, halmos).

## Key Code

The Evidence dataclass — 8 fields, frozen, validated:

```python
# evidence.py:29-44
@dataclass(frozen=True)
class Evidence:
    source: str
    vuln_class: str
    polarity: Polarity
    strength: float           # [0, 1]
    reliability: float       # [0, 1]
    kind: Kind
    deterministic: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength must be in [0,1], got {self.strength}")
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(f"reliability must be in [0,1], got {self.reliability}")
```

Why this matters: `frozen=True` makes Evidence immutable — once created, it can't be accidentally mutated by a downstream node. The `__post_init__` validation means a bug that produces `strength=1.5` fails immediately, not silently flows into a confidence calculation.

The `formal()` constructor — the P8a addition that required zero changes to `fuse()`:

```python
# evidence.py:159-184
@staticmethod
def formal(source: str, vuln_class: str, polarity: "Polarity",
           invariant: str, proven: bool, counterexample: str = "",
           reliability: float = 0.95) -> Evidence:
    return Evidence(
        source=source,
        vuln_class=vuln_class,
        polarity=polarity,
        strength=1.0 if polarity == Polarity.SUPPORTS else 0.9,
        reliability=round(float(reliability), 4),
        kind=Kind.FORMAL,
        deterministic=True,
        detail={"invariant": invariant, "proven": proven,
                "counterexample": counterexample[:200] if counterexample else ""},
    )
```

Why this matters: a formal proof of invariant violation (`SUPPORTS`, strength=1.0) is the strongest possible evidence — `_is_strong_supports()` treats any `kind=FORMAL` as "strong" unconditionally (fuse.py:87-88). A formal proof of safety (`REFUTES`, strength=0.9) is also strong — but capped below 1.0 because the tool's coverage is limited (Halmos might prove the wrong invariant).

The core fusion math — de-correlation, signed aggregation, asymmetry:

```python
# fuse.py:93-139
def _fuse_for_evidence(evidence: list[Evidence]) -> tuple[str, float, list[Evidence]]:
    if not evidence:
        return "SAFE", 0.0, []

    family_counts: dict[str, int] = {}
    for e in evidence:
        family = FAMILIES.get(e.source, e.source)
        family_counts[family] = family_counts.get(family, 0) + 1

    positive_mass = 0.0
    driving: list[Evidence] = []
    for e in evidence:
        family = FAMILIES.get(e.source, e.source)
        discount = 1.0 / max(family_counts.get(family, 1), 1)
        discounted_rel = e.reliability * discount

        if e.polarity == Polarity.SUPPORTS:
            positive_mass += discounted_rel * e.strength
        elif e.polarity == Polarity.REFUTES:
            positive_mass -= discounted_rel * e.strength

        driving.append(e)

    confidence = max(0.0, min(1.0, positive_mass))
    verdict = _band_label(confidence)

    if _is_strong_supports(evidence) and verdict == "SAFE":
        verdict = "DISPUTED"

    return verdict, confidence, driving
```

Why this matters: the family discount (line 115-116) is the de-correlation mechanism. Slither and Aderyn are in the same `STATIC_SYNTAX` family — if both fire on Reentrancy, each gets discounted by 1/2. This prevents correlated syntax tools from double-counting the same pattern. The asymmetry override (line 136-137) is the FN/FP rule: a strong SUPPORTS can never be cleared to SAFE by a REFUTES — it floors at DISPUTED.

The dual-verdict emit — the ZK boundary:

```python
# fuse.py:160-174
for cls, items in sorted(by_class.items()):
    det_items = [e for e in items if e.deterministic]
    verdict_provable, conf_provable, _ = _fuse_for_evidence(det_items)

    verdict_full, conf_full, driving = _fuse_for_evidence(items)

    results[cls] = ClassVerdict(
        cls=cls,
        verdict_provable=verdict_provable,
        verdict_full=verdict_full,
        confidence=round(conf_full, 4),
        driving_evidence=driving,
    )
```

Why this matters: `verdict_provable` runs the same fusion math twice — once on the deterministic subset, once on all evidence. This is the "no trade-off between smart and provable" design (D-B): you get both tiers from the same code path, same weights, same bands. The ZK proof anchors `verdict_provable`; the human report shows `verdict_full`.

The reliability fallback chain — L3 → L1 → L0:

```python
# reliability.py:98-117
l3 = _load_l3_table()
if l3 is not None:
    return {
        (src, cls): round(val * scale, 4) if src == "ml" else round(val, 4)
        for (src, cls), val in l3.items()
    }

table: dict[tuple[str, str], float] = {}
all_classes = set(acc.keys())
for cls in all_classes:
    w = acc.get(cls, defaults)
    table[("ml", cls)] = round(w["ml"] * scale, 4)
    table[("slither", cls)] = w["slither"]
    table[("aderyn", cls)] = w["aderyn"]
return table
```

Why this matters: if `reliability_v3.yaml` is missing or malformed, the system falls back to the L1 hand-set values — not to a crash, not to silent zero. The L1 values are the same numbers that were in effect before P3. A caller can detect the fallback by checking whether `_load_l3_table()` returns None. This is the L0→L1→L3 maturity ladder: the system runs at the highest available level, degrades gracefully.

## Design Decision: Uniform Evidence vs Pairwise Rules vs ML Classifier

> **How to read this section:** The table shows the options. The *elimination reasoning* below it shows how to think about the choice — steel-manning each rejected option before showing why it fails.

### The elimination process

**Step 1: What are the options?** Pairwise rules (the current 8-case reconciler), uniform Evidence + fuse(), or an ML classifier that learns the fusion.

**Step 2: What are the criteria — and why these criteria?**
- *Extensibility* — because 6 more channels are coming (Halmos, Z3, Gigahorse, taint, access-control, economic). The cost of adding a channel is the single most important metric.
- *Interpretability* — because a security auditor needs to answer "why CONFIRMED?" by pointing to the driving evidence. "The model said so" is not acceptable.
- *Training data* — because we have 61-83 labeled contracts, not 10,000. Any approach that needs rich labeled fusion data is eliminated.
- *Determinism* — because the ZK proof requires a deterministic fusion function. Non-deterministic fusion can't be anchored on-chain.

**Step 3: Eliminate by steel-manning, then finding the failure condition.**

**Pairwise rules — steel-man first:** "The 8-case reconciler is correct. It's been tested. It produces the right verdicts. Why fix what isn't broken?"

**Why it fails:** It's not broken *today* — it's broken *when you add the 4th source*. The 8 cases become 28 for 8 sources (C(8,2)=28 pairs, each needing a case). Each case has its own thresholds, its own edge cases, and its own bugs. Worse: there's no de-correlation mechanism — Slither and Aderyn are correlated (they share detector patterns), but pairwise rules treat them as independent. Each pair double-counts the same syntactic observation. The 8-case reconciler was the right design for 3 sources; it's the wrong design for 8.

**ML classifier — steel-man first:** "An ML classifier could learn non-linear interactions between sources — e.g., Slither + Aderyn together are worth more than either alone because their detector overlap confirms the pattern. A weighted sum can't capture this synergy."

**Why it fails on three counts simultaneously:**
1. *Training data:* You need labeled *fusion* examples (not just labeled contracts). 61 contracts × 10 classes × 3 sources = ~1830 cells, but most are zero-sample. You'd need ~10,000+ labeled fusion outcomes to train a classifier without overfitting. We have 61.
2. *Determinism:* The classifier's output depends on model weights, which change across training runs. You can't ZK-prove a verdict from a model you can't reproduce bit-for-bit.
3. *Interpretability:* "Why CONFIRMED?" → "The classifier's attention weights pointed to Slither's reentrancy-eth detector." This is technically an explanation, but not one a security auditor can use. The auditor needs to see: "Slither found reentrancy-eth at line 42, Aderyn found reentrancy-state-change at line 42, ML probability 0.87." That's the `driving_evidence` list — only the weighted sum produces it.

**Uniform Evidence + fuse() — why it survives:** It needs zero training data for fusion (training is only for per-source reliability, which is a simpler 1-D problem). It's deterministic (a weighted sum is reproducible). It's interpretable (each verdict traces to `driving_evidence`). Adding a channel is an append (one constructor, one emit function, one family entry — zero changes to `fuse()`). Its weakness (linear assumption, no synergy modeling) is a *future* problem that triggers only when we have enough data to justify a learned model.

**The reasoning principle:** "When choosing a fusion architecture, eliminate approaches that break on *current* constraints (ZK needs determinism, no training data, need interpretability) before considering approaches that break on *future* desires (synergy modeling, non-linear interactions). The weighted Bayesian is the only approach that doesn't break on any current constraint."

### When this decision would be wrong

**The reversal condition:** If the eval shows systematic fusion errors that a weighted sum can't capture — e.g., Slither + Aderyn synergy is real and measurable (the combined precision is >2× either alone), and we have 10,000+ labeled fusion examples — then extend the family discount to a learned interaction term. Don't jump to a full ML classifier — that's a sledgehammer for a nail. A learned interaction term is a scalpel: it keeps the weighted sum's interpretability and determinism while adding the synergy modeling. The trigger: when the eval shows a consistent pattern of "fuse() says SAFE but the class is actually vulnerable" that correlates with multi-source presence.

## Technology Choice: Weighted Bayesian vs Dempster-Shafer vs Learned Weights

**Category:** Evidence fusion under uncertainty.

**Alternatives:**
| Approach | Strength | Weakness |
|----------|----------|---------|
| Weighted Bayesian (chose) | Interpretable, no training, deterministic, handles correlation via family discount | Linear assumption (no synergy between sources) |
| Dempster-Shafer | Handles ignorance + conflict explicitly | Complex, counterintuitive results (Dempster's paradox), hard to tune |
| Learned weights (NN) | Captures non-linear interactions | Needs labeled fusion data, non-deterministic, black box |

**Why Weighted Bayesian:** SENTINEL has 61-83 labeled contracts — too few to train a fusion classifier without overfitting. The weighted sum is deterministic (ZK-provable), interpretable (each verdict traces to `driving_evidence`), and needs zero training data for the fusion itself (training is only for the per-source reliability, which is a simpler 1-D problem). The family discount handles the main correlation risk (Slither + Aderyn overlapping detectors).

**When you'd choose differently:**
- 10,000+ labeled fusion examples with known non-linear interactions → learned weights
- Need to explicitly model "I don't know" vs "evidence is conflicting" → Dempster-Shafer
- Multiple independent witness types with no overlap → plain weighted average (no family discount needed)

**Migration trigger:** if the eval shows systematic fusion errors that a weighted sum can't capture (e.g., Slither + Aderyn synergy is real and measurable), extend the family discount to a learned interaction term. Don't jump to a full ML classifier — that's a sledgehammer for a nail.

## Anti-Patterns

### ❌ Pairwise rules — "just 2 sources, keep it simple"
**What it looks like:** `if ml_flagged and slither_match: CONFIRMED; elif ml_flagged and not slither_match: DISPUTED; ...` — a branch for every combination.
**Why someone would build this:** It's the fastest way to a working prototype. With 3 sources, 8 cases is manageable. Each case is readable and testable.
**Why it's wrong:**
1. O(n²) growth — 3 sources = 8 cases, 8 sources = 28 cases, 15 sources = 105 cases. Each case is a potential bug.
2. No uniform data model — each source has its own output format, so each case parses a different shape.
3. Can't de-correlate — there's no notion of "Slither and Aderyn are correlated, don't double-count." Each pair is treated as independent.
**The right approach:** Uniform Evidence + single `fuse()`. Each channel emits the same 8-field shape. Fusion is one function, not N² cases. Adding a channel is an append, not a rewrite.

### ❌ ML classifier for fusion — "let ML learn the weights"
**What it looks like:** A neural network that takes all source outputs and produces a verdict. "The model will figure out the optimal combination."
**Why someone would build this:** It sounds smarter than a weighted sum. ML can capture non-linear interactions (synergy, antagonism between sources).
**Why it's wrong:**
1. No training data — you'd need labeled *fusion* examples (not just labeled contracts), and 61 contracts is far too few.
2. Non-deterministic — the classifier's output depends on model weights, which change across training runs. You can't ZK-prove a verdict from a model you can't reproduce.
3. Uninterpretable — "why CONFIRMED?" → "the classifier said so." You can't trace a verdict to its driving evidence.
**The right approach:** Weighted Bayesian with data-derived reliability. The weights are learned (L3), but the fusion function is a deterministic, interpretable formula. You get the benefits of learning (weights adapt to measured precision) without the costs (non-determinism, black box, training data hunger).

## Mistakes & Fixes

### Mistake: The 8-case `_reconcile_verdicts` couldn't scale
**What happened:** The original verdict reconciliation was an 8-case function: for each combination of (ML flagged, Slither match, Aderyn match), a hand-coded branch produced a verdict. It was correct for 3 sources. But the architecture proposal (D-A) identified that Phase B/D would add ~6 more channels — growing the case matrix to 28+ branches, each needing its own thresholds and tests.
**Why it happened:** The pairwise approach was the natural first design. When you have 3 sources, writing 8 cases is simpler than designing a uniform data model and a fusion function.
**How we found it:** The architecture finalization proposal (§5.2) flagged it as a scaling cliff: "the current `consensus_engine` + 8-case `_reconcile_verdicts` is correct but pairwise and hand-cased. Phase B/D add ~6 more channels; a pairwise scheme grows combinatorially and would be hardened by Phase B into an unmaintainable shape."
**The fix:** P2 — generalize to the uniform Evidence model *before* adding channels. Each channel emits Evidence; `fuse()` consumes it. Adding Halmos in P8a was a 40-line emit function and one `FAMILIES` entry. `fuse()` didn't change.
**The lesson:** Generalize the data model before scaling the producers. If your integration code grows O(n²) in the number of producers, you have a scaling cliff — fix it when n is small, not when n is already 8 and the rewrite is a 2-week project.

### Mistake: `fuse()` matched legacy verdicts only 22.9% — but legacy was wrong
**What happened:** During the P2 migration, golden tests compared `fuse()` output against the legacy `_reconcile_verdicts` on 83 contracts × ~7 classes each (524 total). `fuse()` matched on only 120/524 (22.9%). 75 classes where legacy flagged → `fuse()` said SAFE. At first glance, this looked like a `fuse()` bug.
**Why it happened:** The legacy reconciler was over-flagging. 42 of the 75 asymmetry violations were in `DenialOfService` — a class where neither Slither nor Aderyn has a detector. Legacy was saying "DISPUTED" (flagged) for classes with no corroborating evidence; `fuse()` was correctly saying "SAFE" (no positive mass). The legacy verdicts were *wrong*, not `fuse()`.
**How we found it:** Per-class breakdown of the 75 asymmetry violations showed the concentration in `DenialOfService` (42/75). Manual inspection confirmed: no Slither detector, no Aderyn detector, ML probability below threshold → legacy overrode SAFE→DISPUTED anyway. `fuse()` with the same inputs correctly produced SAFE (no evidence = no positive mass).
**The fix:** T2.7 flip — delete the legacy path entirely. `fuse()` is the sole verdict producer. The 75 "mismatches" were actually 75 bugs in the legacy system that `fuse()` fixed for free.
**The lesson:** When a new system disagrees with an old system, don't assume the new one is wrong. Measure *which* system is right on each disagreement. "It was always done this way" is not a measurement.

### Mistake: The `deterministic` flag was missing initially
**What happened:** The first Evidence prototype had 7 fields — no `deterministic` flag. All evidence was treated equally. When the ZK boundary question arose (D-B: "what does the oracle cryptographically attest?"), there was no way to separate reproducible evidence (ML, Slither, Halmos) from non-reproducible evidence (LLM debate) in the fusion function.
**Why it happened:** The ZK boundary wasn't part of the original Evidence design — it was a separate architectural concern (D-B) that was later recognized as requiring a per-evidence flag.
**How we found it:** The proposal (§5.1) added `deterministic` as the 8th field: "True if reproducible (ML, Slither); False if LLM (debate) — D-B." Without it, `verdict_provable` and `verdict_full` would be identical — the ZK tier would include non-reproducible LLM evidence, and the proof would be unrepeatable.
**The fix:** Add `deterministic: bool` to the Evidence dataclass (evidence.py:37). Each constructor sets it: `Evidence.ml()` → `True`, `Evidence.debate()` → `False`, `Evidence.formal()` → `True`. `fuse()` runs the fusion twice — once on `det_items` (provable), once on all items (full).
**The lesson:** When a downstream constraint (ZK provability) requires information that the upstream data model doesn't carry, add the field — don't try to infer it later. A boolean flag on the data record is the simplest way to separate tiers, and it makes the boundary explicit and auditable.

### Mistake: `emit.py` SAFE strength=0.15 caused a false-positive cascade
**What happened:** On 5 Safe reference contracts (ground truth: no vulnerabilities), the pipeline labeled 4-7 classes each as CONFIRMED. The debate judge correctly ruled SAFE for 7 of 8 classes — but the final verdicts showed CONFIRMED. The judge's SAFE ruling was being overridden by Aderyn false positives.
**Why it happened:** `emit_debate_evidence()` assigns a strength of 0.15 to a SAFE verdict (emit.py:166-168). When the judge rules SAFE, the debate evidence is emitted with `strength=0.15` (low). Aderyn's `eth-send-unchecked-address` false positive fires with `strength=1.0` (High impact) and `reliability≈0.24` (L3 fitted). Net: Aderyn's FP out-votes the judge's correct SAFE ruling because 0.15 is artificially tiny.
**How we found it:** The system finalization state-check (2026-06-25) dissected the Safe-contract FP cascade: the debate transcript showed the judge ruling SAFE, but `verdict_full` showed CONFIRMED. Traced to the strength table in `emit.py:166-168`.
**The fix:** This is a known finding, not yet fixed at the time of writing. The fix candidates: (A) bump SAFE strength to 0.50+ so the judge can override a single FP; (B) separate judge evidence from tool evidence with different reliability semantics; (C) lower Aderyn's reliability further. The measurement gate: whatever fix is chosen must show a favorable macro_F1 delta on the eval before shipping (Rule B).
**The lesson:** The strength values assigned by emit helpers are decision-numbers (Rule B). "SAFE = 0.15" is a hand-set L0 constant that should be externalized to config and measured against the eval. A strength value that throws away the information content of a correct LLM ruling is a bug — the judge saying SAFE after reading the code is HIGH-information, not LOW-information.

## What Would Break If You Removed This?

**Remove the `deterministic` flag:** `verdict_provable` and `verdict_full` become identical — the ZK tier includes non-reproducible LLM evidence. The ZK proof becomes unrepeatable (same contract + same model → different debate → different verdict). The entire ZKML anchoring story (D-B, P9) collapses — there's nothing deterministic to prove.

**Remove `fuse()` and go back to pairwise:** adding Halmos requires editing the 8-case reconciler into a 28-case monster. Each new channel multiplies the test surface. The de-correlation that `FAMILIES` provides (Slither + Aderyn in the same family, discounted by 1/N) disappears — correlated syntax tools double-count the same pattern.

**Remove the Evidence model (go back to per-channel verdict fields):** every channel writes its own verdict field (`consensus_verdict`, `debate_verdict`, `ml_verdict`). Downstream code must read all of them and reconcile — which is the pairwise problem all over again, just deferred to the consumer. Two sources of truth for a verdict is a recurring bug factory (the dual-wire shim lesson from Doc 01).

**Remove the family discount:** Slither and Aderyn — both `STATIC_SYNTAX` — fire on the same reentrancy pattern. Without the 1/N discount, each contributes its full `reliability × strength`, double-counting the same syntactic observation. Reentrancy confidence inflates; more contracts hit the CONFIRMED band. The false-positive rate on Safe contracts (already the system's weakest gate, WS2) gets worse.

## At Scale

*Scale metric: number of evidence channels (current: 6 active; P8b adds ~4 more).*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| 6 channels (current) | `fuse()` runs in <1ms | — | — |
| 15 channels (P8b + economic) | Still <1ms — O(n) in evidence count | Family assignments need review (are taint + access_control really the same family as Halmos?) | Add families as needed |
| 50 channels (hypothetical) | Fusion still fast | Evidence list grows large; `driving_evidence` per verdict becomes noisy | Top-K evidence filtering before fusion |
| 100 channels (hypothetical) | Fusion math still O(n) | Reliability matrix becomes sparse (most cells zero-sample) | Hierarchical reliability (per-family prior) |

The fusion function itself is not the scale wall — it's a weighted sum, O(n) in evidence count. The scale wall is the **reliability matrix**: each (source, class) cell needs enough labeled data to fit a meaningful precision. With 6 sources × 10 classes = 60 cells and 61-83 contracts, many cells are zero-sample or tiny-n. Adding channels without adding labeled data creates more zero-sample cells → more L1 fallback → the L3 benefit dilutes.

## Try It Yourself

> TRY IT: `cd agents && python -c "from src.orchestration.verdict.evidence import Evidence, Kind, Polarity; e = Evidence.ml('Reentrancy', 0.85, 0.90); print(f'{e.source} {e.vuln_class} {e.polarity} str={e.strength} rel={e.reliability} det={e.deterministic}')"`

> TRY IT: `cd agents && python -c "from src.orchestration.verdict.evidence import Evidence; e = Evidence.formal('halmos', 'Reentrancy', __import__('src.orchestration.verdict.evidence', fromlist=['Polarity']).Polarity.SUPPORTS, 'no-reentrancy', False, '0xdead'); print(f'{e.source} {e.kind} str={e.strength} det={e.deterministic}')"`

> TRY IT: `cd agents && python -c "from src.orchestration.verdict.reliability import load_reliability; t = load_reliability(); print(f'slither/MishandledException = {t[(\"slither\", \"MishandledException\")]}'); print(f'aderyn/Timestamp = {t[(\"aderyn\", \"Timestamp\")]}')"`

## Limitations & What's Missing

- **Fixed family assignments.** `FAMILIES` is a hardcoded dict (fuse.py:24-38). If a new source doesn't fit an existing family, it gets its own (no discount) — which may over-count if it's actually correlated with an existing family. There's no mechanism to learn family assignments from data.

- **No evidence conflict resolution beyond polarity.** If Slither says SUPPORTS and Aderyn says REFUTES (same family), the signed sum nets them out. But there's no explicit "conflict detected" signal — no `contradictions` field in `ClassVerdict`. The old `cross_validator` had one; `fuse()` doesn't.

- **No uncertainty quantification on fuse() output.** `confidence` is a point estimate, not a distribution. You can't say "85% confident, ±0.05." This matters for the on-chain oracle: a consumer might want to reject low-confidence verdicts, but the threshold is a single number with no variance.

- **Reliability is noisy with 61-83 contracts.** Many cells are tiny-n (e.g., `ml/CallToUnknown`: 1 TP, 0 FP → measured precision 1.0 → fitted 0.97, but n=1 is almost meaningless). The α=5 Bayesian shrinkage pulls toward the prior, but the fitted value is still dominated by the prior for small-n cells. More labeled data is the honest fix (the `CROSS` deferred workstream).

- **The `emit.py:166-168` SAFE strength=0.15 is an unfixed finding.** It causes the false-positive cascade on Safe contracts. The fix is measurement-gated (Rule B) — any change to the strength table must show a favorable macro_F1 delta before shipping.

## Transferable Patterns

1. **Generalize the data model before scaling the producers** — uniform Evidence before adding channels.
   - *Interview story:* "SENTINEL had 3 analyzers and an 8-case reconciliation function. We saw the cliff: 6 more channels were coming, and 8 cases would grow to 28. We generalized to a uniform Evidence dataclass *before* adding any new channel — every analyzer emits the same 8-field record, and one `fuse()` function consumes them. When Halmos landed in P8a, it was a 40-line emit function and zero changes to fusion. The generalization cost 2 weeks; the alternative was a 28-case monster that grows with every channel."
   - *When this pattern is WRONG:* when the generalization is speculative — you're building a uniform interface for producers that may never exist. If you have 3 sources and no concrete plan for more, the 8-case function is simpler and more readable. Generalize when you have a concrete growth plan (we had Halmos, Gigahorse, taint all designed), not when you're guessing.

2. **Dual-tier architecture — provable vs advisory** — `verdict_provable` (deterministic) + `verdict_full` (all evidence).
   - *Interview story:* "SENTINEL's ZK proof anchors only the deterministic evidence (ML + static analysis). The LLM debate is advisory — it enriches the human report but isn't part of the cryptographic claim. We get both tiers from the same `fuse()` function by running it twice: once on `deterministic=True` evidence, once on all evidence. No trade-off between 'smart' and 'provable' — we produce both, anchor one, report the other."
   - *When this pattern is WRONG:* when the deterministic tier is too weak to be useful (e.g., if ML is the only deterministic source and it's unreliable, `verdict_provable` will be mostly SAFE — the ZK proof attests to nothing). The pattern requires a strong deterministic core. If your deterministic evidence is sparse, either strengthen it (better static analysis) or accept that the provable tier is conservative.

3. **Dataclass as contract — frozen, validated, 8 fields** — `Evidence` is immutable and self-checking.
   - *Interview story:* "Every evidence item in SENTINEL is a frozen dataclass with 8 fields and `__post_init__` validation. You can't accidentally set `strength=1.5` — it raises immediately. You can't mutate evidence after creation — it flows through the pipeline immutably. This caught multiple bugs during development where a node tried to amend a strength value post-hoc."
   - *When this pattern is WRONG:* when you need to update evidence in place (e.g., a multi-pass analyzer that refines its own confidence). Frozen dataclasses force you to create a new instance for each update, which is correct but can be verbose. For a single-pass pipeline like SENTINEL's, frozen is right; for an iterative refinement loop, a mutable dataclass with the same validation hooks would be more ergonomic.

---

**Source files verified:**
- `agents/src/orchestration/verdict/evidence.py:15-26, 29-44, 48-60, 120-141, 159-184` — Polarity/Kind enums, Evidence dataclass, ml() + debate() + formal() constructors
- `agents/src/orchestration/verdict/fuse.py:24-38, 70-90, 93-139, 142-177` — FAMILIES, strong-SUPPORTS, core fusion, dual emit
- `agents/src/orchestration/verdict/emit.py:17-64, 153-180, 205-243, 246-298` — emit_ml, emit_debate (SAFE=0.15), emit_halmos, emit_consensus
- `agents/src/orchestration/verdict/reliability.py:33-70, 73-140` — L3 loader, L3→L1→L0 fallback chain
- `agents/src/orchestration/verdict/verdict.py:14-20` — ClassVerdict dataclass
- `agents/configs/reliability_v3.yaml:1-8, 74-107, 109-180` — schema, table, cells_detail

**Verified against commit hash:** `c47898ea5`
