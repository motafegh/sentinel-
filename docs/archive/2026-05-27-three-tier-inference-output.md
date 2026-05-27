# Three-Tier Suspicion Output — Inference Design Proposal

**Date:** 2026-05-27
**Status:** PROPOSED — pending implementation decision
**Motivation:** Analysis of Run 4 test contract evaluation (20 contracts, see `docs/changes/2026-05-26-pipeline-alignment-and-inference-evaluation.md`)

---

## The Problem With Binary Thresholds

The current inference output applies a per-class threshold and reports only classes above it:

```python
vulnerabilities = [
    {"vulnerability_class": cls_name, "probability": round(prob, 4)}
    for cls_name, prob, thresh in zip(self._class_names, probs_list, self.thresholds.cpu())
    if prob >= thresh.item()   # ← hard binary cut
]
```

This creates two problems:

**1. Silent signal suppression.** Classes below threshold vanish from output entirely. A contract
scoring DoS=0.36 (threshold=0.45) is reported as SAFE for DoS. But 0.36 is not "no signal" — it
means "the model found something DoS-like but isn't confident enough to commit." For a security
oracle feeding an agent pipeline, that signal is valuable.

**2. Threshold optimized for the wrong objective.** Per-class thresholds are tuned to maximize
F1-macro on the validation set — a symmetric metric that weights precision and recall equally.
But in security, asymmetric costs apply: missing a real vulnerability (FN) costs money and
security; a false alarm (FP) costs analyst time. F1-optimal thresholds are not risk-optimal.

**Evidence from 20-contract evaluation:**

Of 12 missed classes (FN), most had raw probability just below threshold:

| Contract | Expected | Raw prob | Threshold | Gap |
|----------|----------|----------|-----------|-----|
| 05 | DenialOfService | 0.36 | 0.45 | −0.09 |
| 10 | GasException | 0.36 | 0.40 | −0.04 |
| 07 | TransactionOD | 0.35 | 0.35 | 0.00 (exactly at boundary) |
| 13 | UnusedReturn | 0.32 | 0.35 | −0.03 |
| 14 | Reentrancy | 0.34 | 0.40 | −0.06 |
| 17 | IntegerUO | 0.41 | 0.50 | −0.09 |

The model has the signal. The threshold suppresses it.

---

## The System's Actual Role

SENTINEL is not a verdict system. It is a **suspicion oracle** that feeds a multi-agent pipeline:

```
[ML Model] → probabilities → [Agent Module]
                                   │
                    ┌──────────────┼──────────────┐
                    ↓              ↓               ↓
             [RAG Research] [Static Analysis] [LLM Reasoning]
                    └──────────────┼──────────────┘
                                   ↓
                              [Synthesizer]
                                   ↓
                           [Audit Report + ZK Proof]
```

The model's job is to flag regions of concern. The agents do the verification. A model that says
"I'm 36% confident this has DoS characteristics" is giving the agent *useful information* — not
a verdict. Suppressing that to SAFE is information loss.

This also aligns with what the user observed: "even if we got one and even not even completely
one and we just get suspicions — we fully go for deeply analyze in any way we could and at least
we flag it as not sure or be careful or something like that for next module."

---

## Proposed Design: Three Tiers

### Tier Definitions

| Tier | Threshold | Meaning | Agent action |
|------|-----------|---------|--------------|
| `CONFIRMED` | prob ≥ 0.55 | Model is confident | Hard flag; prioritize for ZK proof |
| `SUSPICIOUS` | prob ≥ 0.25 | Non-trivial signal present | Send to RAG + static analysis deep dive |
| `NOTEWORTHY` | prob ≥ 0.10 | Weak signal, worth noting | Include in full probability vector; low-priority |

These thresholds are starting points — they should be tunable per deployment risk tolerance, not
hardcoded. A high-value DeFi protocol audit might use `SUSPICIOUS ≥ 0.20`; a quick triage scan
might use `SUSPICIOUS ≥ 0.40`.

### Proposed Output Format

```python
{
    "label": "suspicious",          # "safe" | "suspicious" | "confirmed_vulnerable"
    "verdict_summary": "2 CONFIRMED, 3 SUSPICIOUS",

    # Full 10-class probability vector — always present, never filtered
    "probabilities": {
        "CallToUnknown":            0.638,
        "DenialOfService":          0.312,
        "ExternalBug":              0.261,
        "GasException":             0.302,
        "IntegerUO":                0.314,
        "MishandledException":      0.292,
        "Reentrancy":               0.620,
        "Timestamp":                0.197,
        "TransactionOrderDependence": 0.281,
        "UnusedReturn":             0.249,
    },

    # Tiered findings — only classes above their tier threshold
    "confirmed": [
        {"vulnerability_class": "CallToUnknown", "probability": 0.638, "tier": "CONFIRMED"},
        {"vulnerability_class": "Reentrancy",    "probability": 0.620, "tier": "CONFIRMED"},
    ],
    "suspicious": [
        {"vulnerability_class": "DenialOfService", "probability": 0.312, "tier": "SUSPICIOUS"},
    ],

    # Legacy field — preserved for backwards compatibility
    # Contains only CONFIRMED classes at current tuned thresholds
    "vulnerabilities": [
        {"vulnerability_class": "CallToUnknown", "probability": 0.638},
        {"vulnerability_class": "Reentrancy",    "probability": 0.620},
    ],

    # Metadata
    "thresholds": [0.40, 0.45, 0.35, 0.40, 0.50, 0.40, 0.40, 0.30, 0.35, 0.35],
    "tier_thresholds": {
        "confirmed":   0.55,
        "suspicious":  0.25,
        "noteworthy":  0.10,
    },
    "truncated": false,
    "windows_used": 2,
    "num_nodes": 47,
    "num_edges": 89,
}
```

---

## Impact on 20 Test Contracts

Using `SUSPICIOUS ≥ 0.25` tier on the raw probability table:

| Contract | Expected | Currently | With SUSPICIOUS tier |
|----------|----------|-----------|----------------------|
| 05 DoS | DenialOfService | SAFE | SUSPICIOUS: DoS=0.36 ✓ |
| 07 TOD | TransactionOD | SAFE | SUSPICIOUS: TOD=0.35 ✓ |
| 10 Gas | GasException | SAFE | SUSPICIOUS: Gas=0.36, Re=0.37 ✓ |
| 13 UR | Re+TS+UR | Re+TS (miss UR) | UR=0.32 → SUSPICIOUS ✓ |
| 14 Re | Reentrancy | SAFE | SUSPICIOUS: Re=0.34 ✓ |
| 15 TOD | TransactionOD | SAFE | SUSPICIOUS: TOD=0.29 ✓ |
| 16 Gas | GasException | SAFE | SUSPICIOUS: Gas=0.38, Re=0.38 ✓ |
| 17 Int | IntegerUO | SAFE | SUSPICIOUS: Int=0.41 ✓ |

**Only hard misses remaining at `SUSPICIOUS ≥ 0.25`:**
- ExternalBug (prob=0.30, structurally invisible — requires inter-contract analysis)
- UnusedReturn on minimal contracts (prob=0.30–0.34, semantic gap)

This is a significant improvement for zero code changes to the model itself.

---

## Does F1 Matter in Production?

The answer is nuanced. F1-macro on a held-out test set tells you whether the model generalizes
to unseen contracts of similar distribution. It matters for comparing training runs and detecting
overfitting.

But F1 is NOT the right metric for production deployment decisions. What matters operationally:

1. **Recall on high-severity classes** — missing Reentrancy or IntegerUO costs real money.
   A model with Reentrancy recall=0.85 at 2× FP rate is worth more than F1=0.31 that misses 40%.

2. **FP rate that analysts can process** — if every contract fires 4 classes, analysts stop looking.
   The SUSPICIOUS tier helps here: agents auto-handle SUSPICIOUS cases, analysts only see CONFIRMED.

3. **Calibration** — does prob=0.6 actually mean 60% of such contracts are vulnerable?
   (Currently unknown — calibration curves not yet computed.)

4. **Pattern coverage** — vulnerability patterns evolve. New `unchecked{}` arithmetic patterns,
   new DeFi-specific vulnerabilities, new proxy patterns. The model will always lag novel patterns.
   RAG + static analysis agents provide coverage for what the model hasn't seen.

The right framing: **the model does triage, agents do verification, reports document findings.**
Triage quality is measured by recall (did we investigate the right contracts?), not exact-match.

---

## Implementation Plan

### Phase 1 — Output format change (no retraining, immediate)

**File:** `ml/src/inference/predictor.py`, `_format_result()`

Changes:
1. Always include full 10-class `probabilities` dict (no threshold filter)
2. Add `confirmed` and `suspicious` lists using configurable tier thresholds
3. Preserve `vulnerabilities` as legacy field (backwards compat) — CONFIRMED only
4. Add `tier_thresholds` to output so consumers know the boundaries used
5. Add `label` that reflects highest tier: `"safe"` / `"suspicious"` / `"confirmed_vulnerable"`

**File:** `ml/src/inference/preprocess.py`

No changes needed — preprocess returns graphs and tokens, not predictions.

**File:** `ml/api/` (when built)

API response schema updated to include new fields. Legacy consumers reading `vulnerabilities`
continue to work.

### Phase 2 — Agent module integration

**File:** `agents/nodes/ml_assessment.py`

Instead of: `if result["vulnerabilities"]: → route to full analysis`
Use: `route CONFIRMED to ZK proof pipeline; route SUSPICIOUS to RAG + static analysis;
      pass full probability vector to synthesizer for context`

**Evidence routing:** The existing `evidence_router` node can use the probability vector to
decide which tools to invoke, with per-class signal strength informing depth of investigation.

### Phase 3 — Configurable tier thresholds

Add per-deployment tier config (e.g., in predictor constructor or loaded from JSON companion
file alongside the per-class thresholds). This allows tuning for different deployment contexts
without code changes.

---

## Open Questions

1. **What are the right SUSPICIOUS/CONFIRMED thresholds?** 0.25/0.55 are educated guesses
   from the 20-contract evaluation. Proper calibration requires: computing calibration curves
   on the validation set, defining an acceptable FP rate for SUSPICIOUS, finding the threshold
   that achieves it while maximizing recall.

2. **Should NOTEWORTHY (≥ 0.10) be included in agent input?** It adds noise. Probably not in
   the agent prompt, but include in the raw `probabilities` dict for future analysis.

3. **Does the agent module need the full vector or just tiers?** The synthesizer LLM can reason
   over probabilities ("IntegerUO=0.49, almost CONFIRMED") better than discrete tiers. Pass both.

4. **How does this interact with the ZK proof pipeline?** The proxy MLP currently takes the
   full 10-class logit vector (not just above-threshold classes). The proof is over the full
   output. The CONFIRMED/SUSPICIOUS tiers are for human/agent consumption, not the proof circuit.
   No changes to the ZK pipeline needed.
