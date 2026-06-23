# Session 4 — Decision + Synthesis (walkthrough note)

**Date:** 2026-06-22
**Scope:** `consensus.py` (164), `confidence.py` (79), `nodes.py:1013-1456` (cross_validator), `nodes.py:1462-1568` (_reconcile_verdicts), `nodes.py:1571-1943` (synthesizer), `nodes.py:1987-2055` (consensus_engine).

---

## The verdict pipeline (3 stages, every stage fail-soft)

```
                 ┌────────────────────────────────────────┐
                 │  consensus_engine  (A.6/A.7)            │
                 │  • Per-class weighted ML/Slither/Aderyn │
                 │  • track_confidence (multiplicative)    │
                 │  • SAFE → DISPUTED override for flagged │
                 └────────────────┬───────────────────────┘
                                  │ consensus_verdict, confidence_by_class
                                  ▼
                 ┌────────────────────────────────────────┐
                 │  cross_validator  (A.4 debate)         │
                 │  • Selective gating: skip if consensus  │
                 │    already CONFIRMED + 2+ tool hits     │
                 │  • Otherwise: Prosecutor/Defender/Judge  │
                 │  • Falls back silently on LLM failure  │
                 └────────────────┬───────────────────────┘
                                  │ verdicts, confirmations, debate_transcript
                                  ▼
                 ┌────────────────────────────────────────┐
                 │  _reconcile_verdicts  (in synthesizer) │
                 │  • 8-case table, FN/FP asymmetry       │
                 │  • debate can upgrade, never to SAFE    │
                 │  • Default: more-severe wins            │
                 └────────────────┬───────────────────────┘
                                  │ final per-class verdict
                                  ▼
                 ┌────────────────────────────────────────┐
                 │  compute_verdict  (routing.py fallback)│
                 │  • Only reached if neither consensus   │
                 │    nor debate voted on a class         │
                 │  • Returns INCONCLUSIVE for flagged     │
                 └────────────────────────────────────────┘
```

Every node sets `error: str | None` on failure and returns a partial dict. The graph never raises; the synthesizer always produces a final report.

---

## The 8-case reconciliation table (`_reconcile_verdicts`)

This is the most important function in the system. It enforces the **FN/FP asymmetry principle**: a security tool's worst failure is missing a real vulnerability; false positives are cheap. So a flagged class must never be silently cleared.

| Case | consensus verdict | debate verdict | Final | Rule |
|------|-------------------|----------------|-------|------|
| 1a | CONFIRMED | SAFE / WATCH / INCONCLUSIVE | CONFIRMED | `consensus_confirmed_debate_cannot_clear` |
| 1b | CONFIRMED | DISPUTED | DISPUTED | `confirmed_surfaces_debate_uncertainty` |
| 2 | LIKELY | SAFE | DISPUTED | `disagreement_surfaces_as_disputed` |
| 3 | LIKELY | DISPUTED | DISPUTED | `likely_downgraded_to_disputed` |
| 4 | DISPUTED | SAFE | DISPUTED | `disputed_not_cleared_by_debate` |
| 5 | DISPUTED | CONFIRMED / LIKELY | debate verdict | `debate_upgrade` |
| 6 | (voted) | None (silent / timeout) | consensus verdict | consensus stands |
| 7 | None | (debate voted) | debate verdict | debate is only signal |
| 8 | None | None | `compute_verdict()` | last resort |
| default | any | any | more-severe wins | `more_severe_wins_*` |

**The key insight (Case 1b):** the debate can **downgrade** a unanimous tool-corroborated CONFIRMED to DISPUTED if the source-reading LLM saw something the syntactic tools missed. This was a real production fix. The live finding: 3 unanimous tool votes were CONFIRMED on a CEI pattern that wasn't actually a CEI violation (the "state change after external call" was a non-balance array index, not a balance). The debate correctly said DISPUTED; the old code would have kept CONFIRMED.

**The other key insight (Case 1a):** the debate cannot clear a class that consensus CONFIRMED with multi-tool agreement. The debate might say SAFE because the FAST model hallucinates, but the principle is that cheap signals saying "safe" never override expensive corroboration saying "confirmed."

---

## The ML weight discount — `ML_WEIGHT_SCALE=0.5` (Ali directive, 2026-06-21)

```python
# consensus.py:63-79
DEFAULT_ML_WEIGHT_SCALE: float = 0.5

def get_weights(class_name: str) -> dict[str, float]:
    base = ACCURACY_WEIGHTS.get(class_name, DEFAULT_WEIGHTS)
    return {
        "ml":      round(base["ml"] * _ml_scale(), 4),  # ML weight halved
        "slither": base["slither"],
        "aderyn":  base["aderyn"],
    }
```

**Why it exists:** Run 12's ML model is not yet reliable (known FP behavior on ExternalBug — the `s_Form001` contract scored p=0.96 on a 26-line KV store, a clear false positive per the DIVE crosswalk audit). The agent layer must do its OWN analysis and treat ML as a clue, not an authority.

**What it means in practice:** with `ML_WEIGHT_SCALE=0.5`, the maximum ML-only confidence is 0.5 × 0.78 (Reentrancy) = 0.39 — below the LIKELY band (0.50). ML can only contribute to a higher verdict by corroboration with at least one static tool.

**How to raise it:** the comment says "Raise this toward 1.0 once a better-calibrated model ships." Run 13 is DEFERRED per Ali; when it lands, this is the knob.

**This is the single most important design decision for the AI/ML engineer interview.** It demonstrates: calibrated skepticism toward your own model, principled weight assignment, fail-safe design.

---

## Per-class reliability weights (PRINCIPLED DEFAULTS, not fitted)

`consensus.py:46-57` — `ACCURACY_WEIGHTS` is a `{class: {ml, slither, aderyn}}` table. Per the docstring: "derived from SENTINEL Run 12 evaluation findings (47K SmartBugs-Wild eval + manual inspection + DIVE crosswalk audit), NOT a fitted confusion-matrix table."

| Class | ml | slither | aderyn | Why |
|-------|----|---------| ------|------|
| Reentrancy | 0.78 | 0.82 | 0.60 | ML true-positive confirmed; Slither reentrancy-eth strong; Aderyn corroborates but mostly Slither superset |
| IntegerUO | 0.62 | 0.80 | 0.70 | Slither/Aderyn syntactic detectors reliable; ML moderate |
| Timestamp | 0.80 | 0.45 | 0.40 | ML authoritative — static tools miss business-logic misuse of `block.timestamp` by design |
| ExternalBug | 0.45 | 0.50 | 0.45 | All flat — ML FP-prone (s_Form001), tools give no precision signal (3-way precision 3.0%) |
| GasException | 0.40 | 0.65 | 0.55 | ML weak (low training signal); tools better |
| CallToUnknown | 0.60 | 0.70 | 0.60 | Balanced |
| MishandledException | 0.55 | 0.72 | 0.62 | Slither strong on unchecked-send/transfer |
| UnusedReturn | 0.55 | 0.75 | 0.65 | Slither strong on unused-return |
| DenialOfService | 0.65 | 0.55 | 0.50 | ML moderate; tools weaker (DoS hard to statically detect) |
| TransactionOrderDependence | 0.70 | 0.60 | 0.45 | ML best for the ordering pattern |

The DEFAULT_WEIGHTS (used when a class is not in the table) is balanced, slightly tool-leaning: `{"ml": 0.60, "slither": 0.65, "aderyn": 0.55}`. Forward-compatible with new classes or a renamed taxonomy.

---

## Bayesian confidence (A.7) — `track_confidence`

```python
# confidence.py
SLITHER_AGREE = 1.10
SLITHER_DISAGREE = 0.90
ADERYN_AGREE = 1.05
ADERYN_DISAGREE = 0.97
RAG_AGREE = 1.05
RAG_RELEVANCE = 0.70
```

**Multiplicative nudges, not additive** — confidence cannot leave [0, 1] in spirit, and is clamped defensively anyway. Each available signal applies one nudge; absent signals (None) are skipped so a fast-path verdict with only ML evidence returns the clamped ML probability unchanged.

**Why the magnitudes are small (±5-10%):** "Evidence refines, it does not overrule the model." The ML probability is the base; static tools and RAG modulate it slightly. This is the right shape for a tool whose purpose is to surface, not to convict.

**Confidence bands vs verdict bands** (different labels, same boundaries):
| Confidence | Band | Verdict |
|------------|------|---------|
| ≥ 0.70 | high | CONFIRMED |
| ≥ 0.50 | medium | LIKELY |
| ≥ 0.30 | low | DISPUTED |
| < 0.30 | negligible | SAFE |

---

## Selective debate gating (WS4.2)

`nodes.py:1343-1379` — skip the LLM debate when every flagged class is `CONFIRMED` by consensus AND has 2+ tool signals:

```python
_skip_debate = False
if _debate_on and consensus_verdict_state:
    _all_confirmed = True
    for _vuln in all_flagged:
        _cv = consensus_verdict_state.get(_cls, {})
        if _cv.get("verdict") != "CONFIRMED":
            _all_confirmed = False; break
        _tool_count = (
            (_cv.get("ml_signal", 0) or 0)
            + (_cv.get("slither_match", 0) or 0)
            + (_cv.get("aderyn_match", 0) or 0)
        )
        if _tool_count < 2:
            _all_confirmed = False; break
    _skip_debate = _all_confirmed
```

**Why "2+ tool signals" and not "all 3":** in practice, when Slither and Aderyn agree, ML is almost always the same direction. The "2 of 3" check is enough to skip the debate without losing real disagreements.

**Why never skip on consensus SAFE:** the FN/FP asymmetry principle. The skip is gated on positive evidence only.

**Speed impact:** when all 3 tools agree CONFIRMED, the debate is redundant. Saves 20-30s of 3 sequential LLM calls on every "easy" finding. Cumulative across 305+ audits, this is significant.

---

## Hotspot-guided excerpts (WS3, 2026-06-22)

`nodes.py:1186-1260` — replaced the old `[:2000]` raw truncation with targeted excerpts of just the lines flagged by `graph_explain`'s hotspot analysis. The LLM now reads:

```
── Reentrancy ──
  withdraw (lines 42-58, score=0.87)
  Signals: external_call, state_write_after
  ```solidity
    42: function withdraw(uint amount) public {
    43:     require(balances[msg.sender] >= amount);
    44:     (bool success, ) = msg.sender.call{value: amount}("");
    45:     require(success);
    46:     balances[msg.sender] -= amount;
    47: }
  ```

Full contract source (for reference):
```solidity
... first 4000 chars ...
```

**The old prompt** sent the first 2000 chars of the contract regardless of which functions mattered. For a 500-line contract with the bug at line 487, the LLM saw lines 1-80 and made up its mind. The new prompt puts the relevant lines at the top of the LLM's attention budget.

**Fallback:** if no hotspot data (e.g. fast path), the prompt falls back to the raw truncation plus a note about ML windows used.

---

## The synthesizer's narrative prompt — the "hallucination incident" fix

`nodes.py:1748-1764` — the old prompt listed every ML-flagged class without a verdict:

```
- [CONFIRMED] Reentrancy: 87.2%
- [SUSPICIOUS] IntegerUO: 31.5%
- [SUSPICIOUS] GasException: 28.1%
```

The LLM had no way to know which classes had been cleared, and described a "Reentrancy risk" on a contract whose Reentrancy verdict was SAFE (and which contained zero external calls). Live incident — captured in `docs/changes/2026-06-21-agents-manual-verification-real-bugs-found.md`.

The fix attaches the verdict to every line:

```
- [CONFIRMED] Reentrancy: 87.2% → verdict: CONFIRMED
- [SUSPICIOUS] IntegerUO: 31.5% → verdict: DISPUTED
- [SUSPICIOUS] GasException: 28.1% → verdict: SAFE
```

And the system prompt now says: "if a class's verdict is SAFE or DISPUTED, do NOT describe it as a real risk even though it appears in the list."

**This is the kind of bug only live testing catches.** A unit test would have passed because the synthesizer returned a string. Only human reading caught the hallucination.

---

## The synthesizer's two layers

The synthesizer has two independent recommendation generators:

1. **Rule-based recommendation** (lines 1700-1736) — 3 tiers (HIGH/MODERATE/LOW RISK) with structured counts (RAG chunks, Slither High/Medium, prior on-chain audits). Always runs.

2. **LLM narrative** (lines 1738-1864) — STRONG model, 4-section Markdown (Severity / Vulnerability Summary / Exploit Pattern / Recommended Fix). Conditional on `_llm_enabled()`; falls back silently on any failure (LLM unavailable, timeout, malformed response).

Final: `final_recommendation = (narrative or recommendation) + truncated_note`.

**Why both:** the LLM is a senior-auditor simulator; it can write a coherent, contextual narrative. The rule-based path is the safety net — guaranteed to produce something useful even with no LLM. The OR pattern means we always get the LLM's narrative when it's available, and never get an empty report when it's not.

---

## What I did NOT read (and why)

- `reflection`, `explainer`, `visualizer` internals — covered in Session 5.
- The support files `attribution.py` and `visualizer.py` (separate from `nodes.py`) — covered in Session 5.
- `timeouts.py` — the centralized timeout config; the comments in `cross_validator` and `synthesizer` reference it but the file itself is small and can be skimmed when needed.
- Tests that exercise this code: `test_consensus_voting.py`, `test_confidence_tracking.py`, `test_metric_attribution.py`, `test_reflection.py`, `test_visualizer.py`, `test_verdict_reconciliation.py`. All on the Session 10 list.
