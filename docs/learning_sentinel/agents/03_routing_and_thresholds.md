# Step 3 — Routing & Thresholds

## The ML output isn't yes/no

For each of the 10 vulnerability classes, the model outputs a probability (0 to 1) —
a confidence score, not a verdict. Something downstream has to turn 10 numbers into
"which tools run, how deep does this go." That's `routing.py` + `quick_screen` +
`evidence_router`.

## Three lookup tables in `routing.py`

**`DEEP_THRESHOLDS`** — per-class probability cutoff (e.g. `Reentrancy: 0.35`).
Deliberately LOWER than the model's own "I'm confident" cutoff (0.50-0.55) — catches
borderline cases for a second opinion, not just already-confident ones.

**`ROUTING_RULES`** — per class, which tool(s) activate if its threshold is crossed.
`Reentrancy → [static_analysis, rag_research]`; `GasException → [static_analysis]`
only (no point searching exploit write-ups for a gas-cost issue).

**`CLASS_TO_DETECTORS`** — Slither has ~100 checks ("detectors"), each hunting one
specific bug pattern (`reentrancy-eth`, `unchecked-lowlevel`, ...). This table scopes
Slither down to only the detectors relevant to whatever classes were flagged —
faster, same relevant coverage. (This table is also what the registration bug from
the previous session made irrelevant — scoping a list of detectors that were never
actually registered did nothing.)

## `compute_active_tools()`

For each class: probability ≥ its DEEP_THRESHOLD? → union in its ROUTING_RULES
tools. Empty result = fast path. Non-empty = deep path, fanned out in parallel
("fan-out" = one decision triggering several next nodes simultaneously, the
parallel branch in Step 1's diagram, vs a normal one-after-another edge).

## `quick_screen` — catches what ML can't see

Runs Slither + Aderyn on EVERY contract (not gated by ML at all), against its OWN
separate hand-picked "escalate regardless" detector list (different from
`CLASS_TO_DETECTORS`). If it finds something High-impact while the ML-driven logic
says "fast path," `evidence_router` overrides and forces `static_analysis` to run
anyway. This is the two-signal gate: BOTH ML and quick_screen must agree it's safe.

## `compute_verdict()` — oldest fallback in the chain

Rule-based, predates `consensus_engine`. Roughly: prob ≥ 0.50 + Slither/strong-RAG
match → CONFIRMED; prob ≥ 0.50 + weaker match → LIKELY; prob ≥ 0.50 + nothing →
DISPUTED. This is verdict source #3 in the fallback chain (after debate, after
consensus_engine) — the one that disagreed with consensus_engine in the bug fixed
this week.

→ You now know: routing is three lookup tables + a two-signal safety net, and the
verdict that eventually "wins" for a class can come from up to three different,
independently-evolved rule systems.
