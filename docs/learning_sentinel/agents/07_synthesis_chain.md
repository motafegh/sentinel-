# Step 7 — The Synthesis Chain

## Why four nodes, in sequence

By the time `consensus_engine`/`cross_validator` finish, evidence is scattered
across `state`: a vote, maybe a debate transcript, Slither findings, RAG chunks,
hotspot data. Not a report yet. `synthesizer → reflection → explainer →
visualizer` turn that pile into one coherent document, each adding a layer. This
is the part a human actually reads — everything earlier only matters because it
flows through here.

## `synthesizer` — verdict priority + the narrative

Up to 3 different opinions can exist for one class (Step 2). Checked in priority
order, falling through only if the previous is missing:
1. **Debate's verdict** (`cross_validator`) — most informed, read the actual code.
2. **`consensus_engine`'s vote** — fallback if debate didn't run/failed. This
   ordering IS today's Fix 1 — previously `synthesizer` skipped straight to #3,
   the exact bug that let `compute_verdict()` disagree with `consensus_engine` on
   `safe_storage.sol`.
3. **`compute_verdict()`** (Step 3's oldest rule) — last resort, only for classes
   `consensus_engine` never voted on.

Then builds the **narrative** (LLM-written Markdown, 4 fixed sections: Severity /
Vulnerability Summary / Exploit Pattern / Recommended Fix). Today's Fix 2 lives
here: the prompt used to list every flagged class with NO verdict attached, so the
model had no signal a class was already cleared — the literal mechanism behind
the Reentrancy/Multicall hallucination. Fixed: every class now carries its actual
verdict in the prompt; instructions explicitly say "only discuss CONFIRMED/LIKELY."

## `reflection` — self-critique, with a real blind spot

Asks "does this hang together?" — unused evidence, contradictions, low-confidence
verdicts. Partly fixed rules (always runs), partly an optional LLM summary.

Sees verdicts WITH labels attached (same fix as narrative) — but does NOT see the
narrative text itself. Caught a real tension correctly on `safe_storage.sol`, but
structurally cannot catch a hallucination INSIDE the narrative's prose, because
that text never reaches it. Still open, not hypothetical.

## `explainer` — the LIME-style breakdown (Step 1's foreshadowing, paid off)

Computes "how much of this verdict came from ML vs. Slither vs. RAG" as
percentages summing to 100. Folds everything Phase A added
(`confidence_by_class`, `consensus_verdict`, `reflection_notes`) into one
`final_report` dict — single artifact, not scattered state.

## `visualizer` — the last stop

Self-contained, clickable HTML (source one side, verdict cards the other),
written to disk. Last node before END. Nothing downstream depends on it — pure
output for a human.

→ You now know: the synthesis chain is where today's two real bugs lived (verdict
fallback order, narrative grounding), and `reflection`'s blind spot (sees verdicts,
not narrative prose) is a genuinely still-open gap, not yet fixed.
