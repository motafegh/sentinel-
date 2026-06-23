# Step 5 — The Debate

## Why a debate, not just more math

`consensus_engine` is arithmetic over yes/no signals. Some judgments need actual
reading-and-reasoning over the contract text — that's a job for a language model
(LLM), not a weighted vote. `cross_validator` runs that conversation.

## System prompt vs user message

A **system prompt** = instructions telling the model who to act as / how to behave
for the whole exchange, set once behind the scenes. A **user message** = the actual
content to respond to. `cross_validator` sends THREE different system prompts in
sequence, each assigning a different role.

## The three roles (adversarial = opposing sides arguing, courtroom-style)

1. **Prosecutor** — "argue why it HAS the vulnerabilities... treat the ML
   probability as a weak hint only." Given per-class evidence + first 2000 chars of
   the actual contract source — reasons from real code, not just numbers.
2. **Defender** — "argue why these findings may be false positives... the ML model
   is known to over-predict." Gets Prosecutor's argument + same evidence + source,
   told explicitly to push back.
3. **Judge** — gets both arguments (text only, no raw source again) and renders
   structured JSON (a simple text format for structured data) mapping each class
   to CONFIRMED/LIKELY/DISPUTED/WATCH/SAFE.

Throughline from Step 4: Prosecutor told ML is "a weak hint," Defender told ML
"over-predicts" — the discount from consensus_engine's math is reinforced again in
the language given to the AI.

## Why real source code, not just a summary

Per Ali's directive: independent judgment from reading the actual code, not
rubber-stamping the ML/tool summary.

## Real measured cost (not estimated — from today's unbounded-timeout runs)

| Run | Prosecutor | Defender | Judge | Debate total | % of audit |
|---|---|---|---|---|---|
| vulnerable_reentrant.sol | 114.3s | 115.2s | 106.7s | 336.5s | 73% |
| safe_storage.sol | 81.1s | 100.8s | 75.2s | 257.0s | 72% |

Three SEQUENTIAL calls (Defender needs Prosecutor's text first — can't run in
parallel like Step 3's tool fan-out). ~3/4 of total audit wall-clock time, on the
FAST model (chosen because the STRONG model was even slower here).

## Timeout architecture (the bug fixed this week)

ONE budget wraps the entire 3-call sequence (`DEBATE_TIMEOUT_S`, 240s) — not
per-call. Old version: per-call budget allowed 3×90s=270s worst case, exceeding an
outer script's own tighter limit, killing the process mid-debate. Real numbers
above show even 240s is too tight — genuinely needs 257-336s.

On timeout/failure: `cross_validator` returns empty, `synthesizer` falls through to
`consensus_engine`'s vote, then `compute_verdict()` as last resort (Step 2/3 chain).

→ You now know: the debate is 3 sequential LLM role-plays reinforcing the ML
discount in plain language, costs ~3/4 of total audit time, and fails soft into
the same verdict-fallback chain from Steps 2-3.
