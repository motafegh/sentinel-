# Step 4 — "ML Is a Hint," Not an Authority

## The problem

Run 12 (current ML model) isn't reliable enough to trust alone — caught giving
high-confidence false positives (verified live today: 81.8% confident on a bug
class that doesn't exist in that contract). But it's not useless either — right
most of the time. Ali's directive: use the signal without fully trusting it.
`consensus_engine` node + `consensus.py` + `confidence.py` (pure logic, no
network/AI calls) implement this.

## Weighted voting (`consensus.py`)

Three "witnesses" per class: ML, Slither, Aderyn. Not equally trustworthy for
every class — Slither nails Reentrancy syntactically, is useless on Timestamp
misuse (business-logic judgment, not syntax). `ACCURACY_WEIGHTS` gives each
witness a per-class trust level, e.g. Reentrancy: `{ml: 0.78, slither: 0.82,
aderyn: 0.60}`.

`consensus_vote()`: each witness's yes/no × its weight, summed, divided by total
possible weight ("normalizing" — scaling so the result always lands in [0,1]
regardless of how many witnesses spoke). That's the vote's **confidence**, mapped
to a band: ≥0.70 CONFIRMED, ≥0.50 LIKELY, ≥0.30 DISPUTED, below SAFE.

## The actual "ML is a hint" mechanism

`get_weights()` multiplies ML's base weight by `ML_WEIGHT_SCALE` (an environment
variable — a config value living outside the code, changeable without editing
files), default `0.5`. Reentrancy: 0.78 × 0.5 = 0.39 vs Slither's untouched 0.82.

Result: ML alone, even at 99% confidence, normalizes to ~0.39 — below the 0.70
CONFIRMED line. **ML alone can never confirm anything.** Verified directly:
`consensus_vote(0.99, slither=False, aderyn=False, "Reentrancy")` → always SAFE.
One agreeing tool → jumps past LIKELY immediately.

## Two different "confidences" — don't conflate them

- `consensus_vote()`'s confidence = normalized weighted-vote score (above).
- `track_confidence()` (`confidence.py`) = a SEPARATE number. Starts at the ML
  probability, then multiplicatively nudges it as each tool's agreement/
  disagreement arrives (×1.10 if Slither agrees, ×0.90 if it disagrees — "loosely
  Bayesian": each new signal adjusts the existing belief rather than replacing
  it). Always clamped (forced into bounds) to [0,1].

`consensus_engine` (the node) calls both for every class with ML prob ≥ 0.50 OR a
tool hit (skips weak sub-threshold noise with no support), storing results in
`state["consensus_verdict"]` and `state["confidence_by_class"]`.

→ You now know: the entire "ML is a hint" directive boils down to one line —
discount ML's vote weight before tallying — and it's been verified to make
ML-alone-confirmation mathematically impossible.
