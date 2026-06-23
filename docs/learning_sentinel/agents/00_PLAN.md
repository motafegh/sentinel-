# Agents Module Onboarding — Plan

**Style:** ONBOARDING.md "Plan Onboarding" (guided workshop) — frame each step
(problem/intent/why+how) → walk it → checkpoint (what changed, what we learned).
No pop quizzes; proceed on Ali's cue.

**Why this folder exists:** Ali asked to persist this journey, not just the chat —
each step below gets its own file in this folder, written as we go (not batched at
the end), so the context survives past this conversation.

**Goal beyond "what the code does":** every step is anchored to a real decision or
incident from today's session (the registration bug, the timeout investigation, the
ML-as-hint directive) — the agents module's CURRENT shape is a product of those
incidents, and understanding why a piece exists is more durable than memorizing
what it does.

---

## Ground rules (added after Ali's correction, step 1 → step 2)

- **No unexplained jargon, ever.** Any technical/professional term — "LIME-style",
  "TypedDict", "context manager", "subprocess", "fan-out", whatever — gets a plain
  definition the first time it appears, even if it seems basic. Don't assume
  fluency in any of the underlying tech (Python async, HTTP, ML terms, etc.).
- **For important concepts, confirm the point landed**, even briefly — not a quiz,
  but don't just move past something load-bearing without making sure it's clear.
- This applies retroactively too — if a past step used an unexplained term, fix it
  when caught, like the LIME example in step 7's description above.

## Steps

### 1. Big picture — `01_big_picture.md`
The 13-node graph end to end: fast path vs deep path, where ML inference sits,
where this module physically runs (in-process nodes vs 4 separate MCP server
processes vs LM Studio). Context: why the graph grew from 9 nodes to 13 today
(Phase A), and why some nodes run in-process while others are network calls.

### 2. AuditState — `02_state_and_data_flow.md`
The one TypedDict every node reads/writes. How LangGraph merges partial returns.
Context: why `total=False` everywhere, why routing_decisions uses an append
reducer, and which fields are genuinely load-bearing vs which are Phase-A
additions still finding their place (e.g. consensus_verdict vs verdicts).

### 3. Routing — `03_routing_and_thresholds.md`
`quick_screen` (Tier 0, runs on everything) → `evidence_router` → fast/deep
fork. `DEEP_THRESHOLDS`, `ROUTING_RULES`, `CLASS_TO_DETECTORS`. Context: the
two-signal gate (ML + quick_screen must BOTH agree it's safe) and why that
exists — closing the "ML says safe but a static tool disagrees" gap.

### 4. "ML is a hint" design — `04_ml_as_hint_consensus.md`
`consensus_engine`, `confidence.py`'s Bayesian updates, `ML_WEIGHT_SCALE`.
Context: this is Ali's mid-session directive from the Phase A build — Run 12 ML
isn't trustworthy enough to be authoritative, so the agent layer must
corroborate independently. Live proof from today: ML-only never reaches
CONFIRMED; verified against a real false positive (safe_storage.sol).

### 5. The debate — `05_the_debate.md`
`cross_validator`'s Prosecutor → Defender → Judge. Context: why it reads the
contract source directly (not just the ML/tool summary), the timeout
architecture (one outer budget, not per-call — and why that changed), and the
REAL measured cost from today's unbounded test (each role ~75-115s, debate
~257-336s total — this is the most expensive part of an audit by far).

### 6. Static analysis tools — `06_static_analysis_tools.md`
Slither + Aderyn integration in `static_analysis` / `quick_screen`. Context:
this is the most important step for trust calibration — both tools were
SILENTLY non-functional (Slither registered zero detectors; Aderyn was
invoked with wrong arguments) for the module's entire history until today.
Understanding exactly what was broken and how it was verified fixed should
make you comfortable trusting (or distrusting) any given finding.

### 7. Synthesis chain — `07_synthesis_chain.md`
`synthesizer` → `reflection` → `explainer` → `visualizer`. Context: two fixes
from today — the verdict fallback gap (consensus_engine's vote being silently
bypassed) and the narrative hallucination (LLM describing a vulnerability that
every other signal said was SAFE). Shows how a multi-stage pipeline can have
internally-disagreeing stages even when no single stage is "wrong."

### 8. Running it live — `08_running_it_live.md`
The 4 MCP servers + LM Studio + `run_real_audit.py`. How to actually fire off
a real audit yourself, read the log, and find the report. Context: the
timeout/timing infrastructure built today (`timeouts.py`, `timing.py`,
`--unbounded-timeouts`) exists specifically so this is observable, not a black
box.

---

## Tracking

- [ ] 1. Big picture
- [ ] 2. AuditState
- [ ] 3. Routing
- [ ] 4. ML-as-hint design
- [ ] 5. The debate
- [ ] 6. Static analysis tools
- [ ] 7. Synthesis chain
- [ ] 8. Running it live
