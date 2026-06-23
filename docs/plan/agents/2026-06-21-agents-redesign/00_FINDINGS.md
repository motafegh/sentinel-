# Redesign Findings — Debate Scaling, RAG, and Unused ML Preprocessing (2026-06-21)

**Context:** raised when Ali questioned whether
the debate's full-source-reading makes earlier pipeline stages pointless, and
whether RAG has any real data behind it. Both turned out to be real, verified
problems — not just stylistic concerns. This doc captures everything found,
verified against the actual code (not assumed), plus Ali's 4 follow-up ideas,
each checked against the codebase. **No fixes applied yet — findings only,
for a redesign decision.**

**Update (same session):** Ali corrected my initial debate-gating suggestion —
see "FN/FP asymmetry correction" section below. This is the single most important
addition to this doc; read it before acting on anything else here.

---

## Finding 1 — Debate/narrative source truncation breaks at realistic scale

**Verified limits** (exact `grep` of `nodes.py`):
- `cross_validator` (the debate): contract source truncated to **2,000 characters** (`nodes.py:1116`)
- `synthesizer` narrative: truncated to **500 characters** (`nodes.py:1481`)
- `rag_research` query text: truncated to 200 characters (`nodes.py:461`) — lower
  stakes, just used to build a search query, not shown raw to verify.

**Every test contract used today was 373–720 characters** — meaning every debate
run this session saw 100% of the contract with 3-5x headroom to spare. We have
**zero evidence** about behavior on a realistic contract (commonly thousands of
characters, multiple functions, inheritance). At 2,000 chars, a real contract's
debate would plausibly see only import statements and the first function or two
— if the actual vulnerable function isn't in that prefix, Prosecutor and Defender
are arguing over code that doesn't contain the bug.

**No test needed to know this is true** (Ali's point): we're already slow
(257-336s for the debate alone) on tiny contracts; a longer contract can only be
slower AND more incomplete under fixed-character truncation — both facts follow
directly from how the truncation is implemented, no experiment required.

## Finding 2 — RAG's new sources are placeholder data, and directly caused a hallucination

**Verified directly from a saved report** (`test_audit_reports/safe_storage_report.json`,
`rag_evidence` field): the retrieval for `safe_storage.sol` returned 4 real
DeFiHackLabs historical-hack documents (none relevant to this contract) plus:

```
source: solodit, protocol: Multicall
"Arbitrary external call with user-controlled target and calldata,
execute(address target, bytes data) performs target.call(data) without an allowlist"
```

This is **not real scraped data** — it's one of 5 placeholder entries hand-written
during today's Phase A work for the Solodit fetcher (no live API connection was
ever built to Solodit/Code4rena/Sherlock/Immunefi; see
`docs/changes/2026-06-21-agents-phase-a-extended-capability.md`, which explicitly
flagged this as seed data to "replace with full exports for production"). This
EXACT entry is the content that produced the earlier hallucinated narrative
describing a Reentrancy/Multicall risk on a contract with zero external calls.

**Honest state of the RAG corpus:** 726 of 776 chunks are real (DeFiHackLabs, a
real GitHub repo of historical exploit post-mortems, predates this session). The
other 24 chunks (the 5 "new" sources added today) are synthetic filler I wrote —
not fetched from anywhere. For this contract, RAG contributed nothing useful and
actively introduced the hallucination's source material.

---

## Ali's 4 follow-up ideas — each checked against the actual code

### (1) Are the ML model's tokens/graphs (.pt files) reused in the agents system?

**Verified: essentially no.** `graph_explain`'s own docstring (`nodes.py:909-928`)
says explicitly: *"Phase 1 (current): Slither structural proxy for attention...
Phase 2 (future): true GNN attention weights via forward-pass hooking."* The
`ml_hotspots` field it produces today is a **Slither-based approximation**, not
the actual trained model's attention weights, graph tensors, or token embeddings.

Worse: even that approximation is barely used downstream. Searched for every
place `ml_hotspots` is READ (not just written) in `nodes.py` — **zero hits**.
It's only consumed by `visualizer.py` (`visualizer.py:97`) for the final HTML
highlighting — **never fed into `cross_validator`'s debate prompts or
`synthesizer`'s narrative prompts**, the two places that most need a "where to
look" signal instead of (or alongside) raw truncated text.

### (2) Could `data_module`'s existing code become MCP tools for agents?

**Verified: real, substantial, currently-unused opportunity.**
`data_module/sentinel_data/representation/` contains: `graph_extractor.py`,
`cfg_builder.py` (control-flow graph), `call_graph.py`, `pdg_builder.py` (program
dependence graph), `opcode_extractor.py`, `tokenizer.py` — this is the SAME
representation pipeline that builds the graph/token data used to train the ML
model. `data_module/sentinel_data/preprocessing/` adds `segmenter.py`,
`normalizer.py`, `flattener.py`, `compiler.py`. None of this is wired into the
agents module's MCP servers today — agents only talks to 4 purpose-built servers
(inference, RAG, audit, graph-inspector) and otherwise calls Slither/Aderyn raw.
Exposing pieces of this as additional MCP tools is plausible and would reuse
already-tested code rather than reinventing graph/AST parsing inside agents.

### (3a) Should the debate run only in specific situations, not by default?

**Verified current behavior:** the debate runs unconditionally on every contract
that reaches the deep path (`cross_validator` is a fixed node in the graph after
`consensus_engine`, no additional gating). Given the debate is ~3/4 of total audit
time (Step 5 finding), and `consensus_engine` already provides a fast, ML-discounted,
tool-corroborated vote before the debate even starts, there's a real case for
gating the debate behind a narrower trigger (e.g., only when `consensus_engine`'s
confidence is ambiguous/contested, or only for specific high-stakes classes) rather
than running the full 3-role conversation on every deep-path contract regardless.

### (3b) Does the debate harness prevent "over-talking" / excess token usage?

**Verified: no, it does not.** `synthesizer`'s narrative call sets
`max_tokens=4096` (`nodes.py:1537`) and `reflection`'s call sets `max_tokens=1024`
(`nodes.py:1858`) — both have an explicit output-length cap. **`cross_validator`'s
three debate calls (`get_strong_llm()`/`get_fast_llm()` at `nodes.py:1108/1110`) set
no `max_tokens` at all.** Each of Prosecutor/Defender/Judge can generate as much
text as the model wants, with no length constraint and no instruction telling it to
be concise — directly contributing to the measured 75-115s per role. A harness fix
here (explicit max_tokens per role + a "respond in N sentences" instruction) is a
concrete, low-risk lever to both cut cost and reduce rambling.

### (3c) Handle large contracts the way the ML model already does (chunking/windows)?

**Verified: the ML side already solved this properly, and agents doesn't reuse it.**
`ml/src/inference/predictor.py` (`process_source_windowed`, search hits around
lines 465, 554-575) implements **sliding-window tokenization**: short contracts
get 1 real window + zero-padding; long contracts get up to 4 windows of 512 tokens
each, combined via a learned `WindowAttentionPooler` — explicitly documented as
"no silent truncation." This is the exact problem Finding 1 describes, already
solved on the ML side with a real, validated mechanism. The debate's raw
`[:2000]`/`[:500]` character slicing is a much cruder fallback that throws away
everything past the cutoff with no pooling, no windowing, no signal about what
was lost.

### (3d) Use already-preprocessed representations instead of full raw reading?

**This is the strongest idea, and it directly connects (1) and (3c).** Two
existing-but-unused assets could feed the debate a focused, pre-digested view of
the contract instead of (or alongside) a blind text prefix:
- `ml_hotspots` / a real GNN attention pass (per `graph_explain`'s own "Phase 2"
  TODO) — "look at lines X-Y in function Z" instead of "here's the first 2000
  characters, good luck."
- The ML model's windowed-tokenization output — already segments a long contract
  into model-relevant chunks; could plausibly be summarized or re-used to decide
  what the debate sees, rather than re-deriving a worse truncation independently.

---

## FN/FP asymmetry correction (Ali's correction to the debate-gating suggestion)

**My original suggestion was wrong on the part that matters most.** I proposed
skipping the debate when `consensus_engine`'s confidence lands clearly at EITHER
extreme — clearly SAFE or clearly CONFIRMED — treating both as symmetric "we're
sure, save the time" cases. Ali pointed out that in a security-audit context the
cost of a false negative (missing a real vulnerability — potentially millions of
dollars) is wildly asymmetric with the cost of a false positive (flagging a safe
contract — wasted review time). Treating both extremes the same is wrong.

**Corrected logic:**
- **Skip the debate when multiple independent tools agree it's VULNERABLE** — low
  risk. The action doesn't change (flag it either way), and skipping also avoids
  a second risk: the Defender role arguing an already-corroborated finding back
  down to a lower verdict.
- **NEVER skip the debate because cheap signals say "safe."** This is exactly
  backwards from what I originally proposed. The cheap signals (ML, Slither,
  Aderyn) are the LEAST trustworthy part of the system — ML is proven to
  over-predict and is deliberately discounted (Step 4); Slither/Aderyn have
  documented per-class blind spots (e.g. Timestamp business-logic misuse is
  invisible to pattern-matching by design). "Looks safe by cheap signals" is
  precisely the regime where a costly miss could be hiding undetected. Cutting
  the most careful check there is the worst place to save time.

**This is not a new principle for this codebase — it already exists elsewhere,
just never named.** Verified by reading the actual code (not assumed):
- `routing.py`'s `DEEP_THRESHOLDS` are deliberately set BELOW the model's own
  "confident" cutoff (e.g. 0.35 vs 0.50-0.55) — the code comment literally says
  "we want to investigate borderline cases, not skip them." Biases toward more
  false positives (extra investigation) in exchange for fewer false negatives.
- The two-signal fast-path gate requires BOTH ML and `quick_screen` to
  affirmatively agree a contract is safe before skipping deep analysis — either
  one disagreeing forces a closer look.
- **Searched the entire codebase for any explicit statement of this principle —
  none exists.** Every mention of "false positive" in the code is about the
  OPPOSITE problem (ML over-predicting). The asymmetry is implicitly present in
  two routing decisions but has never been written down or used as a checklist
  to audit the rest of the system. This doc is the first time it's named.

## Routing-logic pass — other places this asymmetry might be silently violated

Read `routing.py` and the relevant `nodes.py` functions fully (not skimmed) to
check for other "looks safe → silently skip/clear" spots. Found three:

**1. `compute_verdict()`'s implicit SAFE collapse for the entire sub-0.50 range.**
The function's docstring claims `SAFE ← prob < DEEP_THRESHOLDS[cls] (shouldn't
reach here)` — but the actual code (verified by reading the full function body)
has no branch for the gap between a class's `DEEP_THRESHOLD` (e.g. 0.35) and 0.50.
A class that crossed its deep threshold (real enough to investigate), got
checked, and found NO corroboration lands in this gap and falls through to the
function's final `return "SAFE", sources` — the SAME verdict as a class ML scored
near zero. There's no distinction between "borderline, uncorroborated" and
"clearly nothing here." This collapses exactly the cases `DEEP_THRESHOLDS` was
designed to flag for extra scrutiny back down to a flat SAFE with no flag at all.

**2. `consensus_engine`'s skip-voting condition creates a silent gap with #1.**
`consensus_engine` only votes on a class if `prob >= 0.50` OR a tool hit exists
(`nodes.py:1724`: `if prob < 0.50 and not slither_found and not aderyn_found:
continue`). A class in the 0.35-0.49 band with no tool corroboration gets NO
`consensus_verdict` entry at all — meaning if the debate also fails for this
class (which we measured happens — debate timeouts), the verdict-fallback chain
(Step 2/3) skips straight past `consensus_engine` (nothing there) to
`compute_verdict()`, which — per finding #1 — auto-SAFEs it. **The exact failure
mode Ali is worried about is concretely reachable today**: a class flagged enough
to warrant deep investigation, with weak ML signal and no tool corroboration,
gets zero real scrutiny if the debate also happens to fail, and is marked SAFE
with no flag distinguishing it from "definitely nothing here."

**3. An inconsistency I introduced myself this week, found by this pass.**
`quick_screen`'s Slither escalation checks `impact in ("High", "Medium",
"Critical")` (`nodes.py:260`) — three severity levels. The Aderyn escalation I
rewrote during this week's bug fix checks `impact == "High"` only
(`nodes.py:285`) — one level. Aderyn's Medium-impact findings silently do NOT
trigger the fast-path-override escalation that the equivalent Slither findings
would. Not deliberate — an inconsistency introduced while fixing the larger
registration bug, caught only by this asymmetry-focused pass.

**Not yet fixed — all three are documented here for a deliberate decision, not
silently patched.**

---

## The model's "4 eyes" — are they used as separate evidence anywhere?

Ali's question: the ML model is a "four-eye" architecture — does anything in
agents, or any current plan, use the 4 eyes as separate evidence channels (the
way `consensus_engine` treats ML/Slither/Aderyn as separate witnesses)?

**Verified precisely, not assumed:**
- The model (`ml/src/models/sentinel_model.py`) computes 4 distinct internal
  representations ("eyes"): `gnn_eye` (graph/structural view), `transformer_eye`
  (token-sequence view via CodeBERT), `fused_eye` (cross-attention combination of
  the two), `cfg_eye` (control-flow-graph view). Each has its OWN auxiliary
  classifier — `aux_gnn`, `aux_transformer`, `aux_fused`, `aux_phase2` — meaning
  the model can in principle produce 4 separate per-class probability vectors,
  one per "eye," in addition to the final fused prediction.
- These aux classifiers are computed **unconditionally in the forward pass**
  (verified: no `if self.training:` guard around them in `sentinel_model.py`) —
  they run at inference time too, not just during training. **The compute already
  happens; the result is just thrown away.**
- The inference API's actual response schema (`predictor.py:_format_result`,
  read in full) returns ONLY the fused/final output: `probabilities`,
  `confirmed`, `suspicious`, `tier_thresholds`, `thresholds`, `truncated`,
  `windows_used`, `num_nodes`, `num_edges`. **No per-eye field exists anywhere
  in the schema.** Nothing in `agents/` receives them, and nothing in the Phase
  A-D extended-capability plans mentions exposing them either — this is not a
  "planned but not yet built" item, it's genuinely new.

**Is it a good idea? Conditionally yes, with one prerequisite before trusting it.**

The appeal is real: 4 independent perspectives on the same contract (structural
shape vs. token sequence vs. their fusion vs. control flow) disagreeing with each
other is itself a signal — much like multiple tools disagreeing already triggers
more scrutiny elsewhere in this system. And because the aux heads are already
computed, exposing them costs nothing extra in the forward pass — likely a
classifier of a Python dict field (in `_format_result`) plus an API schema
change. Cheap to expose.

**The caveat:** the aux heads exist for a TRAINING purpose (likely an auxiliary
loss encouraging each eye to learn a well-formed individual representation) —
there is no evidence any of them were evaluated as STANDALONE classifiers at
inference time. Before wiring "the 4 eyes disagree → flag for extra scrutiny"
into the live pipeline as a trusted signal, it needs an offline check first: on
the existing labeled benchmark, does inter-eye disagreement actually correlate
with cases the model got wrong? If yes, this is a strong, nearly-free addition to
`consensus_engine`'s evidence base. If the aux heads turn out to be noisy/
uncalibrated in isolation, exposing them risks adding apparent-but-fake signal —
the same category of mistake as the RAG placeholder problem, just from a
different source. **Validate before wiring in, don't wire in on the strength of
the architecture diagram alone.**

---

## Summary table

| # | Idea/Finding | Verified state | Severity |
|---|---|---|---|
| 1 | Debate/narrative truncation | 2000/500 chars, never tested past trivial contracts | High — design untested at real scale |
| 2 | RAG new-source data | Synthetic placeholders, directly caused a hallucination | High — actively harmful, not just unhelpful |
| 3 | ML tokens/graphs reuse | Not used; `ml_hotspots` is a Slither proxy, unused by debate/narrative | High — real signal sitting unused |
| 4 | `data_module` as MCP tools | Real, substantial, zero current integration | Medium — opportunity, not a bug |
| 5 | Debate should be selective | Currently runs unconditionally on every deep-path contract | Medium — cost/quality tradeoff, asymmetric (see correction) |
| 6 | Debate harness anti-rambling | No `max_tokens` cap on any of the 3 debate roles (unlike synthesizer/reflection) | Medium — concrete, low-risk fix available |
| 7 | Chunking like ML training | ML already has a validated windowed-tokenization solution; debate doesn't use it | Medium-High — proven pattern available to borrow |
| 8 | FN/FP asymmetry never stated | No principle anywhere in code/docs; my own gating suggestion violated it | High — corrected above, but worth a named rule |
| 9 | `compute_verdict()` SAFE collapse | Entire sub-0.50 band (incl. borderline 0.35-0.49) collapses to flat SAFE | High — silent, reachable today |
| 10 | `consensus_engine` skip-vote gap | Combines with #9 — a flagged-but-uncorroborated class can get zero real scrutiny if the debate also fails | High — concretely reachable, not hypothetical |
| 11 | quick_screen Slither/Aderyn inconsistency | Slither escalates on High/Medium/Critical; Aderyn (rewritten this week) only High | Low-Medium — self-introduced, now documented |
| 12 | Model's 4 eyes unused as evidence | Aux per-eye predictions computed but discarded; not in any current plan | Medium — promising, needs offline validation first |

**Not yet decided:** which of these to act on, in what order, or whether to
redesign the debate's source-access mechanism entirely (e.g., windowed/attention-
guided instead of raw truncation) before anything else. This doc is the
fact-finding record for that decision, not the decision itself.
