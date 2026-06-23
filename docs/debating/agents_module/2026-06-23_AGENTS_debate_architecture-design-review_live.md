# SENTINEL Agents Module — Architecture & Design Debate (LIVE)

> **Status:** LIVE working document. Started 2026-06-23. This captures an ongoing
> debate between Ali and Claude about the agents module's design, architecture, and
> the professional/production-grade practices around it. **We add, remove, and modify
> this as the discussion proceeds.** It is a thinking record, not a finalized spec.
>
> **Naming:** follows the 6-part convention (`<date>_<MODULE>_<RUN_or_PHASE>_<WHAT>_<descriptor>.md`).
> **Trust rule:** all code-level claims here were read from source on 2026-06-23
> (graph.py, state.py, routing.py, consensus.py, confidence.py, llm/client.py,
> nodes.py verdict logic, _call_mcp_tool, job_store.py). Re-verify before acting if stale.
>
> **Companion scratch (raw notes):** `~/.claude/scratch/agents_arch_review_20260623.md`

---

## 0. Purpose & how to use this doc

A place to keep the high-value architectural reasoning so we don't lose it across
sessions. Each major topic is a section. Open questions and unmade decisions are
tracked explicitly at the bottom (§9). When a decision is made, move it from §9 into
the relevant section and note the date.

---

## 1. Current state of the agents module (factual baseline)

Read from `docs/plan/agents/2026-06-21-agents-redesign/01_MASTER_PLAN.md` + source + tests.

- **446 tests pass** (the 3 previously-noted solc failures are now resolved).
- **~11,673 LOC** in `agents/src/`. Largest file: `nodes.py` = **2,280 LOC** (god-file).
- Workstreams: **8 of 9 done** (WS1–WS5, WS6a/C.1 gateway, WS6a/C.2 eval framework).
- **Open:** WS6a/**C.3 prompt-injection guards** (not started), **C.4 monitoring** (not started),
  **WS6b** Phase B symbolic/bytecode (tooling unblocked, no implementation),
  **WS6c** Phase D economic/on-chain (tooling unblocked, no implementation).
- Large uncommitted working tree (src/api, src/eval, representation_server, tests, READMEs).

---

## 2. What kind of architecture this *actually* is

**It is a deterministic WORKFLOW (a DAG with LLM-augmented nodes) — NOT an agentic system.**

- Control flow is fixed in `build_graph()`. Nodes do not choose their own tools or
  next action; `evidence_router` branches via hardcoded rules. The LLM only fills in
  *content* at specific nodes (debate, narrative), never *control flow*.
- Anthropic's "workflow vs. agent" distinction: SENTINEL is firmly a **workflow**.
- **This is the correct choice for the domain.** A security oracle needs determinism,
  reproducibility, and auditability — you do not want an autonomous agent improvising
  its analysis path on a contract holding millions.
- **Takeaway:** the name "agents" oversells it. The strong framing is: *"a deterministic
  orchestration graph, chosen deliberately because auditability matters more than
  autonomy in security."* That is more impressive than claiming it's agentic.

### Topology (from graph.py)
```
START → ml_assessment → quick_screen → evidence_router ─┬─(fast)→ synthesizer
                                                        └─(deep, parallel fan-out)→
                                                           rag_research
                                                           static_analysis  → audit_check →
                                                           graph_explain                  consensus_engine →
                                                                                          cross_validator →
                                                                                          synthesizer →
                                                          reflection → explainer → visualizer → END
```
- Two-signal fast-path gate: fast path requires BOTH ML safe AND quick_screen clean (FN safety).
- SqliteSaver checkpoints every node (resume-on-crash). PEP562 lazy graph singleton.

---

## 3. Core design philosophy (sound, senior-level)

Three principles are visibly encoded and they cohere:

1. **Defense-in-depth / evidence accumulation.** No single detector trusted. ML, Slither,
   Aderyn, RAG (later symbolic/economic) each contribute independent evidence; layers
   fuse them. = ensemble with a meta-reasoner.
2. **Cost funnel with escalation.** Cheap signals on everything (ML, quick_screen) → gate
   → expensive analysis (RAG, 3-call debate) only when warranted.
3. **FN/FP asymmetry as the governing invariant.** "A missed vuln costs millions; a wasted
   review costs minutes." Encoded as actual control flow, not a slogan.

### The verdict layer (the crown jewel AND the biggest risk)
- `consensus_engine` — deterministic weighted per-class vote over {ML, Slither, Aderyn}.
  ML discounted by `ML_WEIGHT_SCALE=0.5` → ML alone can never reach CONFIRMED.
- `cross_validator` — LLM debate (Prosecutor/Defender/Judge), semantic reasoning.
- `_reconcile_verdicts` — **8-case hand-written table** fusing consensus vote vs debate.
  Debate can UPGRADE; can only downgrade to DISPUTED, never silently SAFE. Each rule cites
  the real contract it was hardened against. Genuinely well-reasoned safety logic.

---

## 4. Strengths (professional view)

1. **Verdict-integrity design is the standout.** FN/FP asymmetry encoded as control flow;
   distinct INCONCLUSIVE/DISPUTED state ("checked, found nothing" ≠ "couldn't check").
2. **ML treated as a discounted hint, not an authority** — structurally requires
   corroboration. Correct given Run 12's known FP behavior.
3. **Fail-soft everywhere** — every LLM/MCP/tool call degrades to a rule-based path.
4. **Intellectual honesty in the code** — job_store labeled NOT PRODUCTION-READY; dead
   Slither detector names verified against 0.11.5 and removed (not left as silent no-ops).
5. **MCP tool boundary** isolates the GPU-bound ML model behind a service boundary (good).

---

## 5. Weaknesses (professional view)

1. **`nodes.py` is a 2,280-line god-file** — 13 nodes + verdict reconciliation + Aderyn
   parsing in one file. Biggest structural debt. Split into `nodes/` + `verdict/` packages.
   (Other over-long files exist project-wide too.)
2. **Hand-tuned magic numbers, never validated** — DEEP_THRESHOLDS, ACCURACY_WEIGHTS,
   confidence nudges, verdict bands. Honestly labeled, but the C.2 eval framework that
   could calibrate them is built and NOT YET USED for that. **The design is principled
   but unmeasured — the central gap.**
3. **One 2B model does all reasoning** (incl. the debate the verdict trusts). Downgraded
   from Qwen-9B purely for RTX-3070 speed (9B ≈ 2.9 tok/s). Quality ceiling is set here.
4. **RAG is effectively a stub** — only 726 DeFiHackLabs chunks (matched 0 of 2 test
   contracts); fakes removed. RAG nudges rarely fire.
5. **No prompt-injection guard (C.3 not started)** — untrusted contract source flows
   straight into LLM prompts. Real security hole for a *security* tool.
6. **Single-process gateway, in-memory job store, fresh MCP connection per call** — not
   horizontally scalable yet (all documented with swap paths, none done).

---

## 6. Three big DESIGN tensions (the deep, unmade decisions)

### Tension 1 — Fusion architecture doesn't scale to N channels *(highest leverage)*
- Verdict today = ensemble-of-ensemble: `consensus_engine` (deterministic vote) +
  `cross_validator` (debate), fused by an **8-case hand if-table** designed for TWO inputs.
- Roadmap adds: symbolic, bytecode, taint, access-control (Phase B), economic (Phase D)
  = ~8 evidence channels. A pairwise hand-cased reconciliation **will not scale**.
- **Right design:** every channel emits a uniform typed record
  `Evidence{class, signal, confidence, reliability, source}`; **one fusion function**
  consumes the *list*. Adding a channel = append to the list, not rewrite the table.
- **Do this BEFORE Phase B lands**, or Phase B calcifies the un-scalable shape.

### Tension 2 — Determinism vs. ZK-provability *(reaches the product thesis)*
- Product goal: ZK-proved, on-chain audit oracle → verdict must be **reproducible**.
- `consensus_engine` = pure deterministic math → ZK-friendly, anchorable.
- LLM debate = **non-deterministic** (temp=0 ≠ guaranteed across model versions/quant/hardware).
- The more the verdict leans on (and is overridable by) the debate, the harder to prove/anchor.
- **Unmade decision:** is the *provable* verdict the deterministic consensus (debate =
  advisory context only)? Or does only the consensus tier go on-chain? Decide before the
  verdict layer hardens.

### Tension 3 — Agent layer is compensating for the model *(intent question)*
- ML is the thesis centerpiece (GNN+GraphCodeBERT) yet deliberately DISTRUSTED
  (scale=0.5, can't confirm alone). Correct today (Run 12 FP behavior), but it means the
  agent layer does the heavy lifting the model was meant to do.
- **Unmade decision:** as the model improves, does the agent layer THIN OUT (ML earns more
  trust), or is multi-tool defense-in-depth PERMANENT regardless of model quality? The two
  answers imply different architectures.

---

## 7. The "strong model as decider / fast model as worker" idea (Ali's Q)

Real pattern: **orchestrator-worker / model cascade / strong-judge + weak-generator.**
Spend model *quality* where a decision is made; spend model *speed* where text is produced.

Mapping to SENTINEL:
- **Decisions (strong model):** the **Judge** in cross_validator — it's the boss; the whole
  reconciliation trusts its verdict. Highest-value place for a strong model.
- **Production (fast model):** synthesizer narrative, explainer, visualizer (describing an
  already-decided verdict). 2B fine.
- **No model:** evidence_router, consensus_engine (pure code — keep deterministic).

Elegant version: **Prosecutor + Defender on fast 2B (surface arguments); Judge on strong
model (decides); only on hard cases; reading only ml_hotspots, not the whole contract.**

**Caveat (changes the math):** LLM time = prefill (read input) + decode (generate output).
A short verdict is NOT free if the prompt is the whole contract — prefill of a long prompt
is slow on the 3070. Levers: (1) feed the Judge only hotspot excerpts (WS3 pays off here);
(2) convene the strong Judge only on genuinely ambiguous cases (WS4.2 gating already exists).

**Still gated on measurement:** can't tell if a strong Judge is worth it — or even if the
2B Judge is the real weak link — without the eval framework. Measure first.

---

## 8. Implementation/practice answers (Ali's 4 Qs) + the two new rules

### 8.1 Long files vs small modules
- Professional standard = **Single Responsibility per file** ("if you need the word *and*,
  it's two files"). Driver = blast radius of change, not aesthetics. Small focused modules win.
- Heuristics: function ~50 lines; file 200–400; >500 needs a reason; >1000 almost always splittable.
- Why the AI wrote god-files: appending to the current file is the path of least resistance.

### 8.2 Magic numbers — production approach (maturity ladder)
- L0: hand-set constant (prototype, label honestly) ← **where we are**.
- L1: externalize into a versioned config (YAML/JSON) with a schema.
- L2: measure before every change — eval delta on a held-out labeled set (recall-weighted).
- L3: LEARN from data — consensus weights fitted from per-tool confusion matrix; thresholds
  from precision-recall curves at a chosen operating point.
- Mantra: **tests prove the code RUNS; evals prove the system is GOOD.**

### 8.3 Model swap
- Mechanically one line (`MODEL_STRONG`), architecture is model-agnostic (good).
- Catches: speed/hardware (reintroduces timeout problem), can't tell if worth it without
  eval, prompts are model-tuned (gemma quirks). Treat model as a variable the eval measures,
  not a faith-based swap. Order: calibrate eval → then a model bake-off.
- Ali's "speed-first to prove the pipeline works across contracts" was the correct order.

### 8.4 RAG plan review (02_RAG_BUILD_PLAN.md)
- **Strong:** leads with a go/no-go gate (Step 0 — measure before building); A/B split
  (canonical SWC defs vs real finding corpus), ship A first (cheap, never misleads);
  relevance floor (Step 3) internalizes the hallucination lesson; honest effort estimate.
- **Push back:** (1) diagnose *why* DeFiHackLabs matched zero BEFORE sourcing more —
  embedding mismatch? granularity? no overlap? Cheaper than Step 2, should precede it.
  (2) the relevance-floor threshold is itself a number to calibrate, not guess.
  (3) type-A (10 class definitions) may not need RAG at all — fits in a system prompt;
  RAG earns its complexity only when the KB is too big for context (type-B).

### 8.5 Two new rules added to root CLAUDE.md (2026-06-23) — "Professional Coding Standards (always on)"
- **Rule A — One file, one reason to change** (Single Responsibility; small focused modules).
- **Rule B — Separate policy from mechanism; never change a decision-number without measuring
  it** (tests run / evals good; L0→L3 maturity ladder). Scoped to ALL modules.

---

## 9. Open questions / unmade decisions (tracker)

**ALL RESOLVED 2026-06-23** in the finalization proposal:
`docs/proposal/2026-06-23_proposal_SYSTEM_architecture-finalization.md` (§4). Summary:

| # | Question | Decision | Status |
|---|----------|----------|--------|
| D-A | Generalize fusion to a uniform Evidence model BEFORE Phase B? | YES (highest leverage; gates roadmap) | RESOLVED |
| D-B | ZK/determinism boundary — debate in trust path or advisory? | Prove deterministic core (Opt A); debate advisory; Opt C for decentralization later | RESOLVED |
| D-C | Agent layer thins as model improves, or permanent? | Both — ML reliability data-derived (auto-thins); layer persists to a floor | RESOLVED |
| D-D | Close the measurement loop — when? | FIRST (keystone, P0) | RESOLVED |
| D-E | Split nodes.py? | YES, inside the D-A refactor (P2) | RESOLVED |
| D-F | Strong-Judge cascade? | Conditional YES, gated on D-D measurement (P6) | RESOLVED |
| D-G | Keep `reflection`? | Gate behind eval; keep-but-skippable (P6) | RESOLVED |

Beyond-debate decisions added in the proposal: B-1 prompt-injection (mandatory, early),
B-2 externalize config, B-3 reliability replaces hand-set weights, B-4 reproducibility test +
model-hash, B-5 RAG gated (diagnose zero-match, ship SWC cheap), B-6 gateway hardening.

---

## 10. Cross-cutting keystone

A single thread runs through nearly everything above: **the eval/measurement loop is the
keystone.** It makes the magic numbers tunable (§8.2), the model swap a decision not a guess
(§8.3, §7), the RAG go/no-go answerable (§8.4), and `reflection`'s value checkable (D-G).
The C.2 framework is built; the missing move is *closing the loop* (D-D).

---

## Changelog
- **2026-06-23** — Doc created. Captured: current state, workflow-not-agentic framing,
  design philosophy, strengths/weaknesses, three design tensions, strong-Judge cascade idea,
  the 4 implementation answers, the two new CLAUDE.md rules, open-decision tracker.
- **2026-06-23** — Debate round 2 (improvisation-vs-determinism, evidence model, ZK boundary,
  ML-compensation). All decisions D-A…D-G RESOLVED and consolidated into the finalization
  proposal: `docs/proposal/2026-06-23_proposal_SYSTEM_architecture-finalization.md`. §9 updated.
