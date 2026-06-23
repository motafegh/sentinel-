# SENTINEL Agents Module — Mastery Roadmap

> **For:** Ali — going from "I know roughly what it does" to "I can explain, modify, debug,
> and extend any part of the agents module cold."
> **Created:** 2026-06-23. **Style:** self-study. Follow stages in order; don't skip the spine.
> **Rule of the road (from CLAUDE.md):** trust the *source code* first. Every "Read" below
> points at `.py` files. The README/DIAGRAM cross-refs are maps — useful, but if a doc and the
> code ever disagree, the code wins. If you catch one disagreeing, that's a finding — note it.

---

## How to use this document

There are **three roadmaps** plus a **capstone**. Do them in this order:

1. **Roadmap A — Core Pipeline Mastery** — the spine. Everything else assumes it. ~3–5 sessions.
2. **Roadmap B — The "Why"** — the redesign story + supporting systems. Makes the code's *shape* make sense. ~2–3 sessions.
3. **Roadmap C — The Frontier** — production infra + the not-yet-built capability phases. ~2 sessions.
4. **Capstone** — prove mastery by doing, not reading.

**Every stage has the same five parts** — use them as a checklist:

| Part | What it means |
|---|---|
| 🎯 **Concept** | The one idea this stage teaches. If you can't state it in a sentence, you're not done. |
| 📖 **Read** | Exact files (and line ranges) to read, in order. The *primary source*. |
| 🗺️ **Cross-ref** | Section of `agents/README.md` or `agents/DIAGRAM.md` that maps it. Read *after* the code. |
| ✍️ **Exercise** | Do this with your hands. Predict-then-verify beats re-reading. |
| ✅ **Checkpoint** | "You've mastered this stage when you can…" — a concrete test. Don't advance until you pass it. |

**Two reference docs you'll lean on throughout** (both freshly updated 2026-06-23):
- `agents/README.md` — operational map (how to run, env vars, module map, testing).
- `agents/DIAGRAM.md` — architecture (the 13-node graph, state schema, verdict pipeline, file map §12).

**Where the truth about current status lives:** `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`
and `docs/plan/agents/2026-06-21-agents-redesign/01_MASTER_PLAN.md`.

---

## Mental model before you start (read once, the 60-second version)

The agents module takes **Solidity source** and produces a **per-vulnerability-class verdict**
(`CONFIRMED` / `LIKELY` / `DISPUTED` / `SAFE`) plus a human report. It does this by orchestrating
many evidence sources through a **LangGraph state machine of 13 nodes**:

```
                                         ┌─ rag_research ───┐
START → ml_assessment → quick_screen →   │  static_analysis │ → audit_check → consensus_engine
                          evidence_router┤  graph_explain   │                      │
                                (router)  └──────────────────┘                      ▼
                                    │  (FAST PATH: skip the fan-out)         cross_validator
                                    └────────────────────────────────────────────► synthesizer
                                                                                      │
                                                            reflection → explainer → visualizer → END
```

- **ML is a hint, not a vote.** The GNN+CodeBERT model flags classes; tools (Slither, Aderyn,
  RAG, on-chain history) corroborate or contradict; a weighted **consensus engine** and an
  optional **LLM debate** reconcile them into the final verdict.
- **The governing safety principle:** a flagged class only reaches `SAFE` if **both** the consensus
  engine **and** the debate agree it's safe. The debate can *upgrade* a verdict but can only
  *downgrade* to `DISPUTED`, never silently to `SAFE`. (This is the heart of the redesign — Roadmap B.)
- **Everything is fail-soft:** if the LLM or ML API is down, nodes fall back to rule-based logic
  and the graph still completes.

Hold that picture. Now go build it properly.

---
---

# ROADMAP A — Core Pipeline Mastery (the spine)

By the end of Roadmap A you can trace any contract through all 13 nodes, name what each node
reads and writes, and explain why the path branched the way it did.

---

## Stage A0 — Orientation: the map before the territory

🎯 **Concept:** What the module is for, what the major pieces are, and where they live on disk —
so every later file has a home in your head.

📖 **Read** (skim, don't study):
- `agents/README.md` §Overview (lines 5–59) and §Module Map (60–117).
- `agents/DIAGRAM.md` §1 One-Page Overview (9–47) and §2 the 13-node graph (48–133).
- Then `ls agents/src/` and match each folder to a sentence: `orchestration/` (the brain),
  `mcp/servers/` (5 tool servers), `rag/` (knowledge retrieval), `llm/` (model client),
  `ingestion/` (feedback loop), `api/` (HTTP gateway), `eval/` (benchmark metrics).

✍️ **Exercise:** Without looking, draw the 13-node graph from memory. Arrows included.
Then check against DIAGRAM §2. Redraw until it matches.

✅ **Checkpoint:** You can (a) name all 13 nodes in execution order, (b) say which folder
each `src/` package is and what it does in one sentence each.

---

## Stage A1 — The Backbone, Part 1: LangGraph & the shared state

🎯 **Concept:** A LangGraph audit is a message-passing state machine. There is one shared
`AuditState` dict; each node receives it, returns a *partial* update, and LangGraph merges it.
This is THE foundational idea — get it wrong and nothing else makes sense.

📖 **Read** (this is the most important read in Roadmap A — go slowly):
- `agents/src/orchestration/state.py` — **all 203 lines.** Read every field comment. Pay
  special attention to the two `RECALL` blocks at the top (partial-dict merge; reducers) and
  to `routing_decisions` (lines 76–83) — the one field with an *append* reducer
  (`Annotated[list, operator.add]`) instead of replace-on-write.
- Note the field lifecycle table (lines 15–28): which node *sets* each field.

🗺️ **Cross-ref:** `DIAGRAM.md` §4 State Schema (178–235).

✍️ **Exercise:** Make a table with three columns — *field*, *set by which node*, *read by which
node(s)*. Fill it from the comments. You'll use this table for the rest of Roadmap A.

✅ **Checkpoint:** You can answer: *Why does `routing_decisions` need `operator.add` but
`verdicts` doesn't?* (Hint: who writes each, and how many writers run in parallel.) If you can't,
re-read state.py:10–13 and 76–83.

---

## Stage A2 — The Backbone, Part 2: graph wiring & the two paths

🎯 **Concept:** How the 13 nodes are wired, how the **fast path vs deep path** decision is made,
and how parallel branches fan out and fan back in.

📖 **Read:**
- `agents/src/orchestration/graph.py` — **all 281 lines.** The module docstring (1–53) is a
  mini-lecture on the execution model — read it. Then:
  - `_route_from_evidence_router()` (91–137) — the conditional edge. This is the **two-signal
    gate**: fast path needs ML *and* quick_screen to both say safe.
  - `build_graph()` (144–253) — node registration (170–182), edges (185–220), checkpointer (222–244).
  - The lazy `__getattr__` singleton (256–281) — why importing the module is cheap.

🗺️ **Cross-ref:** `DIAGRAM.md` §2 (the fan-out / fan-in and the fast-path gate, 110–133);
`README.md` §Graph Topology (203–256) and §Checkpointing (321–328).

✍️ **Exercise:** Trace two contracts on paper using your A1 field table:
  1. A contract the ML model thinks is totally safe *and* Slither finds nothing → which nodes run?
  2. A contract ML thinks is safe but Slither flags High severity → which nodes run, and why?
  (Answer is in `_route_from_evidence_router` lines 120–135.)

✅ **Checkpoint:** Given any `ml_result` + `quick_screen_hits`, you can predict the exact set of
nodes that will execute and in what order. You can explain *why* `graph_explain` always joins the
deep-path fan-out (graph.py:134).

---

## Stage A3 — Entry signals: ML assessment, quick screen, routing

🎯 **Concept:** The first three nodes turn raw source into the routing decision. ML produces a
**three-tier** signal (confirmed / suspicious / below); quick_screen runs static tools on *every*
contract as a safety net; routing maps probabilities → which tools to activate per class.

📖 **Read** (node functions live in the 2280-line `nodes.py` — read just these spans):
- `nodes.py` `ml_assessment` (361–433) — calls the inference MCP server, normalizes the
  three-tier schema into `ml_result`. Cross-check the schema against `state.py:56–69`.
- `nodes.py` `quick_screen` (191–315) — Tier-0 Slither/Aderyn on all contracts; writes
  `quick_screen_hits`. Understand *why this exists* (state.py:89–94: "ML says safe but tricky").
- `nodes.py` `evidence_router` (316–360) — logs routing rationale to `routing_decisions`.
- `agents/src/orchestration/routing.py` — **all 282 lines.** This is the single source of truth
  for per-class thresholds and `compute_active_tools()`. Note it's called *twice* (once by the
  node for logging, once by the edge function for branching — graph.py:108–111 explains the split).

🗺️ **Cross-ref:** `README.md` §Per-Class Routing (285–306); `DIAGRAM.md` §2 fast-path gate.

✍️ **Exercise:** In `routing.py`, find the threshold table. Pick three classes (e.g., Reentrancy,
Timestamp, GasException) and write down their deep-path thresholds. Then invent an `ml_result`
that activates exactly two of them and verify with the logic in `compute_active_tools`.

✅ **Checkpoint:** You can explain the difference between the ML *tier* thresholds
(confirmed/suspicious, in `ml_result`) and the *routing* thresholds (in `routing.py`), and why
both exist. You can state what `quick_screen` protects against.

---

## Stage A4 — Evidence gathering: the deep-path nodes + the MCP mesh

🎯 **Concept:** On the deep path, three nodes gather independent evidence in parallel, then
`audit_check` pulls on-chain history. Each leans on an **MCP server** — understand the client/server
split.

📖 **Read:**
- `nodes.py` `static_analysis` (729–913) — runs Slither + Aderyn for real, parses findings,
  and (when ExternalBug is flagged) builds `external_call_summary` (state.py:101–107 — the
  "GNN structural gap" compensation). This is the longest evidence node; read it carefully.
- `nodes.py` `rag_research` (434–517) — queries the RAG server for similar known vulns.
- `nodes.py` `graph_explain` (914–1012) — GNN attention / CodeBERT hotspots → `ml_hotspots`.
- `nodes.py` `audit_check` (518–728) — on-chain AuditRegistry history.
- Then the servers they call: skim `agents/src/mcp/servers/` —
  `inference_server.py` (:8010), `rag_server.py` (:8011), `audit_server.py` (:8012),
  `graph_inspector_server.py` (:8013), `representation_server.py` (:8014, the WS5 CFG/CEI server).

🗺️ **Cross-ref:** `DIAGRAM.md` §3 MCP Server Mesh (134–177) and §6 RAG Pipeline (292–334);
`README.md` §MCP Servers (368–389).

✍️ **Exercise:** Pick `representation_server.py`. Find `_detect_cei_violations` and explain, in
two sentences, how a write-before-external-call (the classic reentrancy shape) is detected from
the CFG. (This is WS5 — you'll revisit its rationale in Roadmap B.)

✅ **Checkpoint:** For each deep-path node you can name (a) the state field(s) it writes, (b) the
MCP server it depends on and that server's port, (c) what happens if that server is down (hint:
fail-soft — find the try/except).

---

## Stage A5 — The Verdict Engine (the intellectual heart)

🎯 **Concept:** This is where evidence becomes a verdict. Three mechanisms, in order:
**consensus_engine** (weighted per-class vote: ML is a *hint*, tools vote), **confidence tracking**
(Bayesian update), and **cross_validator** (the optional Prosecutor/Defender/Judge LLM debate),
followed by the **8-case reconciliation** that decides the final label under a strict FN/FP-asymmetry
rule. Spend real time here — this is the part that makes SENTINEL *correct*, not just plumbed.

📖 **Read, in this exact order:**
1. `agents/src/orchestration/consensus.py` — **all 164 lines.** The weighted vote. Find where
   `ML_WEIGHT_SCALE` is applied (MEMORY.md note: it's 0.5 so ML *alone* can never CONFIRM).
2. `nodes.py` `consensus_engine` (1987–2057) — how the node calls consensus.py and writes
   `consensus_verdict` + `confidence_by_class`.
3. `agents/src/orchestration/confidence.py` — **all 79 lines.** The Bayesian confidence tracker.
4. `nodes.py` `cross_validator` (1013–1570) — the big one (~560 lines). It contains:
   - the **selective debate gating** (WS4.2): skip the expensive 3-role debate when *every*
     flagged class already has ≥2-tool CONFIRMED consensus (look around 1343–1364);
   - the **8-case reconciliation** between consensus verdict and debate verdict (~1397–1420):
     the rule that the debate can upgrade but can only downgrade to `DISPUTED`, never `SAFE`.
5. `agents/src/orchestration/attribution.py` — **all 59 lines.** Per-source % breakdown that
   later feeds the explainer.

🗺️ **Cross-ref:** `DIAGRAM.md` §9 Verdict Pipeline (408–466); `README.md` §Verdicts (307–320).
Then read the *design* doc that specifies the rules:
`docs/plan/agents/2026-06-21-agents-redesign/05_VERDICT_RECONCILIATION_PLAN.md` (the 8 cases).

✍️ **Exercise (do this one carefully):** Build a truth table of all 8 reconciliation cases —
columns: *consensus says*, *debate says*, *final verdict*. Fill it from `cross_validator`'s code,
then check it against `05_VERDICT_RECONCILIATION_PLAN.md`. Confirm the invariant: there is **no**
row where (consensus = non-SAFE) AND (final = SAFE) without the debate also clearing it.

✅ **Checkpoint:** You can (a) explain why ML alone can't CONFIRM, (b) recite the FN/FP asymmetry
rule in one sentence, (c) given a consensus verdict + a debate verdict for a class, state the final
verdict without looking, (d) explain when the debate is *skipped* and why that's safe.

---

## Stage A6 — Synthesis & enrichment: turning verdicts into a report

🎯 **Concept:** The last four nodes assemble the human-facing report and add self-critique,
attribution, and a visual. Every run (fast or deep) passes through these.

📖 **Read:**
- `nodes.py` `synthesizer` (1571–1986) — assembles `final_report`; calls the LLM for the
  narrative (fail-soft to rule-based). Note the report schema (state.py:140–149).
- `nodes.py` `reflection` (2058–2207) — self-critique (A.3): unused evidence, contradictions,
  uncertain verdicts. Writes `reflection_notes`.
- `nodes.py` `explainer` (2208–2249) — LIME-style metric attribution (A.8); folds confidence +
  consensus into the report.
- `nodes.py` `visualizer` (2250–end) + `agents/src/orchestration/visualizer.py` (163 lines) —
  the self-contained interactive hotspot HTML.

🗺️ **Cross-ref:** `DIAGRAM.md` §9 (the synthesis tail); `README.md` §Verdicts.

✍️ **Exercise:** Open a real generated report under `agents/test_audit_reports/` (or run one in
the Capstone). Map every top-level field of the report back to the node that produced it, using
your A1 field table.

✅ **Checkpoint:** You can trace a single field in the final report (say `recommendation` or
`metric_attribution`) all the way back to the node and the evidence that produced it.

> **🏁 End of Roadmap A.** You now understand the *machine*. Roadmap B explains *why it's shaped
> the way it is* — which is what separates someone who can read the code from someone who can change
> it safely.

---
---

# ROADMAP B — The "Why": the redesign & supporting systems

The pipeline looks the way it does because of a specific set of discovered failures. Understanding
those failures is what lets you modify the verdict logic without reintroducing them.

---

## Stage B1 — Read the investigation

🎯 **Concept:** The redesign started from 12+3 concrete findings, each verified against code.
Knowing them tells you where the bodies are buried.

📖 **Read** (these are *design docs* — but they were written against the code and are current):
- `docs/plan/agents/2026-06-21-agents-redesign/README.md` — the index (start here).
- `00_FINDINGS.md` — the 12 numbered, severity-rated findings.
- `04_LIVE_BASELINE_FINDINGS.md` — findings #13–15, found empirically on real reports.
  **#14 is the critical one**: the debate's verdict used to silently override a correct,
  tool-corroborated verdict. This is the bug the whole verdict-integrity work exists to kill.

✍️ **Exercise:** For finding #14, go find the code in `cross_validator` that now *prevents* it
(the reconciliation invariant from Stage A5). Connect the finding to the fix.

✅ **Checkpoint:** You can name three of the original findings and point at the code that fixed each.

---

## Stage B2 — The workstreams (WS0–WS5), each as "broken → fixed"

🎯 **Concept:** Each workstream is one coherent fix. You already saw the *code*; now attach each
to its *reason*.

📖 **Read** `docs/plan/agents/2026-06-21-agents-redesign/01_MASTER_PLAN.md` (the master ordering
table + per-WS prose). For each WS, then glance at its test file to see what behavior is locked in:

| WS | What it fixed | Code you already read | Test file |
|---|---|---|---|
| **WS0** | Tests asserted plumbing, not verdict quality → built the eval gate | `src/eval/` | `tests/test_eval_framework.py` |
| **WS1** | Verdict integrity / FN-FP safety net | consensus + reconciliation | `tests/test_verdict_integrity.py` |
| **WS1.5** | The 8-case reconciliation rules | `cross_validator` (1397–1420) | `tests/test_verdict_reconciliation.py` |
| **WS2** | Removed fabricated RAG evidence (kept 752 real DeFiHackLabs) | `src/rag/build_index.py` | `tests/test_rag_fetchers.py` |
| **WS3** | Debate saw too much / wrong stuff → hotspot excerpts + eye clues | `cross_validator` input prep | `tests/test_ws3_hotspot_excerpts.py` |
| **WS4.1 / 4.2** | Debate cost cap; skip debate when consensus is already certain | `cross_validator` gating | `tests/test_ws4_2_selective_gating.py` |
| **WS5** | Reused data_module CFG as an MCP tool (CEI detection) | `representation_server.py` | `tests/test_representation_server.py` |

✍️ **Exercise:** Run the suite for one workstream and read the test names as a spec:
`cd agents && source .venv/bin/activate && python -m pytest tests/test_verdict_reconciliation.py -v`.
The test names *are* the behavior contract — read them like sentences.

✅ **Checkpoint:** For any WS0–WS5 you can say the one-line problem, the one-line fix, and name a
test that would fail if the fix regressed.

---

## Stage B3 — Supporting systems: RAG, ingestion/feedback, LLM client

🎯 **Concept:** The three systems that feed and surround the pipeline. Lower priority than the
spine, but you need them for a complete picture.

📖 **Read:**
- **RAG:** `src/rag/build_index.py` (build), `src/rag/retriever.py` (hybrid FAISS + BM25 with
  RRF), `src/rag/chunker.py`, `src/rag/embedder.py`. Cross-ref `DIAGRAM.md` §6 (292–334),
  `README.md` §RAG Pipeline (329–367).
- **Ingestion / feedback loop:** `src/ingestion/pipeline.py`, `feedback_loop.py`,
  `deduplicator.py`, schedulers. Cross-ref `DIAGRAM.md` §7 (335–377).
- **LLM client:** `src/llm/client.py`. Cross-ref `DIAGRAM.md` §8 (378–407), `README.md` §LLM
  Client (379–389). Note the LM Studio endpoint (MEMORY.md: `http://127.0.0.1:12345/v1`).

✍️ **Exercise:** In `retriever.py`, find where FAISS (semantic) and BM25 (lexical) scores are
fused (RRF). Explain in one sentence why hybrid beats either alone for vuln retrieval.

✅ **Checkpoint:** You can describe how a RAG query flows from `rag_research` → retriever →
ranked chunks, and how the feedback loop would add a newly-confirmed vuln back into the index.

---
---

# ROADMAP C — The Frontier: production infra & extended capability

What's built, what's half-built, and what's planned. This is where *you* will likely add code next,
so know the seams.

---

## Stage C1 — Phase C: production infrastructure

🎯 **Concept:** The pipeline wrapped for real use — an HTTP gateway, an evaluation harness, and
(planned) input guards + monitoring.

📖 **Read / status (verify against MEMORY.md, which is the live status):**
- **C.1 Gateway — DONE.** `src/api/gateway.py` (5 endpoints: POST/GET `/audit`, `/health`, `/`,
  list), `src/api/job_store.py` (in-memory job lifecycle + eviction), `src/api/models.py`
  (Pydantic schemas). Tests: `tests/test_gateway.py` (~44). Run it: `python -m src.api.gateway`
  (port 8000). Read the gateway docstring (1–45) — it explains the async job model and lazy graph import.
- **C.2 Eval framework — DONE.** `src/eval/pipeline_metrics.py` (per-class P/R/F1),
  `benchmarks.py` (loads the 88-contract corpus), `regression.py` (baseline compare).
  Tests: `tests/test_eval_framework.py`. **Dataset decision:** uses the existing 88-contract WS0
  corpus now; expanding to 150–200 real-world contracts is *deferred, not abandoned*
  (see `docs/plan/agents/2026-06-17-extended-capability/03_PHASE_C_EXECUTION_PLAN.md`, C.2 header).
- **C.3 Guards — NOT started.** Prompt-injection defense (comment stripping, pattern detection)
  for untrusted contract source fed to the LLM. Spec in `03_PHASE_C_EXECUTION_PLAN.md`, C.3.
- **C.4 Monitoring — NOT started.** Health checks, alerts, metrics. Spec C.4.

🗺️ **Cross-ref:** `README.md` §Quick Start (118–200) to actually run the gateway + servers.

✍️ **Exercise:** Start the gateway, POST a small contract with `audit_no_llm=true`, poll the
job_id until done, read the JSON. (Commands: `README.md` §Quick Start.)

✅ **Checkpoint:** You can submit an audit over HTTP and explain the job lifecycle (PENDING →
RUNNING → COMPLETED/FAILED) by pointing at `job_store.py`. You can state exactly what C.3 and C.4
still need to add and why they matter.

---

## Stage C2 — Phase B & D: the not-yet-built capability frontier

🎯 **Concept:** The big planned extensions. The *tools* are installed and verified; the *nodes*
that use them aren't written yet. Reading the plans now means you can implement them later.

📖 **Read:**
- `docs/plan/agents/2026-06-17-extended-capability/02_PHASE_B_EXECUTION_PLAN.md` — symbolic
  execution + bytecode: B.1–B.3 Halmos/Z3 formal verification, B.4–B.5 Gigahorse bytecode
  analysis, B.6 taint, B.7 access control, B.8 call-graph reachability, B.9 CVE matching.
  Note the schema placeholders already reserved in `state.py:194–202`
  (`symbolic_findings`, `bytecode_analysis`, `taint_flows`, `permission_graph`).
- `04_PHASE_D_EXECUTION_PLAN.md` — economic + on-chain: D.1–D.4 ItyFuzz/Anvil attack simulation,
  D.5 ZKML proofs, D.6 on-chain AuditRegistry submission.
- **The toolchain:** `~/tools/TOOLCHAIN_ENV.md` — how to actually invoke Halmos, Gigahorse,
  ItyFuzz, Anvil (all installed + verified 2026-06-22).

✍️ **Exercise:** Run one tool yourself end-to-end using `TOOLCHAIN_ENV.md` — e.g., decompile a
contract with Gigahorse, or `ityfuzz --help`. Then look at the matching `state.py` placeholder and
imagine the node that would populate it.

✅ **Checkpoint:** You can explain what *new, independent* evidence Phase B adds that the current
pipeline can't produce (formal proofs + bytecode facts vs. heuristics), and why Phase D is
sequenced last (most specialized, furthest from current correctness gaps).

---
---

# CAPSTONE — Prove mastery by doing

Reading is necessary but not sufficient. Do all five. Each maps back to a roadmap.

1. **Run a full real audit** (tests A0–A6). Start the MCP servers + gateway (`README.md` §Quick
   Start), audit a contract from `agents/test_contracts/`, and read the report end-to-end.
2. **Predict before you run** (A2–A5). Pick a contract, predict: the path taken, which tools
   activate, the per-class consensus, and the final verdicts. *Then* run and diff your prediction
   against the report. Investigate every miss — the misses are where your model of the system is wrong.
3. **Change a threshold** (A3). Lower one class's deep-path threshold in `routing.py`, predict the
   behavior change, run, confirm. Revert.
4. **Add a test** (B2). Write one new test asserting a verdict-reconciliation behavior (e.g., a
   specific consensus+debate combo → expected final verdict). Make it pass. This proves you
   understand the invariant, not just the code.
5. **Explain a node cold** (all of A). Have someone (or future-you) pick a random node; explain
   from memory what it reads, computes, writes, what it depends on, and how it fails soft.

✅ **You are a master of the agents module when:** you can do #5 for *any* of the 13 nodes, you
passed #2 with no surprises you couldn't explain, and you could implement a new Phase-B node by
following the plan + the `state.py` placeholder + the existing node patterns.

---

## Appendix — Fast index of "where things live"

| You want… | Go to |
|---|---|
| The shared state schema | `agents/src/orchestration/state.py` |
| The graph wiring / paths | `agents/src/orchestration/graph.py` |
| All 13 node implementations | `agents/src/orchestration/nodes.py` (2280 lines; line anchors in Stages A3–A6) |
| Per-class routing thresholds | `agents/src/orchestration/routing.py` |
| Weighted consensus voting | `agents/src/orchestration/consensus.py` |
| Verdict reconciliation rules | `cross_validator` in `nodes.py` (~1013–1570) + `05_VERDICT_RECONCILIATION_PLAN.md` |
| Confidence / attribution | `confidence.py`, `attribution.py` |
| The 5 MCP tool servers | `agents/src/mcp/servers/` (ports 8010–8014) |
| RAG | `agents/src/rag/` |
| HTTP gateway | `agents/src/api/` |
| Eval / benchmark | `agents/src/eval/` |
| How to run anything | `agents/README.md` §Quick Start |
| The big architecture picture | `agents/DIAGRAM.md` (file map in §12) |
| The "why" behind the design | `docs/plan/agents/2026-06-21-agents-redesign/` |
| Live project status | `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` |
| External tool invocation | `~/tools/TOOLCHAIN_ENV.md` |

---

*Roadmap maintained alongside the code. If a `file:line` reference drifts, the code moved — update
the reference and treat the drift as a finding (CLAUDE.md Rule 4).*
