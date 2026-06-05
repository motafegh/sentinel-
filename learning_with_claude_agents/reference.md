# Reference — Learning With Claude (Agents Module)

This folder tracks the ongoing deep-dive study of the Sentinel agents module.
It contains 4 files. This file is the entry point — read it first.

---

## The 4 Files

| File | Purpose | When to Update |
|------|---------|----------------|
| `reference.md` | **This file.** How the system works, rules for each file, overall structure. | When meta-rules, current status, or roadmap change. |
| `preferences.md` | All teaching preferences (P1, P2, ...). Controls HOW teaching is delivered. | Immediately when a new preference is stated or observed. Never batch. |
| `audit_flags.md` | All issues found during teaching (A1, A2, ...). Bugs, design problems, missing guards. | Immediately when an `[AUDIT]` flag is raised during teaching. |
| `session_log.md` | Record of what was covered in each session. Progress tracker. | After each chunk is fully delivered and questions posted. |

---

## Spec File Update Protocol

This protocol defines exactly when, how, and what to update in each file.
Claude must follow this on every session — not selectively.

### WHEN to update

| Trigger | File(s) to update | Timing |
|---------|------------------|--------|
| User states a new preference | `preferences.md` | **Immediately** — before continuing teaching |
| Claude observes a new teaching pattern | `preferences.md` | At the end of the current response |
| An `[AUDIT]` flag is raised inline | `audit_flags.md` | **Immediately** — same response that raised it |
| A chunk finishes (teaching delivered, questions posted) | `session_log.md` | End of that response |
| Current status changes (chunk complete, phase done) | `reference.md` (Current Status section) | End of that response |
| A preference is refined or clarified | `preferences.md` | Immediately, with a note on what changed |
| CLAUDE.md project facts change (branch, constraints, etc.) | `CLAUDE.md` | When the change is confirmed |

### HOW to update

- **preferences.md** — Append new `### P#` section at the bottom. Never rewrite existing ones silently; add a "(refined: ...)" note if clarifying.
- **audit_flags.md** — Append new `## A#` entry at the bottom. Full format: File, Location, Issue, Fix, Severity, Status, Raised. Never delete or edit past entries.
- **session_log.md** — Append new `## Session N` block. Include: file/lines, concepts taught, warm-up results, gaps closed, audit flags raised.
- **reference.md** — Update the `Current Status` section inline. Update roadmap phase markers (✅ → 🔄 → pending).

### WHAT must be in each entry

**preferences.md entry minimum:**
```
### P# — Short Title
What the rule is.
When it applies.
Format/example if relevant.
```

**audit_flags.md entry minimum:**
```
## A# — File — Short description
**File:** path
**Location:** function/line
**Issue:** what is wrong and why it matters
**Fix:** concrete fix
**Severity:** Low / Medium / High
**Status:** Open / Noted / Fixed
**Raised:** Session N, Chunk N
```

**session_log.md entry minimum:**
```
## Session N — Phase X: filename (Chunk N)
**File:** path (lines)
**Concepts taught:** bullet list
**Warm-up recall:** pass/fail per question, gaps noted
**Challenge questions:** answered Y/N, gaps closed
**Audit flags raised:** A# list
```

### Commit rule

After any spec file update: `git add learning_with_claude_agents/ && git commit && git push`.
Spec files are the persistent memory of this journey — uncommitted updates are lost if the session ends.

---

## Rules

### For Claude

1. **At the start of every teaching session:** read all 4 files to restore full context.
   No assumptions — state of the journey lives here, not in conversation memory.

2. **Preferences are non-negotiable constraints.** Every teaching response must comply
   with ALL active preferences in `preferences.md`. Check before writing.

3. **Audit flags must be raised inline** during teaching using the format:
   > **[AUDIT] A#** — description
   Then immediately add to `audit_flags.md`. Never delay.

4. **Session log updates** happen after each chunk is fully delivered and questions posted —
   not mid-chunk.

5. **Never delete entries** from audit_flags.md or session_log.md. Only append.

6. **Preferences can be updated or refined** — add new ones, clarify existing ones.
   Never silently override an existing preference; add a note if it evolves.

7. **Follow the Spec File Update Protocol above** on every session, every response that triggers
   an update. No exceptions.

### For the User

- State new preferences at any time — Claude will add them to `preferences.md` immediately.
- Challenge question answers trigger gap-fill teaching (P2) and session log update.
- To resume a session after a break: just say "resume" — Claude reads all 4 files first.

---

## Current Status

- **Active phase:** Not started — Phase 1 is next
- **Current chunk:** None yet
- **Preferences active:** P1 through P16
- **Audit flags raised:** None
- **Files taught so far:** None

---

## Module Architecture Overview

The agents module (~11,274 lines) is a production-scale smart-contract audit orchestrator.
It coordinates ML inference, RAG retrieval, static analysis, on-chain history, and LLM synthesis
through a 9-node LangGraph state machine. Six sub-modules:

```
┌─────────────────────────────────────────────────────────────┐
│         LangGraph StateGraph (9 nodes)                       │
│  ml_assessment → quick_screen → evidence_router → [routes]  │
│              → cross_validator → synthesizer                 │
└─────────────────────────────────────────────────────────────┘
         calls 4 MCP servers (SSE HTTP, ports 8010-8013)
         ├── :8010 inference_server   → Module 1 FastAPI (ML)
         ├── :8011 rag_server          → HybridRetriever
         ├── :8012 audit_server        → Sepolia on-chain
         └── :8013 graph_inspector     → GNN hotspots / Slither

RAG Pipeline: DeFiHackLabs → chunk → embed → FAISS+BM25 → RRF
Ingestion: incremental fetch + dedup + on-chain feedback loop
LLM Client: LM Studio (FAST/STRONG/CODER/EMBED model roles)
```

**Teaching order rationale:**
LLM client first — it's used everywhere; understand it before nodes reference it.
State + routing next — the skeleton of the graph before the flesh.
RAG third — the knowledge base that most nodes depend on.
Ingestion fourth — how RAG stays fresh + the on-chain feedback loop.
MCP servers fifth — how capabilities are exposed to the graph and the outside world.
Orchestration last — graph topology + all 9 nodes, where everything converges.

---

## Teaching Roadmap

Legend: ✅ done · 🔄 in progress · ⬜ pending

```
Phase 1  ⬜  llm/client.py                         (LM Studio client, 4 model roles)
Phase 2  ⬜  orchestration/state.py                 (AuditState TypedDict, 16 fields, reducers)
             orchestration/routing.py               (DEEP_THRESHOLDS, ROUTING_RULES, verdict logic)
Phase 3  ⬜  rag/                                   (RAG pipeline — fetch → chunk → embed → retrieve)
             rag/fetchers/base_fetcher.py           (abstract BaseFetcher, Document dataclass)
             rag/fetchers/github_fetcher.py         (DeFiHackLabs, 3 comment formats → 2 chunks)
             rag/chunker.py                         (RecursiveCharacterTextSplitter)
             rag/embedder.py                        (nomic-embed-text, retry logic)
             rag/retriever.py                       (FAISS+BM25+RRF — HybridRetriever)
             rag/build_index.py                     (atomic writes, rollback snapshots → 2 chunks)
Phase 4  ⬜  ingestion/                             (incremental pipeline + feedback loop)
             ingestion/deduplicator.py              (SHA256 dedup)
             ingestion/pipeline.py                  (IngestionPipeline, incremental fetch)
             ingestion/feedback_loop.py             (OnChainListener + FeedbackIngester → 2 chunks)
             ingestion/scheduler_dagster.py         (Dagster asset, daily schedule — awareness only)
Phase 5  ⬜  mcp/servers/                           (4 MCP servers, SSE HTTP)
             mcp/servers/inference_server.py        (predict, batch_predict → wraps ML API)
             mcp/servers/rag_server.py              (search → wraps HybridRetriever)
             mcp/servers/audit_server.py            (on-chain history → 2 chunks)
             mcp/servers/graph_inspector_server.py  (hotspots, GNN→Slither fallback)
Phase 6  ⬜  orchestration/                         (9-node LangGraph brain)
             orchestration/graph.py                 (StateGraph builder, SqliteSaver checkpoint)
             orchestration/nodes.py                 (1415 lines → 4 chunks)
               Chunk 1: ml_assessment, quick_screen, evidence_router
               Chunk 2: rag_research, static_analysis, graph_explain
               Chunk 3: audit_check, cross_validator
               Chunk 4: synthesizer, report persistence, end-to-end data flow
```

---

## Extended Capability Proposal

**Source:** `docs/agent_proposal/2026-06-04-agent-extended-capability-proposal.md`
**Status:** Design proposal — pending review

### Three missing analysis paradigms (not in current V3 graph)

| Paradigm | What's absent | Proposed agent |
|---|---|---|
| Execution-based | No symbolic execution | `symbolic_exec` — Mythril on hot functions |
| Proof-of-concept | No exploit generation | `poc_generator` — coder LLM + `forge test` |
| Economic | No fork simulation | `economic_sim` — Anvil fork, flash loan / oracle attacks |

### Other proposed additions
- **`reflection` node** — LLM self-critique pass after synthesizer (low effort, high value)
- **Multi-LLM debate** — upgrade `cross_validator` to prosecutor + defender + judge
- **RAG expansion** — Code4rena, Sherlock, Solodit sources (currently only 726 DeFiHackLabs)
- **FastAPI gateway** — entry point for external contract submission (currently none)
- **Pipeline evaluation** — end-to-end benchmark, not just ML training metrics
- **Prompt injection guards** — comment stripping before LLM prompts (security gap)

### Revised graph (proposal)
```
ml_assessment → quick_screen → evidence_router
  └─ FAST PATH → synthesizer → reflection → END
  └─ DEEP PATH:
       rag_research ─┐
       static_analysis ─┤ (parallel)
       graph_explain ─┤
       symbolic_exec ─┘ (NEW — Tier 2)
       ↓
       audit_check → cross_validator (debate) → poc_generator (NEW)
       → economic_sim (NEW, DeFi only) → synthesizer → reflection (NEW) → END
```

### Teaching integration
When we reach the relevant chunk, each proposed addition is taught as:
- What gap it closes (the *problem* the current node doesn't solve)
- How it would be designed (the *solution* — tool choice, trigger conditions, state additions)
- What new concepts it introduces (the *learning exposure*)

---

## File Map — agents/src/

| File | Lines | Complexity | Teaching approach |
|------|-------|------------|-------------------|
| `llm/client.py` | 184 | Medium | Single chunk — 4 model roles, LM Studio wiring |
| `orchestration/state.py` | 160 | Medium | Single chunk — TypedDict, reducers, 16 fields |
| `orchestration/routing.py` | 263 | Medium-High | Single chunk — threshold table, routing rules, verdict logic |
| `orchestration/graph.py` | 241 | Medium | Single chunk — StateGraph builder, checkpointing |
| `orchestration/nodes.py` | 1415 | Very High | 4 chunks — 9 node implementations |
| `rag/fetchers/base_fetcher.py` | 94 | Low | Single chunk — abstract interface |
| `rag/fetchers/github_fetcher.py` | 478 | High | 2 chunks — DeFiHackLabs parser, 3 comment formats |
| `rag/chunker.py` | 199 | Low-Medium | Single chunk — splitter config, overlap |
| `rag/embedder.py` | 228 | Medium | Single chunk — embed client, retry logic |
| `rag/retriever.py` | 334 | High | Single chunk — FAISS+BM25+RRF |
| `rag/build_index.py` | 603 | High | 2 chunks — atomic writes, rollback |
| `ingestion/deduplicator.py` | 136 | Low | Single chunk — SHA256 |
| `ingestion/pipeline.py` | 313 | Medium-High | Single chunk — incremental pipeline |
| `ingestion/feedback_loop.py` | 470 | High | 2 chunks — OnChainListener, FeedbackIngester |
| `ingestion/scheduler_dagster.py` | 140 | Low | Awareness only — Dagster asset/schedule |
| `mcp/servers/inference_server.py` | 480 | High | Single chunk — MCP tool wiring, SSE transport |
| `mcp/servers/rag_server.py` | 353 | Medium-High | Single chunk |
| `mcp/servers/audit_server.py` | 707 | High | 2 chunks — ABI decoding, on-chain calls |
| `mcp/servers/graph_inspector_server.py` | 527 | High | Single chunk — hotspot attribution, fallbacks |
