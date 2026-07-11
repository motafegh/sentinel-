# Documentation Plan — Agents Module

**Date:** 2026-06-26
**Goal:** Comprehensive documentation for all agents sub-modules — READMEs, diagrams, and technical learning documents.

---

## Current State

### Source files: ~64 across 9 sub-modules

| Sub-module | Files | Has README? | Has DIAGRAM? |
|-----------|-------|-------------|--------------|
| `api/` | 4 (gateway, job_store, sqlite_job_store, models) | NO | NO |
| `config/` | 2 (loader, schema) | NO | NO |
| `eval/` | 7 (benchmarks, gates, metrics, regression, reliability) | NO | NO |
| `ingestion/` | 5 (pipeline, deduplicator, feedback, schedulers) | YES (stale) | YES (stale) |
| `llm/` | 1 (client) | YES (stale) | YES (stale) |
| `mcp/` | 8 (5 servers + audit package) | YES (stale) | YES (stale) |
| `orchestration/` | 6 (graph, state, routing, consensus, confidence, attribution) | YES (stale) | YES (stale) |
| `orchestration/nodes/` | 14 node files | NO | NO |
| `orchestration/verdict/` | 5 (evidence, fuse, emit, reliability, verdict) | NO | NO |
| `rag/` | 13 (retriever, embedder, chunker, fetchers, build_index) | YES (stale) | YES (stale) |
| `security/` | 4 (comment_strip, delimit, detect, sanitize) | NO | NO |

**"Stale"** = written before P2-P10 work. Documents the old architecture, not the current one.

---

## Deliverables

### Part 1: READMEs (10 files)

Each sub-module gets a README.md that is:
- **Accurate** — reflects current source code, not old architecture
- **Concise** — what this module does, key files, how to run/test
- **Cross-linked** — links to the learning doc and diagram

| # | Path | Action |
|---|------|--------|
| 1 | `agents/README.md` | UPDATE — top-level overview, all sub-modules |
| 2 | `agents/src/api/README.md` | CREATE — gateway, job store, health monitoring |
| 3 | `agents/src/config/README.md` | CREATE — config schema, externalized decision-numbers |
| 4 | `agents/src/eval/README.md` | CREATE — eval framework, metrics, reliability matrix |
| 5 | `agents/src/ingestion/README.md` | UPDATE — report pipeline, deduplication |
| 6 | `agents/src/llm/README.md` | UPDATE — LLM client, model selection |
| 7 | `agents/src/mcp/README.md` | UPDATE — MCP architecture, 5 servers |
| 8 | `agents/src/orchestration/README.md` | UPDATE — graph, state, routing, 14 nodes |
| 9 | `agents/src/orchestration/verdict/README.md` | CREATE — evidence, fuse, reliability |
| 10 | `agents/src/security/README.md` | CREATE — 3-layer defense (P4) |
| 11 | `agents/src/rag/README.md` | UPDATE — hybrid retrieval, RRF, corpus |

### Part 2: Diagrams (update existing + create new)

All diagrams use ASCII/Mermaid (no external tools needed).

| # | Path | Content |
|---|------|---------|
| 1 | `agents/DIAGRAM.md` | UPDATE — full pipeline with all 14 nodes including formal_verification |
| 2 | `agents/src/orchestration/DIAGRAM.md` | UPDATE — graph topology, routing, evidence flow |
| 3 | `agents/src/orchestration/verdict/DIAGRAM.md` | CREATE — Evidence model, fuse() flow, dual verdict |
| 4 | `agents/src/security/DIAGRAM.md` | CREATE — 3-layer defense pipeline |
| 5 | `agents/src/api/DIAGRAM.md` | CREATE — gateway request flow, job lifecycle, health monitor |

### Part 3: Technical Learning Documents (the core deliverable)

These are **educational notes** — not API docs. They teach the concepts, design decisions, and patterns. Written for Ali to deeply understand what was built and why.

Each learning doc covers:
1. **What** — what this module does and why it exists
2. **How** — walkthrough of key source files, code-level explanations
3. **Why** — design decisions, alternatives considered, tradeoffs
4. **Patterns** — engineering patterns used (and what they teach)
5. **Connection** — how this module connects to the rest of SENTINEL
6. **Lessons** — what was learned building it (mistakes, insights)

| # | Path | Title | ~Pages |
|---|------|-------|--------|
| 1 | `docs/learning/01_orchestration_pipeline.md` | The Audit Pipeline: How 14 Nodes Process a Contract | 8-10 |
| 2 | `docs/learning/02_verdict_fusion.md` | Evidence Model & Fuse(): How 6 Channels Become One Verdict | 6-8 |
| 3 | `docs/learning/03_security_prompt_injection.md` | Prompt Injection Defense: 3 Layers, 8 Patterns, 8 Adversarial Contracts | 4-5 |
| 4 | `docs/learning/04_reproducibility_determinism.md` | Reproducibility & Model Hash: The ZK Boundary | 4-5 |
| 5 | `docs/learning/05_mcp_servers.md` | MCP Architecture: 5 Servers, SSE Transport, Fail-Soft | 5-6 |
| 6 | `docs/learning/06_rag_hybrid_retrieval.md` | RAG: Hybrid FAISS+BM25 Retrieval with Reciprocal Rank Fusion | 5-6 |
| 7 | `docs/learning/07_gateway_production.md` | Gateway Hardening: SQLite Persistence, Health Monitoring, Crash Recovery | 4-5 |
| 8 | `docs/learning/08_eval_framework.md` | Evaluation Framework: F1, Fbeta, Reliability Matrix, Bayesian Shrinkage | 5-6 |
| 9 | `docs/learning/09_formal_verification.md` | Halmos: Symbolic Execution as Evidence | 3-4 |
| 10 | `docs/learning/10_config_decision_numbers.md` | Decision Numbers: From Hand-Set Constants to Data-Derived Reliability | 3-4 |

---

## Execution Order

### Session 1: Core pipeline (most important)
1. `01_orchestration_pipeline.md` — the heart of the system
2. `02_verdict_fusion.md` — how evidence becomes verdicts
3. Update `orchestration/README.md` + `DIAGRAM.md`
4. Create `verdict/README.md` + `DIAGRAM.md`

### Session 2: Security + Reproducibility
5. `03_security_prompt_injection.md`
6. `04_reproducibility_determinism.md`
7. Create `security/README.md` + `DIAGRAM.md`
8. Create `api/README.md` + `DIAGRAM.md`

### Session 3: Infrastructure
9. `05_mcp_servers.md`
10. `06_rag_hybrid_retrieval.md`
11. `07_gateway_production.md`
12. Update `mcp/README.md`, `rag/README.md`, `llm/README.md`

### Session 4: Evaluation + Config + Remaining
13. `08_eval_framework.md`
14. `09_formal_verification.md`
15. `10_config_decision_numbers.md`
16. Create `eval/README.md`, `config/README.md`
17. Update top-level `README.md` + `DIAGRAM.md`

---

## Principles for the Learning Documents

1. **Source code is truth** — every statement verified against actual `.py` files. No guessing.
2. **Teach the "why"** — not just "what the code does" but "why it was designed this way." What alternatives were rejected? What tradeoffs were made?
3. **Real code snippets** — include actual function signatures, key code blocks from the source. Not pseudocode.
4. **Connect to Ali's learning goals** — Ali is targeting Senior AI/ML Engineer and Hybrid AI/Blockchain Engineer roles. Frame the lessons in terms of:
   - System design patterns (fail-soft, defense-in-depth, dual-verdict)
   - ML engineering (reliability matrices, deterministic inference, model hashing)
   - Blockchain engineering (ZK boundary, on-chain anchoring, MCP architecture)
5. **Mistakes and fixes** — document what went wrong (Aderyn silent-skip, RAG zero-match, cascade over-prediction) and how it was fixed. These are the most valuable lessons.

---

## Estimated Effort

| Part | Files | Effort |
|------|-------|--------|
| READMEs | 11 | ~2 hours |
| Diagrams | 5 | ~1 hour |
| Learning docs | 10 | ~8-10 hours (1 hour each) |
| **Total** | **26** | **~11-13 hours** |

Split across 4 sessions, ~3 hours per session.
