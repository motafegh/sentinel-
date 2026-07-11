# Plan: Doc 05 — MCP Architecture: 5 Servers, SSE Transport, Fail-Soft

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/05_mcp_architecture.md`
**Session:** 3 of 5
**Prerequisite docs:** Doc 01 (Pipeline), Doc 02 (Evidence/Fuse)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that the pipeline has 14 nodes, some of which call external tools (Slither, Aderyn, ML model, RAG). The `_call_mcp_tool()` helper in `_helpers.py` is used by all nodes to communicate with MCP servers. The pipeline is fail-soft — if a tool is down, the node returns empty results.

**From Doc 02 (Evidence/Fuse):** You learned that each tool emits `Evidence` with a `source` field ("ml", "slither", "aderyn", "rag", "halmos"). The `tool_status` field in state tracks whether each tool actually ran (`{"ran": True/False, "reason": ...}`) — this is Rule 5C compliance.

**Connection to this doc:** This doc explains the MCP (Model Context Protocol) server architecture that sits between the pipeline nodes and the external tools. Each server isolates a heavy/external tool behind an HTTP/SSE boundary. This is why the pipeline is fail-soft — a crashed MCP server can't take down the pipeline.

**Key concepts carried forward:** `_call_mcp_tool()`, fail-soft principle, `tool_status` field (Rule 5C), `Evidence.source` field, the 5 server ports (8010-8014).

---

## Step 1: Read source files

- [ ] `agents/src/mcp/__init__.py`
- [ ] `agents/src/mcp/servers/__init__.py`
- [ ] `agents/src/mcp/servers/inference_server.py` — ML API proxy (port 8010), mock prediction, model_hash forwarding
- [ ] `agents/src/mcp/servers/rag_server.py` — RAG search (port 8011), HybridRetriever lazy loading
- [ ] `agents/src/mcp/servers/audit_server.py` — re-export shim (delegates to audit/ package)
- [ ] `agents/src/mcp/servers/audit/__init__.py` — audit package (re-exports)
- [ ] `agents/src/mcp/servers/audit/_server.py` — audit registry server (port 8012)
- [ ] `agents/src/mcp/servers/audit/_handlers.py` — tool handlers
- [ ] `agents/src/mcp/servers/audit/_decode.py` — AuditResult decoder
- [ ] `agents/src/mcp/servers/audit/_lifecycle.py` — startup/shutdown
- [ ] `agents/src/mcp/servers/audit/_config.py` — config
- [ ] `agents/src/mcp/servers/graph_inspector_server.py` — graph analysis (port 8013)
- [ ] `agents/src/mcp/servers/representation_server.py` — representation (port 8014)
- [ ] `agents/src/orchestration/nodes/_helpers.py` — `_call_mcp_tool()` function (the single helper all nodes use)

## Step 2: Read scratch files

- [ ] `~/.claude/scratch/system_finalization_statecheck_20260625.md` — Rule 5C findings on audit_server split (717-line god-file → 6-file package), the lazy loading fix

## Step 3: Read existing docs (for staleness comparison)

- [ ] `agents/src/mcp/README.md` — current (stale) README
- [ ] `agents/src/mcp/servers/README.md` — current (stale) servers README
- [ ] `agents/src/mcp/servers/DIAGRAM.md` — current (stale) diagram

## Step 4: Write sections

- [ ] **TL;DR:** 5 MCP servers (ports 8010-8014), each isolates a heavy/external tool behind SSE transport, `_call_mcp_tool()` handles timeout/retry/error, fail-soft via `tool_status` field (Rule 5C)
- [ ] **The Problem:** ML model on GPU, Slither/Aderyn as subprocesses, RAG with FAISS index — can't all be in one process. Need isolation so one crash doesn't kill the pipeline. Need a standard protocol for tool calls
- [ ] **How We Arrived at This Design:** invariant (pipeline always produces report) → constraint (isolate GPU-bound from CPU-bound, isolate crashable tools) → simplest boundary (MCP over SSE, one server per tool) → stress-test (add Halmos as direct Python, not MCP — different pattern) → measure (latency, failure rate)
- [ ] **The Solution:** 5-server architecture diagram:
  ```
  Pipeline Nodes                    MCP Servers (SSE)           External Tools
  ─────────────                    ──────────────────           ──────────────
  ml_assessment ──→ inference_server (8010) ──→ ML API (8001, GPU)
  rag_research  ──→ rag_server (8011)        ──→ FAISS + BM25
  audit_check   ──→ audit_server (8012)      ──→ AuditRegistry.sol
  graph_explain ──→ graph_inspector (8013)   ──→ Graph analysis
  (all nodes)    ──→ representation (8014)    ──→ Representation server
  ```
  SSE transport pattern: `/sse` (persistent event stream) + `/messages/` (JSON-RPC POST). `_call_mcp_tool()` helper. Fail-soft flow: timeout → empty results + `tool_status["ran": False]`
- [ ] **Key Code:**
  - `_call_mcp_tool()` (_helpers.py) — the single helper all nodes use: takes `server_url`, `tool_name`, `arguments`. Returns dict. Handles timeout, connection error, JSON-RPC error
  - Server lifespan pattern — `_on_startup()` lazy loads heavy state (FAISS index, model checkpoint), NOT at import time
  - `audit/` package split — 6 files: `_server.py`, `_handlers.py`, `_decode.py`, `_lifecycle.py`, `_config.py`, `__init__.py`
  - `inference_server.py` mock prediction — when ML API is down, returns mock result with `model_hash`
- [ ] **Design Decision:** MCP over SSE vs gRPC vs REST vs direct import (tradeoff table: isolation, latency, complexity, bidirectional needs, Docker compatibility)
- [ ] **Technology Choice:** MCP protocol (5-question framework: category, alternatives, why MCP, when gRPC is better, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ Direct imports (no service boundary) — "simpler, no network." Breaks: can't isolate GPU, can't scale independently, crashes cascade. Right: MCP servers with SSE
  - ❌ Monolithic server (one server for all tools) — "one process to manage." Breaks: one crash kills everything, can't restart one tool independently. Right: one server per tool
- [ ] **Mistakes & Fixes:**
  - `audit_server.py` was 717 lines — a god-file with server setup, handlers, decoder, config, lifecycle all in one. Fix: P2.5 long-file audit split into `audit/` package (6 files). Each file has single responsibility
  - Importing `HybridRetriever()` at module level crashed in CI (no FAISS index built). Fix: lazy `_on_startup()` loads index when server starts, not at import
  - Aderyn silent-skip: `FileNotFoundError` caught, logged at DEBUG, `[]` returned — indistinguishable from "ran clean." Fix: Rule 5C — `tool_status["aderyn"] = {"ran": False, "reason": "not_found"}`. This finding was the origin of Rule 5C
- [ ] **What Would Break Without This:** Remove MCP boundary → GPU crash kills pipeline. Remove `_call_mcp_tool()` → no timeout/retry/error handling, each node reimplements it badly. Remove `tool_status` → silent failures (Rule 5C violation). Remove lazy loading → CI crashes (no heavy state available)
- [ ] **At Scale:** 5 servers (current) / 15 / 50 / 100 — SSE connection management becomes complex at 50+. At 100+ need connection pooling or gRPC
- [ ] **Try It Yourself:**
  ```
  curl -s localhost:8010/health   # inference proxy
  curl -s localhost:8011/health   # RAG server
  curl -s localhost:8012/health   # audit server
  curl -s localhost:8013/health   # graph inspector
  curl -s localhost:8014/health   # representation
  ```
- [ ] **Limitations:** Sequential health probes (not concurrent — each takes up to 1.5s). No connection pooling. Single-process per server. SSE is one-directional (can't do bidirectional streaming). `tool_status` is in state but not in MCP protocol (nodes set it, not servers)
- [ ] **Transferable Patterns:** (1) Service isolation — one crash doesn't cascade (2) Fail-soft with status flags — `tool_status["ran": False]` not silent `[]` (3) Lazy loading for heavy state — defer index/model loading to startup, not import. Each with interview story + when wrong.

## Step 5: Verify

- [ ] Confirm 5 server ports: 8010 (inference), 8011 (RAG), 8012 (audit), 8013 (graph), 8014 (representation)
- [ ] Verify `audit/` package has 6 files (not a single `audit_server.py` anymore)
- [ ] Confirm `_call_mcp_tool` exists in `_helpers.py` and takes `server_url`, `tool_name`, `arguments`
- [ ] Verify lazy loading pattern (`_on_startup` not `__init__`) in at least one server
