# 05. MCP Architecture: 5 Servers, SSE Transport, Fail-Soft

> **Prerequisites:** [01. The Audit Pipeline] — nodes call external tools via `_call_mcp_tool()`. [02. Evidence Model & Fuse()] — each tool emits Evidence with a `source` field; `tool_status` tracks whether each tool ran (Rule 5C).
> **Next:** [06. Reliability Calibration] covers how the per-(source, class) reliability values are fitted from eval data — these are the weights the MCP-served tools carry into `fuse()`.
> **Cross-ref:** [07. Gateway Production] covers the health-monitoring system that probes these 5 servers every 30s.
> **Scope:** This doc covers the MCP (Model Context Protocol) server architecture, the SSE transport pattern, the `_call_mcp_tool()` helper, lazy loading, and the fail-soft contract. It does NOT cover the tools themselves (ML model in `ml/`, Slither/Aderyn as subprocesses) or the reliability values (see Doc 06).
> **TL;DR:** SENTINEL isolates 5 heavy/external tools behind MCP (Model Context Protocol) servers, each on its own port (8010-8014). Each server wraps a tool behind an SSE transport boundary: the pipeline node calls `_call_mcp_tool(server_url, tool_name, arguments)`, the server proxies to the tool, and returns JSON. If a server crashes, the pipeline node catches the error, writes `tool_status["tool"] = {"ran": False, "reason": ...}`, and continues — the pipeline always produces a report (fail-soft). The `audit_server.py` god-file (717 lines) was split into a 6-file `audit/` package (P2.5). Heavy state (FAISS index, model checkpoint) is loaded lazily at server startup, not at import time — so CI and tests can import server modules without crashing.

---

## The Problem: You Can't Put Everything in One Process

### What needs isolation

SENTINEL's pipeline calls 5 external tools, each with different runtime characteristics:

| Tool | What it needs | Why it can't be in-process |
|------|-------------|--------------------------|
| ML model (GNN + CodeBERT) | GPU (RTX 3070, 8GB VRAM) | GPU crash takes down the process |
| RAG retriever (FAISS + BM25) | ~500MB FAISS index in memory | Index load takes 3s — can't do per-request |
| Slither | Subprocess (`subprocess.run`) | Slither crash → zombie process → resource leak |
| Aderyn | Subprocess (Rust binary) | Same as Slither — external binary, crash-prone |
| Audit registry | HTTP to blockchain node | Network failure → pipeline hangs |

**Teaching: why not just import everything?** If the ML model, FAISS index, and Slither were all in one Python process:
1. A CUDA out-of-memory error in ML inference would crash the entire pipeline — including Slither and the synthesizer, which don't need the GPU.
2. A Slither segfault would kill the process — including the ML model, which has a 30s warmup cost.
3. You can't restart just one tool. If Slither starts producing bad output, you restart the whole process — including reloading the 500MB FAISS index and warming up the ML model.

**The reasoning:** when tools have *different* failure modes (GPU crash, subprocess segfault, network timeout), they need *different* restart boundaries. One process = one restart boundary. Five processes = five restart boundaries. If Slither crashes, you restart only the Slither server — the ML model keeps its warmup, the FAISS index stays loaded.

### Why MCP, not just HTTP?

You could expose each tool as a simple REST endpoint (`POST /predict`, `POST /search`). Why use MCP (Model Context Protocol)?

**Teaching: the protocol choice reasoning.** The question isn't "REST or MCP?" — it's "what does the protocol give you that raw HTTP doesn't?"

| Need | Raw HTTP | MCP |
|------|----------|-----|
| Tool discovery (what tools exist?) | Hardcoded in the client | `list_tools()` — server declares its own tools |
| Schema validation (what arguments?) | Manual JSON schema in client | `inputSchema` in the tool definition — SDK validates |
| Error handling (what went wrong?) | HTTP status codes (coarse) | JSON-RPC error codes + text content |
| Multiple clients | Each client hardcodes the API | Any MCP client (Claude Desktop, Cursor, LangGraph) works |

**The reasoning:** MCP gives us tool discovery and schema validation for free. The server declares its tools (`list_tools()`), the client discovers them at connection time, and the SDK validates arguments before the handler runs. With raw HTTP, we'd hardcode the tool list in every client, validate arguments manually, and update both sides when the API changes. MCP trades a small protocol overhead (the SSE handshake + JSON-RPC envelope) for automatic discovery and validation.

**When raw HTTP would be better:** if you have exactly one client and exactly one tool, and the API never changes. Then MCP's discovery and validation are overhead with no benefit. But SENTINEL has 5 tools and multiple clients (the pipeline, tests, and potentially Claude Desktop for debugging) — the discovery and validation are worth the cost.

---

## How We Arrived at This Design

> **How to read this section:** Each step shows the question, *how to reason about it*, and the chain of logic connecting the answer to the design.

### Step 1 — Identify the invariant (the "must always be true" test)

**The question:** What must always be true about tool calls, even if the tool crashes?

**Applying the "useless or dangerous" test:**

| Candidate property | If violated → | Verdict |
|---|---|---|
| Pipeline produces a report even if a tool crashes | On-chain consumer gets nothing → oracle useless | **Invariant** (from Principle 6, Doc 01) |
| Tool failures are visible (not silent) | Silent failure → biased reliability matrix → wrong verdicts | **Invariant** (Rule 5C) |
| Each tool can be restarted independently | One crash → full pipeline restart → 30s warmup wasted | Preference (important but not dangerous) |
| Tool calls complete within timeout | Pipeline hangs → no report → oracle useless | **Invariant** (timeout enforced) |

**The reasoning chain:** The pipeline must produce a report (Principle 6). This means: a tool crash must not crash the pipeline. This forces: each tool call is wrapped in a try/except that returns a structured failure (Rule 5C: `tool_status["ran"] = False`), not a crash. And the failure must be *visible* — `tool_status` is the contract that says "this tool didn't run" vs "this tool ran and found nothing." Silent failures (`return []` on crash) are forbidden — they're indistinguishable from "ran clean" and they poison the reliability matrix.

### Step 2 — Identify the constraints (what forces a specific shape)

**Constraint A: GPU-bound tools must be isolated from CPU-bound tools.**
- *Why:* The ML model runs on the GPU (RTX 3070, 8GB VRAM). Slither, Aderyn, and the synthesizer run on CPU. If they share a process, a CUDA out-of-memory error kills everything — including the CPU-bound work that doesn't need the GPU.
- *What this forces:* The ML model gets its own server (port 8010). CPU-bound tools get their own servers. A GPU crash only affects the ML server — the rest of the pipeline continues.

**Constraint B: Heavy state must be loaded once, not per-request.**
- *Why:* The FAISS index is ~500MB. Loading it per-request would add 3s latency to every RAG call. The ML model checkpoint is ~200MB — loading it per-request would add 5s.
- *What this forces:* State is loaded at server startup (`_on_startup()`), not per-request. The server process holds the state in memory and reuses it across requests. This is why the server is a long-running process, not a serverless function.

**Constraint C: The pipeline must not hang on a tool that's slow or stuck.**
- *Why:* A Slither hang on a complex contract (infinite loop in the AST parser) would block the pipeline forever — no report is produced.
- *What this forces:* Every tool call has a timeout (`DEFAULT_MODULE1_INFERENCE_TIMEOUT_S`, `SLITHER_TIMEOUT_S`, etc. — all in `timeouts.py`). `_call_mcp_tool()` doesn't enforce the timeout itself — the MCP SDK does, via the SSE connection's read timeout. But the *server* enforces it on the tool side (e.g., Slither subprocess timeout, HTTP client timeout for ML API calls).

### Step 3 — Eliminate alternatives (find what breaks under *current* conditions)

**The three approaches for tool isolation:**

| Approach | How it breaks | When it breaks | Eliminate? |
|---|---|---|---|
| **Direct imports** (no service boundary) | GPU crash kills pipeline. Can't restart one tool independently. No timeout isolation. | Always — these are current conditions. | **Yes** |
| **Monolithic server** (one server, all tools) | One crash kills all tools. Can't restart one tool without restarting all. FAISS index reloads when Slither crashes. | When any tool crashes. | **Yes** |
| **One server per tool** (5 MCP servers) | 5 processes to manage. 5 ports to configure. SSE connection overhead per call (~5ms). | When you have 50+ servers (connection management). | **No** — survives at current scale. |

**The reasoning:** Direct imports fail on the invariant (pipeline must survive crashes) — a GPU crash or Slither segfault kills everything. A monolithic server improves isolation (one crash doesn't kill the pipeline) but still can't restart one tool independently — restarting the monolith reloads the FAISS index and ML model. One server per tool is the only approach where restarting Slither doesn't reload the FAISS index. Its cost (5 processes, 5 ports) is manageable at current scale.

**Steel-manning direct imports:** "Just import Slither and the ML model directly into the pipeline. It's simpler — no network, no serialization, no MCP overhead. The pipeline is one process, one deployment."

**Why it fails:** A CUDA OOM during ML inference kills the process. The synthesizer — which doesn't need the GPU — dies too. No report is produced. The invariant is violated. The simplicity of direct imports is paid for with fragility: one tool's crash takes down everything. For a security oracle where "no report" means "on-chain consumer gets nothing," this is unacceptable.

### Step 4 — Stress-test against future growth

**The test:** "What happens when we add Halmos (P8a) and Gigahorse (P8b)?"

**Tracing through the design:**
- Halmos: runs as a subprocess (Foundry + `halmos` binary). It's in the same category as Slither/Aderyn — external binary, crash-prone. **But it's NOT behind an MCP server** — it's called directly from the `formal_verification` node via subprocess. Why? Because Halmos is called once per audit (not per-class), and its output is parsed directly into Evidence. The MCP overhead (SSE handshake + JSON-RPC envelope) is not justified for a single subprocess call per audit.

**This is an important nuance:** not every tool is behind MCP. The decision is per-tool:
- Tools called multiple times per audit (ML inference, RAG search) → MCP server (amortize connection cost)
- Tools called once per audit via subprocess (Slither, Aderyn, Halmos) → direct subprocess call

**Wait — Slither and Aderyn are called directly too?** Yes. Slither and Aderyn are called as subprocesses from `static_analysis.py`, not through MCP servers. The MCP servers are for: (1) ML API proxy (port 8010), (2) RAG search (port 8011), (3) audit registry (port 8012), (4) graph inspector (port 8013), (5) representation server (port 8014). Slither and Aderyn are subprocess calls, not MCP tools.

**This means the "5 MCP servers" in the TL;DR is the count of MCP servers, not the count of external tools.** The external tools are: ML model (via MCP), RAG (via MCP), Slither (direct subprocess), Aderyn (direct subprocess), Halmos (direct subprocess), graph inspector (via MCP), representation (via MCP), audit registry (via MCP). The MCP boundary is for tools that need persistent state (ML checkpoint, FAISS index) or HTTP proxying (audit registry → blockchain node). Subprocess tools (Slither, Aderyn, Halmos) don't need MCP — they're stateless processes.

**The reasoning principle:** "Use MCP servers for tools with persistent state or HTTP-proxy needs. Use direct subprocess calls for stateless tools. The cost of MCP (SSE handshake, JSON-RPC, serialization) is justified when you amortize it across many calls or need state persistence — not for a single subprocess invocation per audit."

### Step 5 — Measure (latency and failure handling)

**The question:** What's the overhead of MCP vs direct call?

**Measured characteristics:**
- MCP SSE handshake: ~5ms (connect to `/sse`, receive session ID)
- JSON-RPC call + response: ~2ms (serialize args, send, deserialize response)
- Total MCP overhead per call: ~7ms
- Direct subprocess call (Slither): ~50ms (fork + exec + parse JSON output)

**The reasoning:** for the ML API (called once per audit, 500ms inference time), 7ms of MCP overhead is 1.4% — negligible. For RAG (called once per audit, 100ms retrieval), 7ms is 7% — acceptable. For Slither (called once per audit, 5s analysis), a 7ms MCP overhead would be negligible — but Slither is a subprocess anyway, so MCP is not used. The overhead is acceptable for the tools that need it (persistent state, HTTP proxy).

> **The method, summarized:** (1) Find invariants — pipeline must survive crashes, failures must be visible (Rule 5C). (2) Find constraints — GPU isolation, heavy state, timeout enforcement. (3) Eliminate alternatives by finding *current* failure conditions — direct imports fail on crashes, monolith fails on independent restart. (4) Choose per-tool, not globally — MCP for persistent-state tools, subprocess for stateless tools. (5) Measure overhead — 7ms MCP overhead is negligible for 100ms+ tool calls.

---

## The Solution

### The 5-server architecture

```
Pipeline Nodes                    MCP Servers (SSE)           External Tools / State
─────────────                    ──────────────────           ─────────────────────
ml_assessment ──→ inference_server (8010) ──→ ML API (8001, GPU, checkpoint)
rag_research  ──→ rag_server (8011)        ──→ FAISS index + BM25 (500MB in memory)
audit_check   ──→ audit_server (8012)       ──→ AuditRegistry.sol (blockchain node)
graph_explain ──→ graph_inspector (8013)    ──→ Graph analysis (AST → hotspot scores)
(all nodes)    ──→ representation (8014)    ──→ Representation server (AST → PyG graph)

Direct subprocess (NOT MCP):
quick_screen     ──→ slither --detect ... (subprocess)
static_analysis  ──→ slither + aderyn (subprocess)
formal_verif     ──→ halmos --json-output (subprocess)
```

### The SSE transport pattern

Each MCP server exposes two HTTP endpoints:

```
/sse          → persistent event stream (Server-Sent Events)
                Client connects here, receives a session ID, and listens for responses.

/messages/    → JSON-RPC POST endpoint
                Client sends tool calls here: {"method": "call_tool", "params": {...}}
                Server responds via the SSE stream: {"result": {...}}
```

**Teaching: why SSE, not WebSocket?** SSE is one-directional (server → client). WebSocket is bidirectional. MCP uses SSE for responses and POST for requests — the client sends requests via POST, the server pushes responses via SSE. This is simpler than WebSocket (no bidirectional frame negotiation, no connection upgrade handshake) and works through HTTP proxies (SSE is standard HTTP). The trade-off: SSE is one-connection-per-client, so concurrent clients each need their own SSE stream. For SENTINEL (one pipeline, sequential tool calls), this is fine.

### Worked example: tracing an ML prediction call

Let's trace what happens when `ml_assessment` needs an ML prediction:

**Step 1: Pipeline node calls `_call_mcp_tool()`:**
```python
result = await _h._call_mcp_tool(
    server_url="http://localhost:8010/sse",
    tool_name="predict",
    arguments={"contract_code": state["contract_code"]},
)
```

**Step 2: `_call_mcp_tool()` opens an SSE connection:**
```python
async with sse_client(server_url) as (read, write):      # connect to /sse
    async with ClientSession(read, write) as session:    # MCP handshake
        await session.initialize()                        # list_tools + capabilities
        result = await session.call_tool("predict", arguments)  # JSON-RPC call
```

**Step 3: The inference server receives the call:**
The server's `call_tool()` handler routes to `_handle_predict()`, which calls `_call_inference_api()`, which sends an HTTP POST to the ML API at `localhost:8001/predict`. The ML API runs inference and returns probabilities.

**Step 4: The response flows back:**
```
ML API → inference_server (HTTP response) → MCP SSE stream → _call_mcp_tool() → json.loads() → dict
```

**Step 5: The node processes the result:**
```python
return {
    "ml_result":   result,
    "model_hash":  result.get("model_hash", ""),        # P5 provenance
    "tool_status": {"ml": {"ran": True, "label": result.get("label", "?")}},
}
```

**If the ML API is down:** `_call_inference_api()` catches `httpx.RequestError` (connection refused), logs a warning, and returns `_mock_prediction()` — a realistic fake response. The pipeline continues with mock data. The `tool_status` still says `ran: True` because the MCP server responded (with mock data). This is a known limitation — see Limitations.

**Teaching: why the mock fallback exists.** In development (no GPU available), the ML API isn't running. Without the mock fallback, every pipeline call would fail with a connection error, and no development or testing could happen. The mock provides a realistic response (plausible probabilities, correct schema) so the pipeline can be tested end-to-end without a GPU. In production, `MODULE1_MOCK=false` is set, and the mock is never used. If the ML API crashes in production, the mock fallback is a *degraded mode* — the pipeline produces a report, but the ML evidence is fake. This is fail-soft: the report is produced, but it's marked as degraded (the `model_hash` is `"mock_model_hash_..."` so the provenance chain reveals it).

### The fail-soft contract (Rule 5C)

Every tool call in the pipeline follows this pattern:

```
┌──────────────────────────────────────────────────────────────────┐
│  Tool call flow (every node that calls an external tool):        │
│                                                                  │
│  try:                                                            │
│      result = await _call_mcp_tool(...) or subprocess.run(...)   │
│      return {                                                    │
│          "tool_status": {"tool": {"ran": True, ...}},            │
│          ...result...                                            │
│      }                                                           │
│  except (TimeoutError, ConnectionError, FileNotFoundError):     │
│      return {                                                    │
│          "tool_status": {"tool": {"ran": False, "reason": ...}}, │
│          ...empty defaults...                                    │
│      }                                                           │
│                                                                  │
│  NEVER: return [] (silent — indistinguishable from "ran clean")  │
│  ALWAYS: return tool_status with ran: True/False                │
└──────────────────────────────────────────────────────────────────┘
```

**The reasoning:** `[]` (empty list) is ambiguous — it could mean "tool ran and found nothing" (correct) or "tool crashed and returned nothing" (Rule 5C violation). The `tool_status` field disambiguates: `ran: False` means the tool didn't run (crash, timeout, not installed); `ran: True` with empty results means the tool ran and found nothing. Downstream code (synthesizer, eval) checks `ran` before interpreting results.

### The lazy loading pattern

Heavy state is loaded at server startup, not at import time:

```python
# rag_server.py:97-111
_retriever: HybridRetriever | None = None    # module-level, starts as None

def _on_startup() -> None:                     # called from run_server(), NOT at import
    global _retriever
    _retriever = HybridRetriever()             # loads 500MB FAISS index + BM25 corpus
```

**Teaching: why not load at import time?** If `HybridRetriever()` were called at module level (e.g., `_retriever = HybridRetriever()` on line 97), then every `import rag_server` would load the FAISS index — including in CI (no index built), in unit tests (don't need the index), and in documentation generation (importing the module for introspection). The lazy pattern defers the load to `run_server()`, which is only called when the server actually starts. Importing the module is always safe — it just sets `_retriever = None`.

**The bug this fixed:** Before the lazy pattern, importing `rag_server` in CI (where the FAISS index wasn't built) crashed with `FileNotFoundError`. CI tests that imported the module to check its API failed — not because the code was wrong, but because the import triggered a 500MB file load. The fix: move the load to `_on_startup()`, keep the import side-effect-free.

### The `audit/` package split (P2.5)

The `audit_server.py` was a 717-line god-file with 5 responsibilities: server setup, tool handlers, AuditResult decoder, config, and lifecycle. It was split into a 6-file package:

```
agents/src/mcp/servers/audit/
├── __init__.py      (774 bytes)  — public API re-exports
├── _server.py       (3160 bytes) — SSE server, /health endpoint
├── _handlers.py     (13491 bytes)— tool call handlers (register_audit, get_audit, etc.)
├── _decode.py       (4384 bytes) — AuditResult decoder (bytes → dict)
├── _lifecycle.py    (4117 bytes) — startup/shutdown (web3 provider init)
└── _config.py       (3974 bytes) — config constants (RPC URL, contract address)
```

**The reasoning:** Each file has one responsibility (Rule A: Single Responsibility). `_handlers.py` handles tool calls; `_decode.py` decodes blockchain responses; `_lifecycle.py` manages startup/shutdown; `_config.py` holds configuration. Before the split, changing the decoder risked breaking the server setup. After the split, the decoder can be tested independently. The 6-file package is ~28KB total — each file is 3-13KB, well under the 500-line guideline (CLAUDE.md Rule 5A).

## Key Code

The `_call_mcp_tool()` helper — the single function all MCP-calling nodes use:

```python
# _helpers.py:44-62
async def _call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            raw = result.content[0].text
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"MCP tool '{tool_name}' returned non-JSON response: {raw[:200]}"
                ) from exc
```

Why this matters: every MCP-calling node uses this one function. No node reimplements SSE connection management, MCP handshake, or JSON parsing. If the MCP SDK changes (e.g., new transport), only this function needs updating — not 5 different node files. The function is 18 lines. It opens a connection, calls the tool, parses the response, and returns a dict. Errors (non-JSON response) raise `RuntimeError` — the caller catches it and writes `tool_status`.

The mock fallback — fail-soft for development:

```python
# inference_server.py:318
"model_hash": "mock_model_hash_" + "0" * 46,
```

Why this matters: when the ML API is unreachable (development, CI, GPU down), the inference server returns a mock prediction. The mock includes a `model_hash` that is visibly fake (`"mock_model_hash_..."`). This means: the provenance chain still works — the report shows the mock hash, and a human reviewer immediately sees "this verdict was produced with mock data, not a real model." The mock is fail-soft: the pipeline produces a report, but the report is honestly marked as degraded.

## Design Decision: MCP over SSE vs gRPC vs REST vs Direct Import

> **How to read this section:** The table shows the options. The *elimination reasoning* shows how to think about the choice.

### The elimination process

**Step 1: What are the options?** (a) MCP over SSE (chosen), (b) gRPC, (c) raw REST, (d) direct import (no service boundary).

**Step 2: Eliminate by finding failure conditions.**

**Direct import — steel-man first:** "Direct import is the simplest. No network, no serialization, no protocol overhead. The ML model is a Python object — just call `model.predict(code)`. Why add a server?"

**Why it fails:** A CUDA OOM kills the process — including the CPU-bound synthesizer. Slither's subprocess crash takes down the FAISS index. No independent restart. The invariant (pipeline must survive crashes) is violated. Direct import is only viable if *no tool ever crashes* — which is never true for external tools (Slither, Aderyn, Halmos are all crash-prone subprocesses).

**Raw REST — steel-man first:** "REST is universal. Every language has an HTTP client. No special SDK needed. Just `POST /predict` with JSON. Why use MCP?"

**Why it fails on two counts:**
1. *No tool discovery:* the client must hardcode the tool list and argument schemas. When the server adds a new tool, every client needs a code update. MCP's `list_tools()` makes the server self-describing.
2. *No schema validation:* the client must validate arguments manually. MCP's `inputSchema` (JSON Schema) is validated by the SDK before the handler runs — malformed requests never reach the handler.

**gRPC — steel-man first:** "gRPC is faster than SSE (binary protocol, connection pooling, bidirectional streaming). For high-throughput systems, gRPC is the standard. Why use SSE when gRPC is faster?"

**Why it fails for SENTINEL:**
1. *Overkill at current scale:* gRPC's advantage is throughput (millions of calls/sec) and bidirectional streaming. SENTINEL makes ~5 MCP calls per audit, ~100 audits/day. SSE's 7ms overhead × 5 calls × 100 audits = 3.5s/day. gRPC would save ~2s/day. Not worth the complexity.
2. *Code generation burden:* gRPC requires `.proto` files and generated stubs for every language. MCP is JSON-over-HTTP — any HTTP client works, no code generation.
3. *MCP ecosystem compatibility:* MCP is designed for agent-tool communication (Claude Desktop, Cursor, LangGraph). gRPC is not MCP-compatible. Using MCP means SENTINEL's tools are accessible from any MCP client, not just the pipeline.

**MCP over SSE — why it survives:** It provides tool discovery, schema validation, and MCP-ecosystem compatibility. Its overhead (7ms/call) is negligible. Its weakness (one-connection-per-client, no bidirectional streaming) doesn't matter at current scale (one pipeline, sequential calls). The migration to gRPC (or WebSocket) is a future option when scale demands it.

**The reasoning principle:** "When choosing a service protocol, eliminate options that break on *current* needs first. Direct import breaks on crash isolation. Raw REST breaks on tool discovery. gRPC doesn't break — but it's overkill. MCP over SSE is the minimum protocol that doesn't break on any current need. Choose the minimum, not the maximum."

### When this decision would be wrong

**The reversal condition:** If SENTINEL scales to 50+ concurrent audits (each making 5 MCP calls), SSE's one-connection-per-client model creates 250 concurrent SSE connections. At that point, switch to gRPC (connection pooling, multiplexed streams) or WebSocket (bidirectional, one connection per client). The trigger: when SSE connection management code becomes a maintenance burden (>50 lines of connection pooling logic).

## Technology Choice: MCP Protocol

**The 5-question framework:**

1. **What category?** Agent-to-tool communication protocol.
2. **What alternatives?** (a) MCP (Model Context Protocol), (b) gRPC, (c) raw REST, (d) direct import.
3. **Why MCP?** Tool discovery (`list_tools`), schema validation (`inputSchema`), MCP ecosystem compatibility (Claude Desktop, Cursor, LangGraph can all use SENTINEL's tools).
4. **When is gRPC better?** High-throughput (>1000 calls/sec), bidirectional streaming needs, polyglot microservices architecture.
5. **Migration trigger:** SSE connection count exceeds ~50 concurrent — switch to gRPC or WebSocket.

## Anti-Patterns

### ❌ Direct imports (no service boundary) — "simpler, no network"
**What it looks like:** `from ml.model import predict; result = predict(code)` — the ML model is imported directly into the pipeline process.
**Why someone would build this:** It's the fastest path to a working prototype. No server, no serialization, no protocol. One process, one deployment.
**Why it's wrong:**
1. *GPU crash kills pipeline* — CUDA OOM takes down the synthesizer (which doesn't need the GPU).
2. *No independent restart* — if Slither starts producing bad output, you restart the whole process (reloading the 200MB ML checkpoint + 500MB FAISS index).
3. *No timeout isolation* — a hung Slither subprocess blocks the entire event loop.
**The right approach:** MCP servers with SSE. Each tool is a separate process. A GPU crash only affects the ML server. Slither is a subprocess that can be killed and restarted independently. Timeouts are enforced at the process boundary.

### ❌ Monolithic server (one server for all tools) — "one process to manage"
**What it looks like:** One Starlette app that exposes all tools: `/predict`, `/search`, `/register_audit`, `/graph_hotspots`, `/represent`. One process, one port.
**Why someone would build this:** "One process is easier to deploy, monitor, and restart. One health check instead of five."
**Why it's wrong:**
1. *One crash kills everything* — if the FAISS index corrupts and the RAG handler crashes, the ML predict endpoint dies too. No isolation between tools.
2. *Can't restart one tool* — restarting the monolith reloads the 200MB ML checkpoint and 500MB FAISS index. A 10s restart for a Slither config change becomes a 30s restart.
3. *Mixed resource profiles* — the ML server needs GPU, the RAG server needs 500MB RAM, the audit server needs network. One process can't optimize for all three.
**The right approach:** One server per tool (or per resource profile). The ML server has GPU access; the RAG server has the FAISS index; the audit server has the blockchain connection. Each can be restarted, scaled, and monitored independently.

## Mistakes & Fixes

### Mistake: `audit_server.py` was a 717-line god-file
**What happened:** The audit registry server was a single 717-line file containing: server setup (Starlette routes), tool handlers (register_audit, get_audit, list_audits), AuditResult decoder (bytes → dict from blockchain), config constants (RPC URL, contract address), and lifecycle management (web3 provider initialization). All 5 responsibilities in one file.
**Why it happened:** The server grew organically — each new tool was added to the same file. There was no initial architecture decision to split; the file just grew with each feature.
**How we found it:** P2.5 long-file audit (CLAUDE.md Rule 5A). The file was 717 lines — past the 500-line guideline, and describing what it does required "and" (server setup *and* handlers *and* decoder *and* config *and* lifecycle).
**The fix:** Split into the `audit/` package (6 files): `_server.py` (SSE setup), `_handlers.py` (tool handlers), `_decode.py` (decoder), `_lifecycle.py` (startup/shutdown), `_config.py` (constants), `__init__.py` (re-exports). Each file has one responsibility. The decoder can now be tested independently. Changing config doesn't risk breaking handlers.
**The lesson:** When a file passes 500 lines and its description requires "and," split it. The split is mechanical (the functions already exist — just move them to focused files), and the test suite catches regressions. A 717-line file is not "complex" — it's "unstructured complexity." Splitting it doesn't reduce complexity; it *organizes* it so each piece is independently testable.

### Mistake: Importing `HybridRetriever()` at module level crashed in CI
**What happened:** `rag_server.py` had `_retriever = HybridRetriever()` at module level (line 97). In CI (where the FAISS index wasn't built), this crashed with `FileNotFoundError`. Every test that imported `rag_server` — even to check its API — failed.
**Why it happened:** The retriever was loaded at import time because "the server needs it." But import time and server startup time are different: import happens in tests, CI, and introspection; startup happens only when the server actually runs.
**How we found it:** CI failures on `import rag_server` — `FileNotFoundError: faiss_index.bin not found`.
**The fix:** Move the load to `_on_startup()`: `_retriever = None` at module level, `_retriever = HybridRetriever()` inside `_on_startup()`. The handler checks `if _retriever is None` and returns a structured error.
**The lesson:** Import should be side-effect-free. Heavy state (FAISS index, model checkpoint, database connection) should be loaded at startup, not at import. The test is: "can I import this module without any external files, network, or GPU?" If no, the import has side effects — move them to an explicit startup function.

### Mistake: Aderyn silent-skip (the origin of Rule 5C)
**What happened:** When Aderyn's binary wasn't found, `_run_aderyn_on_file()` caught `FileNotFoundError`, logged at `DEBUG` level, and returned `[]`. This was indistinguishable from "Aderyn ran and found no issues." The pipeline continued as if Aderyn had run clean. 83 contracts × 10 classes produced zero Aderyn signal — and the reliability matrix treated this as "Aderyn has 0% precision" instead of "Aderyn didn't run."
**Why it happened:** The exception handler was written defensively — "don't crash, return empty." But the return value (`[]`) was the same shape as a successful empty result. The caller had no way to distinguish "ran and found nothing" from "didn't run."
**How we found it:** P0 baseline macro_F1=0.1958 — surprisingly low. Investigation showed Aderyn's reliability was 0.0 across all classes. Root cause: the binary wasn't being found (PATH issue), so every Aderyn call returned `[]`, and the reliability matrix saw 0 true positives and 0 false positives — consistent with "Aderyn never ran."
**The fix:** Rule 5C (CLAUDE.md §C). `_resolve_aderyn_binary()` raises `FileNotFoundError` with a precise message. Callers catch it and write `tool_status["aderyn"] = {"ran": False, "reason": "binary not found"}`. The synthesizer and eval layer check `ran` before interpreting results. An empty return is no longer "tool absent"; `ran=False` is.
**The lesson:** A silent failure manufactures a rabbit hole. The Aderyn bug cost days of investigation because the symptom (low macro_F1) pointed at the ML model, not at Aderyn. The root cause (missing binary) was invisible because the error was swallowed. Rule 5C: never return a value that's indistinguishable from success. Always carry the failure — `ran: False` in `tool_status`, or raise.

## What Would Break If You Removed This?

**Remove the MCP boundary:** GPU crash kills the pipeline. No independent restart. The invariant (always produce a report) is violated.

**Remove `_call_mcp_tool()`:** each node reimplements SSE connection, MCP handshake, JSON parsing, and error handling. 5 copies of the same 18-line function, each with its own bugs. Timeout handling is inconsistent — one node hangs, another doesn't.

**Remove `tool_status` (Rule 5C):** silent failures return. Aderyn's missing binary looks like "Aderyn found no issues." The reliability matrix is biased. The eval reports wrong numbers. Decisions are made on bad data.

**Remove lazy loading:** CI crashes on import. Every test that imports a server module triggers a 500MB FAISS index load or a 200MB model checkpoint load. Test suite takes 10 minutes instead of 30 seconds.

**Remove the `audit/` package split (go back to 717-line file):** changing the decoder risks breaking server setup. The file is 717 lines — reading it to find one function takes minutes. The test surface is the whole file, not focused modules.

## At Scale

*Scale metric: number of concurrent audits (current: 1-5; each makes ~5 MCP calls).*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| 1-5 concurrent (current) | 5 SSE connections, <1ms overhead | — | — |
| 20 concurrent | 100 SSE connections | Connection management becomes noticeable | Connection pooling in `_call_mcp_tool()` |
| 50 concurrent | 250 SSE connections | SSE overhead visible; one-connection-per-client is wasteful | Switch to WebSocket or gRPC |
| 100+ concurrent | 500+ SSE connections | SSE is the bottleneck | gRPC with connection pooling + multiplexing |

The MCP/SSE architecture scales to ~50 concurrent audits before SSE's one-connection-per-client model becomes a bottleneck. At that point, the migration is: (a) switch `_call_mcp_tool()` to use a connection pool (reuse SSE connections), or (b) switch to gRPC (multiplexed streams, one connection per client). Both are changes to `_call_mcp_tool()` — the server side doesn't change. This is the benefit of having a single helper function: the protocol migration is one file, not 5.

## Try It Yourself

> TRY IT: `for port in 8010 8011 8012 8013 8014; do echo -n "Port $port: "; curl -s --max-time 2 localhost:$port/health | python3 -m json.tool 2>/dev/null || echo "not running"; done` — health-check all 5 MCP servers.

> TRY IT: `cd agents && python -c "import asyncio; from src.orchestration.nodes._helpers import _call_mcp_tool; print('helper loaded — would connect to MCP server on call')"` — verify the helper imports without crashing (no side effects).

> TRY IT: `cd agents && ls -la src/mcp/servers/audit/` — see the 6-file package split (was 717-line single file).

## Limitations & What's Missing

- **Mock fallback hides failures in development.** When `MODULE1_MOCK=true`, the inference server returns fake predictions. The `tool_status` says `ran: True` because the MCP server responded — but the response is mock data, not real inference. In production, `MODULE1_MOCK=false` prevents this. But if the ML API crashes in production and the mock fallback activates, the `model_hash` is `"mock_model_hash_..."` — visibly fake. The provenance chain reveals it, but the `tool_status` doesn't flag it. A stricter design would set `tool_status["ml"] = {"ran": True, "mock": True}` when mock fallback is used.

- **Sequential health probes.** The gateway's health monitor probes 5 servers sequentially (each with a 1.5s timeout). If 3 servers are down, the health check takes 4.5s instead of 1.5s. A concurrent probe (`asyncio.gather`) would fix this but adds complexity.

- **No connection pooling.** `_call_mcp_tool()` opens a new SSE connection for every call. For 5 calls per audit, this is 5 × 5ms = 25ms of handshake overhead. Connection pooling (reuse the SSE connection across calls) would save this, but SSE connections are stateful (one session per connection) and pooling requires session management.

- **`tool_status` is in state but not in the MCP protocol.** The MCP server doesn't know about `tool_status` — it's set by the pipeline node, not the server. If a node forgets to set `tool_status`, the failure is silent. The contract is enforced by code review and tests, not by the protocol.

- **Single-process per server.** Each MCP server is one uvicorn process. If the server's event loop blocks (e.g., a synchronous FAISS search), all clients wait. At higher concurrency, multiple workers (uvicorn `--workers 4`) would help — but the heavy state (FAISS index, model checkpoint) would need to be shared or replicated.

## Transferable Patterns

1. **Service isolation — one crash doesn't cascade** — 5 MCP servers, one per resource profile.
   - *Interview story:* "SENTINEL has 5 external tools with different failure modes — GPU crash (ML), subprocess segfault (Slither), network timeout (blockchain node). We isolated each behind an MCP server. When the ML API crashes, the pipeline continues with Slither + Aderyn + Halmos evidence. The synthesizer checks `tool_status['ml']['ran']` and produces a partial report. The on-chain consumer still gets a verdict — it just knows the ML tier was unavailable."
   - *When this pattern is WRONG:* when the tools are lightweight (no heavy state, no GPU, no network) and crash rarely. Then MCP's overhead (7ms/call + 5 processes to manage) is not justified. Direct import is simpler. Use service isolation when tools have *different* failure modes or *different* resource profiles — not just because "microservices are best practice."

2. **Fail-soft with status flags — `tool_status["ran"] = False`, not silent `[]`** — Rule 5C compliance.
   - *Interview story:* "SENTINEL's Aderyn integration had a silent-skip bug: when the binary wasn't found, the handler returned `[]` — the same shape as 'ran and found nothing.' 83 contracts produced zero Aderyn signal, and the reliability matrix treated this as 'Aderyn has 0% precision.' The root cause was invisible. We fixed it with Rule 5C: every tool failure writes `tool_status['aderyn'] = {'ran': False, 'reason': 'binary not found'}`. An empty return is no longer 'tool absent'; `ran=False` is. The eval layer checks `ran` before computing metrics."
   - *When this pattern is WRONG:* when the status flag is never checked downstream. If `tool_status` is written but the synthesizer doesn't read it, the flag is theater — the failure is still silent in practice. The contract requires both: the caller writes the flag, AND the consumer checks it. Test both sides.

3. **Lazy loading for heavy state — defer to startup, not import** — FAISS index, model checkpoint.
   - *Interview story:* "SENTINEL's RAG server loads a 500MB FAISS index. Initially, it loaded at import time — every `import rag_server` in CI crashed with `FileNotFoundError` because the index wasn't built in CI. The fix: `_retriever = None` at module level, `_retriever = HybridRetriever()` inside `_on_startup()`. Now import is side-effect-free; the index loads only when the server actually starts. CI tests can import the module to check its API without loading 500MB of data."
   - *When this pattern is WRONG:* when the state is lightweight (a config dict, a small lookup table). Lazy loading adds a `_on_startup()` function, a `None` check in the handler, and a startup error path. For a 1KB config, this complexity is not justified — just load at import time. Use lazy loading when the state takes >100ms to load or requires external files.

---

**Source files verified:**
- `agents/src/orchestration/nodes/_helpers.py:44-62` — `_call_mcp_tool()` SSE client pattern
- `agents/src/mcp/servers/inference_server.py:48-62, 155-172, 197-242, 318, 427-499` — config, startup/shutdown, HTTP bridge, mock fallback, SSE server
- `agents/src/mcp/servers/rag_server.py:70-111, 289-349` — lazy loading, SSE server
- `agents/src/mcp/servers/audit/` — 6-file package (verified: `__init__.py`, `_config.py`, `_decode.py`, `_handlers.py`, `_lifecycle.py`, `_server.py`)
- `agents/src/orchestration/nodes/_helpers.py:78-91` — `_resolve_aderyn_binary()` (Rule 5C origin)

**Verified against commit hash:** `c47898ea5`
