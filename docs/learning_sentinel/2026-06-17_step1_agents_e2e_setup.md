# Learning Sentinel — 2026-06-17 — Agents E2E Setup

> DEEP DIVE mode (trigger: "teach me"). Walkthrough of the setup phase
> of `docs/plan/agents/2026-06-17-agents-real-e2e-test/`.

---

## 0. Big Picture (3 layers)

This run has **3 layers** that are easy to confuse:

| Layer | What | Why it matters |
|---|---|---|
| **1. Plan docs** | 5 markdown files in `docs/plan/agents/2026-06-17-agents-real-e2e-test/` | Source of truth for WHAT we'll do |
| **2. Source code** | `.py` files in `agents/src/` and `ml/src/` | Source of truth for HOW it actually works |
| **3. Running services** | Processes bound to ports :8001, :8010-13, :1256 | Source of truth for "is it really up?" |

**Rule of thumb (per CLAUDE.md):** layer 2 trumps layer 1. The plan said the 4 MCP servers
were "all confirmed" — but `inference_server.py:463` and `audit_server.py:688` were
written for an old Starlette API and crashed at startup. The plan was right that the
**ports are in code**; the plan was wrong that the **servers start cleanly**.

→ You now know: when you read a plan, treat "confirmed" as "I read this in the source",
not "I verified this at runtime". Run a `curl` after every code change.

---

## 1. Environment Audit — what we checked and why

We probed 5 components before touching any code. Here's the order and rationale:

### 1.1 LM Studio (port 1256)
```bash
timeout 3 curl -s -o /dev/null -w "%{http_code}\n" http://localhost:1256/v1/models
```

**Why first:** LM Studio is the slowest moving part. If it's not running, we can't run
the plan at all — the LLM calls in `cross_validator` need it. Probe before doing
anything else so we know the gate.

**What we found:** Running on `:1256` (NOT `:1234` and NOT `:4567`). The port is
**fluid** — LM Studio's desktop app picks a free port automatically to avoid collisions.

**What this means for the plan:** The plan's hardcoded `http://localhost:1234/v1` is
wrong. Real code reads from env var `LM_STUDIO_BASE_URL`. The .env had `4567` (also wrong)
plus a stale WSL2 gateway IP.

### 1.2 ML API (port 8001)
```bash
source ml/.venv/bin/activate
uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8001
```

**Why:** Run 12 checkpoint is the brain. Without the API, `ml_assessment` and
`graph_explain` (via `inference_server`) have nothing to call.

**What we found:** Cold start takes ~6 seconds (4s checkpoint load + ~2s warmup). The
warmup forward pass exercises a 3-node graph with `FUNCTION:` prefix path — the actual
production code path. The 8GB VRAM RTX 3070 handles it without OOM.

**Verification:** `curl /predict` returned a 10-class probability distribution for a
trivial contract. CONFIRMED the API is alive.

### 1.3 WSL2 Gateway IP — the silent killer
```bash
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# Returns: 10.255.255.254
```

**Why this matters:** LM Studio runs as a Windows app (not WSL). The agents code in
WSL reaches it via the WSL2 gateway IP. This IP **changes on every Windows reboot**.

**The .env had:** `LM_STUDIO_BASE_URL=http://172.19.48.1:4567/v1` — a frozen-in-time
gateway IP. Connect refused silently. The 172.x address was valid 6 months ago; today
the gateway is 10.255.255.254.

**Fix:** One-line edit. From now on, anyone running this needs to verify the gateway
IP first.

→ You now know: WSL2's `resolv.conf` gateway IP is dynamic. Hardcoding it in `.env`
creates a 6-month time bomb. Always `cat /etc/resolv.conf` before debugging "LM Studio
not responding".

### 1.4 Tool versions
```bash
poetry run slither --version    # 0.11.5 ✓
poetry run aderyn --version     # NOT installed ✗
```

**Why version-check:** Run 12's SmartBugs Wild eval (47K contracts) found 6,782 errors
that were all from pre-0.4.21 Slither (not model bugs). If we run E2E with a broken
Slither, we'll chase phantom bugs for hours.

**Aderyn not installed:** Reading the code (`nodes.py:265, 613`), the static analysis
node wraps aderyn calls in try/except and **skips silently** if not installed. So this
is non-fatal — we just lose aderyn's findings, slither still works.

### 1.5 RAG index
```bash
ls -lh agents/data/index/faiss.index
# 2.3M
```

**Why check:** RAG server crashes at startup if the index files are missing. 2.3M is
small but matches the "752 chunks" log line we saw later from the RAG server — DeFiHackLabs
corpus is intentionally small (focused on real exploits, not breadth).

---

## 2. The Critical Bug We Hit

### 2.1 What failed
```python
# agents/src/mcp/servers/inference_server.py:462 (BEFORE fix)
starlette_app = Starlette(
    on_startup=[_on_startup],    # creates shared HTTP client
    on_shutdown=[_on_shutdown],  # closes it cleanly on Ctrl+C / SIGTERM
    routes=[...]
)
```

**Error at startup:**
```
TypeError: Starlette.__init__() got an unexpected keyword argument 'on_startup'
```

### 2.2 Why it failed
Starlette **1.0.0** (released 2025) **removed** `on_startup` and `on_shutdown` as
constructor kwargs. The replacement is the `lifespan` context manager:

```python
# Old (pre-1.0)
Starlette(on_startup=[fn], on_shutdown=[fn], routes=...)

# New (1.0+)
@asynccontextmanager
async def lifespan(app):
    await setup()
    yield
    await teardown()

Starlette(lifespan=lifespan, routes=...)
```

The MCP server code was written for Starlette 0.x. When the venv got Starlette 1.0.0
installed, every startup crashed.

**Two of four servers were affected:** `inference_server.py` and `audit_server.py`.
`rag_server.py` and `graph_inspector_server.py` happened to NOT use `on_startup` —
`rag_server.py:307` calls `_on_startup()` directly before `uvicorn.run()`, and
`graph_inspector_server.py` doesn't need setup at all.

### 2.3 Why the plan v1.1 missed it
The audit checked the **port number** in each file (`.py` line where `int = 8010`)
and called it "confirmed". It did NOT run `python server.py` to verify the server
actually starts. **A static audit of port numbers is not a startup test.**

### 2.4 The fix
```python
# AFTER (in both files)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    await _on_startup()
    try:
        yield
    finally:
        await _on_shutdown()

starlette_app = Starlette(
    lifespan=lifespan,
    routes=[...]
)
```

This is the standard Starlette 1.0+ pattern. The try/finally guarantees `_on_shutdown`
runs on Ctrl+C / SIGTERM (the whole reason for using a lifespan vs calling directly).

**Verification:** All 4 servers now respond to `/health` with status `ok`.

→ You now know: the lifespan pattern is the modern Starlette way. If you write a new
MCP server, use lifespan from the start. If you see `on_startup` in old code, the file
predates Starlette 1.0 and needs a migration.

---

## 3. Background Process Trap (Second Issue)

After fixing the Starlette bug, the servers still wouldn't start in the background:

```
INFO:     Application startup complete.
ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8010)
INFO:     Waiting for application shutdown.
```

`ss -tlnp` showed no process on `:8010`. The port was genuinely free. But uvicorn
thought it was in use.

### 3.1 Root cause
When you background a process with `&` + `nohup` + `disown` from a shell, the process
**inherits the shell's file descriptors** (including stdin). When the shell exits
(because the bash tool's command "completes"), the kernel sends **SIGHUP** to the
process group. uvicorn catches SIGHUP, begins graceful shutdown, releases the bind,
then logs the EADDRINUSE because another restart attempt is already in the pipeline.

**The fix:** `setsid` creates a new session and process group, fully detaching from
the controlling shell:

```bash
setsid poetry run python src/mcp/servers/inference_server.py \
  > /tmp/opencode/mcp_inference.log 2>&1 < /dev/null &
disown
```

Two key changes:
- `setsid` — new session, immune to SIGHUP from parent
- `< /dev/null` — explicit stdin redirection so the kernel doesn't close it later

**Foreground worked perfectly** (`timeout 5 poetry run python server.py`) — confirming
this is a backgrounding issue, not a server issue.

→ You now know: when starting a long-lived server in a background shell tool, use
`setsid` + redirect all three streams (`>log 2>&1 < /dev/null`). `nohup` + `disown`
isn't enough on modern kernels.

---

## 4. State at End of Setup Phase

| Service | Port | Status | Notes |
|---|---|---|---|
| LM Studio | 1256 | ✓ UP | All 4 required models loaded |
| ML API | 8001 | ✓ UP | Run 12 FINAL, /predict verified |
| inference_server | 8010 | ✓ UP | Wraps ML API |
| rag_server | 8011 | ✓ UP | 752 chunks indexed |
| audit_server | 8012 | ✓ UP | MOCK MODE (no real Sepolia) |
| graph_inspector_server | 8013 | ✓ UP | Phase 2 (GNN attention) |

**Changes made to code:**
1. `agents/src/mcp/servers/inference_server.py` — `on_startup` → `lifespan`
2. `agents/src/mcp/servers/audit_server.py` — `on_startup` → `lifespan`

**Changes made to config:**
1. `agents/.env` — `LM_STUDIO_BASE_URL` corrected (gateway IP + port)

**Net effect:** 0 known critical bugs remain. Real E2E can now proceed.

---

## 5. What's next

Phase 2 (Execution): extract test contracts from `tests/test_smoke_e2e.py:44-79`, build
a `run_real_audit.py` harness, run the graph on 2 contracts, capture JSON reports.

The contracts are 14-line Solidity snippets (Vault with reentrancy + SafeStorage baseline).
This is the same input that the mocked smoke test uses, so any verdict difference between
mocked and real is attributable to real services, not input drift.
