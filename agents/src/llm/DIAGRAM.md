# LLM Client — LM Studio Router

> **Scope:** `agents/src/llm/` — LangChain-compatible wrapper around
> LM Studio. 4 model roles (FAST / STRONG / CODER / EMBED), all
> OpenAI-compatible. Source-of-truth: the code. Last verified: 2026-06-23.

---

## 1. One-Page Overview

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Single file: client.py (204 lines)                                      │
  │  Single concern: provide a factory for ChatOpenAI / OpenAIEmbeddings      │
  │                  pointed at a local LM Studio server                      │
  └──────────────────────────────────────────────────────────────────────────┘

  4 model roles (each a thin factory wrapper):
  ┌────────────────┬────────────────────────────────────┬─────────────────┐
  │ MODEL_FAST     │ gemma-4-e2b-it          (2B)      │ ~17-30s/tasks   │
  │ MODEL_STRONG   │ gemma-4-e2b-it          (2B)      │ (was qwen3.5-9B│
  │                │                                     │  but swapped    │
  │                │                                     │  2026-06-17 —   │
  │                │                                     │  see FIX-18)    │
  │ MODEL_CODER    │ qwen2.5-coder-7b-instruct          │ code-specific   │
  │ MODEL_EMBED    │ text-embedding-nomic-embed-text-v1.5│ 768-dim embed   │
  └────────────────┴────────────────────────────────────┴─────────────────┘

  LangChain  ──▶  ChatOpenAI / OpenAIEmbeddings
                       │
                       │ httpx → http://<gateway>:<port>/v1
                       ▼
                  LM Studio (local GPU server, OpenAI-compatible)
                       │
                       ▼
                  4 GGUF models on RTX 3070 8GB
```

---

## 2. The 4 Model Roles

### 2.1 `MODEL_FAST` — gemma-4-e2b-it (2B)

```
  Speed:     ~12 tok/s on RTX 3070 (3.18 GB VRAM, fully on GPU)
  Used by:   simple tasks, tool selection, API-only flows
             cross_validator debate (A.4 — was 2.91 tok/s, fixed 2026-06-17)
             reflection optional LLM summary (A.3)
             consensus_engine LLM assistance (A.6)
  Default:   temperature=0.0 (deterministic for security)
  Why:       fast enough for 3 sequential debate calls within 240s timeout
```

### 2.2 `MODEL_STRONG` — gemma-4-e2b-it (2B) — *was qwen3.5-9B*

```
  Speed:     ~17-30s for typical tasks (was 23min for 4096 tokens on qwen3.5-9B!)
  Used by:   RAG synthesis, report generation, narrative
             synthesizer final_report Markdown narrative
  Default:   temperature=0.0
  Note:      FIX-18 (2026-06-17) — changed FROM "qwen3.5-9b-ud" because it ran
             at 2.91 tok/s (LM Studio log: 23:17:51 [INFO] tg = 2.91 t/s).
             4096 tokens would take 23 minutes. Switched to gemma-4-e2b-it
             (also used for FAST) — quality sufficient for 4-section narrative.
             If higher quality needed later: try "qwen2.5-coder-7b-instruct".

  ⚠ BOTH "FAST" and "STRONG" now point to the same model
    (gemma-4-e2b-it). Only CODER and EMBED are distinct.
```

### 2.3 `MODEL_CODER` — qwen2.5-coder-7b-instruct

```
  Used by:   Solidity analysis, code logic review
             cross_validator Prosecutor/Defender/Judge debate (A.4)
             when reading the contract source is the dominant task
  Default:   temperature=0.0
  Why a code-specific model?
    General LLMs understand Solidity WORDS but not PATTERNS.
    Qwen2.5-Coder was trained on Solidity source — it understands:
      • access control  (onlyOwner, require, modifier)
      • state machine transitions (locked/unlocked)
      • reentrancy patterns (external call before state update)
      • upgrade proxy patterns (delegatecall storage layout)
```

### 2.4 `MODEL_EMBED` — text-embedding-nomic-embed-text-v1.5

```
  Used by:   RAG index building + query embedding
  Output:    768-dim vectors
  Note:      not a chat model — OpenAIEmbeddings() not ChatOpenAI()
  Why nomic (not a code embed)?
    We embed natural language descriptions (Solidity comments, exploit
    write-ups), not raw Solidity. Text embed > code embed for natural language.
```

---

## 3. The Factory Pattern

```python
# client.py:75-108 — generic factory
def get_llm(model, temperature=0.0, max_tokens=None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=LM_STUDIO_BASE_URL,     # env: LM_STUDIO_BASE_URL
        api_key=LM_STUDIO_API_KEY,       # env: LM_STUDIO_API_KEY (any non-empty)
        temperature=temperature,         # 0.0 default — deterministic
        timeout=LM_STUDIO_TIMEOUT,       # env: LM_STUDIO_TIMEOUT (default 60s)
        **({"max_tokens": max_tokens} if max_tokens else {}),
    )

# Thin role-specific wrappers
def get_fast_llm()     -> ChatOpenAI:         return get_llm(MODEL_FAST)
def get_strong_llm(max_tokens=None) -> ChatOpenAI: return get_llm(MODEL_STRONG, max_tokens=max_tokens)
def get_coder_llm()    -> ChatOpenAI:         return get_llm(MODEL_CODER)
def get_embedding_model() -> OpenAIEmbeddings:  return OpenAIEmbeddings(model=MODEL_EMBED, ...)

# AGENT_MODEL_MAP — forward declaration for future M4.x agents
AGENT_MODEL_MAP = {
    "static_analyzer":  MODEL_CODER,   # reads Solidity structure
    "ml_intelligence":  MODEL_FAST,    # calls Module 1 API only
    "rag_researcher":   MODEL_STRONG,  # reasons over text descriptions
    "code_logic":       MODEL_CODER,   # understands Solidity logic
    "synthesizer":      MODEL_STRONG,  # generates structured report
}
```

---

## 4. LM Studio Connection

```
  ┌────────────────────┐         ┌──────────────────────┐
  │  LangChain         │         │  LM Studio           │
  │  ChatOpenAI()      │ ──────► │  http://gateway:4567 │
  │  OpenAIEmbeddings()│  /v1    │  /v1/chat/completions│
  └────────────────────┘         │  /v1/embeddings      │
                                 │                      │
                                 │  4 GGUF models on    │
                                 │  RTX 3070 8GB        │
                                 └──────────────────────┘
   OpenAI-compatible API
   - same request/response format as api.openai.com
   - api_key can be any non-empty string (LM Studio doesn't validate)
   - all chat models: temperature=0.0 for determinism
```

### Why LM Studio and not real OpenAI?

- **No API costs** — local GPU inference
- **No data leaves the box** — proprietary contracts stay on-prem
- **No rate limits** — auditor can run hundreds of audits/day
- **Model choice is local** — pick the right model for the task
- **OpenAI-compat means low code change** — drop-in `ChatOpenAI` swap

### WSL2 Gateway IP Gotcha (FIX-15, 2026-04-11)

WSL2's gateway IP changes on every Windows reboot. Old code hardcoded
`http://172.21.16.1:1234/v1` — broke on every reboot. Fix: read from
`LM_STUDIO_BASE_URL` env var in `agents/.env`.

```bash
# To find current WSL2 gateway IP:
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# → 172.21.x.x (changes on reboot)
```

Set in `agents/.env`:
```bash
LM_STUDIO_BASE_URL=http://172.21.x.x:4567/v1
```

---

## 5. Configuration

```
  ┌─── LM Studio (agents/.env) ─────────┬────────────────────────────────┐
  │ LM_STUDIO_BASE_URL                  │ http://<gateway>:4567/v1        │
  │ LM_STUDIO_API_KEY                   │ "lm-studio" (any non-empty str)  │
  │ LM_STUDIO_TIMEOUT                   │ 60 (seconds)                    │
  └─────────────────────────────────────┴────────────────────────────────┘

  All 4 roles use the same base_url + api_key + timeout.
  Models are identified by their LM Studio model IDs (exact strings).
```

---

## 6. Timeouts (cross-reference with orchestration)

| Context | Env var | Default | Used by |
|---------|---------|---------|---------|
| Per-LLM call | `LM_STUDIO_TIMEOUT` | 60s | every `get_llm()` call |
| Single-pass cross_validator | `CROSS_VALIDATOR_TIMEOUT_S` | 90s | `cross_validator` node (DEBATE_MODE=off) |
| Full 3-role debate | `DEBATE_TIMEOUT_S` | 240s | `cross_validator` node (DEBATE_MODE=on, default) |
| Synthesizer narrative | `SYNTHESIZER_TIMEOUT_S` | 120s | `synthesizer` node |
| Synthesizer max tokens | `SYNTHESIZER_MAX_TOKENS` | 4096 | `synthesizer` node |
| LLM kill-switch | `AGENTS_DISABLE_LLM` | 0 | all LLM-calling nodes fall back to rule-based |

The LM Studio timeout (60s) is the **innermost** timeout. Outer
timeouts (90s, 240s, 120s) wrap multiple LLM calls.

---

## 7. FIX History

| Fix | Date | What | Why |
|-----|------|------|-----|
| **FIX-15** | 2026-04-11 | `LM_STUDIO_BASE_URL` from env | Hardcoded WSL2 IP broke on every reboot |
| **FIX-16** | 2026-04-11 | `timeout=60` on all chat/embed models | LM Studio hang (model load, GPU OOM) → caller blocks forever |
| **FIX-17** | 2026-06-17 | `max_tokens` pass-through to `get_strong_llm` | LM Studio default too low for 4-section narrative → `finish_reason="length"`, empty content |
| **FIX-18** | 2026-06-17 | `MODEL_STRONG` changed from `qwen3.5-9b-ud` to `gemma-4-e2b-it` | Qwen 9B ran at 2.91 tok/s (would take 23min for 4096 tokens); Gemma 2B runs in 17-30s, sufficient quality |
| **ADD-2** | 2026-04-11 | `AGENT_MODEL_MAP` forward declaration | M4.x milestone agents not yet built — central place to change role → model mapping |

---

## 8. Consumers — Who Calls the LLM Client

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Consumed by graph nodes (orchestration/nodes.py):                       │
  │                                                                          │
  │  • ml_assessment          NO direct LLM call (calls Module 1 via MCP)    │
  │                                                                          │
  │  • evidence_router        NO LLM                                          │
  │                                                                          │
  │  • rag_research           NO LLM (builds query from ml_result)            │
  │                                                                          │
  │  • static_analysis        NO LLM (Slither direct)                         │
  │                                                                          │
  │  • graph_explain          NO LLM (calls MCP :8013)                       │
  │                                                                          │
  │  • audit_check            NO LLM (Web3 via MCP :8012)                     │
  │                                                                          │
  │  • consensus_engine       LLM-assisted (STRONG, A.6)                      │
  │                                                                          │
  │  • cross_validator        LLM debate (FAST × 3 sequential, A.4)          │
  │                                                                          │
  │  • synthesizer            LLM narrative (STRONG, 4096 tokens)             │
  │                                                                          │
  │  • reflection             LLM self-critique (STRONG, A.3) — optional     │
  │                                                                          │
  │  • explainer              NO LLM (LIME-style pure logic, A.8)             │
  │                                                                          │
  │  • visualizer             NO LLM (HTML generation, A.9)                   │
  │                                                                          │
  │  Consumed by other modules:                                              │
  │                                                                          │
  │  • rag/embedder.py        get_embedding_model() — wraps OpenAIEmbeddings │
  │  • rag/build_index.py     (via embedder.py)                                │
  │  • ingestion/pipeline.py  (via embedder.py)                                │
  │  • mcp/servers/rag_server.py (via HybridRetriever)                         │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Test-Mode Kill-Switch

```bash
# In agents/.env
AGENTS_DISABLE_LLM=1
```

When set, every LLM-calling node (cross_validator, synthesizer narrative,
reflection, consensus_engine LLM) consults `_llm_enabled()` and falls back to
rule-based logic. The graph runs:

- **fast** — no LLM call overhead
- **deterministic** — same input → same output (great for tests)
- **without LM Studio running** — useful in CI

See `orchestration/nodes.py:_llm_enabled()` for the exact check.

---

## 10. Smoke Test

```bash
cd agents
poetry run python -m src.llm.client
# Tests all three chat models + confirms LM Studio connectivity
# Output:
#   LM Studio URL: http://<gateway>:4567/v1
#   Timeout:       60s
#   Testing LM Studio connection — all three models...
#   Fast model   (gemma-4-e2b-it):     FAST_OK
#   Strong model (gemma-4-e2b-it):     STRONG_OK
#   Coder model  (qwen2.5-coder-7b):   CODER_OK
#   All models responding. LM Studio connection confirmed.
#   Agent routing map: {'static_analyzer': 'qwen2.5-coder-7b-instruct', ...}
```

If any model returns empty or errors:
- `LM Studio not running` → start LM Studio + load models
- `Wrong model ID` → check LM Studio's `/v1/models` for exact ID
- `Connection refused` → wrong gateway IP or LM Studio on different port

---

## 11. Why Two Roles Now Point to the Same Model (FIX-18)

```
  Before FIX-18 (2026-06-17):
  ┌──────────────────┐
  │ MODEL_FAST       │ gemma-4-e2b-it     (2B, ~12 tok/s)
  │ MODEL_STRONG     │ qwen3.5-9b-ud      (9B, 2.91 tok/s on RTX 3070)  ← TOO SLOW
  │ MODEL_CODER      │ qwen2.5-coder-7b   (7B, code-specialized)
  │ MODEL_EMBED      │ nomic-embed-text   (embedding only)
  └──────────────────┘
  Qwen 9B: 4096 tokens × 2.91 tok/s ≈ 23 minutes. Real audit testing showed
  this exceeded outer timeouts (120s synthesizer, 240s debate) and caused
  process kills (the abandoned asyncio.to_thread() HTTP call kept running
  in its OS thread since to_thread cannot be cancelled).

  After FIX-18:
  ┌──────────────────┐
  │ MODEL_FAST       │ gemma-4-e2b-it     (2B, ~17-30s)
  │ MODEL_STRONG     │ gemma-4-e2b-it     (2B, ~17-30s)    ← same as FAST
  │ MODEL_CODER      │ qwen2.5-coder-7b   (7B, code-specialized)
  │ MODEL_EMBED      │ nomic-embed-text   (embedding only)
  └──────────────────┘
  Quality is sufficient for the 4-section Markdown narrative.
  If higher quality needed later: try "qwen2.5-coder-7b-instruct" —
  faster than 9B Qwen, code-specialized.
```

This is a **temporary state** — once a faster strong model is available
(or quality bottleneck appears), `MODEL_STRONG` will diverge again. The
wrapper functions (`get_strong_llm()`) abstract the model choice, so
caller code doesn't change when the model swaps.

---

## 12. File Map

```
  agents/src/llm/
  │
  └── client.py            233 lines  the entire module
                                  get_llm()                generic factory
                                  get_fast_llm(max_t)      gemma-4-e2b-it (FAST)
                                  get_strong_llm(max_t)    gemma-4-e2b-it (STRONG, FIX-18)
                                  get_coder_llm()          qwen2.5-coder-7b (CODER)
                                  get_embedding_model()    nomic-embed-text-v1.5 (EMBED)
                                  AGENT_MODEL_MAP          forward declaration (M4.x)
                                  __main__                 smoke test (3 models)
```

---

## 13. Quick Reference

| Concept | File:Line |
|---------|-----------|
| `LM_STUDIO_BASE_URL` (env-driven) | `agents/src/llm/client.py:46` |
| `LM_STUDIO_API_KEY` (env-driven) | `client.py:49` |
| `LM_STUDIO_TIMEOUT` (env-driven) | `client.py:54` |
| `MODEL_FAST` | `client.py:57` |
| `MODEL_STRONG` (FIX-18 swap) | `client.py:64` |
| `MODEL_CODER` | `client.py:69` |
| `MODEL_EMBED` | `client.py:72` |
| `get_llm()` generic factory | `client.py:75-108` |
| `max_tokens` pass-through (FIX-17) | `client.py:97-99` |
| `get_fast_llm()` | `client.py:111-117` |
| `get_strong_llm(max_tokens)` | `client.py:120-129` |
| `get_coder_llm()` | `client.py:132-145` |
| `get_embedding_model()` | `client.py:148-168` |
| `AGENT_MODEL_MAP` (forward decl) | `client.py:177-183` |
| Smoke test | `client.py:186-203` |
| LLM enable guard (`_llm_enabled()`) | `agents/src/orchestration/nodes.py:90-92` |
| Cross_validator debate | `agents/src/orchestration/nodes.py` (search for `cross_validator`) |
| Synthesizer narrative | `agents/src/orchestration/nodes.py` (search for `synthesizer`) |
| RAG embedder | `agents/src/rag/embedder.py` |

---

## 14. See Also

- `~/projects/sentinel/agents/DIAGRAM.md` — top-level module diagram
- `~/projects/sentinel/agents/src/llm/README.md` — text companion
- `~/projects/sentinel/agents/src/orchestration/DIAGRAM.md` — LLM consumers (graph nodes)
- `~/projects/sentinel/agents/src/orchestration/README.md` — model role assignments per node
- `~/projects/sentinel/agents/src/rag/DIAGRAM.md` — embedder
