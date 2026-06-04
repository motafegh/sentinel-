# LLM Client

LM Studio connection and model router for SENTINEL agents. Provides LangChain-compatible `ChatOpenAI` and `OpenAIEmbeddings` instances pointed at a local LM Studio server.

## File

| File | Lines | Purpose |
|------|-------|---------|
| `client.py` | 184 | LLM client factory with model routing |

## Models

| Role | Model ID | Use Case | Speed (RTX 3070) |
|------|----------|----------|-------------------|
| `FAST` | `gemma-4-e2b-it` | Simple tasks, tool selection, API calls | ~12 tok/s |
| `STRONG` | `qwen3.5-9b-ud` | Reasoning, RAG synthesis, report generation | ~37 tok/s |
| `CODER` | `qwen2.5-coder-7b-instruct` | Solidity analysis, code logic review | — |
| `EMBED` | `text-embedding-nomic-embed-text-v1.5` | RAG embeddings (text descriptions) | — |

### Why a Code-Specific Model

General LLMs understand Solidity words but not patterns. `qwen2.5-coder-7b` was trained on Solidity source and understands:
- Access control (`onlyOwner`, `require`, `modifier`)
- State machine transitions (locked/unlocked)
- Reentrancy patterns (external call before state update)
- Upgrade proxy patterns (delegatecall storage layout)

### Why Nomic-Embed for RAG

We embed natural language descriptions (from Solidity comments and exploit write-ups), not raw Solidity code. Text embedding outperforms code embedding for natural language descriptions.

## Usage

```python
from src.llm.client import get_fast_llm, get_strong_llm, get_coder_llm, get_embedding_model

# Chat models (all return LangChain ChatOpenAI)
fast_llm   = get_fast_llm()    # gemma-4-e2b-it
strong_llm = get_strong_llm()  # qwen3.5-9b-ud
coder_llm  = get_coder_llm()   # qwen2.5-coder-7b-instruct

# Embedding model (LangChain OpenAIEmbeddings)
embedder   = get_embedding_model()  # nomic-embed-text-v1.5

# Generic factory
from src.llm.client import get_llm, MODEL_FAST, MODEL_STRONG
llm = get_llm(model=MODEL_FAST, temperature=0.0)
```

All models use `temperature=0.0` by default — deterministic output correct for security audit tasks.

## Agent Model Routing

```python
AGENT_MODEL_MAP = {
    "static_analyzer":  MODEL_CODER,    # reads Solidity structure
    "ml_intelligence":  MODEL_FAST,     # calls Module 1 API only
    "rag_researcher":   MODEL_STRONG,   # reasons over text descriptions
    "code_logic":       MODEL_CODER,    # understands Solidity logic
    "synthesizer":      MODEL_STRONG,   # generates structured report
}
```

Forward declaration — agent classes not yet implemented (M4.x milestone). Change a model here and all agents using that role update automatically.

## Configuration (`.env`)

```bash
LM_STUDIO_BASE_URL=http://172.21.16.1:4567/v1   # WSL2 gateway IP
LM_STUDIO_API_KEY=lm-studio                       # required by LangChain (not validated)
LM_STUDIO_TIMEOUT=60                              # seconds — prevents indefinite hangs
```

### WSL2 Gateway IP

On WSL2, the Windows host gateway IP changes on reboot. Find it with:
```bash
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
```
Set `LM_STUDIO_BASE_URL` explicitly in `agents/.env`.

## Connection

```
LangChain ChatOpenAI  ──▶  LM Studio (OpenAI-compatible API)  ──▶  Local GPU
                           http://localhost:4567/v1
```

LM Studio exposes an OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoint. LangChain connects via `ChatOpenAI` and `OpenAIEmbeddings` pointed at the local server instead of `api.openai.com`.

## Smoke Test

```bash
cd agents
poetry run python -m src.llm.client
# Tests all three chat models + confirms LM Studio connectivity
```
