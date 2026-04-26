"""
client.py

LM Studio connection + model router for SENTINEL agents.
LM Studio exposes an OpenAI-compatible API — LangChain connects
via ChatOpenAI pointed at localhost instead of api.openai.com.

Model routing strategy (match model to task):
  FAST   — gemma-4-e2b-it          — simple tasks, tool selection, API calls
  STRONG — qwen3.5-9b-ud           — reasoning, RAG synthesis, report generation
  CODER  — qwen2.5-coder-7b        — Solidity analysis, code logic review
  EMBED  — nomic-embed-text-v1.5   — RAG embeddings (text descriptions)

CHANGES (2026-04-11):
  FIX-15: LM Studio URL now read from .env via os.getenv().
          Old: hardcoded "http://172.21.16.1:1234/v1" — broke on every WSL2 reboot
               (gateway IP changes) requiring a manual code edit each time.
          New: LM_STUDIO_BASE_URL env var with the old IP as fallback.
          Add to .env: L
          M_STUDIO_BASE_URL=http://172.21.16.1:1234/v1
  FIX-16: timeout=60 added to ChatOpenAI and OpenAIEmbeddings.
          Old: no timeout — if LM Studio hung (model loading, GPU OOM),
               all callers blocked indefinitely with no recovery path.
          New: 60s timeout raises an exception the caller can handle.
  ADD-2:  AGENT_MODEL_MAP marked as forward declaration — the agent
          classes it references are not yet implemented (M4.x milestone).
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

# Load .env from agents/ directory so LM_STUDIO_BASE_URL is available
# whether this module is imported by a larger process OR run directly as a script.
# load_dotenv() is a no-op if variables are already set in the environment,
# so it is safe to call early at module level.
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# FIX-15: Read from environment — no more hardcoded IPs that break on reboot.
# To find your current WSL2 gateway IP:
#   cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# Then set in agents/.env:
#   LM_STUDIO_BASE_URL=http://<that-ip>:<port>/v1
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://172.21.16.1:4567/v1")

# LM Studio doesn't require a real API key, but LangChain requires a non-empty string.
LM_STUDIO_API_KEY  = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# FIX-16: Request timeout in seconds.
# If LM Studio hangs (model loading, GPU OOM), callers get an exception
# after this many seconds instead of blocking indefinitely.
LM_STUDIO_TIMEOUT  = int(os.getenv("LM_STUDIO_TIMEOUT", "60"))

# ── Model IDs (exactly as LM Studio reports from /v1/models) ─────────────────
MODEL_FAST   = "gemma-4-e2b-it"
MODEL_STRONG = "qwen3.5-9b-ud"

# Code-specific LLM — trained on 80+ languages including Solidity.
# Understands: access control patterns, state machine transitions,
# reentrancy patterns, upgrade proxy layouts — better than general models.
MODEL_CODER  = "qwen2.5-coder-7b-instruct"

# Embedding only — not a chat model
MODEL_EMBED  = "text-embedding-nomic-embed-text-v1.5"


def get_llm(model: str = MODEL_FAST, temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI instance pointed at LM Studio.

    FIX-15: Uses LM_STUDIO_BASE_URL from env (not hardcoded IP).
    FIX-16: timeout=LM_STUDIO_TIMEOUT prevents indefinite hangs.

    Args:
        model:       LM Studio model ID — use MODEL_* constants above
        temperature: 0.0 = deterministic (correct for security audit tasks)

    Returns:
        ChatOpenAI instance ready for LangChain chains and agents
    """
    logger.debug(f"Initialising LLM — model: {model} | temp: {temperature} | timeout: {LM_STUDIO_TIMEOUT}s")

    return ChatOpenAI(
        model=model,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=temperature,
        timeout=LM_STUDIO_TIMEOUT,    # FIX-16
    )


def get_fast_llm() -> ChatOpenAI:
    """
    Gemma-4-E2B — fast, lightweight.
    Use for: MLIntelligenceAgent (API calls only, no code reasoning needed).
    Speed: ~12 tokens/sec on RTX 3070 (fully on GPU, 3.18 GB VRAM).
    """
    return get_llm(model=MODEL_FAST, temperature=0.0)


def get_strong_llm() -> ChatOpenAI:
    """
    Qwen3.5-9B — strong reasoning.
    Use for: RAGResearcherAgent, SynthesizerAgent.
    Speed: ~37 tokens/sec on RTX 3070 (most layers on GPU).
    """
    return get_llm(model=MODEL_STRONG, temperature=0.0)


def get_coder_llm() -> ChatOpenAI:
    """
    Qwen2.5-Coder-7B — code-specific LLM.
    Use for: StaticAnalyzerAgent, CodeLogicAgent.

    RECALL — Why a code-specific model:
    General LLMs understand Solidity words but not patterns.
    Qwen2.5-Coder was trained on Solidity source — it understands:
      - access control (onlyOwner, require, modifier)
      - state machine transitions (locked/unlocked)
      - reentrancy patterns (external call before state update)
      - upgrade proxy patterns (delegatecall storage layout)
    """
    return get_llm(model=MODEL_CODER, temperature=0.0)


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Nomic-embed-text-v1.5 — text embedding model.
    Use for: RAG index building and query embedding.

    FIX-15: Uses LM_STUDIO_BASE_URL from env.
    FIX-16: timeout=LM_STUDIO_TIMEOUT prevents indefinite hangs on model load.

    RECALL — Why this model not a code embedding model:
    We embed natural language descriptions (from Solidity comments),
    not raw Solidity code. Text embedding outperforms code embedding
    for natural language descriptions.
    """
    logger.debug(f"Initialising embedding model: {MODEL_EMBED}")

    return OpenAIEmbeddings(
        model=MODEL_EMBED,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        timeout=LM_STUDIO_TIMEOUT,    # FIX-16
    )


# ── Agent → model routing reference ──────────────────────────────────────────
# ADD-2: This map is a FORWARD DECLARATION.
# The agent classes listed here (static_analyzer, ml_intelligence, etc.)
# are not yet implemented — they are planned for milestone M4.x.
# When agents are built, import this dict to keep model assignments centralised.
# Change a model here → all agents using that role update automatically.
AGENT_MODEL_MAP = {
    "static_analyzer":  MODEL_CODER,    # reads Solidity structure
    "ml_intelligence":  MODEL_FAST,     # calls Module 1 API only
    "rag_researcher":   MODEL_STRONG,   # reasons over text descriptions
    "code_logic":       MODEL_CODER,    # understands Solidity logic
    "synthesizer":      MODEL_STRONG,   # generates structured report
}


if __name__ == "__main__":
    logger.info(f"LM Studio URL: {LM_STUDIO_BASE_URL}")
    logger.info(f"Timeout:       {LM_STUDIO_TIMEOUT}s")
    logger.info("Testing LM Studio connection — all three models...")

    fast     = get_fast_llm()
    response = fast.invoke("Reply with exactly: FAST_OK")
    logger.info(f"Fast model   ({MODEL_FAST}):  {response.content}")

    strong   = get_strong_llm()
    response = strong.invoke("Reply with exactly: STRONG_OK")
    logger.info(f"Strong model ({MODEL_STRONG}): {response.content}")

    coder    = get_coder_llm()
    response = coder.invoke("Reply with exactly: CODER_OK")
    logger.info(f"Coder model  ({MODEL_CODER}):  {response.content}")

    logger.info("All models responding. LM Studio connection confirmed.")
    logger.info(f"Agent routing map: {AGENT_MODEL_MAP}")
