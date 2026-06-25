"""
client.py

LM Studio connection + model router for SENTINEL agents.
LM Studio exposes an OpenAI-compatible API — LangChain connects
via ChatOpenAI pointed at localhost instead of api.openai.com.

Model routing strategy (match model to task):
  FAST   — gemma-4-e2b-it          — simple tasks, tool selection, API calls
  STRONG — gemma-4-e2b-it           — reasoning, RAG synthesis, report generation
  CODER  — qwen2.5-coder-7b-instruct        — Solidity analysis, code logic review
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
# Default centralized in src/orchestration/timeouts.py (2026-06-21) — this is
# the ONLY timeout in the pipeline read at IMPORT time (module-level), so to
# override it for a given run the env var must be set BEFORE this module is
# first imported. run_real_audit.py's `_resolve_timeouts()` does this.
from src.orchestration.timeouts import DEFAULT_LM_STUDIO_TIMEOUT_S
# int(float(...)) tolerates BOTH "60" and "3600.0" — every other timeout in the
# pipeline is resolved as a float (see timeouts.get_timeout); CLI/--unbounded-
# timeouts writes float-formatted strings, which plain int() rejects.
LM_STUDIO_TIMEOUT  = int(float(os.getenv("LM_STUDIO_TIMEOUT", str(DEFAULT_LM_STUDIO_TIMEOUT_S))))

# ── Model IDs (exactly as LM Studio reports from /v1/models) ─────────────────
MODEL_FAST   = "gemma-4-e2b-it"
# 2026-06-17 FIX-18: Changed MODEL_STRONG from "qwen3.5-9b-ud" to "gemma-4-e2b-it".
# The Qwen 9B runs at 2.91 tok/sec on this RTX 3070 with Q4_K_XL quantization
# (LM Studio log: 23:17:51 [INFO] tg = 2.91 t/s). 4096 tokens would take 23 minutes.
# gemma-4-e2b-it (2B) runs at ~17-30s for similar tasks. Quality is sufficient for
# the 4-section Markdown narrative. If higher quality is needed later, try
# "qwen2.5-coder-7b-instruct" (7B, code-specialized, faster than 9B Qwen).
MODEL_STRONG = "gemma-4-e2b-it"

# Code-specific LLM — trained on 80+ languages including Solidity.
# Understands: access control patterns, state machine transitions,
# reentrancy patterns, upgrade proxy layouts — better than general models.
MODEL_CODER  = "qwen2.5-coder-7b-instruct"

# Embedding only — not a chat model
MODEL_EMBED  = "text-embedding-nomic-embed-text-v1.5"


def get_llm(model: str = MODEL_FAST, temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI instance pointed at LM Studio.

    FIX-15: Uses LM_STUDIO_BASE_URL from env (not hardcoded IP).
    FIX-16: timeout=LM_STUDIO_TIMEOUT prevents indefinite hangs.
    FIX-17 (2026-06-17): max_tokens parameter. LM Studio's default is too low for
            the synthesizer narrative (4 Markdown sections + reasoning content).
            Without explicit max_tokens, the model hits finish_reason="length"
            and returns content="" — LangChain then raises an empty exception.

    Args:
        model:       LM Studio model ID — use MODEL_* constants above
        temperature: 0.0 = deterministic (correct for security audit tasks)
        max_tokens:  Output token limit. None = LM Studio default (~2-4K).
                     Pass 4096+ for long generations (narrative, reports).

    Returns:
        ChatOpenAI instance ready for LangChain chains and agents
    """
    logger.debug(f"Initialising LLM — model: {model} | temp: {temperature} | timeout: {LM_STUDIO_TIMEOUT}s | max_tokens: {max_tokens}")

    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(
        model=model,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=temperature,
        timeout=LM_STUDIO_TIMEOUT,    # FIX-16
        **kwargs,
    )


def get_fast_llm(max_tokens: int | None = None) -> ChatOpenAI:
    """
    Gemma-4-E2B — fast, lightweight.
    Use for: MLIntelligenceAgent (API calls only, no code reasoning needed),
    cross_validator debate (Prosecutor/Defender/Judge).
    Speed: ~12 tokens/sec on RTX 3070 (fully on GPU, 3.18 GB VRAM).

    WS4.1 (2026-06-22): max_tokens pass-through. The 3 debate roles previously
    had no output-length cap — each could generate unlimited text, contributing
    to 75-115s per role. After a 384/512/768/1024 sweep on vulnerable_reentrant.sol:

      384/512 → LM Studio returns content="" (model's internal preamble hits the
                cap before producing output). Debate effectively doesn't run —
                verdict falls through to consensus.
      768    → Sweet spot. Prosecutor 657 chars, defender 900 chars, both
                non-empty, judge produces valid JSON. ~28s debate.
      1024   → Verbose (2021/1682 chars), no verdict improvement over 768.

    Actual defaults live in `nodes.py` cross_validator:
      DEBATE_PROSECUTOR_MAX_TOKENS = 768
      DEBATE_DEFENDER_MAX_TOKENS  = 768
      DEBATE_JUDGE_MAX_TOKENS     = 0    (0 = uncapped — judge needs the room
                                            for its reasoning before the JSON;
                                            capping makes it return empty)

    Overridable per call via this `max_tokens` parameter.
    """
    return get_llm(model=MODEL_FAST, temperature=0.0, max_tokens=max_tokens)


def get_strong_llm(max_tokens: int | None = None) -> ChatOpenAI:
    """
    Gemma-4-e2b-it — strong reasoning (same model as FAST post-FIX-18).
    Use for: RAGResearcherAgent, SynthesizerAgent.

    FIX-17 (2026-06-17): max_tokens pass-through. Synthesizer needs 4096+ for
    the 4-section narrative. Without it, LM Studio truncates mid-response.
    """
    return get_llm(model=MODEL_STRONG, temperature=0.0, max_tokens=max_tokens)


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
