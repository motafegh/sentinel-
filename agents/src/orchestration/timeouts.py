"""
timeouts.py — single source of truth for every timeout in the audit pipeline.

WHY THIS FILE EXISTS (2026-06-21): every LLM-call timeout and subprocess timeout
used to be a bare magic number duplicated at its call site (`timeout=90`,
`os.getenv("X", "120")`, ...). Tracing where a given default came from required
grepping the whole module, and changing it meant finding every occurrence. All
defaults now live ONLY here — every other module imports the constant by name
and reads it via `os.getenv(ENV_VAR_NAME, str(DEFAULT))` at call time (not
cached at import), so existing override mechanisms keep working unchanged:
  - tests: `monkeypatch.setenv("DEBATE_TIMEOUT_S", "0.2")`
  - shell: `DEBATE_TIMEOUT_S=400 poetry run python scripts/run_real_audit.py ...`
  - CLI:   `python scripts/run_real_audit.py --debate-timeout-s 400 ...`
            (run_real_audit.py's `_resolve_timeouts()` sets the env var for you)

Each constant's ENV VAR NAME is unchanged from before this refactor — no .env
file or existing test needs to change variable names, only where the default
value is defined.
"""

from __future__ import annotations

import os

# ── LLM call timeouts ────────────────────────────────────────────────────────

# LM Studio HTTP client timeout — the floor under every single LLM call made
# anywhere in the pipeline (ChatOpenAI/OpenAIEmbeddings request timeout). If
# this fires, it fires before any asyncio.wait_for wrapping the call ever gets
# a chance to. Read at import time by src/llm/client.py (module-level), so to
# change it for a given run you must set the env var BEFORE that module is
# first imported (run_real_audit.py's `_resolve_timeouts()` does this).
ENV_LM_STUDIO_TIMEOUT_S = "LM_STUDIO_TIMEOUT"
DEFAULT_LM_STUDIO_TIMEOUT_S = 60.0

# cross_validator, DEBATE_MODE=off (single classification call).
ENV_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S = "CROSS_VALIDATOR_TIMEOUT_S"
DEFAULT_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S = 90.0

# cross_validator, DEBATE_MODE=on — bounds the ENTIRE 3-role sequence
# (Prosecutor+Defender+Judge) as ONE unit, not per call. See nodes.py
# cross_validator docstring for the 2026-06-21 incident this fixed.
ENV_DEBATE_TIMEOUT_S = "DEBATE_TIMEOUT_S"
DEFAULT_DEBATE_TIMEOUT_S = 240.0

# synthesizer's LLM narrative generation call.
ENV_SYNTHESIZER_NARRATIVE_TIMEOUT_S = "SYNTHESIZER_TIMEOUT_S"
DEFAULT_SYNTHESIZER_NARRATIVE_TIMEOUT_S = 120.0

# reflection's optional LLM self-critique summary call.
ENV_REFLECTION_TIMEOUT_S = "REFLECTION_TIMEOUT_S"
DEFAULT_REFLECTION_TIMEOUT_S = 120.0

# ── Subprocess timeouts (static analysis tools) ─────────────────────────────

# Aderyn subprocess invocation (_run_aderyn_on_file, used by both
# static_analysis and quick_screen).
ENV_ADERYN_TIMEOUT_S = "ADERYN_TIMEOUT_S"
DEFAULT_ADERYN_TIMEOUT_S = 90.0

# ── MCP / HTTP client timeouts (separate long-running server processes) ────
# These are read at import time by their respective server modules — to
# change them you must restart that server process with the new env var set,
# CLI flags from run_real_audit.py cannot reach into an already-running
# separate process.

ENV_GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT_S = "GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT"
DEFAULT_GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT_S = 60.0

ENV_MODULE1_INFERENCE_TIMEOUT_S = "MODULE1_TIMEOUT"
DEFAULT_MODULE1_INFERENCE_TIMEOUT_S = 30.0

# Used by run_real_audit.py's --unbounded-timeouts flag: every timeout above
# is set to this value so a step only ever stops at natural completion (or a
# genuine hang), making true per-step timing visible with nothing artificially
# truncated. Not "infinite" — a hang should still eventually surface.
UNBOUNDED_TIMEOUT_S = 3600.0


def get_timeout(env_var: str, default: float) -> float:
    """Read a timeout from the environment, falling back to `default`. Read at
    call time (never cached) so env-var overrides set at any point — shell,
    test monkeypatch, or CLI-injected before this process's first use — apply."""
    try:
        return float(os.getenv(env_var, str(default)))
    except ValueError:
        return default
