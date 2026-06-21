"""
Shared pytest configuration for the agents test suite.

Phase A (2026-06-21): the LLM-calling nodes (cross_validator, synthesizer
narrative, reflection) now consult `_llm_enabled()`. Historically the e2e
smoke tests relied on the LM Studio URL being unreachable so the LLM calls
failed fast into rule-based fallback. Now that the URL can point at a live
LM Studio, disable LLM for the whole test session so tests stay fast and
deterministic. Real runs (scripts/run_real_audit.py) do not set this flag.
"""

import os


def pytest_configure(config):  # noqa: ARG001
    os.environ.setdefault("AGENTS_DISABLE_LLM", "1")
