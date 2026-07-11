"""
security/ — Prompt-injection defense for LLM prompts (P4).

Three layers:
  1. comment_strip  — remove Solidity comments (preserves line count)
  2. prompt_delimit — wrap source in <<CONTRACT_SOURCE>> delimiters
  3. injection_detect — pattern-match 8 known injection signatures (log-only)

Orchestrator: prompt_sanitize.sanitize_for_prompt()
"""

from src.security.comment_strip import strip_comments
from src.security.injection_detect import InjectionMatch, detect_injections
from src.security.prompt_delimit import delimit_contract_source
from src.security.prompt_sanitize import sanitize_for_prompt

__all__ = [
    "strip_comments",
    "delimit_contract_source",
    "detect_injections",
    "sanitize_for_prompt",
    "InjectionMatch",
]
