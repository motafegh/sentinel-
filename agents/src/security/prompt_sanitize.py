"""
security/prompt_sanitize.py — Orchestrator: applies all 3 layers in sequence (P4).

Pipeline:
  1. detect_injections(source) — Layer 3 on ORIGINAL source (before stripping)
  2. strip_comments(source)    — Layer 1
  3. delimit_contract_source() — Layer 2
"""

from __future__ import annotations

from src.security.comment_strip import strip_comments
from src.security.injection_detect import InjectionMatch, detect_injections
from src.security.prompt_delimit import delimit_contract_source


def sanitize_for_prompt(
    source: str,
    *,
    detect: bool = True,
) -> tuple[str, list[InjectionMatch]]:
    """
    Apply all 3 injection-defense layers to contract source.

    Args:
        source: raw Solidity source code
        detect: if True, run injection detection on original source (Layer 3)

    Returns:
        (sanitized_and_delimited_source, injection_matches)
    """
    matches: list[InjectionMatch] = []
    if detect:
        matches = detect_injections(source)

    stripped = strip_comments(source)
    delimited = delimit_contract_source(stripped)
    return delimited, matches
