"""
security/injection_detect.py — Layer 3: pattern detection, log-only canary (P4).

Detects 8 known prompt-injection patterns in Solidity source code.
Runs on the ORIGINAL source (before comment stripping) so evidence is preserved.
Returns a list of InjectionMatch — never blocks, never alters the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionMatch:
    pattern: str
    location: str
    snippet: str
    confidence: str


_INSTRUCTION_VERBS = re.compile(
    r"\b(?:ignore|override|disregard|forget|pretend|bypass|skip|mark|set|change|force)\b",
    re.IGNORECASE,
)

_SAFE_MARKERS = re.compile(
    r"\b(?:SAFE|safe|no\s+vulnerabilit|not\s+vulnerable|clean|approved)\b",
    re.IGNORECASE,
)

_ROLE_NAMES = re.compile(
    r"\b(?:Prosecutor|Defender|Judge|judge|auditor|AI|assistant|model|LLM|system)\b",
)

_EXTRACTION_KEYWORDS = re.compile(
    r"\b(?:system\s*prompt|reveal|print\s*(?:the|out)|output\s*(?:the|all)|"
    r"show\s*(?:me\s*)?(?:the|your)|what\s*(?:are|is)\s*your\s*instructions|"
    r"repeat\s*(?:the|your)|recite)\b",
    re.IGNORECASE,
)

_URL_IMPORT = re.compile(
    r'import\s+["\'](?:https?://|ftp://|//)',
    re.IGNORECASE,
)

_NATSPEC_LINE = re.compile(r"^\s*///")
_NATSPEC_BLOCK_START = re.compile(r"/\*\*")


def _extract_lines(source: str) -> list[str]:
    return source.splitlines()


def _truncate(text: str, limit: int = 80) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _detect_comment_injection(source: str) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    lines = _extract_lines(source)
    in_block = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if in_block:
            if "*/" in line:
                in_block = False
                segment = line[: line.index("*/")]
            else:
                segment = line
            if _INSTRUCTION_VERBS.search(segment) and _SAFE_MARKERS.search(segment):
                matches.append(InjectionMatch(
                    pattern="comment",
                    location=f"line {lineno}",
                    snippet=_truncate(segment),
                    confidence="high",
                ))
            continue
        line_comment_match = re.search(r"(?<![:/])//(.*)", line)
        if line_comment_match:
            comment_text = line_comment_match.group(1)
            if _INSTRUCTION_VERBS.search(comment_text) and _SAFE_MARKERS.search(comment_text):
                matches.append(InjectionMatch(
                    pattern="comment",
                    location=f"line {lineno}",
                    snippet=_truncate(comment_text),
                    confidence="high",
                ))
            elif _INSTRUCTION_VERBS.search(comment_text):
                matches.append(InjectionMatch(
                    pattern="comment",
                    location=f"line {lineno}",
                    snippet=_truncate(comment_text),
                    confidence="medium",
                ))
        if "/*" in line:
            start = line.index("/*") + 2
            if "*/" in line:
                end = line.index("*/")
                segment = line[start:end]
            else:
                in_block = True
                segment = line[start:]
            if _INSTRUCTION_VERBS.search(segment):
                matches.append(InjectionMatch(
                    pattern="comment",
                    location=f"line {lineno}",
                    snippet=_truncate(segment),
                    confidence="medium",
                ))
    return matches


def _detect_string_injection(source: str) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    lines = _extract_lines(source)
    string_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
    for lineno, line in enumerate(lines, 1):
        for m in string_pattern.finditer(line):
            content = m.group(1)
            if _INSTRUCTION_VERBS.search(content) and _SAFE_MARKERS.search(content):
                matches.append(InjectionMatch(
                    pattern="string",
                    location=f"line {lineno}",
                    snippet=_truncate(content),
                    confidence="high",
                ))
            elif _INSTRUCTION_VERBS.search(content):
                matches.append(InjectionMatch(
                    pattern="string",
                    location=f"line {lineno}",
                    snippet=_truncate(content),
                    confidence="medium",
                ))
    return matches


def _detect_role_swap(source: str) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    lines = _extract_lines(source)
    for lineno, line in enumerate(lines, 1):
        if _ROLE_NAMES.search(line) and _INSTRUCTION_VERBS.search(line):
            is_comment = "//" in line or "/*" in line or "*/" in line
            is_string = False
            for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', line):
                if _ROLE_NAMES.search(m.group(1)) and _INSTRUCTION_VERBS.search(m.group(1)):
                    is_string = True
            if is_comment or is_string:
                matches.append(InjectionMatch(
                    pattern="role-swap",
                    location=f"line {lineno}",
                    snippet=_truncate(line),
                    confidence="high",
                ))
    return matches


def _detect_extraction(source: str) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    lines = _extract_lines(source)
    for lineno, line in enumerate(lines, 1):
        if not _EXTRACTION_KEYWORDS.search(line):
            continue
        is_comment = "//" in line or "/*" in line
        is_string = False
        for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', line):
            if _EXTRACTION_KEYWORDS.search(m.group(1)):
                is_string = True
        if is_comment or is_string:
            matches.append(InjectionMatch(
                pattern="extraction",
                location=f"line {lineno}",
                snippet=_truncate(line),
                confidence="high",
            ))
    return matches


def _detect_identifier_injection(source: str) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    ident_pattern = re.compile(
        r"\b(function|variable|event|modifier|struct)\s+"
        r"((?:ignore|override|disregard|forget|pretend|bypass|skip|mark|set|change|force)"
        r"[A-Za-z0-9_]*)",
        re.IGNORECASE,
    )
    lines = _extract_lines(source)
    for lineno, line in enumerate(lines, 1):
        for m in ident_pattern.finditer(line):
            matches.append(InjectionMatch(
                pattern="identifier",
                location=f"line {lineno}",
                snippet=_truncate(m.group(0)),
                confidence="high",
            ))
    return matches


def _detect_natspec_injection(source: str) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    lines = _extract_lines(source)
    in_natspec_block = False
    for lineno, line in enumerate(lines, 1):
        if _NATSPEC_BLOCK_START.search(line) and "*/" not in line:
            in_natspec_block = True
            continue
        if in_natspec_block:
            if "*/" in line:
                in_natspec_block = False
                segment = line[: line.index("*/")]
            else:
                segment = line
            if _INSTRUCTION_VERBS.search(segment):
                matches.append(InjectionMatch(
                    pattern="NatSpec",
                    location=f"line {lineno}",
                    snippet=_truncate(segment),
                    confidence="high",
                ))
            continue
        if _NATSPEC_LINE.search(line):
            comment_text = line[line.index("///") + 3:]
            if _INSTRUCTION_VERBS.search(comment_text):
                matches.append(InjectionMatch(
                    pattern="NatSpec",
                    location=f"line {lineno}",
                    snippet=_truncate(comment_text),
                    confidence="high",
                ))
    return matches


def _detect_import_injection(source: str) -> list[InjectionMatch]:
    matches: list[InjectionMatch] = []
    lines = _extract_lines(source)
    for lineno, line in enumerate(lines, 1):
        if _URL_IMPORT.search(line):
            matches.append(InjectionMatch(
                pattern="import",
                location=f"line {lineno}",
                snippet=_truncate(line),
                confidence="high",
            ))
    return matches


def detect_injections(source: str) -> list[InjectionMatch]:
    """
    Detect 8 known prompt-injection patterns in Solidity source.

    Runs on the ORIGINAL source (before comment stripping).
    Returns a list of InjectionMatch — never blocks, never alters the pipeline.

    The 8 patterns:
      1. comment — instruction-like phrases in comments
      2. string — instruction-like phrases in string literals
      3. role-swap — text addressing the LLM by role name
      4. extraction — keywords attempting to extract system prompts
      5. identifier — function/variable names with instruction-like phrases
      6. NatSpec — NatSpec tags with instruction-like phrases
      7. multi — 2+ distinct patterns detected (meta-pattern)
      8. import — import paths containing URLs
    """
    all_matches: list[InjectionMatch] = []
    all_matches.extend(_detect_comment_injection(source))
    all_matches.extend(_detect_string_injection(source))
    all_matches.extend(_detect_role_swap(source))
    all_matches.extend(_detect_extraction(source))
    all_matches.extend(_detect_identifier_injection(source))
    all_matches.extend(_detect_natspec_injection(source))
    all_matches.extend(_detect_import_injection(source))

    distinct_patterns = {m.pattern for m in all_matches}
    if len(distinct_patterns) >= 2:
        all_matches.append(InjectionMatch(
            pattern="multi",
            location="contract-level",
            snippet=f"{len(distinct_patterns)} distinct patterns: {', '.join(sorted(distinct_patterns))}",
            confidence="high",
        ))

    return all_matches
