"""Solidity-safe source normalization primitives.

Historical preprocessing used regular expressions for comment removal.  That
can corrupt valid Solidity when ``//`` or ``/*`` appears inside a string.  The
comment scanner is now lexical for every caller.  R4 repaired preprocessing
additionally opts into line-preserving mode so graph/site provenance is not
shifted by blank-line compaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NORMALIZER_VERSION = "r4-lexical-v2"
_TRAIL_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTI_NL = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class NormalizeResult:
    """Output of normalization with auditable line-count metadata."""

    content: str
    n_lines_before: int
    n_lines_after: int
    normalizer_version: str = NORMALIZER_VERSION
    comment_characters_removed: int = 0
    line_structure_preserved: bool = False


def strip_comments_lexically(source: str) -> tuple[str, int]:
    """Remove Solidity comments without interpreting markers inside strings.

    Non-newline comment characters are replaced with spaces.  Newlines inside
    comments are always preserved, escaped quote/backslash sequences inside
    strings remain exact, and adjacent code tokens cannot be fused by deletion.
    """

    CODE, SINGLE, DOUBLE, LINE_COMMENT, BLOCK_COMMENT = range(5)
    state = CODE
    out: list[str] = []
    removed = 0
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if state == CODE:
            if ch == "/" and nxt == "/":
                out.extend((" ", " "))
                removed += 2
                i += 2
                state = LINE_COMMENT
                continue
            if ch == "/" and nxt == "*":
                out.extend((" ", " "))
                removed += 2
                i += 2
                state = BLOCK_COMMENT
                continue
            out.append(ch)
            if ch == "'":
                state = SINGLE
            elif ch == '"':
                state = DOUBLE
            i += 1
            continue

        if state in (SINGLE, DOUBLE):
            quote = "'" if state == SINGLE else '"'
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(source[i + 1])
                i += 2
                continue
            if ch == quote:
                state = CODE
            i += 1
            continue

        if state == LINE_COMMENT:
            if ch == "\n":
                out.append("\n")
                state = CODE
            else:
                out.append(" ")
                removed += 1
            i += 1
            continue

        if ch == "*" and nxt == "/":
            out.extend((" ", " "))
            removed += 2
            i += 2
            state = CODE
            continue
        if ch == "\n":
            out.append("\n")
        else:
            out.append(" ")
            removed += 1
        i += 1

    return "".join(out), removed


def normalize(source: str, *, preserve_line_structure: bool = False) -> NormalizeResult:
    """Return deterministic, Solidity-safe normalized text.

    ``preserve_line_structure=False`` retains the historical blank-line
    compaction contract for compatibility while using the safe lexical comment
    scanner.  R4 repaired preprocessing MUST pass ``True`` so source locations
    remain stable through promotion.
    """

    n_before = source.count("\n") + 1
    out, removed = strip_comments_lexically(source)
    out = _TRAIL_WS.sub("", out)
    if not preserve_line_structure:
        out = _MULTI_NL.sub("\n\n", out)
        out = out.strip()
    if out and not out.endswith("\n"):
        out += "\n"

    return NormalizeResult(
        content=out,
        n_lines_before=n_before,
        n_lines_after=out.count("\n") + 1,
        comment_characters_removed=removed,
        line_structure_preserved=preserve_line_structure,
    )
