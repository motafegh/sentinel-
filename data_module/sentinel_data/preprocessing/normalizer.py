"""Solidity-safe source normalization for repaired DATA builds.

Historical preprocessing used regular expressions for comment removal.  That
can corrupt valid Solidity when ``//`` or ``/*`` appears inside a string and it
can shift source locations by deleting block-comment newlines.  R4 repaired
preprocessing therefore uses a small lexical state machine and preserves line
structure.

This module is deliberately dependency-free so the same lexical primitive can
be reused by deduplication and repository-safe tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NORMALIZER_VERSION = "r4-lexical-v2"
_TRAIL_WS = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass(frozen=True)
class NormalizeResult:
    """Output of normalization with auditable line-count metadata."""

    content: str
    n_lines_before: int
    n_lines_after: int
    normalizer_version: str = NORMALIZER_VERSION
    comment_characters_removed: int = 0


def strip_comments_lexically(source: str) -> tuple[str, int]:
    """Remove Solidity comments without interpreting comment markers in strings.

    Non-newline comment characters are replaced with spaces rather than simply
    deleted.  This prevents adjacent tokens from being accidentally joined and
    preserves every newline inside both line and block comments.  Solidity does
    not define nested block comments, so the first ``*/`` closes a block comment.

    Single- and double-quoted strings are preserved byte-for-character at the
    Python ``str`` level, including escaped quote and backslash sequences.
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
                # Preserve the escaped character and do not let it terminate the
                # current string state.
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

        # BLOCK_COMMENT
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


def normalize(source: str) -> NormalizeResult:
    """Return deterministic, Solidity-safe normalized text.

    R4 v2 normalization intentionally does *not* collapse blank lines or strip
    leading/trailing source lines.  Source-line stability is part of provenance
    and later graph/site alignment.  The only whitespace normalization is
    trailing horizontal whitespace.  A missing final newline is added for a
    deterministic text-file boundary.
    """

    n_before = source.count("\n") + 1
    out, removed = strip_comments_lexically(source)
    out = _TRAIL_WS.sub("", out)
    if out and not out.endswith("\n"):
        out += "\n"

    return NormalizeResult(
        content=out,
        n_lines_before=n_before,
        n_lines_after=out.count("\n") + 1,
        comment_characters_removed=removed,
    )
