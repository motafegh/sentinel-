"""
security/comment_strip.py — Layer 1: Solidity comment removal (P4).

State-machine scanner that removes Solidity comments while preserving line count.
Handles:
  - Line comments: // ...
  - Block comments: /* ... */
  - NatSpec line comments: /// ...
  - NatSpec block comments: /** ... */

Preserves string literals (does NOT strip comments inside strings).
Preserves line count (replaces comments with spaces/newlines).
"""

from __future__ import annotations

import enum


class _State(enum.Enum):
    CODE = "CODE"
    LINE_COMMENT = "LINE_COMMENT"
    BLOCK_COMMENT = "BLOCK_COMMENT"
    STRING_DOUBLE = "STRING_DOUBLE"
    STRING_SINGLE = "STRING_SINGLE"


def strip_comments(source: str) -> str:
    """
    Remove Solidity comments from source code.

    Returns sanitized source with comments replaced by whitespace.
    Line count is preserved (newlines inside comments are kept).
    String literals are preserved (comments inside strings are not stripped).

    Args:
        source: Solidity source code

    Returns:
        Source with comments removed, line count preserved
    """
    if not source:
        return ""

    output: list[str] = []
    state = _State.CODE
    i = 0
    n = len(source)

    while i < n:
        c = source[i]
        next_c = source[i + 1] if i + 1 < n else ""

        if state == _State.CODE:
            if c == "/" and next_c == "/":
                state = _State.LINE_COMMENT
                output.append(" ")
                output.append(" ")
                i += 2
                continue
            elif c == "/" and next_c == "*":
                state = _State.BLOCK_COMMENT
                output.append(" ")
                output.append(" ")
                i += 2
                continue
            elif c == '"':
                state = _State.STRING_DOUBLE
                output.append(c)
                i += 1
                continue
            elif c == "'":
                state = _State.STRING_SINGLE
                output.append(c)
                i += 1
                continue
            else:
                output.append(c)
                i += 1
                continue

        elif state == _State.LINE_COMMENT:
            if c == "\n":
                output.append("\n")
                state = _State.CODE
                i += 1
                continue
            else:
                output.append(" ")
                i += 1
                continue

        elif state == _State.BLOCK_COMMENT:
            if c == "*" and next_c == "/":
                output.append(" ")
                output.append(" ")
                state = _State.CODE
                i += 2
                continue
            elif c == "\n":
                output.append("\n")
                i += 1
                continue
            else:
                output.append(" ")
                i += 1
                continue

        elif state == _State.STRING_DOUBLE:
            if c == "\\" and next_c == '"':
                output.append(c)
                output.append(next_c)
                i += 2
                continue
            elif c == '"':
                output.append(c)
                state = _State.CODE
                i += 1
                continue
            else:
                output.append(c)
                i += 1
                continue

        elif state == _State.STRING_SINGLE:
            if c == "\\" and next_c == "'":
                output.append(c)
                output.append(next_c)
                i += 2
                continue
            elif c == "'":
                output.append(c)
                state = _State.CODE
                i += 1
                continue
            else:
                output.append(c)
                i += 1
                continue

    return "".join(output)
