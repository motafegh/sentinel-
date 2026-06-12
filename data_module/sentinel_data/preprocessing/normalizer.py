"""Normalizer — strip comments, SPDX headers, normalize whitespace."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NormalizeResult:
    """Output of source normalization with line-count metadata."""

    content: str
    n_lines_before: int
    n_lines_after: int


_SPDX_RE    = re.compile(r'^//\s*SPDX-License-Identifier:[^\n]*\n?', re.MULTILINE)
_LINE_CMT   = re.compile(r'//[^\n]*')
_BLOCK_CMT  = re.compile(r'/\*.*?\*/', re.DOTALL)
_MULTI_NL   = re.compile(r'\n{3,}')
_TRAIL_WS   = re.compile(r'[ \t]+$', re.MULTILINE)


def normalize(source: str) -> NormalizeResult:
    """Strip SPDX headers, line/block comments, trailing whitespace, and collapse blank lines."""

    n_before = source.count('\n') + 1
    out = _SPDX_RE.sub('', source)
    out = _BLOCK_CMT.sub('', out)
    out = _LINE_CMT.sub('', out)
    out = _TRAIL_WS.sub('', out)
    out = _MULTI_NL.sub('\n\n', out)
    out = out.strip() + '\n'
    return NormalizeResult(
        content=out,
        n_lines_before=n_before,
        n_lines_after=out.count('\n') + 1,
    )
