"""Segmenter + VersionBucketer — split multi-contract files and tag Solidity era.

Version buckets:
  legacy       < 0.6  (pre-SafeMath era; integer overflow is implicit)
  transitional 0.6–0.7
  modern       >= 0.8 (SafeMath by default; `unchecked{}` exists)

Also records has_unchecked_block for 0.8+ files (relevant for IntegerUO detection
in Stage 4; this was feat[11] / in_unchecked_block in the v9 schema).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_CONTRACT_RE  = re.compile(r'\bcontract\s+(\w+)\s*(?:is\s+[^{]+)?\{', re.MULTILINE)
_PRAGMA_VER   = re.compile(r'(\d+)\.(\d+)\.(\d+)')
_UNCHECKED_RE = re.compile(r'\bunchecked\s*\{')


@dataclass
class Segment:
    """Segmentation result: contract names, version bucket, and unchecked-block flag."""

    contract_name: str
    source: str              # the full file source (we keep the whole file per contract unit)
    version_bucket: str      # "legacy" | "transitional" | "modern"
    has_unchecked_block: bool
    pragma_raw: str
    contract_names: list[str] = field(default_factory=list)  # all contracts in the file


def segment_and_bucket(source: str, pragma_raw: str) -> Segment:
    """Return one Segment representing the file.

    We treat multi-contract files as one unit (the full file) — splitting them
    into per-contract files would break import chains. Instead we record all
    contract names found so Stage 4 can match labels by contract name.
    """
    names = _CONTRACT_RE.findall(source)
    primary = names[0] if names else "Unknown"
    bucket = _bucket(pragma_raw)
    has_unchecked = bool(_UNCHECKED_RE.search(source))

    return Segment(
        contract_name=primary,
        source=source,
        version_bucket=bucket,
        has_unchecked_block=has_unchecked,
        pragma_raw=pragma_raw,
        contract_names=names,
    )


def _bucket(pragma_raw: str) -> str:
    """Map a pragma version string to a Solidity era bucket."""

    m = _PRAGMA_VER.search(pragma_raw)
    if not m:
        return "legacy"
    major, minor = int(m.group(1)), int(m.group(2))
    if major == 0 and minor < 6:
        return "legacy"
    if major == 0 and minor < 8:
        return "transitional"
    return "modern"
