"""Deduplicator — three-level dedup: exact SHA-256 → address-level → text-normalized hash.

The BCCC dataset had 38.8% duplication, mostly from the same contract appearing in
multiple class folders with minor edits. The 0.85 threshold catches copy-paste-with-edits.

Level 3 detects contracts that are identical after stripping comments and collapsing
whitespace. This catches copy-paste-with-comment-edits near-dups that Level 1 misses.
Identifier lowercasing is intentionally NOT applied — it would collapse semantically
distinct function names (e.g. reentrantWithdraw ≠ reentrancyWithdraw).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_ADDRESS_RE  = re.compile(r'0x[0-9a-fA-F]{40}')
_BLOCK_CMT   = re.compile(r'/\*.*?\*/', re.DOTALL)
_LINE_CMT    = re.compile(r'//[^\n]*')
_WHITESPACE  = re.compile(r'\s+')


@dataclass
class DedupRecord:
    """Outcome of deduplication for a single file."""

    sha256: str
    dedup_group_id: str   # canonical representative sha256 for this dedup group
    is_duplicate: bool
    duplicate_of: str     # sha256 of the canonical representative, or "" if canonical


def _normalize_for_dedup(content: str) -> str:
    """Strip comments and collapse all whitespace for Level-3 dedup hash.

    Intentionally does NOT lowercase identifiers — that would create false-positive
    groups across different vulnerability classes (e.g. Reentrancy vs ReentrancyGuard).
    """
    out = _BLOCK_CMT.sub('', content)
    out = _LINE_CMT.sub('', out)
    out = _WHITESPACE.sub(' ', out).strip()
    return out


class Deduplicator:
    """Stateful deduplicator — call process() for each file in sequence."""

    def __init__(self):
        # sha256 → canonical path
        self._seen_sha: dict[str, Path] = {}
        # ethereum address → first sha256 that had this address
        self._seen_addr: dict[str, str] = {}
        # normalized-text hash → first sha256 with that normalized content
        self._seen_norm: dict[str, str] = {}

    def process(self, content: str, path: Path) -> DedupRecord:
        """Check content against seen SHA-256 hashes, Ethereum addresses, and normalized text.

        Returns a DedupRecord indicating whether this file is a duplicate.
        """

        sha = _sha256(content)

        # Level 1: exact SHA-256 match
        if sha in self._seen_sha:
            return DedupRecord(
                sha256=sha,
                dedup_group_id=sha,
                is_duplicate=True,
                duplicate_of=sha,
            )
        self._seen_sha[sha] = path

        # Level 2: address-level — same Ethereum address in two different files
        addrs = set(_ADDRESS_RE.findall(content))
        for addr in addrs:
            if addr in self._seen_addr:
                canonical_sha = self._seen_addr[addr]
                return DedupRecord(
                    sha256=sha,
                    dedup_group_id=canonical_sha,
                    is_duplicate=True,
                    duplicate_of=canonical_sha,
                )
        for addr in addrs:
            self._seen_addr[addr] = sha

        # Level 3: text-normalized hash — catches copy-paste-with-comment-edits near-dups
        norm_hash = _sha256(_normalize_for_dedup(content))
        if norm_hash in self._seen_norm:
            canonical_sha = self._seen_norm[norm_hash]
            return DedupRecord(
                sha256=sha,
                dedup_group_id=canonical_sha,
                is_duplicate=True,
                duplicate_of=canonical_sha,
            )
        self._seen_norm[norm_hash] = sha

        return DedupRecord(
            sha256=sha,
            dedup_group_id=sha,
            is_duplicate=False,
            duplicate_of="",
        )


def _sha256(content: str) -> str:
    """Compute hex-encoded SHA-256 of a string."""

    return hashlib.sha256(content.encode()).hexdigest()
