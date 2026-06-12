"""Deduplicator — three-level dedup: exact SHA-256 → address-level → AST near-dup (stub).

The BCCC dataset had 38.8% duplication, mostly from the same contract appearing in
multiple class folders with minor edits. The 0.85 threshold catches copy-paste-with-edits.

Level 3 (AST near-dup) is stubbed — requires Slither which is installed in the pipeline
group. Mark files with dedup_group_id=sha256 for now; Stage 2 will add Slither-based
similarity clustering.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_ADDRESS_RE = re.compile(r'0x[0-9a-fA-F]{40}')


@dataclass
class DedupRecord:
    """Outcome of deduplication for a single file."""

    sha256: str
    dedup_group_id: str   # = sha256 in Stage 1; similarity cluster id in Stage 2+
    is_duplicate: bool
    duplicate_of: str     # sha256 of the canonical representative, or "" if canonical


class Deduplicator:
    """Stateful deduplicator — call process() for each file in sequence."""

    def __init__(self):
        # sha256 → canonical path
        self._seen_sha: dict[str, Path] = {}
        # ethereum address → first sha256 that had this address
        self._seen_addr: dict[str, str] = {}

    def process(self, content: str, path: Path) -> DedupRecord:
        """Check content against seen SHA-256 hashes and Ethereum addresses.

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

        # Level 3: AST near-dup — STUB (requires Slither; deferred to Stage 2)
        return DedupRecord(
            sha256=sha,
            dedup_group_id=sha,
            is_duplicate=False,
            duplicate_of="",
        )


def _sha256(content: str) -> str:
    """Compute hex-encoded SHA-256 of a string."""

    return hashlib.sha256(content.encode()).hexdigest()
