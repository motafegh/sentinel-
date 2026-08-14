"""Deterministic duplicate identity for repaired DATA builds.

Historical preprocessing treated a shared Ethereum address literal as proof
that two files were duplicates and deleted the later record.  The Phase-8
real-data audit proved that this erased content-distinct positive contracts.
A shared address is now provenance/family *evidence only*; it is never a
deletion criterion.

Duplicate identity is intentionally split into:

* exact text identity (SHA-256 of the input text), and
* normalized-code identity (comments removed lexically, code whitespace
  canonicalized without changing string literals).

Leakage-family construction is a later, explicit stage.  Address literals are
returned as signals so that stage can inspect them without conflating them with
identity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sentinel_data.preprocessing.normalizer import strip_comments_lexically

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class DedupRecord:
    """Identity/deduplication result for one source record."""

    sha256: str
    normalized_sha256: str
    dedup_group_id: str
    is_duplicate: bool
    duplicate_of: str
    duplicate_kind: str  # "" | "exact" | "normalized"
    address_literals: tuple[str, ...] = ()


def _canonicalize_code_whitespace(source: str) -> str:
    """Collapse code whitespace while preserving string literal contents.

    Comments are first removed by the lexical normalizer primitive.  Whitespace
    outside quoted strings is collapsed to one ASCII space; whitespace and
    escapes inside strings remain exact.  This lets formatting/comment-only
    variants share a normalized-code identity without corrupting URLs or other
    string data.
    """

    stripped, _ = strip_comments_lexically(source)
    CODE, SINGLE, DOUBLE = range(3)
    state = CODE
    out: list[str] = []
    pending_space = False
    i = 0

    while i < len(stripped):
        ch = stripped[i]
        if state == CODE:
            if ch.isspace():
                pending_space = bool(out)
                i += 1
                continue
            if pending_space and out and out[-1] != " ":
                out.append(" ")
            pending_space = False
            out.append(ch)
            if ch == "'":
                state = SINGLE
            elif ch == '"':
                state = DOUBLE
            i += 1
            continue

        quote = "'" if state == SINGLE else '"'
        out.append(ch)
        if ch == "\\" and i + 1 < len(stripped):
            out.append(stripped[i + 1])
            i += 2
            continue
        if ch == quote:
            state = CODE
        i += 1

    return "".join(out).strip()


def normalized_code_sha256(content: str) -> str:
    """Return the repaired normalized-code identity hash."""

    return _sha256(_canonicalize_code_whitespace(content))


class Deduplicator:
    """Stateful exact/normalized deduplicator with no address deletion."""

    def __init__(self) -> None:
        self._seen_sha: dict[str, Path] = {}
        self._seen_norm: dict[str, str] = {}

    def process(self, content: str, path: Path) -> DedupRecord:
        """Classify duplicate identity for ``content``.

        Exact and normalized duplicates collapse to one content identity.  Shared
        addresses are reported but never set ``is_duplicate=True``.
        """

        sha = _sha256(content)
        norm_hash = normalized_code_sha256(content)
        addresses = tuple(sorted({a.lower() for a in _ADDRESS_RE.findall(content)}))

        if sha in self._seen_sha:
            return DedupRecord(
                sha256=sha,
                normalized_sha256=norm_hash,
                dedup_group_id=sha,
                is_duplicate=True,
                duplicate_of=sha,
                duplicate_kind="exact",
                address_literals=addresses,
            )

        if norm_hash in self._seen_norm:
            canonical_sha = self._seen_norm[norm_hash]
            self._seen_sha[sha] = path
            return DedupRecord(
                sha256=sha,
                normalized_sha256=norm_hash,
                dedup_group_id=canonical_sha,
                is_duplicate=True,
                duplicate_of=canonical_sha,
                duplicate_kind="normalized",
                address_literals=addresses,
            )

        self._seen_sha[sha] = path
        self._seen_norm[norm_hash] = sha
        return DedupRecord(
            sha256=sha,
            normalized_sha256=norm_hash,
            dedup_group_id=sha,
            is_duplicate=False,
            duplicate_of="",
            duplicate_kind="",
            address_literals=addresses,
        )


def _sha256(content: str) -> str:
    """Compute UTF-8 SHA-256 for a Python string."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()
