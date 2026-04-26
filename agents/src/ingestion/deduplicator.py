"""
deduplicator.py

Tracks which documents have already been indexed.
Prevents re-embedding documents on every pipeline run.

RECALL — Why deduplication matters:
  Without it: every run re-indexes all 726 documents → 12s embedding always.
  With it: only new documents get embedded → 4 new exploits = ~0.5s.

CHANGES (2026-04-11):
  FIX-25: filter_new() and mark_seen() type hints tightened.
          filter_new() previously typed as (list) -> list — bare list
          accepted any iterable and mypy/pyright couldn't catch passing
          the wrong type from the two call sites in pipeline.py and
          feedback_loop.py. Now documented precisely; Document imported
          under TYPE_CHECKING to avoid circular import at runtime.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger

# TYPE_CHECKING block: imported only by static analysis tools (mypy, pyright),
# never at runtime — avoids the circular import that would result from
# deduplicator.py → base_fetcher.py → (nothing) vs
# pipeline.py → deduplicator.py AND pipeline.py → base_fetcher.py.
if TYPE_CHECKING:
    from ..rag.fetchers.base_fetcher import Document


class Deduplicator:
    """
    Tracks indexed document IDs in a JSON file.

    Interface:
      seen(doc_id)        → bool
      filter_new(docs)    → new docs only       (list[Document] → list[Document])
      mark_seen(doc_ids)  → None                (list[str] → None)

    RECALL — doc_id is SHA256(file_path)[:16] from BaseFetcher.
    Stable across content edits, deterministic across runs.

    Design choice: JSON not database.
    Simple, human-readable, git-trackable.
    For 100K+ documents: upgrade to SQLite (same interface, different _load/_save).
    """

    def __init__(self, seen_hashes_path: Path):
        self.path    = seen_hashes_path
        self._hashes: dict[str, str] = self._load()
        logger.debug(f"Deduplicator loaded — {len(self._hashes)} documents already indexed")

    def _load(self) -> dict[str, str]:
        """Load seen hashes from disk. Returns empty dict if file missing."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load seen hashes: {e} — starting fresh")
            return {}

    def _save(self) -> None:
        """Persist current hashes to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._hashes, f, indent=2)

    def seen(self, doc_id: str) -> bool:
        """Return True if this document has already been indexed."""
        return doc_id in self._hashes

    def filter_new(self, documents: "list[Document]") -> "list[Document]":
        """
        Return only documents not yet indexed.

        FIX-25: Properly typed as list[Document] → list[Document].
                The TYPE_CHECKING guard above makes this safe at runtime
                (no actual import of Document occurs).

        Called by pipeline before chunking/embedding — only new documents
        proceed to the expensive embedding step.

        Args:
            documents: list[Document] from any fetcher

        Returns:
            Subset of documents whose doc_id is not in seen_hashes
        """
        new_docs: list = []
        for doc in documents:
            if self.seen(doc.doc_id):
                # A-23: log each skip at DEBUG so we can trace exactly which documents
                # were dropped. Without this, a mid-run crash loses the skip reason.
                # DEBUG is off by default — enable with: logger.add(..., level="DEBUG")
                logger.debug(
                    "skip | doc_id={} | source={}", doc.doc_id, getattr(doc, "source", "?")
                )
            else:
                new_docs.append(doc)

        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            logger.info(f"Deduplication: {skipped} already indexed, {len(new_docs)} new")
        else:
            logger.info(f"Deduplication: all {len(new_docs)} documents are new")

        return new_docs

    def mark_seen(self, doc_ids: list[str]) -> None:
        """
        Mark documents as indexed. Call AFTER successful index update.

        RECALL — Checkpoint pattern: mark AFTER success, not before.
        If embedding fails mid-batch, those docs are NOT marked seen.
        Next run retries them automatically.

        FIX-25: Typed as list[str] (was untyped in the original).

        Args:
            doc_ids: list of doc_id strings to mark as seen
        """
        timestamp = datetime.now().isoformat()
        for doc_id in doc_ids:
            self._hashes[doc_id] = timestamp
        self._save()
        logger.debug(f"Marked {len(doc_ids)} documents as indexed")

    @property
    def total_indexed(self) -> int:
        """Total number of documents indexed across all runs."""
        return len(self._hashes)
