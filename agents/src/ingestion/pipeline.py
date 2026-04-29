"""
pipeline.py

Orchestrates the full RAG ingestion pipeline.
Called by all three schedulers: cron, Dagster, GitHub Actions.

RECALL — Pipeline vs build_index.py:
  build_index.py:  builds index from scratch (first time only)
  pipeline.py:     incremental updates (every scheduled run)

  build_index.py:  always re-embeds everything
  pipeline.py:     only embeds NEW documents (deduplication)

CHANGES (2026-04-11):
  FIX-23: Paths anchored to __file__ — no longer CWD-dependent.
  FIX-9:  BM25Okapi import moved to module level — fail-fast if missing.
  FIX-8:  FileLock guards all index writes — concurrent runs can't corrupt.
  FIX-7:  Atomic writes via temp-file + rename — crash-safe index updates.

CHANGES (2026-04-29):
  FIX-BugA: tmp_faiss.rename(FAISS_PATH) → tmp_faiss.replace(FAISS_PATH).
            .rename() raises FileExistsError on Windows/WSL2 when the
            destination already exists. .replace() is POSIX-atomic on Linux
            AND silently overwrites on Windows — correct on both platforms.
            Same fix applied to feedback_loop.py (cross-project issue).

Run from agents/ directory:
  poetry run python -m src.ingestion.pipeline
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from filelock import FileLock, Timeout
from loguru import logger
from rank_bm25 import BM25Okapi

from ..rag.fetchers.base_fetcher import BaseFetcher, Document
from ..rag.fetchers.github_fetcher import DeFiHackLabsFetcher
from ..rag.chunker import Chunk, Chunker
from ..rag.embedder import Embedder
from .deduplicator import Deduplicator

# ── Paths ─────────────────────────────────────────────────────────────────────
_AGENTS_DIR       = Path(__file__).parent.parent.parent

INDEX_DIR         = _AGENTS_DIR / "data" / "index"
FAISS_PATH        = INDEX_DIR / "faiss.index"
BM25_PATH         = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH       = INDEX_DIR / "chunks.pkl"
METADATA_PATH     = INDEX_DIR / "index_metadata.json"
SEEN_HASHES_PATH  = INDEX_DIR / "seen_hashes.json"

DEFIHACKLABS_DIR  = _AGENTS_DIR / "data" / "defihacklabs"
EXPLOITS_DIR      = _AGENTS_DIR / "data" / "exploits"

INDEX_LOCK_PATH    = INDEX_DIR / ".index.lock"
INDEX_LOCK_TIMEOUT = 300


def _atomic_write_binary(path: Path, write_fn) -> None:
    """
    FIX-7: Write a binary file atomically.

    Writes to a .tmp sibling first, then replaces the real path.
    On POSIX (Linux/WSL2), Path.replace() is atomic — no reader ever sees a
    partially-written file. On Windows, .replace() silently overwrites the
    destination (unlike .rename() which raises FileExistsError).
    On crash mid-write, only the .tmp is orphaned; the real file is intact.
    """
    tmp = path.with_suffix(".tmp")
    try:
        write_fn(tmp)
        tmp.replace(path)   # FIX-BugA: was .rename() — fails on Windows when dest exists
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class IngestionPipeline:
    """
    Incremental RAG ingestion pipeline.

    Open/Closed principle:
      Open for extension — pass new fetchers at init time.
      Closed for modification — pipeline code unchanged per source.
    """

    def __init__(self, fetchers: Optional[list[BaseFetcher]] = None):
        self.fetchers = fetchers or [
            DeFiHackLabsFetcher(
                repo_path=DEFIHACKLABS_DIR,
                data_dir=EXPLOITS_DIR,
            )
        ]

        self.chunker      = Chunker()
        self.embedder     = Embedder(batch_size=32)
        self.deduplicator = Deduplicator(SEEN_HASHES_PATH)

        logger.info(
            f"Pipeline initialised — "
            f"{len(self.fetchers)} fetcher(s) | "
            f"{self.deduplicator.total_indexed} documents already indexed"
        )

    def run(self) -> dict:
        """
        Run the full incremental ingestion pipeline.

        Returns:
            dict with run statistics:
              fetched, new_docs, new_chunks, new_vectors, skipped, errors, duration_sec
        """
        logger.info("=" * 60)
        logger.info("SENTINEL INGESTION PIPELINE — START")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 60)

        run_start = time.time()
        stats = {
            "fetched":      0,
            "new_docs":     0,
            "new_chunks":   0,
            "new_vectors":  0,
            "skipped":      0,
            "errors":       [],
            "duration_sec": 0,
        }

        # ── Step 1: Fetch ────────────────────────────────────────────────
        all_documents: list[Document] = []

        for fetcher in self.fetchers:
            try:
                logger.info(f"Fetching from {fetcher.source_name}...")
                if not fetcher.health_check():
                    logger.warning(f"{fetcher.source_name} health check failed — skipping")
                    stats["errors"].append(f"{fetcher.source_name}: health check failed")
                    continue

                docs = fetcher.fetch()
                all_documents.extend(docs)
                logger.info(f"  {fetcher.source_name}: {len(docs)} documents fetched")

            except Exception as e:
                logger.error(f"{fetcher.source_name} failed: {e}")
                stats["errors"].append(f"{fetcher.source_name}: {str(e)}")

        stats["fetched"] = len(all_documents)

        if not all_documents:
            logger.warning("No documents fetched — pipeline complete (nothing to do)")
            return stats

        # ── Step 2: Deduplicate ──────────────────────────────────────────
        new_documents     = self.deduplicator.filter_new(all_documents)
        stats["skipped"]  = len(all_documents) - len(new_documents)
        stats["new_docs"] = len(new_documents)

        if not new_documents:
            logger.info("No new documents — index is already up to date")
            stats["duration_sec"] = round(time.time() - run_start, 1)
            return stats

        logger.info(f"New documents to index: {len(new_documents)}")

        # ── Step 3: Chunk ────────────────────────────────────────────────
        logger.info("Chunking new documents...")
        new_chunks          = self.chunker.chunk_documents(new_documents)
        stats["new_chunks"] = len(new_chunks)

        # ── Step 4: Embed ────────────────────────────────────────────────
        logger.info("Embedding new chunks...")
        new_vectors         = self.embedder.embed_chunks(new_chunks)
        new_vectors_np      = np.array(new_vectors, dtype=np.float32)
        stats["new_vectors"] = len(new_vectors)

        # ── Steps 5-8: Write index (locked + atomic) ─────────────────────
        try:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            with FileLock(str(INDEX_LOCK_PATH), timeout=INDEX_LOCK_TIMEOUT):
                self._write_index(new_chunks, new_vectors_np, new_documents)
        except Timeout:
            raise RuntimeError(
                f"Could not acquire index lock after {INDEX_LOCK_TIMEOUT}s. "
                f"Another pipeline process may be hung. "
                f"Delete {INDEX_LOCK_PATH} to reset."
            )

        # ── Step 9: Update metadata ──────────────────────────────────────
        duration            = round(time.time() - run_start, 1)
        stats["duration_sec"] = duration

        try:
            with open(CHUNKS_PATH, "rb") as f:
                total_chunks = len(pickle.load(f))
        except Exception:
            total_chunks = stats["new_chunks"]

        metadata = {
            "last_run":         datetime.now().isoformat(),
            "total_chunks":     total_chunks,
            "total_documents":  self.deduplicator.total_indexed,
            "vector_dimension": int(new_vectors_np.shape[1]),
            "sources":          [f.source_name for f in self.fetchers],
            "embedding_model":  "text-embedding-nomic-embed-text-v1.5",
            "chunk_size":       self.chunker.chunk_size,
            "chunk_overlap":    self.chunker.chunk_overlap,
            "faiss_type":       "IndexFlatL2",
            "last_run_stats":   stats,
        }

        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Fetched:      {stats['fetched']} documents")
        logger.info(f"  Skipped:      {stats['skipped']} (already indexed)")
        logger.info(f"  New docs:     {stats['new_docs']}")
        logger.info(f"  New chunks:   {stats['new_chunks']}")
        logger.info(f"  New vectors:  {stats['new_vectors']}")
        logger.info(f"  Duration:     {duration}s")
        if stats["errors"]:
            logger.warning(f"  Errors:       {stats['errors']}")
        logger.info("=" * 60)

        return stats

    def _write_index(
        self,
        new_chunks: list[Chunk],
        new_vectors_np: np.ndarray,
        new_documents: list[Document],
    ) -> None:
        """
        Write FAISS / chunks.pkl / bm25.pkl under the held FileLock.

        FIX-7:    Every write goes to .tmp first, then replaces atomically.
        FIX-8:    Only called from inside FileLock context.
        FIX-BugA: _atomic_write_binary now uses .replace() — cross-platform safe.
        """
        # ── Step 5: FAISS ────────────────────────────────────────────────
        logger.info("Updating FAISS index...")
        if FAISS_PATH.exists():
            index     = faiss.read_index(str(FAISS_PATH))
            old_count = index.ntotal
            index.add(new_vectors_np)
            logger.info(f"FAISS: {old_count} → {index.ntotal} vectors")
        else:
            dimension = new_vectors_np.shape[1]
            index     = faiss.IndexFlatL2(dimension)
            index.add(new_vectors_np)
            logger.info(f"FAISS: built from scratch — {index.ntotal} vectors")

        tmp_faiss = FAISS_PATH.with_suffix(".tmp")
        faiss.write_index(index, str(tmp_faiss))
        tmp_faiss.replace(FAISS_PATH)   # FIX-BugA: was .rename(), fails on Windows

        # ── Step 6: Chunks ───────────────────────────────────────────────
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, "rb") as f:
                all_chunks: list[Chunk] = pickle.load(f)
        else:
            all_chunks = []

        prev_len = len(all_chunks)
        all_chunks.extend(new_chunks)

        def _write_chunks(tmp: Path) -> None:
            with open(tmp, "wb") as f:
                pickle.dump(all_chunks, f)

        _atomic_write_binary(CHUNKS_PATH, _write_chunks)
        logger.info(f"Chunks: {prev_len} → {len(all_chunks)}")

        # ── Step 7: BM25 ─────────────────────────────────────────────────
        logger.info("Rebuilding BM25 index...")
        corpus = [chunk.content.lower().split() for chunk in all_chunks]
        bm25   = BM25Okapi(corpus)

        def _write_bm25(tmp: Path) -> None:
            with open(tmp, "wb") as f:
                pickle.dump(bm25, f)

        _atomic_write_binary(BM25_PATH, _write_bm25)
        logger.info(f"BM25 rebuilt — {len(corpus)} documents")

        # ── Step 8: Mark seen ────────────────────────────────────────────
        self.deduplicator.mark_seen([doc.doc_id for doc in new_documents])


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    result   = pipeline.run()
    logger.info(f"Run result: {result}")
