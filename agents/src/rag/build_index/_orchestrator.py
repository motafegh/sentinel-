# agents/src/rag/build_index/_orchestrator.py
"""
build_index() — the full RAG index rebuild orchestrator.

Six-step pipeline: Fetch → Chunk → Embed → FAISS → BM25 → Save.

This is the *full rebuild* path (re-embeds every document). Incremental
updates should use `src/ingestion/pipeline.py` instead. Used for:
    - First-time RAG setup
    - Forced rebuild after chunking/embedding/BM25 tokenizer changes
    - Recovery from corrupted index artifacts

Production hardening (carried from the original god-file):
    - Paths anchored to __file__
    - Shared index lock compatible with ingestion pipeline
    - Atomic writes for FAISS, BM25, chunks, metadata, and seen_hashes
    - Rollback snapshot if artifact replacement fails
    - build_id + schema_version + config_hash in metadata
    - Staleness detection checks config, not only document count
    - Artifact SHA256 checksums recorded in metadata

Run from agents/ directory:
    poetry run python -m src.rag.build_index
"""

from __future__ import annotations

import json
import time
from typing import Any

import faiss
import numpy as np
from filelock import FileLock, Timeout
from loguru import logger
from rank_bm25 import BM25Okapi

from ..chunker import Chunker
from ..embedder import Embedder
from ..fetchers.github_fetcher import DeFiHackLabsFetcher

from ._paths import (
    BM25_TOKENIZER_VERSION,
    DEFIHACKLABS_DIR,
    EMBEDDING_MODEL_NAME,
    EXPLOITS_DIR,
    FAISS_TYPE,
    INDEX_DIR,
    INDEX_LOCK_PATH,
    INDEX_LOCK_TIMEOUT,
    INDEX_SCHEMA_VERSION,
)
from ._metadata import (
    _config_hash,
    _expected_config,
    _index_is_current,
    _new_build_id,
    _source_file_count,
    _utc_now_iso,
)
from ._pipeline import (
    _collect_extra_documents,
    _validate_build_outputs,
    _write_artifacts,
)


def build_index(force_rebuild: bool = False) -> dict[str, Any]:
    """
    Build the full RAG index from scratch.

    Args:
        force_rebuild:
            If True, rebuild even when the existing index metadata appears
            current. Direct CLI execution always uses True.

    Returns:
        Metadata dict for the current or newly-built index.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    build_id = _new_build_id()
    chunker_for_config = Chunker()
    expected = _expected_config(chunker_for_config)

    if not force_rebuild:
        is_current, existing_metadata, reason = _index_is_current(expected)
        if is_current and existing_metadata is not None:
            logger.info("Index is current — skipping rebuild ({})", reason)
            return {**existing_metadata, "skipped": True, "skip_reason": reason}

        logger.warning("Index rebuild required: {}", reason)

    logger.info("=" * 60)
    logger.info("SENTINEL RAG INDEX BUILD")
    logger.info("Build ID: {}", build_id)
    logger.info("=" * 60)

    try:
        with FileLock(str(INDEX_LOCK_PATH), timeout=INDEX_LOCK_TIMEOUT):
            # Re-check after acquiring the lock in case another process rebuilt
            # the index while this process waited.
            if not force_rebuild:
                is_current, existing_metadata, reason = _index_is_current(expected)
                if is_current and existing_metadata is not None:
                    logger.info("Index became current while waiting for lock — skipping")
                    return {**existing_metadata, "skipped": True, "skip_reason": reason}

            build_start = time.time()

            # ── Step 1: Fetch ─────────────────────────────────────────────
            logger.info("Step 1/6 — Fetching documents...")
            fetcher = DeFiHackLabsFetcher(
                repo_path=DEFIHACKLABS_DIR,
                data_dir=EXPLOITS_DIR,
            )
            source_count = _source_file_count(fetcher)
            documents = fetcher.fetch()
            logger.info("  Fetched {} documents from {}", len(documents), fetcher.source_name)

            # Phase A (A.5): augment with curated audit/finding corpora.
            extra_docs, extra_sources = _collect_extra_documents()
            if extra_docs:
                documents = documents + extra_docs
                logger.info(
                    "  + {} documents from {} extra corpora: {}",
                    len(extra_docs), len(extra_sources), extra_sources,
                )

            # ── Step 2: Chunk ─────────────────────────────────────────────
            logger.info("Step 2/6 — Chunking documents...")
            chunker = Chunker()
            chunks = chunker.chunk_documents(documents)
            logger.info("  Created {} chunks from {} documents", len(chunks), len(documents))

            # ── Step 3: Embed ─────────────────────────────────────────────
            logger.info("Step 3/6 — Embedding chunks...")
            embedder = Embedder(batch_size=32)
            vectors = embedder.embed_chunks(chunks)
            vectors_np = np.array(vectors, dtype=np.float32)
            dimension = int(vectors_np.shape[1])
            logger.info("  Generated {} vectors of dimension {}", len(vectors), dimension)

            # ── Step 4: Build FAISS ───────────────────────────────────────
            logger.info("Step 4/6 — Building FAISS index...")
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors_np)
            logger.info("  FAISS index built — {} vectors indexed", index.ntotal)

            # ── Step 5: Build BM25 ────────────────────────────────────────
            logger.info("Step 5/6 — Building BM25 index...")
            corpus = [chunk.content.lower().split() for chunk in chunks]
            bm25 = BM25Okapi(corpus)
            logger.info("  BM25 index built — {} documents", len(corpus))

            # ── Validate before writing ───────────────────────────────────
            _validate_build_outputs(
                documents=documents,
                chunks=chunks,
                vectors_np=vectors_np,
                index=index,
            )

            build_time = time.time() - build_start
            built_at = _utc_now_iso()
            config = _expected_config(chunker)
            config_hash = _config_hash(config)

            seen_hashes = {
                doc.doc_id: built_at
                for doc in documents
            }

            metadata: dict[str, Any] = {
                # New production metadata
                "schema_version": INDEX_SCHEMA_VERSION,
                "build_id": build_id,
                "config_hash": config_hash,
                "config": config,
                "source_file_count": source_count,
                "bm25_tokenizer": BM25_TOKENIZER_VERSION,

                # Backward-compatible fields expected by existing tooling
                "built_at": built_at,
                "build_time_sec": round(build_time, 1),
                "total_chunks": len(chunks),
                "total_documents": len(documents),
                "vector_dimension": dimension,
                "sources": [fetcher.source_name] + extra_sources,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "chunk_size": chunker.chunk_size,
                "chunk_overlap": chunker.chunk_overlap,
                "faiss_type": FAISS_TYPE,

                # Validation snapshot
                "validation": {
                    "faiss_vectors": int(index.ntotal),
                    "chunks_count": len(chunks),
                    "vectors_count": int(vectors_np.shape[0]),
                    "bm25_documents": len(corpus),
                    "seen_hashes_count": len(seen_hashes),
                },
            }

            # ── Step 6: Save artifacts ────────────────────────────────────
            logger.info("Step 6/6 — Saving index artifacts...")
            metadata = _write_artifacts(
                build_id=build_id,
                index=index,
                bm25=bm25,
                chunks=chunks,
                metadata=metadata,
                seen_hashes=seen_hashes,
            )

            logger.info("=" * 60)
            logger.info("INDEX BUILD COMPLETE")
            logger.info("  Build ID:    {}", build_id)
            logger.info("  Documents:   {}", len(documents))
            logger.info("  Chunks:      {}", len(chunks))
            logger.info("  Vectors:     {} × {}d", len(vectors), dimension)
            logger.info("  Chunk size:  {}", chunker.chunk_size)
            logger.info("  Overlap:     {}", chunker.chunk_overlap)
            logger.info("  Build time:  {:.1f}s", build_time)
            logger.info("  Index dir:   {}", INDEX_DIR.absolute())
            logger.info("=" * 60)

            return metadata

    except Timeout as exc:
        raise RuntimeError(
            f"Could not acquire index lock after {INDEX_LOCK_TIMEOUT}s. "
            f"Another process may be writing the RAG index. "
            f"Lock path: {INDEX_LOCK_PATH}"
        ) from exc