"""
retriever.py

Hybrid FAISS + BM25 retriever with Reciprocal Rank Fusion.

RECALL — Why hybrid search:
  FAISS (semantic): finds chunks with similar MEANING
    Good for: "borrowing exploit" finds "flash loan attack"
    Bad for:  exact terms like CVE numbers, tx hashes, addresses

  BM25 (keyword): finds chunks with matching WORDS
    Good for: "Euler Finance 0xc310a0af" exact match
    Bad for:  synonyms, paraphrases, conceptual similarity

  RRF combines both: chunks ranking high in EITHER system get boosted.

CHANGES (2026-04-11):
  FIX-5:  KeyError on metadata['built_at'] after any pipeline.py run.
          pipeline.py writes "last_run"; build_index.py writes "built_at".
          Now uses .get() with fallback — works with both writers.
  FIX-10: FAISS ↔ chunks sync validated on init. If the index is corrupted
          (pipeline crash mid-write), the retriever raises immediately
          instead of silently returning wrong chunks to agents.
  FIX-12: _apply_filters() now logs a warning when results are empty after
          filtering — previously silent, agents got no signal.
  FIX-23: Paths anchored to __file__ — no longer CWD-dependent.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .embedder import Embedder

# ── Paths ─────────────────────────────────────────────────────────────────────
# FIX-23: Absolute paths anchored to this file's location.
# agents/src/rag/retriever.py → parent.parent.parent = agents/
_AGENTS_DIR   = Path(__file__).parent.parent.parent

INDEX_DIR     = _AGENTS_DIR / "data" / "index"
FAISS_PATH    = INDEX_DIR / "faiss.index"
BM25_PATH     = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH   = INDEX_DIR / "chunks.pkl"
METADATA_PATH = INDEX_DIR / "index_metadata.json"

# RRF constant. Higher = smaller score differences between ranks (smoother).
# Lower = rank 1 dominates more. 60 is empirically tuned for this corpus.
RRF_K = 60


class HybridRetriever:
    """
    Hybrid FAISS + BM25 retriever with Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("flash loan attack on lending protocol", k=5)
        for chunk in results:
            print(chunk.content)
            print(chunk.metadata["protocol"])
    """

    def __init__(self):
        """
        Load all index files from disk and validate sync.

        FIX-10: After loading FAISS and chunks, validate that their sizes
                match. A pipeline crash between writing faiss.index and
                chunks.pkl (FIX-7 prevents this going forward, but old
                indexes may be corrupted) would cause FAISS index N to map
                to the wrong chunk — silent wrong audit results.
                Now we detect and raise immediately on startup.
        """
        logger.info("Loading RAG index...")

        if not FAISS_PATH.exists():
            raise FileNotFoundError(
                f"Index not found at {FAISS_PATH}. "
                f"Run: poetry run python -m src.rag.build_index"
            )

        self.faiss_index: faiss.IndexFlatL2 = faiss.read_index(str(FAISS_PATH))
        logger.debug(f"FAISS loaded — {self.faiss_index.ntotal} vectors")

        with open(BM25_PATH, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        logger.debug("BM25 loaded")

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks: list[Chunk] = pickle.load(f)
        logger.debug(f"Chunks loaded — {len(self.chunks)} chunks")

        # FIX-10: Validate FAISS ↔ chunks sync on every startup.
        # FAISS position N must map to self.chunks[N]. If sizes differ,
        # every retrieval result is potentially wrong — fail loud and early.
        if self.faiss_index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index corruption detected: FAISS has {self.faiss_index.ntotal} vectors "
                f"but chunks list has {len(self.chunks)} entries. "
                f"Re-run build_index.py to rebuild from scratch."
            )

        with open(METADATA_PATH) as f:
            self.metadata = json.load(f)

        self.embedder = Embedder()

        # FIX-5: pipeline.py writes "last_run"; build_index.py writes "built_at".
        # Using .get() with fallback so both writers work correctly.
        built = self.metadata.get("built_at") or self.metadata.get("last_run", "unknown")

        logger.info(
            f"RAG index ready — "
            f"{self.faiss_index.ntotal} vectors | "
            f"built/updated: {built[:10]}"
        )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
        faiss_candidates: int = 20,
    ) -> list[Chunk]:
        """
        Hybrid search: FAISS semantic + BM25 keyword → RRF → top k.

        Args:
            query:            natural language search query
            k:                number of results to return
            filters:          optional metadata filters (see _apply_filters)
            faiss_candidates: candidates fetched from each system before merge.
                              Higher = better recall, slower filtering. Default 20.

        Returns:
            List of up to k Chunk objects ranked by relevance.
            May return fewer than k if filters are strict.
        """
        logger.debug(f"Searching: '{query[:60]}' | k={k}")

        # ── Step 1: FAISS semantic search ────────────────────────────────
        query_vector = self.embedder.embed_query(query)
        query_np     = np.array([query_vector], dtype=np.float32)

        _, indices = self.faiss_index.search(query_np, faiss_candidates)  # distances unused; we rank by RRF not raw L2

        faiss_ranked = [
            (int(idx), rank)
            for rank, idx in enumerate(indices[0])
            if idx != -1   # FAISS returns -1 for empty slots
        ]

        # ── Step 2: BM25 keyword search ───────────────────────────────────
        query_tokens = query.lower().split()
        bm25_scores  = self.bm25.get_scores(query_tokens)

        bm25_ranked = [
            (int(idx), rank)
            for rank, idx in enumerate(
                sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
                [:faiss_candidates]
            )
        ]

        # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────
        # RRF formula: score(chunk) = Σ 1/(rank + RRF_K)
        rrf_scores: dict[int, float] = {}

        for chunk_idx, rank in faiss_ranked:
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (rank + RRF_K)

        for chunk_idx, rank in bm25_ranked:
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (rank + RRF_K)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # ── Step 4: Apply metadata filters ────────────────────────────────
        # Post-retrieval filtering: retrieve 20, filter, return k.
        # We don't pre-filter the index (more complex, not needed at this scale).
        results = [self.chunks[i] for i in sorted_indices]

        if filters:
            results = self._apply_filters(results, filters, query)

        final = results[:k]
        logger.debug(f"Found {len(final)} results (from {len(sorted_indices)} candidates)")
        return final

    def _apply_filters(
        self,
        chunks: list[Chunk],
        filters: dict,
        query: str = "",
    ) -> list[Chunk]:
        """
        Filter chunks by metadata fields.

        Supported filters:
            vuln_type:   str  — exact match on vulnerability type
            date_gte:    str  — "YYYY-MM-DD" — exclude older documents
            loss_gte:    int  — minimum loss in USD
            source:      str  — exact match on source name
            has_summary: bool — only chunks from @Summary documents

        FIX-12: Logs a warning when filtering produces zero results.
                Old code was silent — callers had no way to distinguish
                "no matching exploits exist" from "filter was too aggressive".
        """
        filtered = []
        for chunk in chunks:
            meta = chunk.metadata

            if "vuln_type" in filters:
                if meta.get("vuln_type") != filters["vuln_type"]:
                    continue

            if "date_gte" in filters:
                doc_date = meta.get("date", "")
                if doc_date and doc_date < filters["date_gte"]:
                    continue

            if "loss_gte" in filters:
                if (meta.get("loss_usd") or 0) < filters["loss_gte"]:
                    continue

            if "source" in filters:
                if meta.get("source") != filters["source"]:
                    continue

            if "has_summary" in filters:
                if meta.get("has_summary") != filters["has_summary"]:
                    continue

            filtered.append(chunk)

        # FIX-12: Warn when filters eliminate all candidates.
        if not filtered and chunks:
            logger.warning(
                f"_apply_filters returned 0 results from {len(chunks)} candidates. "
                f"Query: '{query[:60]}' | Filters: {filters}. "
                f"Consider relaxing filters or increasing faiss_candidates."
            )

        logger.debug(f"Filtered {len(chunks)} → {len(filtered)} chunks")
        return filtered

    def get_index_info(self) -> dict[str, Any]:  # A-22: was untyped `dict`, now specific
        """Return index metadata for logging and debugging."""
        return {
            **self.metadata,
            "faiss_vectors": self.faiss_index.ntotal,
            "chunks_loaded": len(self.chunks),
        }


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        ("flash loan attack on lending protocol",     None),
        ("reentrancy vulnerability in vault",         {"vuln_type": "reentrancy"}),
        ("access control privilege escalation",       {"loss_gte": 1_000_000}),
    ]

    for query, filters in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query:   {query}")
        logger.info(f"Filters: {filters}")
        logger.info(f"{'='*60}")

        results = retriever.search(query, k=3, filters=filters)

        for i, chunk in enumerate(results):
            logger.info(f"\nResult {i+1}:")
            logger.info(f"  Protocol:  {chunk.metadata.get('protocol', 'unknown')}")
            logger.info(f"  Date:      {chunk.metadata.get('date', 'unknown')}")
            logger.info(f"  Vuln type: {chunk.metadata.get('vuln_type', 'unknown')}")
            if chunk.metadata.get("loss_usd"):
                logger.info(f"  Loss:      ${chunk.metadata['loss_usd']:,}")
            logger.info(f"  Preview:   {chunk.content[:120]}...")
