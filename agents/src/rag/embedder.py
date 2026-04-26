"""
embedder.py

Converts Chunk content into embedding vectors via LM Studio.

RECALL — Why we call the underlying client directly (not LangChain wrapper):
  LangChain's embed_documents() adds extra formatting that LM Studio
  doesn't support — causes 400 BadRequestError.
  Calling self.embedding_model.client.create() directly sends a plain
  list of strings which LM Studio's /v1/embeddings endpoint expects.

CHANGES (2026-04-11):
  FIX-13: assert replaced with explicit raise.
          assert is silently stripped by Python's -O flag (common in production).
          A vector/chunk count mismatch would have corrupted the FAISS index
          without any error. Now uses RuntimeError which is never disabled.
  FIX-14: Retry logic on LM Studio batch failures.
          Old code: a single transient HTTP error at batch 40 of 42 killed
          the entire run — discarding 40 seconds of successful embedding work.
          New code: 3 attempts per batch with exponential backoff (1s, 2s, 4s).
          Only raises after all retries are exhausted.
"""

import time
from loguru import logger
from langchain_openai import OpenAIEmbeddings

from .chunker import Chunk
from ..llm.client import get_embedding_model

# FIX-14: Retry configuration for LM Studio HTTP calls.
MAX_EMBED_RETRIES    = 3
EMBED_RETRY_BASE_SEC = 1   # seconds — doubles each attempt: 1s, 2s, 4s


class Embedder:
    """
    Converts chunk content to embedding vectors via LM Studio.

    Production note: in a cloud deployment this would call
    OpenAI's text-embedding-3-large or Voyage AI's voyage-code-2.
    We use LM Studio's local nomic-embed-text — same interface,
    no API cost, no data leaving your machine.
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size      = batch_size
        self.embedding_model = get_embedding_model()
        logger.debug(f"Embedder initialised — batch_size: {batch_size}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Convert chunk content to embedding vectors.

        Returns parallel list: vectors[i] corresponds to chunks[i].
        Order is preserved — critical for building the FAISS index correctly.

        FIX-13: Count mismatch raises RuntimeError (not assert).
        FIX-14: Each batch is retried up to MAX_EMBED_RETRIES times with
                exponential backoff before the overall call fails.

        Args:
            chunks: List of Chunk objects to embed

        Returns:
            List of embedding vectors (each = list of 768 floats)
        """
        if not chunks:
            logger.warning("No chunks to embed — returning empty list")
            return []

        logger.info(f"Embedding {len(chunks)} chunks in batches of {self.batch_size}...")
        start_time  = time.time()
        all_vectors = []

        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end    = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_texts  = [chunk.content for chunk in batch_chunks]
            batch_num    = (batch_start // self.batch_size) + 1

            # FIX-14: Retry loop per batch.
            # A transient LM Studio blip (model reload, GPU spike) shouldn't
            # discard all work done in previous batches. We retry the failing
            # batch up to MAX_EMBED_RETRIES times before propagating the error.
            batch_vectors = self._embed_batch_with_retry(
                batch_texts, batch_num, total_batches
            )
            all_vectors.extend(batch_vectors)

            if batch_num % 5 == 0 or batch_end == len(chunks):
                elapsed = time.time() - start_time
                logger.info(
                    f"Embedded {batch_end}/{len(chunks)} chunks "
                    f"({batch_num}/{total_batches} batches) "
                    f"— {elapsed:.1f}s elapsed"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"Embedding complete — {len(all_vectors)} vectors "
            f"in {elapsed:.1f}s "
            f"({elapsed / len(chunks) * 1000:.0f}ms per chunk)"
        )

        # FIX-13: Use explicit raise instead of assert.
        # assert is disabled by Python's -O (optimize) flag — common in production
        # deployments. A silently-passing mismatch here corrupts the FAISS index.
        if len(all_vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: produced {len(all_vectors)} vectors "
                f"for {len(chunks)} chunks. FAISS index would be corrupted. "
                f"This is a bug — report it."
            )

        return all_vectors

    def _embed_batch_with_retry(
        self,
        batch_texts: list[str],
        batch_num: int,
        total_batches: int,
    ) -> list[list[float]]:
        """
        FIX-14: Embed one batch with exponential backoff retry.

        Attempts: MAX_EMBED_RETRIES (3)
        Backoff:  EMBED_RETRY_BASE_SEC * 2^attempt  → 1s, 2s, 4s

        Raises RuntimeError after all retries are exhausted,
        including the batch number so the caller can locate the failure.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_EMBED_RETRIES):
            try:
                response = self.embedding_model.client.create(
                    input=batch_texts,
                    model=self.embedding_model.model,
                )
                return [item.embedding for item in response.data]

            except Exception as e:
                last_error = e
                if attempt < MAX_EMBED_RETRIES - 1:
                    wait = EMBED_RETRY_BASE_SEC * (2 ** attempt)
                    logger.warning(
                        f"Batch {batch_num}/{total_batches} failed "
                        f"(attempt {attempt + 1}/{MAX_EMBED_RETRIES}): {e} "
                        f"— retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"Batch {batch_num}/{total_batches} failed after "
                        f"{MAX_EMBED_RETRIES} attempts. Last error: {e}"
                    )

        raise RuntimeError(
            f"Embedding batch {batch_num}/{total_batches} failed after "
            f"{MAX_EMBED_RETRIES} attempts: {last_error}"
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        A-14 fix: retry logic added to match embed_chunks behaviour.
        embed_chunks used _embed_batch_with_retry (3 attempts, exp. backoff)
        but embed_query had a bare single-shot API call — if the model server
        hiccupped for any reason the whole retrieval pipeline would crash.
        Now we reuse the same helper with batch_num=1/1 for logging clarity.

        RECALL — Must use the SAME model that embedded the documents.
        Different model → different vector space → garbage retrieval results.

        Args:
            query: The search query string

        Returns:
            Single embedding vector (768 floats)

        Raises:
            RuntimeError: if the embedding API fails after MAX_EMBED_RETRIES attempts
        """
        if query:
            logger.debug(f"Embedding query: '{query[:50]}'")

        # Reuse the same retry helper that embed_chunks uses for batches.
        # batch_num/total_batches = 1/1 so log messages read "Batch 1/1 failed".
        vectors = self._embed_batch_with_retry(
            batch_texts=[query],
            batch_num=1,
            total_batches=1,
        )
        return vectors[0]


if __name__ == "__main__":
    from pathlib import Path
    from .fetchers.github_fetcher import DeFiHackLabsFetcher
    from .chunker import Chunker

    fetcher = DeFiHackLabsFetcher(
        repo_path=Path("data/defihacklabs"),
        data_dir=Path("data/exploits"),
    )
    docs    = fetcher.fetch()
    chunker = Chunker()
    chunks  = chunker.chunk_documents(docs)

    sample_chunks = chunks[:10]
    logger.info(f"Testing embedder with {len(sample_chunks)} chunks...")

    embedder = Embedder(batch_size=32)
    vectors  = embedder.embed_chunks(sample_chunks)

    logger.info(f"Vector count:     {len(vectors)}")
    logger.info(f"Vector dimension: {len(vectors[0])}")
    logger.info(f"Sample values:    {vectors[0][:5]}")

    query_vector = embedder.embed_query("reentrancy attack on DeFi vault")
    logger.info(f"Query vector dim: {len(query_vector)}")
    logger.info(f"Query values:     {query_vector[:5]}")

    logger.info("Embedder smoke test complete")
