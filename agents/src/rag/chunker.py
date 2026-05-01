"""
chunker.py

Splits Document objects into fixed-size chunks for embedding.

RECALL — Why we chunk:
  1. Embedding models have token limits (~8192 for nomic-embed)
  2. Smaller chunks → more precise retrieval matches
  3. Overlap prevents information loss at chunk boundaries

CHANGES (2026-04-11):
  FIX-26: Default chunk_size increased from 512 to 1536 characters.
          Old: 512 chars ≈ 100–150 tokens — most DeFiHackLabs descriptions
               (header + attack summary + URLs) are 800–1500 chars, so the
               default split them unnecessarily. avg ~1.8 chunks/doc meant
               the protocol header landed in chunk 0 and the attack steps
               were cut off in chunk 1 — reducing semantic coherence.
          New: 1536 chars ≈ 300 tokens — well within nomic-embed's 8192-token
               limit, keeps most descriptions whole, avg closer to 1.0–1.2
               chunks/doc for richer single-chunk retrieval.
          NOTE: After changing this value you must re-run build_index.py to
                rebuild the index with the new chunk sizes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .fetchers.base_fetcher import Document

# FIX-26: Increased from 512 to 1536 — see module docstring.
DEFAULT_CHUNK_SIZE    = 1536
DEFAULT_CHUNK_OVERLAP = 128   # scaled up proportionally from 64


@dataclass
class Chunk:
    """
    A single embeddable unit of text.

    RECALL — Chunk vs Document:
      Document = one exploit file (full content, becomes N chunks)
      Chunk    = one piece of that file (partial content, one vector)

    Separate dataclass because chunks have additional fields (chunk_id,
    total_chunks) that documents don't need.
    """
    content:      str
    doc_id:       str    # parent document ID — links back to source
    chunk_id:     int    # position within parent document (0-indexed)
    total_chunks: int    # total chunks from parent document
    metadata:     dict = field(default_factory=dict)
    # Inherits all parent metadata plus:
    #   chunk_id:     int  — position in document
    #   total_chunks: int  — how many chunks this document produced

    # RRF retrieval score — populated by HybridRetriever.search().
    # 0.0 for chunks that came from the index without going through search
    # (e.g., directly constructed in tests or loaded from old pickles).
    score: float = 0.0


class Chunker:
    """
    Splits Document objects into Chunk objects for embedding.

    RECALL — RecursiveCharacterTextSplitter split priority:
      1. \\n\\n  (paragraph break — preserves most meaning)
      2. \\n    (line break)
      3. ". "   (sentence end with space)
      4. " "    (word boundary)
      5. ""     (character — last resort, never splits mid-meaning)

    This hierarchy means we never split mid-sentence if avoidable.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        # chunk_size is in characters, not tokens.
        # Characters are faster to count (no tokeniser needed).
        # 1536 chars ≈ 300 tokens for English text.
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.debug(
            f"Chunker initialised — "
            f"chunk_size: {chunk_size} | overlap: {chunk_overlap}"
        )

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Split a single Document into Chunks.

        Returns:
            List of Chunk objects ready for embedding.
            Returns [] if the document produces no text after splitting.
        """
        texts = self.splitter.split_text(doc.content)

        if not texts:
            logger.warning(f"Document {doc.doc_id} produced no chunks — skipping")
            return []

        chunks = []
        for i, text in enumerate(texts):
            # Every chunk inherits parent metadata — when we retrieve chunk 2
            # of Euler Finance, we still know protocol/date/vuln_type/loss_usd.
            chunk_metadata = {
                **doc.metadata,
                "chunk_id":     i,
                "total_chunks": len(texts),
                "doc_id":       doc.doc_id,
                "source":       doc.source,
            }
            chunks.append(Chunk(
                content=text,
                doc_id=doc.doc_id,
                chunk_id=i,
                total_chunks=len(texts),
                metadata=chunk_metadata,
            ))

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """
        Split a list of Documents into Chunks.

        Returns:
            Flat list of all Chunks across all documents.
        """
        logger.info(f"Chunking {len(docs)} documents...")

        all_chunks: list[list[Chunk]] = []
        skipped = 0

        for doc in docs:
            chunks = self.chunk_document(doc)
            if not chunks:
                skipped += 1
                continue
            all_chunks.append(chunks)

        flat_chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]

        logger.info(
            f"Chunking complete — "
            f"{len(flat_chunks)} chunks from {len(docs) - skipped} documents "
            f"({skipped} skipped)"
        )

        if flat_chunks:
            sizes = [len(c.content) for c in flat_chunks]
            logger.debug(
                f"Chunk sizes — "
                f"min: {min(sizes)} | max: {max(sizes)} | avg: {sum(sizes) // len(sizes)}"
            )

        return flat_chunks


if __name__ == "__main__":
    from .fetchers.github_fetcher import DeFiHackLabsFetcher

    _agents_dir = Path(__file__).parent.parent.parent
    fetcher = DeFiHackLabsFetcher(
        repo_path=_agents_dir / "data" / "defihacklabs",
        data_dir=_agents_dir / "data" / "exploits",
    )
    docs    = fetcher.fetch()
    chunker = Chunker()
    chunks  = chunker.chunk_documents(docs)

    multi_chunk_docs = len([
        d for d in docs
        if any(c.total_chunks > 1 for c in chunks if c.doc_id == d.doc_id)
    ])

    logger.info(f"Total chunks:          {len(chunks)}")
    logger.info(f"Avg chunks/document:   {len(chunks) / len(docs):.2f}")
    logger.info(f"Multi-chunk documents: {multi_chunk_docs}")

    euler_chunks = [c for c in chunks if "Euler" in c.metadata.get("protocol", "")]
    if euler_chunks:
        logger.info(f"\n--- Euler Finance chunks: {euler_chunks[0].total_chunks} ---")
        for c in euler_chunks:
            logger.info(f"  Chunk {c.chunk_id}: {len(c.content)} chars")
            logger.info(f"  Preview: {c.content[:100]}...")
