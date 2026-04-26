"""
tests/test_chunker.py

Unit tests for Chunker — no LM Studio, no FAISS, no network needed.
Tests splitting, metadata inheritance, edge cases, and default sizes.

Run:
  cd ~/projects/sentinel/agents
  poetry run pytest tests/test_chunker.py -v
"""

import pytest
from unittest.mock import MagicMock

from src.rag.chunker import Chunker, Chunk, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.rag.fetchers.base_fetcher import Document


def _make_doc(content: str, doc_id: str = "test_doc", metadata: dict = None) -> Document:
    return Document(
        content=content,
        source="test",
        doc_id=doc_id,
        metadata=metadata or {"protocol": "TestProtocol", "date": "2023-01-01"},
    )


class TestChunkerDefaults:
    def test_default_chunk_size_is_1536(self):
        """FIX-26: Ensure default was increased from 512 to 1536."""
        assert DEFAULT_CHUNK_SIZE == 1536

    def test_chunker_uses_default(self):
        c = Chunker()
        assert c.chunk_size == DEFAULT_CHUNK_SIZE
        assert c.chunk_overlap == DEFAULT_CHUNK_OVERLAP

    def test_chunker_accepts_custom_size(self):
        c = Chunker(chunk_size=256, chunk_overlap=32)
        assert c.chunk_size == 256
        assert c.chunk_overlap == 32


class TestChunkDocument:
    def test_short_doc_produces_one_chunk(self):
        c   = Chunker()
        doc = _make_doc("Short content that fits in one chunk.")
        chunks = c.chunk_document(doc)
        assert len(chunks) == 1

    def test_long_doc_produces_multiple_chunks(self):
        c       = Chunker(chunk_size=100, chunk_overlap=10)
        content = "word " * 100   # 500 chars — larger than chunk_size=100
        doc     = _make_doc(content)
        chunks  = c.chunk_document(doc)
        assert len(chunks) > 1

    def test_empty_doc_returns_empty_list(self):
        c      = Chunker()
        doc    = _make_doc("")
        chunks = c.chunk_document(doc)
        assert chunks == []

    def test_chunk_ids_are_sequential(self):
        c       = Chunker(chunk_size=50, chunk_overlap=5)
        content = "word " * 50
        doc     = _make_doc(content)
        chunks  = c.chunk_document(doc)
        ids = [ch.chunk_id for ch in chunks]
        assert ids == list(range(len(chunks)))

    def test_total_chunks_is_consistent(self):
        c       = Chunker(chunk_size=50, chunk_overlap=5)
        content = "word " * 50
        doc     = _make_doc(content)
        chunks  = c.chunk_document(doc)
        assert all(ch.total_chunks == len(chunks) for ch in chunks)

    def test_doc_id_preserved_in_all_chunks(self):
        c      = Chunker(chunk_size=50, chunk_overlap=5)
        doc    = _make_doc("word " * 50, doc_id="my_unique_id")
        chunks = c.chunk_document(doc)
        assert all(ch.doc_id == "my_unique_id" for ch in chunks)

    def test_chunks_are_Chunk_instances(self):
        c      = Chunker()
        doc    = _make_doc("Some content here.")
        chunks = c.chunk_document(doc)
        assert all(isinstance(ch, Chunk) for ch in chunks)


class TestMetadataInheritance:
    """Verify that every chunk inherits all parent document metadata."""

    def test_protocol_inherited(self):
        c   = Chunker()
        doc = _make_doc("content", metadata={"protocol": "Euler Finance", "date": "2023-03-01"})
        chunks = c.chunk_document(doc)
        assert all(ch.metadata["protocol"] == "Euler Finance" for ch in chunks)

    def test_date_inherited(self):
        c   = Chunker()
        doc = _make_doc("content", metadata={"protocol": "X", "date": "2024-06-15"})
        chunks = c.chunk_document(doc)
        assert all(ch.metadata["date"] == "2024-06-15" for ch in chunks)

    def test_chunk_specific_metadata_added(self):
        c   = Chunker(chunk_size=50, chunk_overlap=5)
        doc = _make_doc("word " * 50)
        chunks = c.chunk_document(doc)
        for i, ch in enumerate(chunks):
            assert ch.metadata["chunk_id"] == i
            assert ch.metadata["total_chunks"] == len(chunks)
            assert ch.metadata["doc_id"] == doc.doc_id
            assert ch.metadata["source"] == doc.source

    def test_parent_metadata_not_mutated(self):
        """Chunk metadata must be a copy — not share references with the doc."""
        original_meta = {"protocol": "Test", "date": "2023-01-01"}
        c   = Chunker()
        doc = _make_doc("content", metadata=original_meta)
        chunks = c.chunk_document(doc)
        # Mutating chunk metadata should not affect original
        chunks[0].metadata["protocol"] = "MUTATED"
        assert original_meta["protocol"] == "Test"


class TestChunkDocuments:
    def test_processes_multiple_docs(self):
        c    = Chunker()
        docs = [_make_doc(f"Content for doc {i}.", doc_id=f"doc_{i}") for i in range(5)]
        chunks = c.chunk_documents(docs)
        assert len(chunks) >= 5   # at least one chunk per doc

    def test_empty_docs_skipped(self):
        c    = Chunker()
        docs = [_make_doc("real content"), _make_doc("")]
        chunks = c.chunk_documents(docs)
        assert all(ch.content != "" for ch in chunks)

    def test_returns_flat_list(self):
        c    = Chunker()
        docs = [_make_doc("content " * 100, doc_id=f"doc_{i}") for i in range(3)]
        chunks = c.chunk_documents(docs)
        assert isinstance(chunks, list)
        assert all(isinstance(ch, Chunk) for ch in chunks)

    def test_chunk_sizes_within_limit(self):
        limit = 200
        c     = Chunker(chunk_size=limit, chunk_overlap=20)
        docs  = [_make_doc("word " * 200, doc_id=f"doc_{i}") for i in range(3)]
        chunks = c.chunk_documents(docs)
        # All chunks should be at most chunk_size characters
        # (splitter may produce slightly larger on hard boundaries — allow 10% slack)
        assert all(len(ch.content) <= limit * 1.1 for ch in chunks)
