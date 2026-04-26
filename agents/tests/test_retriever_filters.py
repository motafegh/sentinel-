"""
tests/test_retriever_filters.py

Unit tests for HybridRetriever._apply_filters() and the FAISS↔chunks
sync validation — no LM Studio, no FAISS search, no network needed.
The retriever is instantiated with mocks so the index files are never needed.

Run:
  cd ~/projects/sentinel/agents
  poetry run pytest tests/test_retriever_filters.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.rag.chunker import Chunk


def _make_chunk(
    content: str = "test content",
    doc_id: str = "doc1",
    vuln_type: str = "flash_loan",
    date: str = "2023-06-01",
    loss_usd: int = 1_000_000,
    source: str = "DeFiHackLabs",
    has_summary: bool = True,
) -> Chunk:
    return Chunk(
        content=content,
        doc_id=doc_id,
        chunk_id=0,
        total_chunks=1,
        metadata={
            "vuln_type":    vuln_type,
            "date":         date,
            "loss_usd":     loss_usd,
            "source":       source,
            "has_summary":  has_summary,
        },
    )


@pytest.fixture
def retriever():
    """
    Create a HybridRetriever with mocked internals — no index files needed.
    """
    with patch("src.rag.retriever.FAISS_PATH") as mock_faiss_path, \
         patch("src.rag.retriever.BM25_PATH"),                      \
         patch("src.rag.retriever.CHUNKS_PATH"),                    \
         patch("src.rag.retriever.METADATA_PATH"),                  \
         patch("src.rag.retriever.faiss"),                          \
         patch("src.rag.retriever.pickle"),                         \
         patch("src.rag.retriever.Embedder"),                       \
         patch("builtins.open", MagicMock()):

        mock_faiss_path.exists.return_value = True

        from src.rag.retriever import HybridRetriever
        r = HybridRetriever.__new__(HybridRetriever)

        # Set up minimal attributes directly
        mock_index = MagicMock()
        mock_index.ntotal = 3
        r.faiss_index = mock_index
        r.bm25        = MagicMock()
        r.chunks      = [
            _make_chunk("flash loan on Compound",  "d1", "flash_loan",       "2023-06-01", 5_000_000),
            _make_chunk("reentrancy in vault",      "d2", "reentrancy",       "2022-01-15", 200_000),
            _make_chunk("access control bug",       "d3", "access_control",   "2024-02-01", 50_000_000),
        ]
        r.metadata    = {"built_at": "2026-01-01T00:00:00", "last_run": "2026-01-01T00:00:00"}
        r.embedder    = MagicMock()
        yield r


class TestApplyFilters:
    def test_no_filters_returns_all(self, retriever):
        result = retriever._apply_filters(retriever.chunks, {})
        assert len(result) == 3

    def test_vuln_type_filter(self, retriever):
        result = retriever._apply_filters(retriever.chunks, {"vuln_type": "reentrancy"})
        assert len(result) == 1
        assert result[0].metadata["vuln_type"] == "reentrancy"

    def test_date_gte_filter(self, retriever):
        result = retriever._apply_filters(retriever.chunks, {"date_gte": "2023-01-01"})
        # 2023-06-01 and 2024-02-01 pass; 2022-01-15 does not
        assert len(result) == 2
        assert all(ch.metadata["date"] >= "2023-01-01" for ch in result)

    def test_loss_gte_filter(self, retriever):
        result = retriever._apply_filters(retriever.chunks, {"loss_gte": 1_000_000})
        # 5M and 50M pass; 200K does not
        assert len(result) == 2
        assert all(ch.metadata["loss_usd"] >= 1_000_000 for ch in result)

    def test_source_filter(self, retriever):
        # Add a chunk with a different source
        extra = _make_chunk(source="SENTINEL_ONCHAIN")
        chunks = retriever.chunks + [extra]
        result = retriever._apply_filters(chunks, {"source": "SENTINEL_ONCHAIN"})
        assert len(result) == 1
        assert result[0].metadata["source"] == "SENTINEL_ONCHAIN"

    def test_has_summary_filter(self, retriever):
        # One chunk without summary
        no_summary = _make_chunk(has_summary=False)
        chunks = retriever.chunks + [no_summary]
        result = retriever._apply_filters(chunks, {"has_summary": True})
        assert all(ch.metadata["has_summary"] for ch in result)

    def test_combined_filters(self, retriever):
        result = retriever._apply_filters(
            retriever.chunks,
            {"vuln_type": "flash_loan", "loss_gte": 1_000_000}
        )
        assert len(result) == 1
        assert result[0].metadata["vuln_type"] == "flash_loan"

    def test_strict_filter_returns_empty_with_warning(self, retriever):
        """FIX-12: Empty filter result must log a warning.

        Loguru writes to its own fd-level handler — not through sys.stderr —
        so neither caplog nor capsys captures it reliably. The correct pattern
        is to add a temporary loguru sink (a list) for the duration of the test.
        """
        from loguru import logger

        messages: list[str] = []
        handler_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
        try:
            result = retriever._apply_filters(
                retriever.chunks,
                {"vuln_type": "nonexistent_type"},
                query="test query",
            )
        finally:
            logger.remove(handler_id)

        assert result == []
        full_log = " ".join(messages)
        assert "0 results" in full_log or "filter" in full_log.lower()


class TestFAISSChunksSyncValidation:
    """FIX-10: Retriever must detect FAISS ↔ chunks count mismatch on init."""

    def test_raises_on_count_mismatch(self, tmp_path):
        """
        If FAISS has N vectors but chunks list has M entries (N ≠ M),
        every retrieval result is wrong — detect immediately on startup.
        """
        import json, pickle
        import numpy as np

        # Create minimal index files
        index_dir = tmp_path / "data" / "index"
        index_dir.mkdir(parents=True)

        # Write 3 chunks but FAISS will have 5 vectors (mismatch)
        chunks = [
            _make_chunk("chunk A", "d1"),
            _make_chunk("chunk B", "d2"),
            _make_chunk("chunk C", "d3"),
        ]
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        # Write metadata
        with open(index_dir / "index_metadata.json", "w") as f:
            json.dump({"built_at": "2026-01-01T00:00:00"}, f)

        # Mock FAISS index with 5 vectors (mismatch with 3 chunks)
        with patch("src.rag.retriever.FAISS_PATH", index_dir / "faiss.index"), \
             patch("src.rag.retriever.BM25_PATH",  index_dir / "bm25.pkl"),    \
             patch("src.rag.retriever.CHUNKS_PATH", index_dir / "chunks.pkl"), \
             patch("src.rag.retriever.METADATA_PATH", index_dir / "index_metadata.json"), \
             patch("src.rag.retriever.faiss") as mock_faiss,                   \
             patch("src.rag.retriever.Embedder"):

            (index_dir / "faiss.index").touch()
            # MagicMock is not picklable — write a real picklable placeholder.
            # The retriever stores self.bm25 = pickle.load(...) and never calls
            # it during __init__, so any picklable value works here.
            (index_dir / "bm25.pkl").write_bytes(pickle.dumps({"__mock__": True}))

            mock_index = MagicMock()
            mock_index.ntotal = 5   # mismatch: chunks has 3, FAISS has 5
            mock_faiss.read_index.return_value = mock_index

            from src.rag.retriever import HybridRetriever

            with pytest.raises(RuntimeError, match="corruption"):
                HybridRetriever()
