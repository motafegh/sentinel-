"""
tests/test_deduplicator.py

Unit tests for Deduplicator — no LM Studio, no FAISS, no network needed.
Tests the seen/filter/mark cycle, persistence, and edge cases.

Run:
  cd ~/projects/sentinel/agents
  poetry run pytest tests/test_deduplicator.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.deduplicator import Deduplicator


def _make_doc(doc_id: str):
    """Helper: create a minimal mock Document with a doc_id."""
    doc = MagicMock()
    doc.doc_id = doc_id
    return doc


class TestDeduplicatorInit:
    def test_starts_empty_when_no_file(self, tmp_path):
        path = tmp_path / "seen.json"
        d    = Deduplicator(path)
        assert d.total_indexed == 0

    def test_loads_existing_hashes(self, tmp_path):
        path = tmp_path / "seen.json"
        path.write_text(json.dumps({"abc123": "2026-01-01T00:00:00"}))
        d = Deduplicator(path)
        assert d.total_indexed == 1
        assert d.seen("abc123")

    def test_handles_corrupted_json_gracefully(self, tmp_path):
        path = tmp_path / "seen.json"
        path.write_text("{ invalid json {{")
        d = Deduplicator(path)   # should not raise
        assert d.total_indexed == 0


class TestSeen:
    def test_returns_false_for_unknown_doc(self, tmp_path):
        d = Deduplicator(tmp_path / "seen.json")
        assert not d.seen("unknown_id")

    def test_returns_true_after_mark_seen(self, tmp_path):
        d = Deduplicator(tmp_path / "seen.json")
        d.mark_seen(["doc_abc"])
        assert d.seen("doc_abc")

    def test_does_not_affect_other_ids(self, tmp_path):
        d = Deduplicator(tmp_path / "seen.json")
        d.mark_seen(["doc_x"])
        assert not d.seen("doc_y")


class TestFilterNew:
    def test_returns_all_when_none_seen(self, tmp_path):
        d    = Deduplicator(tmp_path / "seen.json")
        docs = [_make_doc("a"), _make_doc("b"), _make_doc("c")]
        result = d.filter_new(docs)
        assert len(result) == 3

    def test_filters_already_seen_docs(self, tmp_path):
        d = Deduplicator(tmp_path / "seen.json")
        d.mark_seen(["a", "b"])
        docs   = [_make_doc("a"), _make_doc("b"), _make_doc("c")]
        result = d.filter_new(docs)
        assert len(result) == 1
        assert result[0].doc_id == "c"

    def test_returns_empty_when_all_seen(self, tmp_path):
        d = Deduplicator(tmp_path / "seen.json")
        d.mark_seen(["x", "y"])
        docs   = [_make_doc("x"), _make_doc("y")]
        result = d.filter_new(docs)
        assert result == []

    def test_empty_input_returns_empty(self, tmp_path):
        d      = Deduplicator(tmp_path / "seen.json")
        result = d.filter_new([])
        assert result == []


class TestMarkSeen:
    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "seen.json"
        d    = Deduplicator(path)
        d.mark_seen(["id1", "id2"])

        # Reload from disk — a new instance should see the same IDs
        d2 = Deduplicator(path)
        assert d2.seen("id1")
        assert d2.seen("id2")
        assert d2.total_indexed == 2

    def test_mark_empty_list_does_not_crash(self, tmp_path):
        d = Deduplicator(tmp_path / "seen.json")
        d.mark_seen([])   # should not raise
        assert d.total_indexed == 0

    def test_increments_total_indexed(self, tmp_path):
        d = Deduplicator(tmp_path / "seen.json")
        assert d.total_indexed == 0
        d.mark_seen(["a"])
        assert d.total_indexed == 1
        d.mark_seen(["b", "c"])
        assert d.total_indexed == 3

    def test_idempotent_mark(self, tmp_path):
        """Marking the same ID twice should not change the count."""
        d = Deduplicator(tmp_path / "seen.json")
        d.mark_seen(["a"])
        d.mark_seen(["a"])
        assert d.total_indexed == 1

    def test_timestamp_recorded(self, tmp_path):
        path = tmp_path / "seen.json"
        d    = Deduplicator(path)
        d.mark_seen(["doc_ts"])

        data = json.loads(path.read_text())
        assert "doc_ts" in data
        assert "T" in data["doc_ts"]   # ISO timestamp contains 'T'


class TestCheckpointPattern:
    """
    Verifies the checkpoint pattern: docs are NOT marked seen until
    explicitly told to be. If pipeline crashes before mark_seen, the
    next run re-processes those docs.
    """

    def test_docs_not_marked_until_mark_seen_called(self, tmp_path):
        d    = Deduplicator(tmp_path / "seen.json")
        docs = [_make_doc("doc1")]

        # filter_new returns docs but does NOT mark them seen
        result = d.filter_new(docs)
        assert len(result) == 1
        assert not d.seen("doc1")   # still not seen

    def test_after_mark_seen_doc_is_excluded(self, tmp_path):
        d    = Deduplicator(tmp_path / "seen.json")
        docs = [_make_doc("doc1"), _make_doc("doc2")]

        new = d.filter_new(docs)
        assert len(new) == 2

        d.mark_seen(["doc1"])
        new2 = d.filter_new(docs)
        assert len(new2) == 1
        assert new2[0].doc_id == "doc2"
