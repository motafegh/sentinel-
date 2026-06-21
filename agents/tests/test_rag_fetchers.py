"""Tests for A.5 RAG corpus-expansion fetchers (src/rag/fetchers/)."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.fetchers.base_fetcher import Document
from src.rag.fetchers.code4rena_fetcher import Code4renaFetcher
from src.rag.fetchers.sherlock_fetcher import SherlockFetcher
from src.rag.fetchers.solodit_fetcher import SoloditFetcher
from src.rag.fetchers.immunefi_fetcher import ImmunefiFetcher
from src.rag.fetchers.swc_registry_fetcher import SWCRegistryFetcher

ALL_FETCHERS = [
    Code4renaFetcher, SherlockFetcher, SoloditFetcher,
    ImmunefiFetcher, SWCRegistryFetcher,
]


class TestSeedCorpora:
    @pytest.mark.parametrize("cls", ALL_FETCHERS)
    def test_ships_seed_corpus_and_fetches(self, cls):
        f = cls()
        assert f.health_check() is True, f"{f.source_name} seed corpus missing"
        docs = f.fetch()
        assert docs, f"{f.source_name} returned no documents"
        for d in docs:
            assert isinstance(d, Document)
            assert d.content.strip()
            assert d.doc_id
            assert d.metadata.get("source") == f.source_name

    @pytest.mark.parametrize("cls", ALL_FETCHERS)
    def test_source_name_is_set(self, cls):
        assert cls().source_name


class TestJsonCorpusBehaviour:
    def test_missing_corpus_returns_empty(self, tmp_path):
        f = Code4renaFetcher(corpus_path=tmp_path / "does_not_exist.json")
        assert f.health_check() is False
        assert f.fetch() == []

    def test_custom_corpus_parsed(self, tmp_path):
        corpus = tmp_path / "c.json"
        corpus.write_text(json.dumps([
            {"title": "Test finding", "content": "body", "vuln_type": "reentrancy",
             "severity": "high", "date": "2024-01-01"},
        ]))
        f = SherlockFetcher(corpus_path=corpus)
        docs = f.fetch()
        assert len(docs) == 1
        assert docs[0].metadata["vuln_type"] == "reentrancy"
        assert "Test finding" in docs[0].content

    def test_doc_id_derived_when_absent(self, tmp_path):
        corpus = tmp_path / "c.json"
        corpus.write_text(json.dumps([{"title": "X", "content": "y"}]))
        f = SoloditFetcher(corpus_path=corpus)
        assert f.fetch()[0].doc_id.startswith("solodit-")

    def test_fetch_since_filters_by_date(self, tmp_path):
        corpus = tmp_path / "c.json"
        corpus.write_text(json.dumps([
            {"title": "old", "content": "a", "date": "2020-01-01"},
            {"title": "new", "content": "b", "date": "2024-06-01"},
        ]))
        f = ImmunefiFetcher(corpus_path=corpus)
        recent = f.fetch_since(datetime(2023, 1, 1))
        titles = [d.content.split("\n")[0] for d in recent]
        assert "new" in titles
        assert "old" not in titles

    def test_malformed_json_returns_empty(self, tmp_path):
        corpus = tmp_path / "bad.json"
        corpus.write_text("{not valid json")
        f = SWCRegistryFetcher(corpus_path=corpus)
        assert f.fetch() == []

    def test_wrapper_dict_findings_key(self, tmp_path):
        corpus = tmp_path / "c.json"
        corpus.write_text(json.dumps({"findings": [{"title": "T", "content": "c"}]}))
        f = Code4renaFetcher(corpus_path=corpus)
        assert len(f.fetch()) == 1


class TestSWCRegistry:
    def test_has_canonical_swc_ids(self):
        ids = {d.doc_id for d in SWCRegistryFetcher().fetch()}
        assert "SWC-107" in ids  # Reentrancy
        assert "SWC-101" in ids  # Integer over/underflow
