"""
json_corpus_fetcher.py

Shared base for the Phase-A RAG corpus expansion fetchers (Code4rena, Sherlock,
Solodit, Immunefi, SWC). Each public audit/finding source is materialised as a
local JSON corpus under `agents/data/knowledge/<key>.json` — a list of records:

    {
      "doc_id":   "code4rena-2023-...",     # optional; derived if absent
      "title":    "Reentrancy in withdraw()",
      "content":  "full finding text ...",
      "vuln_type":"reentrancy",
      "severity": "high",
      "protocol": "SomeProtocol",
      "date":     "2023-05-01",
      "url":      "https://...",
      "chain":    "ethereum"
    }

Rationale for a local-JSON design (vs live scraping):
  - Deterministic + offline → unit-testable, no network flakiness in CI.
  - Each source's live ingestion (API/scrape) can later write into the same
    JSON path; the fetcher contract does not change.
The repo ships small *curated seed* corpora so the index build and tests work
out of the box; production runs replace these with full exports.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from .base_fetcher import BaseFetcher, Document

# agents/src/rag/fetchers/json_corpus_fetcher.py → parents[3] = agents/
_AGENTS_DIR = Path(__file__).resolve().parents[3]
KNOWLEDGE_DIR = _AGENTS_DIR / "data" / "knowledge"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


class JsonCorpusFetcher(BaseFetcher):
    """
    Base fetcher reading a curated JSON corpus from `data/knowledge/<key>.json`.

    Subclasses set `corpus_key` and `source_name`. Override `corpus_path` (or
    pass `corpus_path=` to __init__) to point at a different file — used by
    tests to inject fixtures.
    """

    corpus_key: str = ""
    _source_name: str = "json-corpus"

    def __init__(self, data_dir: Path | None = None, corpus_path: Path | None = None):
        super().__init__(data_dir or KNOWLEDGE_DIR)
        self._corpus_path = corpus_path or (KNOWLEDGE_DIR / f"{self.corpus_key}.json")

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def corpus_path(self) -> Path:
        return self._corpus_path

    def health_check(self) -> bool:
        return self._corpus_path.exists()

    def _records(self) -> list[dict]:
        if not self._corpus_path.exists():
            logger.debug("{} | no corpus at {} — 0 docs", self.source_name, self._corpus_path)
            return []
        try:
            data = json.loads(self._corpus_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("{} | could not read corpus {}: {}", self.source_name, self._corpus_path, exc)
            return []
        if isinstance(data, dict):  # tolerate {"findings": [...]} wrapper
            data = data.get("findings") or data.get("records") or []
        return data if isinstance(data, list) else []

    def _to_document(self, rec: dict) -> Document:
        title = rec.get("title", "")
        body = rec.get("content", rec.get("description", ""))
        content = f"{title}\n\n{body}".strip() if title else body
        raw_id = rec.get("doc_id") or f"{self.corpus_key}:{title}:{rec.get('url', '')}"
        doc_id = rec.get("doc_id") or (
            f"{self.corpus_key}-" + hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        )
        return Document(
            content=content,
            source=self.source_name,
            doc_id=doc_id,
            metadata={
                "protocol":  rec.get("protocol", "unknown"),
                "date":      rec.get("date", ""),
                "vuln_type": rec.get("vuln_type", rec.get("type", "")),
                "severity":  rec.get("severity", ""),
                "loss_usd":  rec.get("loss_usd", 0),
                "chain":     rec.get("chain", ""),
                "url":       rec.get("url", ""),
                "source":    self.source_name,
            },
        )

    def fetch(self) -> list[Document]:
        docs = [self._to_document(r) for r in self._records() if r]
        logger.info("{} | fetched {} document(s)", self.source_name, len(docs))
        return docs

    def fetch_since(self, since: datetime) -> list[Document]:
        out: list[Document] = []
        for rec in self._records():
            d = _parse_date(rec.get("date"))
            if d is None or d >= since:
                out.append(self._to_document(rec))
        logger.info("{} | fetched {} document(s) since {}", self.source_name, len(out), since.date())
        return out
