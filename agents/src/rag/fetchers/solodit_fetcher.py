"""
solodit_fetcher.py — Solodit aggregated findings (Phase A, A.5).

Solodit aggregates findings across many audit firms/contests. Reads the curated
JSON corpus at data/knowledge/solodit.json.
"""

from __future__ import annotations

from .json_corpus_fetcher import JsonCorpusFetcher


class SoloditFetcher(JsonCorpusFetcher):
    corpus_key = "solodit"
    _source_name = "solodit"
