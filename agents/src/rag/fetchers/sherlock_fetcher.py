"""
sherlock_fetcher.py — Sherlock audit findings (Phase A, A.5).

Sherlock contests skew toward oracle manipulation, MEV, and state-management
bugs. Reads the curated JSON corpus at data/knowledge/sherlock.json.
"""

from __future__ import annotations

from .json_corpus_fetcher import JsonCorpusFetcher


class SherlockFetcher(JsonCorpusFetcher):
    corpus_key = "sherlock"
    _source_name = "sherlock"
