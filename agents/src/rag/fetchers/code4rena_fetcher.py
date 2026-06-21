"""
code4rena_fetcher.py — Code4rena audit-contest findings (Phase A, A.5).

Code4rena publishes graded findings (High/Medium) from public audit contests.
Reads the curated JSON corpus at data/knowledge/code4rena.json. Replace that
file with a full Code4rena export (e.g. from the c4-findings dataset) for
production scale; the fetcher contract is unchanged.
"""

from __future__ import annotations

from .json_corpus_fetcher import JsonCorpusFetcher


class Code4renaFetcher(JsonCorpusFetcher):
    corpus_key = "code4rena"
    _source_name = "code4rena"
