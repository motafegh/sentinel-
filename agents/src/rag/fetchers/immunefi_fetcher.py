"""
immunefi_fetcher.py — Immunefi bug-bounty disclosures (Phase A, A.5).

Immunefi publishes post-mortems of paid bounty disclosures (root cause, impact,
fix). Reads the curated JSON corpus at data/knowledge/immunefi.json.
"""

from __future__ import annotations

from .json_corpus_fetcher import JsonCorpusFetcher


class ImmunefiFetcher(JsonCorpusFetcher):
    corpus_key = "immunefi"
    _source_name = "immunefi"
