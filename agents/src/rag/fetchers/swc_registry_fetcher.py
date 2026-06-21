"""
swc_registry_fetcher.py — SWC (Smart Contract Weakness Classification) registry.

The SWC registry is a fixed taxonomy of weakness types (SWC-100 … SWC-136),
each with a description, remediation, and CWE link. Unlike the contest sources,
this is a stable reference corpus — it ships in data/knowledge/swc_registry.json
and needs no live fetch. Useful for grounding RAG queries in canonical
weakness definitions.
"""

from __future__ import annotations

from .json_corpus_fetcher import JsonCorpusFetcher


class SWCRegistryFetcher(JsonCorpusFetcher):
    corpus_key = "swc_registry"
    _source_name = "swc_registry"
