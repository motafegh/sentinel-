"""Labeling submodule — multi-source verified label pipeline.

Implements the canonical 10-class taxonomy, source-specific crosswalk
mappings, per-source parsers, multi-source merger with conflict resolution,
and Go/No-Go minimum-viable-corpus gate. Produces the canonical
.labels.json files consumed by sentinel-ml training.
"""
