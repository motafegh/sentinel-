"""Ingestion submodule — pull raw .sol contracts from multiple source types.

Provides connectors for Git repos, HuggingFace datasets, Zenodo records,
Etherscan API, and manual (pre-downloaded) sources. Each pull writes a
per-source ingestion manifest with SHA-256 verification for reproducibility.
"""
