"""
cache.py — Content-Addressed Inference Cache (T1-A)

WHY THIS EXISTS
───────────────
Slither + CodeBERT tokenization takes 3–5 s per contract. Repeated contracts
in CI pipelines, audit feedback loops, and smoke tests all pay this cost.
This module provides a transparent disk cache keyed on source content so that
the second call for any given contract returns in < 50 ms.

KEY DESIGN
──────────
Cache key format: "{content_md5}_{FEATURE_SCHEMA_VERSION}"
  • content_md5:           MD5 of the Solidity source text (same string = same key)
  • FEATURE_SCHEMA_VERSION: suffix from graph_schema.py — any feature-engineering
    change bumps this version and automatically invalidates all cached graphs.

This is the same key that process_source() already builds; the cache layer
reads it directly from the ContractPreprocessor, adding zero coordination burden.

Storage layout under cache_dir/:
    {key}_graph.pt   — serialised PyG Data object (graph)
    {key}_tokens.pt  — serialised token dict (input_ids, attention_mask, …)

TTL: entries older than ttl_seconds (default 86400 = 24h) are treated as
expired and re-computed. mtime of the graph file is the TTL clock.

THREAD SAFETY
─────────────
Reads are safe from multiple threads (separate open() calls).
Writes use a tmp file + atomic rename so a partial write never corrupts
an existing cache entry. Multiple simultaneous writers for the same key
may both complete — the last one wins, which is fine (both are identical).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM

# Register PyG classes so weights_only=True works for cached graph files
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])


class InferenceCache:
    """
    Disk-backed content-addressed cache for (graph, tokens) pairs.

    Args:
        cache_dir:    Directory for cache files (created if absent).
        ttl_seconds:  Expiry age in seconds. Default 86400 (24 h).
                      Set to 0 to disable TTL (entries live indefinitely).
    """

    def __init__(
        self,
        cache_dir: str | Path = Path.home() / ".cache" / "sentinel" / "preprocess",
        ttl_seconds: int = 86_400,
    ) -> None:
        self.cache_dir   = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"InferenceCache initialised — dir={self.cache_dir} ttl={ttl_seconds}s"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[tuple[Data, dict]]:
        """
        Return (graph, tokens) for this key, or None on miss/expiry.

        A cache miss is not an error — callers should re-compute and call put().
        """
        graph_path  = self._graph_path(key)
        tokens_path = self._tokens_path(key)

        if not graph_path.exists() or not tokens_path.exists():
            return None

        if self.ttl_seconds > 0:
            age = max(0.0, time.time() - graph_path.stat().st_mtime)
            if age > self.ttl_seconds:
                logger.debug(f"Cache expired ({age:.0f}s > {self.ttl_seconds}s): {key[:16]}…")
                self._evict(key)
                return None

        try:
            graph  = torch.load(graph_path,  map_location="cpu", weights_only=True)
            tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)

            # Validate schema: catch stale cached files that predate a schema change.
            if not hasattr(graph, "x") or graph.x.shape[1] != NODE_FEATURE_DIM:
                raise ValueError(
                    f"Cached graph has x.shape={tuple(graph.x.shape) if hasattr(graph, 'x') else '?'}, "
                    f"expected [N, {NODE_FEATURE_DIM}]. Schema may have changed — evicting."
                )
            expected_tok = torch.Size([1, 512])
            if tokens.get("input_ids", torch.empty(0)).shape != expected_tok:
                raise ValueError(
                    f"Cached tokens input_ids shape {tokens['input_ids'].shape} != {expected_tok}."
                )

            logger.debug(f"Cache hit: {key[:16]}…")
            return graph, tokens
        except Exception as exc:
            logger.warning(f"Cache load failed for {key[:16]}…: {exc} — treating as miss")
            self._evict(key)
            return None

    def put(self, key: str, graph: Data, tokens: dict) -> None:
        """
        Persist (graph, tokens) under this key.

        Uses tmp-file + rename so partial writes never corrupt existing entries.
        """
        graph_path  = self._graph_path(key)
        tokens_path = self._tokens_path(key)

        try:
            self._atomic_save(graph,  graph_path)
            self._atomic_save(tokens, tokens_path)
            logger.debug(f"Cache stored: {key[:16]}…")
        except Exception as exc:
            logger.warning(f"Cache write failed for {key[:16]}…: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _graph_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}_graph.pt"

    def _tokens_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}_tokens.pt"

    def _evict(self, key: str) -> None:
        """Remove both files for a key (best-effort)."""
        for path in (self._graph_path(key), self._tokens_path(key)):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Cache eviction failed for {path.name}: {exc}")

    @staticmethod
    def _atomic_save(obj: object, dest: Path) -> None:
        """Save obj to a tmp file then rename to dest (atomic on POSIX)."""
        tmp = dest.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.rename(dest)
