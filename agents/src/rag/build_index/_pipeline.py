# agents/src/rag/build_index/_pipeline.py
"""
Fetchers + build-step glue for the RAG index build pipeline.

_extra_fetchers: the Phase A (A.5) corpora-expansion fetchers. WS2
disabled the 5 synthetic corpora (Code4rena/Sherlock/Solodit/Immunefi/
SWC) — empty source is honest, a fake one is a liability. The fetcher
code is kept here for when real data is wired (per 02_RAG_BUILD_PLAN.md).
Re-enable by returning the list below after real data sources are built.

_validate_build_outputs + _write_artifacts: the consistency gate +
atomic-write stage between producing the in-memory artifacts and
persisting them to disk. The validate-then-write ordering protects the
critical invariant: FAISS position N ↔ chunks[N].
"""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from ..chunker import Chunk
from ..embedder import Embedder
from ..fetchers.github_fetcher import DeFiHackLabsFetcher
# Phase A (A.5) — curated audit/finding corpora (offline JSON-backed).
from ..fetchers.code4rena_fetcher import Code4renaFetcher
from ..fetchers.sherlock_fetcher import SherlockFetcher
from ..fetchers.solodit_fetcher import SoloditFetcher
from ..fetchers.immunefi_fetcher import ImmunefiFetcher
from ..fetchers.swc_registry_fetcher import SWCRegistryFetcher

from ._paths import (
    BM25_PATH,
    CHUNKS_PATH,
    FAISS_PATH,
    METADATA_PATH,
    SEEN_HASHES_PATH,
)
from ._io import (
    _atomic_write_faiss,
    _atomic_write_json,
    _atomic_write_pickle,
    _restore_snapshot,
    _sha256_file,
    _snapshot_existing_artifacts,
)


def _extra_fetchers() -> list[Any]:
    """
    Phase A (A.5) corpus-expansion fetchers. Each reads a curated JSON corpus
    under data/knowledge/ and returns [] gracefully if its file is absent, so
    the index build degrades to DeFiHackLabs-only when corpora are missing.

    WS2 (2026-06-22): the 5 extra fetchers (Code4rena/Sherlock/Solodit/Immunefi/
    SWC) are DISABLED — their seed corpora are synthetic placeholder data I
    hand-wrote during Phase A (no live API connection was ever built), and one
    directly caused a hallucinated verdict (the Solodit Multicall chunk made
    the narrative describe a Reentrancy risk on a contract with zero external
    calls — see 00_FINDINGS.md Finding #2). An empty source is honest; a fake
    one is a liability. The fetcher code is kept for when real data is wired
    (per 02_RAG_BUILD_PLAN.md); the index build simply excludes them now.
    Re-enable by returning the list below after real data sources are built.
    """
    return []  # WS2: disabled — see 02_RAG_BUILD_PLAN.md for the real build
    # return [
    #     Code4renaFetcher(),
    #     SherlockFetcher(),
    #     SoloditFetcher(),
    #     ImmunefiFetcher(),
    #     SWCRegistryFetcher(),
    # ]


def _collect_extra_documents() -> tuple[list[Any], list[str]]:
    """Fetch from all extra corpora; return (documents, source_names_used)."""
    docs: list[Any] = []
    sources: list[str] = []
    for f in _extra_fetchers():
        try:
            fetched = f.fetch()
        except Exception as exc:  # never let one corpus break the build
            logger.warning("build_index | fetcher {} failed (skipped): {}", f.source_name, exc)
            continue
        if fetched:
            docs.extend(fetched)
            sources.append(f.source_name)
    return docs, sources


def _validate_build_outputs(
    documents: list[Any],
    chunks: list[Chunk],
    vectors_np: np.ndarray,
    index: faiss.Index,
) -> None:
    """
    Validate artifact consistency before writing anything to disk.

    These checks protect the critical invariant:
        FAISS position N ↔ chunks[N]
    """
    if not documents:
        raise RuntimeError("No documents fetched; refusing to build an empty index.")

    if not chunks:
        raise RuntimeError("Chunker produced zero chunks; refusing to build an empty index.")

    if vectors_np.ndim != 2:
        raise RuntimeError(f"Expected 2D vectors array, got shape {vectors_np.shape}.")

    if vectors_np.shape[0] != len(chunks):
        raise RuntimeError(
            f"Vector/chunk count mismatch: {vectors_np.shape[0]} vectors for "
            f"{len(chunks)} chunks."
        )

    if vectors_np.shape[1] <= 0:
        raise RuntimeError(f"Invalid vector dimension: {vectors_np.shape[1]}.")

    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS/chunk count mismatch: FAISS has {index.ntotal} vectors but "
            f"chunks list has {len(chunks)} entries."
        )


def _write_artifacts(
    *,
    build_id: str,
    index: faiss.Index,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    metadata: dict[str, Any],
    seen_hashes: dict[str, str],
) -> dict[str, Any]:
    """
    Write the complete index artifact set under the held FileLock.

    Metadata is written last and includes checksums of the already-written
    artifact files. If any write fails, the previous artifact set is restored.
    """
    snapshot = _snapshot_existing_artifacts(build_id)

    try:
        _atomic_write_faiss(FAISS_PATH, index, build_id)
        _atomic_write_pickle(BM25_PATH, bm25, build_id)
        _atomic_write_pickle(CHUNKS_PATH, chunks, build_id)
        _atomic_write_json(SEEN_HASHES_PATH, seen_hashes, build_id)

        metadata["artifact_sha256"] = {
            "faiss.index": _sha256_file(FAISS_PATH),
            "bm25.pkl": _sha256_file(BM25_PATH),
            "chunks.pkl": _sha256_file(CHUNKS_PATH),
            "seen_hashes.json": _sha256_file(SEEN_HASHES_PATH),
        }

        _atomic_write_json(METADATA_PATH, metadata, build_id)

        logger.info("Index artifacts written atomically")
        return metadata

    except Exception:
        logger.exception("Failed while writing index artifacts")
        _restore_snapshot(snapshot)
        raise