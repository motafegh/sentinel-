"""
build_index.py

Full RAG index build from scratch.

Pipeline:
    Fetch → Chunk → Embed → Build FAISS → Build BM25 → Save

When to use:
    - First-time RAG setup
    - Forced rebuild after chunking changes
    - Forced rebuild after embedding model changes
    - Forced rebuild after BM25 tokenizer changes
    - Recovery from corrupted index artifacts

Why separate from ingestion/pipeline.py:
    build_index.py:
        Full rebuild from scratch.
        Re-embeds every document.
        Used after schema/config changes.

    ingestion/pipeline.py:
        Incremental update.
        Embeds only new documents.
        Used by cron/Dagster/GitHub Actions.

Production hardening:
    - Paths anchored to __file__
    - Shared index lock compatible with ingestion pipeline
    - Atomic writes for FAISS, BM25, chunks, metadata, and seen_hashes
    - Rollback snapshot if artifact replacement fails
    - build_id + schema_version + config_hash in metadata
    - Staleness detection checks config, not only document count
    - Artifact SHA256 checksums recorded in metadata

Run from agents/ directory:
    poetry run python -m src.rag.build_index
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from filelock import FileLock, Timeout
from loguru import logger
from rank_bm25 import BM25Okapi

from .chunker import Chunk, Chunker
from .embedder import Embedder
from .fetchers.github_fetcher import DeFiHackLabsFetcher


# ── Paths ─────────────────────────────────────────────────────────────────────
# Absolute paths anchored to this file.
# agents/src/rag/build_index.py → parent.parent.parent = agents/
_AGENTS_DIR = Path(__file__).parent.parent.parent

INDEX_DIR = _AGENTS_DIR / "data" / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"
METADATA_PATH = INDEX_DIR / "index_metadata.json"
SEEN_HASHES_PATH = INDEX_DIR / "seen_hashes.json"

DEFIHACKLABS_DIR = _AGENTS_DIR / "data" / "defihacklabs"
EXPLOITS_DIR = _AGENTS_DIR / "data" / "exploits"

# Shared with ingestion/pipeline.py by convention.
# Full rebuild and incremental ingestion must never write the index at the same time.
INDEX_LOCK_PATH = INDEX_DIR / ".index.lock"
INDEX_LOCK_TIMEOUT = 300  # seconds


# ── Index schema / config identity ────────────────────────────────────────────

INDEX_SCHEMA_VERSION = "rag_index_v2"
EMBEDDING_MODEL_NAME = "text-embedding-nomic-embed-text-v1.5"
BM25_TOKENIZER_VERSION = "lower_whitespace_v1"
FAISS_TYPE = "IndexFlatL2"

_REQUIRED_ARTIFACTS = [
    FAISS_PATH,
    BM25_PATH,
    CHUNKS_PATH,
    METADATA_PATH,
    SEEN_HASHES_PATH,
]


# ── Time / IDs ────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_build_id() -> str:
    """
    Create a unique build ID for this index rebuild.

    Format is sortable by time and unique enough for local rebuilds:
        20260424T221530Z-a1b2c3d4
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


# ── Atomic write helpers ──────────────────────────────────────────────────────

def _tmp_path(path: Path, build_id: str) -> Path:
    """
    Return a temp path in the same directory as the final artifact.

    Same-directory temp files are required because Path.replace() is atomic
    only within the same filesystem.
    """
    return path.with_name(f".{path.name}.{build_id}.tmp")


def _fsync_directory(directory: Path) -> None:
    """
    Best-effort fsync for the parent directory after atomic replace.

    On POSIX filesystems this makes the rename durable. On platforms where
    directory fsync is unsupported, this silently degrades to normal replace.
    """
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        return

    fd: int | None = None
    try:
        fd = os.open(str(directory), os.O_DIRECTORY)
        os.fsync(fd)
    except OSError:
        # Not all filesystems allow directory fsync. Atomic replace still happened.
        return
    finally:
        if fd is not None:
            os.close(fd)


def _atomic_write_json(path: Path, payload: dict[str, Any], build_id: str) -> None:
    """Write JSON atomically using temp-file + replace."""
    tmp = _tmp_path(path, build_id)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        tmp.replace(path)
        _fsync_directory(path.parent)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_pickle(path: Path, payload: Any, build_id: str) -> None:
    """Write a pickle artifact atomically using temp-file + replace."""
    tmp = _tmp_path(path, build_id)
    try:
        with tmp.open("wb") as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())

        tmp.replace(path)
        _fsync_directory(path.parent)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_faiss(path: Path, index: faiss.Index, build_id: str) -> None:
    """
    Write a FAISS index atomically.

    FAISS writes to a filesystem path, so we write to a same-directory temp
    file first, then atomically replace the final artifact.
    """
    tmp = _tmp_path(path, build_id)
    try:
        faiss.write_index(index, str(tmp))
        tmp.replace(path)
        _fsync_directory(path.parent)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Backup / rollback helpers ────────────────────────────────────────────────

def _snapshot_existing_artifacts(build_id: str) -> dict[Path, Path | None]:
    """
    Copy existing artifacts before replacing them.

    This protects against partial replacement if an exception occurs while
    writing the new artifact set. It is not a substitute for atomic writes;
    it is a rollback safety net.
    """
    backup_dir = INDEX_DIR / "backups" / build_id
    snapshot: dict[Path, Path | None] = {}

    for final_path in _REQUIRED_ARTIFACTS:
        if final_path.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / final_path.name
            shutil.copy2(final_path, backup_path)
            snapshot[final_path] = backup_path
        else:
            snapshot[final_path] = None

    if backup_dir.exists():
        logger.info("Existing index artifacts backed up to {}", backup_dir)

    return snapshot


def _restore_snapshot(snapshot: dict[Path, Path | None]) -> None:
    """
    Restore artifacts from a pre-write snapshot.

    If an artifact did not exist before the failed write, remove any newly
    created artifact at that path.
    """
    logger.warning("Restoring previous index artifacts from rollback snapshot")

    for final_path, backup_path in snapshot.items():
        try:
            if backup_path is not None and backup_path.exists():
                shutil.copy2(backup_path, final_path)
            elif final_path.exists():
                final_path.unlink()
        except Exception as exc:
            logger.error("Rollback failed for {}: {}", final_path, exc)


# ── Metadata / validation helpers ────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """Return SHA256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_config(chunker: Chunker) -> dict[str, Any]:
    """
    Return the index-affecting configuration.

    Any change here means the existing index may be semantically stale and
    should be rebuilt.
    """
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "chunk_size": chunker.chunk_size,
        "chunk_overlap": chunker.chunk_overlap,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "bm25_tokenizer": BM25_TOKENIZER_VERSION,
        "faiss_type": FAISS_TYPE,
    }


def _config_hash(config: dict[str, Any]) -> str:
    """Stable hash of index-affecting configuration."""
    raw = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_metadata() -> dict[str, Any] | None:
    """Load existing metadata if present and valid."""
    if not METADATA_PATH.exists():
        return None

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read existing index metadata: {}", exc)
        return None


def _source_file_count(fetcher: DeFiHackLabsFetcher) -> int:
    """
    Count source .sol files currently available in DeFiHackLabs.

    This is a cheap staleness signal. The actual indexed document count is
    still measured from fetcher.fetch() during the real build.
    """
    try:
        return len(list(fetcher.src_path.rglob("*.sol")))
    except Exception as exc:
        logger.warning("Could not count DeFiHackLabs source files: {}", exc)
        return 0


def _index_is_current(expected: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str]:
    """
    Determine whether the existing index can be reused.

    Checks:
        - required artifacts exist
        - metadata exists
        - schema/config hash matches current code config
        - source file count has not changed

    Returns:
        (is_current, metadata_or_none, reason)
    """
    missing = [path.name for path in _REQUIRED_ARTIFACTS if not path.exists()]
    if missing:
        return False, None, f"missing artifacts: {missing}"

    metadata = _load_metadata()
    if metadata is None:
        return False, None, "missing or invalid metadata"

    expected_hash = _config_hash(expected)
    if metadata.get("config_hash") != expected_hash:
        return False, metadata, "index config changed or old metadata has no config_hash"

    fetcher = DeFiHackLabsFetcher(
        repo_path=DEFIHACKLABS_DIR,
        data_dir=EXPLOITS_DIR,
    )
    current_source_count = _source_file_count(fetcher)
    indexed_source_count = metadata.get("source_file_count")

    if current_source_count == 0:
        return False, metadata, "source file count is 0; cannot prove index freshness"

    if indexed_source_count != current_source_count:
        return (
            False,
            metadata,
            f"source file count changed: current={current_source_count}, indexed={indexed_source_count}",
        )

    return True, metadata, "index artifacts and config are current"


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


# ── Main build function ──────────────────────────────────────────────────────

def build_index(force_rebuild: bool = False) -> dict[str, Any]:
    """
    Build the full RAG index from scratch.

    Args:
        force_rebuild:
            If True, rebuild even when the existing index metadata appears
            current. Direct CLI execution always uses True.

    Returns:
        Metadata dict for the current or newly-built index.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    build_id = _new_build_id()
    chunker_for_config = Chunker()
    expected = _expected_config(chunker_for_config)

    if not force_rebuild:
        is_current, existing_metadata, reason = _index_is_current(expected)
        if is_current and existing_metadata is not None:
            logger.info("Index is current — skipping rebuild ({})", reason)
            return {**existing_metadata, "skipped": True, "skip_reason": reason}

        logger.warning("Index rebuild required: {}", reason)

    logger.info("=" * 60)
    logger.info("SENTINEL RAG INDEX BUILD")
    logger.info("Build ID: {}", build_id)
    logger.info("=" * 60)

    try:
        with FileLock(str(INDEX_LOCK_PATH), timeout=INDEX_LOCK_TIMEOUT):
            # Re-check after acquiring the lock in case another process rebuilt
            # the index while this process waited.
            if not force_rebuild:
                is_current, existing_metadata, reason = _index_is_current(expected)
                if is_current and existing_metadata is not None:
                    logger.info("Index became current while waiting for lock — skipping")
                    return {**existing_metadata, "skipped": True, "skip_reason": reason}

            build_start = time.time()

            # ── Step 1: Fetch ─────────────────────────────────────────────
            logger.info("Step 1/6 — Fetching documents...")
            fetcher = DeFiHackLabsFetcher(
                repo_path=DEFIHACKLABS_DIR,
                data_dir=EXPLOITS_DIR,
            )
            source_count = _source_file_count(fetcher)
            documents = fetcher.fetch()
            logger.info("  Fetched {} documents", len(documents))

            # ── Step 2: Chunk ─────────────────────────────────────────────
            logger.info("Step 2/6 — Chunking documents...")
            chunker = Chunker()
            chunks = chunker.chunk_documents(documents)
            logger.info("  Created {} chunks from {} documents", len(chunks), len(documents))

            # ── Step 3: Embed ─────────────────────────────────────────────
            logger.info("Step 3/6 — Embedding chunks...")
            embedder = Embedder(batch_size=32)
            vectors = embedder.embed_chunks(chunks)
            vectors_np = np.array(vectors, dtype=np.float32)
            dimension = int(vectors_np.shape[1])
            logger.info("  Generated {} vectors of dimension {}", len(vectors), dimension)

            # ── Step 4: Build FAISS ───────────────────────────────────────
            logger.info("Step 4/6 — Building FAISS index...")
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors_np)
            logger.info("  FAISS index built — {} vectors indexed", index.ntotal)

            # ── Step 5: Build BM25 ────────────────────────────────────────
            logger.info("Step 5/6 — Building BM25 index...")
            corpus = [chunk.content.lower().split() for chunk in chunks]
            bm25 = BM25Okapi(corpus)
            logger.info("  BM25 index built — {} documents", len(corpus))

            # ── Validate before writing ───────────────────────────────────
            _validate_build_outputs(
                documents=documents,
                chunks=chunks,
                vectors_np=vectors_np,
                index=index,
            )

            build_time = time.time() - build_start
            built_at = _utc_now_iso()
            config = _expected_config(chunker)
            config_hash = _config_hash(config)

            seen_hashes = {
                doc.doc_id: built_at
                for doc in documents
            }

            metadata: dict[str, Any] = {
                # New production metadata
                "schema_version": INDEX_SCHEMA_VERSION,
                "build_id": build_id,
                "config_hash": config_hash,
                "config": config,
                "source_file_count": source_count,
                "bm25_tokenizer": BM25_TOKENIZER_VERSION,

                # Backward-compatible fields expected by existing tooling
                "built_at": built_at,
                "build_time_sec": round(build_time, 1),
                "total_chunks": len(chunks),
                "total_documents": len(documents),
                "vector_dimension": dimension,
                "sources": [fetcher.source_name],
                "embedding_model": EMBEDDING_MODEL_NAME,
                "chunk_size": chunker.chunk_size,
                "chunk_overlap": chunker.chunk_overlap,
                "faiss_type": FAISS_TYPE,

                # Validation snapshot
                "validation": {
                    "faiss_vectors": int(index.ntotal),
                    "chunks_count": len(chunks),
                    "vectors_count": int(vectors_np.shape[0]),
                    "bm25_documents": len(corpus),
                    "seen_hashes_count": len(seen_hashes),
                },
            }

            # ── Step 6: Save artifacts ────────────────────────────────────
            logger.info("Step 6/6 — Saving index artifacts...")
            metadata = _write_artifacts(
                build_id=build_id,
                index=index,
                bm25=bm25,
                chunks=chunks,
                metadata=metadata,
                seen_hashes=seen_hashes,
            )

            logger.info("=" * 60)
            logger.info("INDEX BUILD COMPLETE")
            logger.info("  Build ID:    {}", build_id)
            logger.info("  Documents:   {}", len(documents))
            logger.info("  Chunks:      {}", len(chunks))
            logger.info("  Vectors:     {} × {}d", len(vectors), dimension)
            logger.info("  Chunk size:  {}", chunker.chunk_size)
            logger.info("  Overlap:     {}", chunker.chunk_overlap)
            logger.info("  Build time:  {:.1f}s", build_time)
            logger.info("  Index dir:   {}", INDEX_DIR.absolute())
            logger.info("=" * 60)

            return metadata

    except Timeout as exc:
        raise RuntimeError(
            f"Could not acquire index lock after {INDEX_LOCK_TIMEOUT}s. "
            f"Another process may be writing the RAG index. "
            f"Lock path: {INDEX_LOCK_PATH}"
        ) from exc


if __name__ == "__main__":
    # Direct execution is intentionally a full rebuild.
    # Incremental updates should use src.ingestion.pipeline instead.
    metadata = build_index(force_rebuild=True)
    logger.info("Build metadata: {}", json.dumps(metadata, indent=2))