# agents/src/rag/build_index/_metadata.py
"""
Index-metadata identity + staleness detection.

A metadata file is written alongside the index and records the schema
version + config hash at build time. Staleness is decided by comparing
the current code's expected config against the recorded one and the
recorded source file count against the live count.

Time/ID helpers (_utc_now_iso, _new_build_id) live here because they
appear in the metadata records built by the orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from ._paths import (
    BM25_TOKENIZER_VERSION,
    EMBEDDING_MODEL_NAME,
    FAISS_TYPE,
    INDEX_SCHEMA_VERSION,
    METADATA_PATH,
    _REQUIRED_ARTIFACTS,
    DEFIHACKLABS_DIR,
    EXPLOITS_DIR,
)
from ..chunker import Chunker  # sibling at agents/src/rag/chunker.py


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


# ── Metadata / validation helpers ────────────────────────────────────────────

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


def _source_file_count(fetcher) -> int:
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
    from ..fetchers.github_fetcher import DeFiHackLabsFetcher

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