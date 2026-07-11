# agents/src/rag/build_index/_paths.py
"""
Path constants + index-schema identity for the RAG index build.

All paths absolute, anchored to this file's location so the build can be
launched from any CWD. Schema/version constants record *what* makes an
index stale (a change here means the index must be rebuilt).

Shared by both the full-rebuild orchestrator (`_orchestrator.build_index`)
and the incremental ingestion pipeline (`src/ingestion/pipeline.py`)
through the module-level constants below.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Absolute paths anchored to this file.
# build_index/_paths.py → agents/src/rag/build_index/_paths.py
# parents[0]=build_index [1]=rag [2]=src [3]=agents
_AGENTS_DIR = Path(__file__).resolve().parents[3]

INDEX_DIR = _AGENTS_DIR / "data" / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"
METADATA_PATH = INDEX_DIR / "index_metadata.json"
SEEN_HASHES_PATH = INDEX_DIR / "seen_hashes.json"

DEFIHACKLABS_DIR = _AGENTS_DIR / "data" / "defihacklabs"
EXPLOITS_DIR = _AGENTS_DIR / "data" / "exploits"
KNOWLEDGE_DIR = _AGENTS_DIR / "data" / "knowledge"

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