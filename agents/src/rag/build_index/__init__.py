# agents/src/rag/build_index/__init__.py
"""
build_index package — full RAG index rebuild (P2.5 split, 2026-06-25).

Split from the original `build_index.py` god-file (661 LOC) into focused
modules by reason-to-change:
    _paths.py         — path constants + index-schema identity
    _io.py            — atomic-write / fsync / sha256 / snapshot-rollback
    _metadata.py      — expected-config + staleness check + time/ID helpers
    _pipeline.py      — fetchers + validate + atomic artifact write stage
    _orchestrator.py  — build_index() main 6-step pipeline (Fetch→Chunk→…)

`python -m src.rag.build_index` runs a full rebuild (see __main__.py).
Incremental updates use `src/ingestion/pipeline.py` instead.
"""

from ._orchestrator import build_index

__all__ = ["build_index"]