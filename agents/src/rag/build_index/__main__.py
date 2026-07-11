# agents/src/rag/build_index/__main__.py
"""
CLI entrypoint: `python -m src.rag.build_index`.

Direct execution is intentionally a full rebuild (force_rebuild=True).
Incremental updates should use `src.ingestion.pipeline` instead.
"""

import json

from loguru import logger

from ._orchestrator import build_index


if __name__ == "__main__":
    metadata = build_index(force_rebuild=True)
    logger.info("Build metadata: {}", json.dumps(metadata, indent=2))