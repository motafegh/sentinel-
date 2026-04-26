"""
base_fetcher.py

Abstract base class for all RAG knowledge base fetchers.
Every fetcher implements this interface — the ingestion pipeline
talks to BaseFetcher only, never to concrete implementations.
This is the Strategy pattern: swap data sources without changing
pipeline code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger


@dataclass
class Document:
    """
    A single document ready for chunking and indexing.

    RECALL — Every piece of knowledge in SENTINEL's RAG system
    is a Document. The metadata dict is critical — it enables
    filtered retrieval: "find reentrancy exploits from 2023 only"
    """
    content:   str                      # raw text to embed
    source:    str                      # where it came from
    doc_id:    str                      # unique identifier for dedup
    metadata:  dict = field(default_factory=dict)
    # Expected metadata keys:
    #   protocol:   str   — "Euler Finance"
    #   date:       str   — "2023-03-13"
    #   vuln_type:  str   — "flash_loan, reentrancy"
    #   severity:   str   — "critical"
    #   loss_usd:   int   — 197000000
    #   chain:      str   — "ethereum"
    #   url:        str   — source URL


class BaseFetcher(ABC):
    """
    Abstract base class all fetchers must implement.

    Production pattern: fetchers are stateless workers.
    State (last_run, seen_hashes) lives in the pipeline,
    not in individual fetchers.
    """

    def __init__(self, data_dir: Path):
        # RECALL — data_dir is where fetchers store raw downloaded data
        # before processing. Keeps raw source separate from processed index.
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"{self.__class__.__name__} initialised — data_dir: {data_dir}")

    @abstractmethod
    def fetch(self) -> list[Document]:
        """
        Fetch all available documents from this source.
        Used for initial index build.

        Returns:
            List of Document objects ready for chunking
        """
        pass

    @abstractmethod
    def fetch_since(self, since: datetime) -> list[Document]:
        """
        Fetch only documents newer than since datetime.
        Used for incremental updates — don't re-process old docs.

        Args:
            since: only return documents created/updated after this time

        Returns:
            List of new Document objects since last run
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of this source — used in metadata."""
        pass

    def health_check(self) -> bool:
        """
        Verify the source is reachable and data is accessible.
        Called by the pipeline before starting ingestion.
        Override in subclasses that need network access.
        """
        return True