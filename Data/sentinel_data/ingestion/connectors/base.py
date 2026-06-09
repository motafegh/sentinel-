"""BaseConnector — interface all source connectors implement."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceConfig:
    """Parsed entry from config.yaml sources_critical_path / sources_additive."""
    name: str
    connector: str
    url: str = ""
    pin: str = ""
    hf_dataset: str = ""
    zenodo_record: str = ""
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PullResult:
    source: str
    local_dir: Path
    resolved_pin: str
    sol_files: list[Path]
    fetched_at: str
    duration_s: float
    extra: dict[str, Any] = field(default_factory=dict)


class ConnectorError(Exception):
    """Raised when a connector cannot complete a pull."""


class BaseConnector(ABC):
    """All connectors share this interface."""

    def pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        """Pull the source to `dest` and return a PullResult.

        `dest` is the per-source raw dir, e.g. data/raw/defihacklabs/.
        The connector creates it if needed.
        """
        import datetime
        dest.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        result = self._pull(cfg, dest)
        result.duration_s = time.monotonic() - start
        result.fetched_at = datetime.datetime.utcnow().isoformat() + "Z"
        return result

    @abstractmethod
    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        """Subclass implements the actual pull logic."""

    @staticmethod
    def find_sol_files(root: Path) -> list[Path]:
        return sorted(root.rglob("*.sol"))
