"""Zenodo connector — download a Zenodo record zip and extract .sol files. [STUB — Stage 1]"""

from pathlib import Path
from sentinel_data.ingestion.connectors.base import BaseConnector, ConnectorError, PullResult, SourceConfig


class ZenodoConnector(BaseConnector):
    """Download a Zenodo record zip and extract .sol files (not yet implemented)."""

    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        raise NotImplementedError(
            "ZenodoConnector is a stub. Implement in Stage 1 for zenodo_clear (record 16910242)."
        )
