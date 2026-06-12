"""HuggingFace connector — load a HF dataset and write .sol files to disk. [STUB — Stage 1]"""

from pathlib import Path
from sentinel_data.ingestion.connectors.base import BaseConnector, ConnectorError, PullResult, SourceConfig


class HuggingFaceConnector(BaseConnector):
    """Download a HuggingFace dataset and extract .sol files (not yet implemented)."""

    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        raise NotImplementedError(
            "HuggingFaceConnector is a stub. Implement in Stage 1 for slither_audited / solidity_defi_vulns."
        )
