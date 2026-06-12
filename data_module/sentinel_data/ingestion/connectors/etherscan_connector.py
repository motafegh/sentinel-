"""Etherscan connector — fetch verified contract source from Etherscan API. [STUB — Stage 1]"""

from pathlib import Path
from sentinel_data.ingestion.connectors.base import BaseConnector, ConnectorError, PullResult, SourceConfig


class EtherscanConnector(BaseConnector):
    """Fetch verified contract source from the Etherscan API (not yet implemented)."""

    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        raise NotImplementedError(
            "EtherscanConnector is a stub. Implement in Stage 1 for disl (514K unlabeled)."
        )
