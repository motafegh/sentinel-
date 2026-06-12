"""Connector factory — map connector type string to class."""

from sentinel_data.ingestion.connectors.base import BaseConnector, ConnectorError, PullResult, SourceConfig
from sentinel_data.ingestion.connectors.git_connector import GitConnector
from sentinel_data.ingestion.connectors.huggingface_connector import HuggingFaceConnector
from sentinel_data.ingestion.connectors.zenodo_connector import ZenodoConnector
from sentinel_data.ingestion.connectors.etherscan_connector import EtherscanConnector
from sentinel_data.ingestion.connectors.manual_connector import ManualConnector

_REGISTRY: dict[str, type[BaseConnector]] = {
    "git":          GitConnector,
    "huggingface":  HuggingFaceConnector,
    "zenodo":       ZenodoConnector,
    "etherscan":    EtherscanConnector,
    "manual":       ManualConnector,
    "audit_report": ManualConnector,
    "rekt_scraper": ManualConnector,
}


def get_connector(connector_type: str) -> BaseConnector:
    """Instantiate and return the connector for *connector_type*.

    Raises ConnectorError if the type is not in the registry.
    """
    cls = _REGISTRY.get(connector_type)
    if cls is None:
        raise ConnectorError(
            f"Unknown connector type {connector_type!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return cls()


__all__ = [
    "BaseConnector", "ConnectorError", "PullResult", "SourceConfig",
    "GitConnector", "get_connector",
]
