"""Manual connector — for sources that require a manual download step. [STUB — Stage 1]"""

from pathlib import Path
from sentinel_data.ingestion.connectors.base import BaseConnector, ConnectorError, PullResult, SourceConfig


class ManualConnector(BaseConnector):
    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        raise NotImplementedError(
            f"[{cfg.name}] requires a manual download. "
            "Place the .sol files in data/raw/{cfg.name}/repo/ and re-run "
            "`sentinel-data ingest --source {cfg.name} --skip-clone`."
        )
