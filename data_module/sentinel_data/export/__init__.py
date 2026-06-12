"""Export submodule — sharded graph/token/label/metadata writers.

Produces the consumable artifact for sentinel-ml: 4 file types per shard
(graphs as PyG Batch, tokens as torch.Tensor, labels and metadata as
parquet). Default shard size is 5,000 contracts. The ``SentinelDatasetExport``
class is the consumer-facing API that the ML module's ``SentinelDataset``
wraps.

Format spec: ``format_schema/v1.yaml``.
"""

from sentinel_data.export.chunker import ExportManifest, chunk_export
from sentinel_data.export.export import SentinelDatasetExport
from sentinel_data.export.label_writer import write_labels_parquet
from sentinel_data.export.metadata_writer import write_metadata_parquet
from sentinel_data.export.graph_writer import write_graphs_shards
from sentinel_data.export.token_writer import write_tokens_shards

__all__ = [
    "ExportManifest",
    "SentinelDatasetExport",
    "chunk_export",
    "write_labels_parquet",
    "write_metadata_parquet",
    "write_graphs_shards",
    "write_tokens_shards",
]
