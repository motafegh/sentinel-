"""Export submodule — sharded graph/token/label/metadata writers.

Produces the consumable artifact for sentinel-ml: 4 file types per shard
(graphs as PyG Batch, tokens as torch.Tensor, labels and metadata as
parquet). Default shard size is 5,000 contracts. The ``SentinelDatasetExport``
class is the consumer-facing API that the ML module's ``SentinelDataset``
wraps.

Format spec: ``format_schema/v1.yaml``.
"""
