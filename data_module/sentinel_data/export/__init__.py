"""Export submodule — sharded graph/token/label/metadata writers.

Produces the consumable artifact for sentinel-ml: 4 file types per shard
(graphs as PyG Batch, tokens as torch.Tensor, labels and metadata as
parquet). Default shard size is 5,000 contracts. The ``SentinelDatasetExport``
class is the consumer-facing API that the ML module's ``SentinelDataset``
wraps.

Format spec: ``format_schema/v1.yaml``.

R0.5 additions:
- ``release_descriptor.json`` — per-file SHA-256 checksums authenticating
  the entire export (including manifest.json).  Written by ``chunk_export``
  and verified by ``verify_release()``.
- ``pickle_safe`` module — restricted ``SafeUnpickler`` that rejects
  ``__reduce__``-based code execution exploits.
"""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_NAMES = {
    "ExportManifest": "sentinel_data.export.chunker",
    "RELEASE_DESCRIPTOR_FILENAME": "sentinel_data.export.release_descriptor",
    "SafeUnpickler": "sentinel_data.export.pickle_safe",
    "SentinelDatasetExport": "sentinel_data.export.export",
    "chunk_export": "sentinel_data.export.chunker",
    "safe_load": "sentinel_data.export.pickle_safe",
    "safe_loads": "sentinel_data.export.pickle_safe",
    "safe_torch_load": "sentinel_data.export.pickle_safe",
    "verify_release": "sentinel_data.export.release_descriptor",
    "write_graphs_shards": "sentinel_data.export.graph_writer",
    "write_labels_parquet": "sentinel_data.export.label_writer",
    "write_metadata_parquet": "sentinel_data.export.metadata_writer",
    "write_release_descriptor": "sentinel_data.export.release_descriptor",
    "write_tokens_shards": "sentinel_data.export.token_writer",
}

__all__ = sorted(_MODULE_NAMES)


def __getattr__(name: str) -> Any:
    module_name = _MODULE_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(module_name)
    result = getattr(mod, name)
    # Cache on the module so subsequent lookups don't re-import
    setattr(__import__(__name__), name, result)
    return result
