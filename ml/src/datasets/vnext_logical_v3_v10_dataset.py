"""Fail-closed training adapter for a future accepted logical-V3/v10 lineage.

The current v10 population is only a candidate.  This adapter is intentionally
unusable with the diagnostic candidate-binding report: it requires both a
separate physical-acceptance record and a later explicit training decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.src.datasets.vnext_logical_v3_dataset import LogicalV3TrainingDataset
from ml.src.datasets.vnext_repaired_dataset import _sha256, vnext_collate_fn
from sentinel_data.preprocessing.r4_versions import (
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.graph_schema_versions import get_graph_schema
from sentinel_data.vnext.policy import CLASS_NAMES
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)

V10_TRAINING_AUTHORIZED_STATUS = "LOGICAL_V3_V10_TRAINING_AUTHORIZED"
V10_PHYSICAL_ACCEPTANCE_SCHEMA = "sentinel-r4-v10-physical-acceptance-v1"


def validate_v10_training_manifest(
    *,
    manifest: dict,
    overlay_dir: Path,
    expected_binding_digest: str | None,
) -> str:
    """Validate exact physical and decision authority for v10 training use."""

    if manifest.get("dataset_version") != DATASET_VERSION_V3:
        raise ValueError("v10 training requires logical-v3 dataset version")
    if manifest.get("export_schema_version") != "v2":
        raise ValueError("v10 training requires export schema v2")
    if manifest.get("partition_version") != ROLE_PARTITION_VERSION_V3:
        raise ValueError("v10 training role partition mismatch")
    if manifest.get("grouping_version") != GROUPING_VERSION_V3:
        raise ValueError("v10 training grouping version mismatch")
    if manifest.get("address_literal_grouping_authority") is not False:
        raise ValueError("v10 training unexpectedly enables address grouping authority")
    if manifest.get("confirmed_negative_rows") != 0:
        raise ValueError("v10 training unexpectedly contains confirmed negatives")
    if list(manifest.get("class_order") or []) != list(CLASS_NAMES):
        raise ValueError("v10 training class order mismatch")
    if manifest.get("graph_schema_version") != V10_GRAPH_SCHEMA_VERSION:
        raise ValueError("v10 training graph schema mismatch")
    if manifest.get("representation_extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError("v10 training extractor version mismatch")
    if manifest.get("representation_root_version") != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError("v10 training representation root version mismatch")
    if manifest.get("status") != V10_TRAINING_AUTHORIZED_STATUS:
        raise ValueError("v10 lineage has no explicit training-authorized status")

    binding = manifest.get("representation_binding_report") or {}
    digest = str(binding.get("binding_digest_sha256") or "")
    if not digest:
        raise ValueError("v10 manifest lacks representation binding digest")
    if expected_binding_digest is not None and digest != expected_binding_digest:
        raise ValueError("v10 expected representation binding digest mismatch")
    report_name = str(binding.get("path") or "representation_binding_report.json")
    report_path = Path(overlay_dir) / report_name
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    if not binding.get("sha256") or _sha256(report_path) != binding.get("sha256"):
        raise ValueError("v10 physical acceptance report hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != V10_PHYSICAL_ACCEPTANCE_SCHEMA:
        raise ValueError("v10 physical acceptance report schema mismatch")
    if report.get("passed") is not True or report.get("physical_acceptance") is not True:
        raise ValueError("v10 representation has not been physically accepted")
    if report.get("training_authorized") is not False:
        raise ValueError("v10 physical acceptance must not authorize training itself")
    if report.get("graph_schema_version") != V10_GRAPH_SCHEMA_VERSION:
        raise ValueError("v10 physical report graph schema mismatch")
    if report.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError("v10 physical report extractor mismatch")
    if report.get("binding_digest_sha256") != digest:
        raise ValueError("v10 manifest/report binding digest mismatch")

    authorization = manifest.get("training_authorization") or {}
    if authorization.get("authorized") is not True:
        raise ValueError("v10 manifest has no explicit training authorization")
    if not str(authorization.get("decision_id") or ""):
        raise ValueError("v10 training authorization lacks a decision ID")
    if authorization.get("binding_digest_sha256") != digest:
        raise ValueError("v10 training decision is not bound to the representation")
    return digest


class LogicalV3V10TrainingDataset(LogicalV3TrainingDataset):
    """Read v10 only after physical acceptance plus explicit run authorization."""

    def __init__(self, *, representations_root: Path, **kwargs) -> None:
        if Path(representations_root).name != V10_REPRESENTATION_ROOT_NAME:
            raise ValueError("v10 dataset requires the exact candidate lineage root name")
        super().__init__(representations_root=representations_root, **kwargs)

    def _validate_manifest(self, expected_binding_digest: str | None) -> None:
        self.binding_digest = validate_v10_training_manifest(
            manifest=self.manifest,
            overlay_dir=self.overlay_dir,
            expected_binding_digest=expected_binding_digest,
        )

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        graph = sample[0]
        if getattr(graph, "graph_schema_version", None) != V10_GRAPH_SCHEMA_VERSION:
            raise ValueError("loaded graph payload is not v10")
        if (
            getattr(graph, "representation_extractor_version", None)
            != V10_REPRESENTATION_EXTRACTOR_VERSION
        ):
            raise ValueError("loaded graph payload has the wrong v10 extractor")
        edge_attr = getattr(graph, "edge_attr", None)
        num_edge_types = get_graph_schema(V10_GRAPH_SCHEMA_VERSION).num_edge_types
        if edge_attr is None or (
            edge_attr.numel()
            and (int(edge_attr.min()) < 0 or int(edge_attr.max()) >= num_edge_types)
        ):
            raise ValueError("loaded graph payload has out-of-range v10 edge IDs")
        return sample


__all__ = [
    "LogicalV3V10TrainingDataset",
    "V10_PHYSICAL_ACCEPTANCE_SCHEMA",
    "V10_TRAINING_AUTHORIZED_STATUS",
    "validate_v10_training_manifest",
    "vnext_collate_fn",
]
