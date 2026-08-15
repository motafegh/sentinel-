"""Phase-8 dataset adapter for corrected logical DATA lineage V3.

V3 reuses the accepted repaired-v2 graph/token files but consumes the corrected
V3 role partition and publication.  Supervision tensor semantics and the model
architecture remain unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.src.datasets.vnext_repaired_dataset import (
    RepairedVNextTrainingDataset,
    _sha256,
    vnext_collate_fn,
)
from sentinel_data.preprocessing.r4_versions import GRAPH_SCHEMA_VERSION
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)


class LogicalV3TrainingDataset(RepairedVNextTrainingDataset):
    """Read active Phase-8 samples from the physically bound V3 publication."""

    def _validate_manifest(self, expected_binding_digest: str | None) -> None:
        if self.manifest.get("dataset_version") != DATASET_VERSION_V3:
            raise ValueError("logical-v3 Phase-8 dataset version mismatch")
        if self.manifest.get("export_schema_version") != "v2":
            raise ValueError("logical-v3 Phase-8 requires export schema v2")
        if self.manifest.get("partition_version") != ROLE_PARTITION_VERSION_V3:
            raise ValueError("logical-v3 Phase-8 role partition mismatch")
        if self.manifest.get("grouping_version") != GROUPING_VERSION_V3:
            raise ValueError("logical-v3 Phase-8 grouping version mismatch")
        if self.manifest.get("address_literal_grouping_authority") is not False:
            raise ValueError("logical-v3 unexpectedly enables address grouping authority")
        if self.manifest.get("confirmed_negative_rows") != 0:
            raise ValueError("logical-v3 Phase-8 unexpectedly contains confirmed negatives")
        if (
            self.manifest.get("status")
            != "LOGICAL_V3_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED"
        ):
            raise ValueError("logical-v3 publication has not passed representation binding")

        binding = self.manifest.get("representation_binding_report") or {}
        digest = str(binding.get("binding_digest_sha256") or "")
        if not digest:
            raise ValueError("logical-v3 manifest lacks representation binding digest")
        if expected_binding_digest is not None and digest != expected_binding_digest:
            raise ValueError(
                f"logical-v3 representation binding mismatch: {digest} != {expected_binding_digest}"
            )
        self.binding_digest = digest

        report_path = Path(self.overlay_dir) / "representation_binding_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        if not binding.get("sha256") or _sha256(report_path) != binding.get("sha256"):
            raise ValueError("logical-v3 representation binding report hash mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise ValueError("logical-v3 representation binding report is not passing")
        if report.get("dataset_version") != DATASET_VERSION_V3:
            raise ValueError("logical-v3 binding report dataset version mismatch")
        if report.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
            raise ValueError("logical-v3 representation graph schema mismatch")
        if report.get("binding_digest_sha256") != digest:
            raise ValueError("logical-v3 manifest/report binding digest mismatch")
        if report.get("address_literal_grouping_authority") is not False:
            raise ValueError("logical-v3 binding report lost grouping policy boundary")


__all__ = ["LogicalV3TrainingDataset", "vnext_collate_fn"]
