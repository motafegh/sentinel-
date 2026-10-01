"""Fail-closed Phase-8 adapter for accepted logical-V3 + R4-D-013 tokens."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ml.src.datasets.vnext_logical_v3_dataset import LogicalV3TrainingDataset
from ml.src.datasets.vnext_repaired_dataset import _sha256, vnext_collate_fn
from sentinel_data.preprocessing.r4_versions import (
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_TOKEN_LINEAGE_VERSION,
    GUARDED_TOKEN_SELECTOR_VERSION,
    GUARDED_TOKEN_TRANSFORMERS_VERSION,
    HISTORICAL_TOKEN_SELECTOR_VERSION,
    TOKEN_TENSOR_SHAPE,
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.representation.graph_schema_versions import get_graph_schema
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)

R4_D009_LOGICAL_ACCEPTANCE_SCHEMA = "sentinel-r4-logical-v3-acceptance-v1"
R4_D013_ACCEPTANCE_SCHEMA = "sentinel-r4-guarded-token-physical-acceptance-v1"
R4_D013_DECISION_ID = "R4-D-013"
R4_D013_DECISION = "ACCEPTED_IMMUTABLE_LOCAL_GUARDED_TOKEN_REPRESENTATION"
R4_D013_BINDING_DIGEST = "9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881"
R4_D011_DECISION_ID = "R4-D-011"
R4_D011_BINDING_DIGEST = "d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd"
EXPECTED_GUARDED_CONTRACTS = 22540


def _load_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_logical_v3_acceptance(*, manifest_path: Path, acceptance_path: Path) -> dict:
    acceptance = _load_json(acceptance_path)
    if acceptance.get("schema") != R4_D009_LOGICAL_ACCEPTANCE_SCHEMA:
        raise ValueError("logical-v3 acceptance schema mismatch")
    if acceptance.get("status") != "PASS":
        raise ValueError("logical-v3 acceptance is not passing")
    if acceptance.get("training_authorized") is not False:
        raise ValueError("logical-v3 acceptance must not grant training by itself")
    versions = acceptance.get("versions") or {}
    if versions.get("dataset") != DATASET_VERSION_V3:
        raise ValueError("logical-v3 accepted dataset version mismatch")
    if versions.get("grouping") != GROUPING_VERSION_V3:
        raise ValueError("logical-v3 accepted grouping version mismatch")
    if versions.get("partition") != ROLE_PARTITION_VERSION_V3:
        raise ValueError("logical-v3 accepted partition version mismatch")
    if (acceptance.get("checks") or {}).get("confirmed_negative_rows_zero") is not True:
        raise ValueError("logical-v3 acceptance no longer proves zero confirmed negatives")
    expected_manifest_sha = str((acceptance.get("lineage") or {}).get("publication_manifest_sha256") or "")
    if not expected_manifest_sha or _sha256(Path(manifest_path)) != expected_manifest_sha:
        raise ValueError("logical-v3 publication manifest no longer matches accepted R4-D-009 evidence")
    return acceptance


def validate_guarded_physical_acceptance(
    *,
    repo_root: Path,
    representations_root: Path,
    acceptance_path: Path,
    expected_binding_digest: str = R4_D013_BINDING_DIGEST,
) -> tuple[dict, dict, dict[tuple[str, str], dict]]:
    repo_root = Path(repo_root).resolve()
    representations_root = Path(representations_root).resolve()
    acceptance = _load_json(acceptance_path)
    if acceptance.get("schema") != R4_D013_ACCEPTANCE_SCHEMA:
        raise ValueError("R4-D-013 acceptance schema mismatch")
    if acceptance.get("decision_id") != R4_D013_DECISION_ID:
        raise ValueError("guarded physical acceptance decision ID mismatch")
    if acceptance.get("status") != "PASS" or acceptance.get("physical_acceptance") is not True:
        raise ValueError("guarded token lineage is not physically accepted")
    if acceptance.get("decision") != R4_D013_DECISION:
        raise ValueError("guarded physical acceptance decision mismatch")
    if acceptance.get("training_authorized") is not False:
        raise ValueError("R4-D-013 must not grant full training authority")

    lineage = acceptance.get("accepted_lineage") or {}
    digest = str(lineage.get("binding_digest_sha256") or "")
    if digest != expected_binding_digest or digest != R4_D013_BINDING_DIGEST:
        raise ValueError("R4-D-013 guarded binding digest mismatch")
    if int(lineage.get("contracts", -1)) != EXPECTED_GUARDED_CONTRACTS:
        raise ValueError("R4-D-013 guarded contract population mismatch")
    if lineage.get("representation_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
        raise ValueError("R4-D-013 token lineage mismatch")
    if lineage.get("selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
        raise ValueError("R4-D-013 selector policy mismatch")
    if lineage.get("control_selector") != HISTORICAL_TOKEN_SELECTOR_VERSION:
        raise ValueError("R4-D-013 control selector mismatch")
    if lineage.get("transformers_version") != GUARDED_TOKEN_TRANSFORMERS_VERSION:
        raise ValueError("R4-D-013 Transformers version mismatch")
    if lineage.get("graph_schema_version") != V10_GRAPH_SCHEMA_VERSION:
        raise ValueError("R4-D-013 graph schema mismatch")
    if lineage.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError("R4-D-013 extractor version mismatch")
    if tuple(lineage.get("token_shape") or ()) != TOKEN_TENSOR_SHAPE:
        raise ValueError("R4-D-013 token shape mismatch")
    if lineage.get("graph_parent_decision_id") != R4_D011_DECISION_ID:
        raise ValueError("R4-D-013 graph-parent decision mismatch")
    if lineage.get("graph_parent_binding_digest_sha256") != R4_D011_BINDING_DIGEST:
        raise ValueError("R4-D-013 graph-parent digest mismatch")

    physical_root = str(lineage.get("physical_root") or "")
    if not physical_root:
        raise ValueError("R4-D-013 acceptance lacks physical root")
    expected_root = (repo_root / physical_root).resolve()
    if representations_root != expected_root:
        raise ValueError(
            "guarded dataset root is not the exact R4-D-013 accepted root: "
            f"{representations_root} != {expected_root}"
        )
    if representations_root.name != GUARDED_REPRESENTATION_ROOT_NAME:
        raise ValueError("guarded representation root basename mismatch")

    candidate_manifest_path = representations_root / "guarded_candidate_manifest.json"
    expected_candidate_sha = str(lineage.get("candidate_manifest_sha256") or "")
    if not expected_candidate_sha or _sha256(candidate_manifest_path) != expected_candidate_sha:
        raise ValueError("R4-D-013 candidate manifest SHA-256 mismatch")
    candidate = _load_json(candidate_manifest_path)
    if candidate.get("schema") != "sentinel-r4-guarded-token-candidate-manifest-v1":
        raise ValueError("guarded candidate manifest schema mismatch")
    if candidate.get("status") != "FULL_GUARDED_TOKEN_CANDIDATE":
        raise ValueError("guarded candidate is not the full D5 population")
    if candidate.get("full_population") is not True:
        raise ValueError("guarded candidate manifest is not full-population")
    if candidate.get("physical_acceptance") is not False:
        raise ValueError("candidate manifest must remain pre-acceptance historical evidence")
    if candidate.get("training_authorized") is not False:
        raise ValueError("candidate manifest must not claim training authority")
    if int(candidate.get("contracts_written", -1)) != EXPECTED_GUARDED_CONTRACTS:
        raise ValueError("guarded candidate written population mismatch")
    if candidate.get("binding_digest_sha256") != digest:
        raise ValueError("guarded candidate/acceptance binding digest mismatch")
    if candidate.get("representation_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
        raise ValueError("guarded candidate token lineage mismatch")
    if candidate.get("selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
        raise ValueError("guarded candidate selector mismatch")
    if candidate.get("control_selector") != HISTORICAL_TOKEN_SELECTOR_VERSION:
        raise ValueError("guarded candidate control selector mismatch")

    records = candidate.get("records") or []
    if len(records) != EXPECTED_GUARDED_CONTRACTS:
        raise ValueError("guarded candidate record population mismatch")
    by_identity: dict[tuple[str, str], dict] = {}
    for raw in records:
        row = dict(raw)
        key = (str(row.get("source") or ""), str(row.get("contract_id") or ""))
        if not all(key) or key in by_identity:
            raise ValueError(f"invalid/duplicate guarded candidate identity: {key}")
        by_identity[key] = row
    return acceptance, candidate, by_identity


class LogicalV3GuardedTrainingDataset(LogicalV3TrainingDataset):
    """Read accepted logical-V3 roles from exact R4-D-013 physical artifacts."""

    def __init__(
        self,
        *,
        repo_root: Path,
        logical_acceptance_path: Path,
        physical_acceptance_path: Path,
        representations_root: Path,
        expected_binding_digest: str = R4_D013_BINDING_DIGEST,
        verify_active_artifacts: bool = True,
        **kwargs,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.logical_acceptance_path = Path(logical_acceptance_path).resolve()
        self.physical_acceptance_path = Path(physical_acceptance_path).resolve()
        self.expected_guarded_binding_digest = str(expected_binding_digest)
        self.verify_active_artifacts = bool(verify_active_artifacts)
        if Path(representations_root).name != GUARDED_REPRESENTATION_ROOT_NAME:
            raise ValueError("guarded dataset requires exact R4-D-013 root basename")
        super().__init__(
            representations_root=representations_root,
            expected_binding_digest=None,
            **kwargs,
        )
        if self.verify_active_artifacts:
            self._verify_active_artifacts()

    def _validate_manifest(self, expected_binding_digest: str | None) -> None:
        LogicalV3TrainingDataset._validate_manifest(self, None)
        self.logical_v3_parent_binding_digest = self.binding_digest
        validate_logical_v3_acceptance(
            manifest_path=Path(self.overlay_dir) / "manifest.json",
            acceptance_path=self.logical_acceptance_path,
        )
        acceptance, candidate, records = validate_guarded_physical_acceptance(
            repo_root=self.repo_root,
            representations_root=self.representations_root,
            acceptance_path=self.physical_acceptance_path,
            expected_binding_digest=self.expected_guarded_binding_digest,
        )
        self.guarded_acceptance = acceptance
        self.guarded_candidate_manifest = candidate
        self._accepted_record_by_identity = records
        self.binding_digest = self.expected_guarded_binding_digest

    def _verify_active_artifacts(self) -> None:
        for row in self._rows:
            source = str(row["source"])
            contract_id = str(row["contract_id"])
            record = self._accepted_record_by_identity.get((source, contract_id))
            if record is None:
                raise ValueError(f"active identity absent from R4-D-013 manifest: {source}/{contract_id}")
            source_dir = self.representations_root / source
            paths = {
                "graph_sha256": source_dir / f"{contract_id}.pt",
                "tokens_sha256": source_dir / f"{contract_id}.tokens.pt",
                "sidecar_sha256": source_dir / f"{contract_id}.rep.json",
            }
            for field, artifact in paths.items():
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
                if _sha256(artifact) != record.get(field):
                    raise ValueError(f"R4-D-013 active artifact hash mismatch: {source}/{contract_id} {field}")

    def __getitem__(self, index: int):
        row = self._rows[index]
        contract_id = str(row["contract_id"])
        source = str(row["source"])
        source_dir = self.representations_root / source
        graph_path = source_dir / f"{contract_id}.pt"
        tokens_path = source_dir / f"{contract_id}.tokens.pt"
        graph = torch.load(graph_path, weights_only=False)
        token_payload = torch.load(tokens_path, weights_only=True)

        if getattr(graph, "graph_schema_version", None) != V10_GRAPH_SCHEMA_VERSION:
            raise ValueError(f"loaded R4-D-013 graph is not v10: {source}/{contract_id}")
        if getattr(graph, "representation_extractor_version", None) != V10_REPRESENTATION_EXTRACTOR_VERSION:
            raise ValueError(f"loaded R4-D-013 graph extractor mismatch: {source}/{contract_id}")
        edge_attr = getattr(graph, "edge_attr", None)
        num_edge_types = get_graph_schema(V10_GRAPH_SCHEMA_VERSION).num_edge_types
        if edge_attr is None or (edge_attr.numel() and (int(edge_attr.min()) < 0 or int(edge_attr.max()) >= num_edge_types)):
            raise ValueError(f"loaded R4-D-013 graph has invalid v10 edge IDs: {source}/{contract_id}")

        if token_payload.get("sha256") != contract_id or token_payload.get("source") != source:
            raise ValueError(f"R4-D-013 token identity mismatch: {source}/{contract_id}")
        if token_payload.get("token_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
            raise ValueError(f"R4-D-013 token lineage mismatch: {source}/{contract_id}")
        if token_payload.get("token_selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
            raise ValueError(f"R4-D-013 token selector mismatch: {source}/{contract_id}")
        if token_payload.get("transformers_version") != GUARDED_TOKEN_TRANSFORMERS_VERSION:
            raise ValueError(f"R4-D-013 token runtime mismatch: {source}/{contract_id}")
        decision = token_payload.get("selector_decision") or {}
        if decision.get("requested_selector") != GUARDED_TOKEN_SELECTOR_VERSION:
            raise ValueError(f"R4-D-013 requested selector mismatch: {source}/{contract_id}")
        if decision.get("control_selector") != HISTORICAL_TOKEN_SELECTOR_VERSION:
            raise ValueError(f"R4-D-013 control selector mismatch: {source}/{contract_id}")
        if decision.get("effective_selector") not in {GUARDED_TOKEN_SELECTOR_VERSION, HISTORICAL_TOKEN_SELECTOR_VERSION}:
            raise ValueError(f"R4-D-013 effective selector mismatch: {source}/{contract_id}")

        input_ids = token_payload.get("input_ids")
        attention_mask = token_payload.get("attention_mask")
        if tuple(input_ids.shape) != TOKEN_TENSOR_SHAPE or tuple(attention_mask.shape) != TOKEN_TENSOR_SHAPE:
            raise ValueError(f"R4-D-013 token tensor shape mismatch: {source}/{contract_id}")
        if input_ids.dtype != torch.long or attention_mask.dtype != torch.long:
            raise ValueError(f"R4-D-013 token tensor dtype mismatch: {source}/{contract_id}")

        supervision = {key: value.clone() for key, value in self._supervision[contract_id].items()}
        return (
            graph,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            supervision,
            contract_id,
            str(row["role"]),
            str(row["group_id"]),
        )


__all__ = [
    "LogicalV3GuardedTrainingDataset",
    "R4_D009_LOGICAL_ACCEPTANCE_SCHEMA",
    "R4_D011_BINDING_DIGEST",
    "R4_D013_ACCEPTANCE_SCHEMA",
    "R4_D013_BINDING_DIGEST",
    "R4_D013_DECISION_ID",
    "validate_guarded_physical_acceptance",
    "validate_logical_v3_acceptance",
    "vnext_collate_fn",
]
