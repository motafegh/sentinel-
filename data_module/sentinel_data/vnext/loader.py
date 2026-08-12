"""Read-only loader for DATA vNext v2 semantic overlays.

The loader is intentionally incompatible with the historical v1 export seam.
It never fills missing v2 semantics from legacy binary label columns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .builder import DATASET_VERSION, EXPORT_SCHEMA_VERSION
from .publication import verify_publication_bindings
from .validator import validate_vnext_overlay


class VNextExport:
    """Explicit read-only view of one DATA vNext semantic overlay."""

    def __init__(self, export_dir: Path, *, require_representation_binding: bool = False) -> None:
        self.export_dir = Path(export_dir)
        self.manifest_path = self.export_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest_surface(require_representation_binding=require_representation_binding)

    def _validate_manifest_surface(self, *, require_representation_binding: bool) -> None:
        schema_version = self.manifest.get("export_schema_version")
        if schema_version != EXPORT_SCHEMA_VERSION:
            raise ValueError(
                f"DATA vNext loader requires export_schema_version={EXPORT_SCHEMA_VERSION!r}; "
                f"got {schema_version!r}. Historical v1 exports require the legacy loader."
            )
        if self.manifest.get("dataset_version") != DATASET_VERSION:
            raise ValueError(f"unexpected DATA vNext dataset version: {self.manifest.get('dataset_version')!r}")
        if self.manifest.get("graph_schema_version") != "v9":
            raise ValueError("DATA vNext overlay does not bind graph schema v9")
        if self.manifest.get("historical_artifacts_mutated") is not False:
            raise ValueError("DATA vNext manifest does not preserve historical immutability")
        if require_representation_binding and self.manifest.get("status") != "VALIDATED_G7_CANDIDATE":
            raise ValueError("DATA vNext overlay has not passed the complete local G7 binding/validation cycle")

    @property
    def label_states_path(self) -> Path:
        return self.export_dir / "label_states.parquet"

    @property
    def ml_targets_path(self) -> Path:
        return self.export_dir / "ml_targets.parquet"

    @property
    def source_registry_path(self) -> Path:
        return self.export_dir / "source_registry.json"

    @property
    def crosswalk_registry_path(self) -> Path:
        return self.export_dir / "crosswalk_registry.json"

    @property
    def representation_requirements_path(self) -> Path:
        return self.export_dir / "representation_requirements.json"

    @property
    def role_counts(self) -> dict[str, int]:
        return dict(self.manifest.get("role_contract_counts") or {})

    def verify(self, *, require_representation_binding: bool = False) -> dict[str, Any]:
        """Run semantic and publication-binding verification as one read-only check."""
        semantic = validate_vnext_overlay(
            self.export_dir,
            require_representation_binding=require_representation_binding,
        )
        publication = verify_publication_bindings(self.export_dir)
        return {
            "passed": bool(semantic["passed"] and publication["passed"]),
            "semantic_validation": semantic,
            "publication_bindings": publication,
        }

    def _pq(self):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DATA vNext loader requires pyarrow") from exc
        return pq

    def load_ml_targets(self, columns: Iterable[str] | None = None):
        """Return the Arrow table backing the per-contract v2 ML projection."""
        if not self.ml_targets_path.is_file():
            raise FileNotFoundError(self.ml_targets_path)
        cols = list(columns) if columns is not None else None
        return self._pq().read_table(self.ml_targets_path, columns=cols)

    def load_label_states(self, columns: Iterable[str] | None = None):
        """Return the canonical contract×class Arrow table."""
        if not self.label_states_path.is_file():
            raise FileNotFoundError(self.label_states_path)
        cols = list(columns) if columns is not None else None
        return self._pq().read_table(self.label_states_path, columns=cols)

    def get_role_contract_ids(self, role: str) -> list[str]:
        """Return sorted contract IDs assigned to one frozen Phase-6 role."""
        table = self.load_ml_targets(columns=["contract_id", "role"])
        rows = table.to_pylist()
        known = set(self.role_counts)
        if role not in known:
            raise KeyError(f"unknown role {role!r}; available: {sorted(known)}")
        return sorted(str(r["contract_id"]) for r in rows if str(r["role"]) == role)

    def __repr__(self) -> str:
        pop = self.manifest.get("population") or {}
        return (
            "VNextExport("
            f"dataset_version={self.manifest.get('dataset_version')!r}, "
            f"status={self.manifest.get('status')!r}, "
            f"contracts={pop.get('contracts')}, "
            f"contract_class_rows={pop.get('contract_class_rows')}"
            ")"
        )


__all__ = ["VNextExport"]
