"""DATA vNext repaired-v2 adapter for the bounded Phase-8 GPU smoke.

The historical :mod:`vnext_dataset` remains bound to the G7-passed
``sentinel-r4-vnext-v1`` population.  This module consumes only the physically
bound ``sentinel-r4-vnext-v2`` local publication and derives population counts
dynamically from its frozen repaired roles.

It reuses the same positive-only supervision tensor semantics and collate
function; no model architecture or loss semantics are changed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from ml.src.datasets.vnext_dataset import (
    MODEL_SELECTION_ROLES,
    TRAIN_ROLES,
    _row_to_supervision,
    vnext_collate_fn,
)
from sentinel_data.preprocessing.r4_versions import (
    GRAPH_SCHEMA_VERSION,
    REPAIRED_DATA_PUBLICATION_ID,
    REPAIRED_ROLE_PARTITION_ID,
)

_ALLOWED_PHASE8_ROLES = TRAIN_ROLES | MODEL_SELECTION_ROLES


class RepairedVNextTrainingDataset(Dataset):
    """Read repaired-v2 samples from the local physically bound publication."""

    def __init__(
        self,
        *,
        overlay_dir: Path,
        representations_root: Path,
        roles: Iterable[str],
        expected_binding_digest: str | None = None,
    ) -> None:
        self.overlay_dir = Path(overlay_dir)
        self.representations_root = Path(representations_root)
        self.roles = frozenset(str(role) for role in roles)

        if not self.roles:
            raise ValueError("repaired Phase-8 dataset requires an explicit role")
        invalid = self.roles - _ALLOWED_PHASE8_ROLES
        if invalid:
            raise ValueError(f"repaired Phase-8 dataset refuses roles {sorted(invalid)}")
        if self.roles & TRAIN_ROLES and self.roles & MODEL_SELECTION_ROLES:
            raise ValueError("training and MODEL_SELECTION must use separate datasets")
        if not self.representations_root.is_dir():
            raise FileNotFoundError(self.representations_root)

        manifest_path = self.overlay_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(expected_binding_digest)

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("repaired Phase-8 dataset requires pyarrow") from exc
        ml_targets = self.overlay_dir / "ml_targets.parquet"
        rows = pq.read_table(ml_targets).to_pylist()
        selected = [row for row in rows if str(row["role"]) in self.roles]
        selected.sort(key=lambda row: (str(row["group_id"]), str(row["contract_id"])))
        if not selected:
            raise ValueError(f"no repaired DATA rows for roles={sorted(self.roles)}")

        frozen_role_counts = Counter(str(row["role"]) for row in selected)
        frozen_groups = {str(row["group_id"]) for row in selected}
        self._rows: list[dict] = []
        self._supervision: dict[str, dict[str, torch.Tensor]] = {}
        self._group_to_indices: dict[str, list[int]] = defaultdict(list)
        active_role_counts: Counter[str] = Counter()
        skipped_no_signal_counts: Counter[str] = Counter()

        for row in selected:
            contract_id = str(row["contract_id"])
            role = str(row["role"])
            if not bool(row["representation_required"]):
                raise ValueError(
                    f"repaired supervised role unexpectedly lacks representation: "
                    f"{contract_id} role={role}"
                )
            supervision = _row_to_supervision(row)
            loss_mask = supervision["effective_loss_mask"]
            metric_mask = supervision["outcome_metric_mask"]

            if role in TRAIN_ROLES:
                if not loss_mask.any():
                    skipped_no_signal_counts[role] += 1
                    continue
                codes = supervision["strength_codes"][loss_mask]
                expected_code = 2 if role == "TRAIN_STRONG" else 1
                if not torch.all(codes == expected_code):
                    raise ValueError(
                        f"repaired role/strength mismatch for {contract_id} ({role})"
                    )
            elif role == "MODEL_SELECTION":
                if loss_mask.any():
                    raise ValueError(
                        f"repaired MODEL_SELECTION has optimizer cells: {contract_id}"
                    )
                if not metric_mask.any():
                    skipped_no_signal_counts[role] += 1
                    continue

            index = len(self._rows)
            self._rows.append(row)
            self._supervision[contract_id] = supervision
            self._group_to_indices[str(row["group_id"])].append(index)
            active_role_counts[role] += 1

        if not self._rows:
            raise ValueError("repaired role population has no authorized active cells")
        active_groups = set(self._group_to_indices)
        missing_groups = frozen_groups - active_groups
        if missing_groups:
            raise ValueError(
                "repaired frozen supervised groups lost all authorized cells: "
                f"count={len(missing_groups)} preview={sorted(missing_groups)[:5]}"
            )

        self.frozen_role_counts = dict(sorted(frozen_role_counts.items()))
        self.frozen_group_count = len(frozen_groups)
        self.role_counts = dict(sorted(active_role_counts.items()))
        self.group_count = len(active_groups)
        self.skipped_no_signal_counts = dict(sorted(skipped_no_signal_counts.items()))
        self.skipped_no_signal_contracts = sum(skipped_no_signal_counts.values())

        expected_roles = dict(self.manifest.get("role_contract_counts") or {})
        for role, count in self.frozen_role_counts.items():
            if int(expected_roles.get(role, -1)) != int(count):
                raise ValueError(
                    f"repaired manifest/ML role count mismatch for {role}: "
                    f"manifest={expected_roles.get(role)} rows={count}"
                )

    def _validate_manifest(self, expected_binding_digest: str | None) -> None:
        if self.manifest.get("dataset_version") != REPAIRED_DATA_PUBLICATION_ID:
            raise ValueError("repaired Phase-8 dataset version mismatch")
        if self.manifest.get("export_schema_version") != "v2":
            raise ValueError("repaired Phase-8 requires export schema v2")
        if self.manifest.get("partition_version") != REPAIRED_ROLE_PARTITION_ID:
            raise ValueError("repaired Phase-8 role partition mismatch")
        if self.manifest.get("confirmed_negative_rows") != 0:
            raise ValueError("repaired Phase-8 unexpectedly contains confirmed negatives")
        if self.manifest.get("status") != "REPAIRED_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED":
            raise ValueError(
                "repaired DATA publication has not passed physical representation binding"
            )
        binding = self.manifest.get("representation_binding_report") or {}
        digest = str(binding.get("binding_digest_sha256") or "")
        if not digest:
            raise ValueError("repaired DATA manifest lacks representation binding digest")
        if expected_binding_digest is not None and digest != expected_binding_digest:
            raise ValueError(
                f"repaired representation binding mismatch: {digest} != {expected_binding_digest}"
            )
        self.binding_digest = digest
        # The repaired builder intentionally retains the frozen v9 graph schema
        # even though its manifest schema differs from historical v1.
        report_path = self.overlay_dir / "representation_binding_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise ValueError("repaired representation binding report is not passing")
        if report.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
            raise ValueError("repaired representation graph schema mismatch")
        if report.get("binding_digest_sha256") != digest:
            raise ValueError("repaired manifest/report binding digest mismatch")

    @property
    def group_to_indices(self) -> dict[str, tuple[int, ...]]:
        return {key: tuple(values) for key, values in self._group_to_indices.items()}

    @property
    def contract_ids(self) -> tuple[str, ...]:
        return tuple(str(row["contract_id"]) for row in self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        contract_id = str(row["contract_id"])
        source = str(row["source"])
        source_dir = self.representations_root / source
        graph_path = source_dir / f"{contract_id}.pt"
        tokens_path = source_dir / f"{contract_id}.tokens.pt"
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        if not tokens_path.is_file():
            raise FileNotFoundError(tokens_path)

        graph = torch.load(graph_path, weights_only=False)
        token_payload = torch.load(tokens_path, weights_only=True)
        supervision = {
            key: value.clone() for key, value in self._supervision[contract_id].items()
        }
        return (
            graph,
            {
                "input_ids": token_payload["input_ids"],
                "attention_mask": token_payload["attention_mask"],
            },
            supervision,
            contract_id,
            str(row["role"]),
            str(row["group_id"]),
        )


__all__ = ["RepairedVNextTrainingDataset", "vnext_collate_fn"]
