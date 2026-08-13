"""DATA vNext training adapter for R4 Phase 8.

This module is intentionally separate from :mod:`sentinel_dataset`, which is the
historical dense-binary/v1 compatibility path.  The Phase-8 adapter consumes the
G7-passed semantic overlay plus the existing per-contract graph/token files.

Unknown cells are represented as NaN.  They are *not* converted to zero.  The
trainer must use the explicit masks returned by this dataset; attempting to pass
the target tensor into an ordinary dense binary loss therefore fails closed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.serialization
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from sentinel_data.vnext.loader import VNextExport
from sentinel_data.vnext.policy import CLASS_NAMES

# PyTorch 2.6+ safe-global registration for PyG representation deserialization.
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

CANONICAL_G7_BINDING_DIGEST = (
    "7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420"
)
TRAIN_ROLES = frozenset({"TRAIN_STRONG", "TRAIN_WEAK"})
MODEL_SELECTION_ROLES = frozenset({"MODEL_SELECTION"})
_ALLOWED_PHASE8_ROLES = TRAIN_ROLES | MODEL_SELECTION_ROLES
_STRENGTH_CODE = {"NONE": 0, "WEAK": 1, "STRONG": 2}
_EXCLUDE_KEYS = [
    "contract_hash",
    "contract_path",
    "contract_name",
    "node_metadata",
    "num_edges",
    "num_nodes",
    "y",
]


def _row_to_supervision(row: dict) -> dict[str, torch.Tensor]:
    """Convert one vNext ML-projection row to fail-closed tensors."""
    targets: list[float] = []
    loss_mask: list[bool] = []
    metric_mask: list[bool] = []
    strength_codes: list[int] = []

    for i in range(len(CLASS_NAMES)):
        value = row[f"target_{i}"]
        if value is None:
            targets.append(float("nan"))
        else:
            numeric = float(value)
            # Policy v1 has no confirmed-negative target.  A zero here would mean
            # the ML adapter is about to recreate the exact R4 corruption class.
            if numeric == 0.0:
                raise ValueError(
                    f"DATA vNext Phase-8 adapter refuses target 0: "
                    f"{row['contract_id']} class={CLASS_NAMES[i]}"
                )
            if numeric != 1.0:
                raise ValueError(
                    f"unexpected DATA vNext target {numeric!r}: "
                    f"{row['contract_id']} class={CLASS_NAMES[i]}"
                )
            targets.append(1.0)

        strength = str(row[f"strength_{i}"])
        if strength not in _STRENGTH_CODE:
            raise ValueError(f"unknown training strength {strength!r}")
        strength_codes.append(_STRENGTH_CODE[strength])
        loss_mask.append(bool(row[f"effective_loss_mask_{i}"]))
        metric_mask.append(bool(row[f"outcome_metric_mask_{i}"]))

    target_t = torch.tensor(targets, dtype=torch.float32)
    loss_t = torch.tensor(loss_mask, dtype=torch.bool)
    metric_t = torch.tensor(metric_mask, dtype=torch.bool)
    strength_t = torch.tensor(strength_codes, dtype=torch.uint8)

    if loss_t.any():
        if not torch.isfinite(target_t[loss_t]).all():
            raise ValueError(f"loss-eligible target is null for {row['contract_id']}")
        if not torch.all(target_t[loss_t] == 1.0):
            raise ValueError(f"loss-eligible target is not positive for {row['contract_id']}")
        if torch.any(strength_t[loss_t] == _STRENGTH_CODE["NONE"]):
            raise ValueError(f"loss-eligible cell has NONE strength for {row['contract_id']}")

    if metric_t.any():
        if not torch.isfinite(target_t[metric_t]).all():
            raise ValueError(f"metric-eligible target is null for {row['contract_id']}")
        if not torch.all(target_t[metric_t] == 1.0):
            raise ValueError(f"metric-eligible target is not positive for {row['contract_id']}")

    return {
        "targets": target_t,
        "effective_loss_mask": loss_t,
        "outcome_metric_mask": metric_t,
        "strength_codes": strength_t,
    }


class VNextTrainingDataset(Dataset):
    """Read Phase-8 samples from the G7 semantic overlay + local representations.

    Parameters
    ----------
    overlay_dir:
        G7-passed ``sentinel-r4-vnext-v1`` directory.
    representations_root:
        Existing local ``data_module/data/representations`` directory.
    roles:
        Explicit Phase-6 roles to expose.  Only TRAIN_STRONG/TRAIN_WEAK and
        MODEL_SELECTION are legal in Phase 8.
    expected_binding_digest:
        Fail-closed binding to the exact G7 physical representation population.
        Pass ``None`` only in synthetic unit tests.
    verify_publication:
        Re-run semantic/publication verification at construction time.  Production
        Phase-8 entry points keep this True; tests may disable it for tiny fixtures.
    """

    def __init__(
        self,
        *,
        overlay_dir: Path,
        representations_root: Path,
        roles: Iterable[str],
        expected_binding_digest: str | None = CANONICAL_G7_BINDING_DIGEST,
        verify_publication: bool = True,
    ) -> None:
        self.overlay_dir = Path(overlay_dir)
        self.representations_root = Path(representations_root)
        self.roles = frozenset(str(r) for r in roles)

        if not self.roles:
            raise ValueError("Phase-8 dataset requires at least one explicit role")
        invalid_roles = self.roles - _ALLOWED_PHASE8_ROLES
        if invalid_roles:
            raise ValueError(
                f"Phase-8 dataset refuses roles {sorted(invalid_roles)}; "
                f"allowed={sorted(_ALLOWED_PHASE8_ROLES)}"
            )
        if self.roles & TRAIN_ROLES and self.roles & MODEL_SELECTION_ROLES:
            raise ValueError("training and MODEL_SELECTION roles must use separate datasets")
        if not self.representations_root.is_dir():
            raise FileNotFoundError(self.representations_root)

        self.export = VNextExport(
            self.overlay_dir,
            require_representation_binding=True,
        )
        if verify_publication:
            verification = self.export.verify(require_representation_binding=True)
            if not verification["passed"]:
                raise ValueError("DATA vNext publication verification failed")

        manifest_binding = (
            self.export.manifest.get("representation_binding_report") or {}
        ).get("binding_digest_sha256")
        if expected_binding_digest is not None and manifest_binding != expected_binding_digest:
            raise ValueError(
                "DATA vNext representation binding digest mismatch: "
                f"{manifest_binding!r} != {expected_binding_digest!r}"
            )
        self.binding_digest = str(manifest_binding or "")

        rows = self.export.load_ml_targets().to_pylist()
        selected = [r for r in rows if str(r["role"]) in self.roles]
        selected.sort(key=lambda r: (str(r["group_id"]), str(r["contract_id"])))
        if not selected:
            raise ValueError(f"no DATA vNext rows for roles={sorted(self.roles)}")

        # Phase-6 roles are frozen at GROUP level.  Therefore some contracts
        # legitimately inherit a supervised group role while carrying no
        # optimizer/metric-eligible cell themselves.
        frozen_role_counts = Counter(str(r["role"]) for r in selected)
        frozen_groups = {str(r["group_id"]) for r in selected}

        self._rows: list[dict] = []
        self._supervision: dict[str, dict[str, torch.Tensor]] = {}
        self._group_to_indices: dict[str, list[int]] = defaultdict(list)

        active_role_counts: Counter[str] = Counter()
        skipped_no_signal_counts: Counter[str] = Counter()

        for row in selected:
            cid = str(row["contract_id"])
            role = str(row["role"])

            if not bool(row["representation_required"]):
                raise ValueError(
                    f"Phase-8 role {role} unexpectedly lacks representation: {cid}"
                )

            supervision = _row_to_supervision(row)
            loss_mask = supervision["effective_loss_mask"]
            metric_mask = supervision["outcome_metric_mask"]

            if role in TRAIN_ROLES:
                # A training GROUP can contain sibling contracts without direct
                # supervision.  Those siblings remain in the frozen partition
                # but must not enter the optimizer.
                if not loss_mask.any():
                    skipped_no_signal_counts[role] += 1
                    continue

                codes = supervision["strength_codes"][loss_mask]
                expected_code = _STRENGTH_CODE[
                    "STRONG" if role == "TRAIN_STRONG" else "WEAK"
                ]
                if not torch.all(codes == expected_code):
                    raise ValueError(
                        f"role/strength mismatch for {cid} ({role})"
                    )

            elif role == "MODEL_SELECTION":
                if loss_mask.any():
                    raise ValueError(
                        f"MODEL_SELECTION contract unexpectedly has optimizer cells: {cid}"
                    )

                # Same group-level rule: only contracts with authorized metric
                # cells participate in checkpoint selection.
                if not metric_mask.any():
                    skipped_no_signal_counts[role] += 1
                    continue

            idx = len(self._rows)
            self._rows.append(row)
            self._supervision[cid] = supervision
            self._group_to_indices[str(row["group_id"])].append(idx)
            active_role_counts[role] += 1

        if not self._rows:
            raise ValueError(
                f"roles={sorted(self.roles)} contain no contracts with "
                "authorized Phase-8 cells"
            )

        active_groups = set(self._group_to_indices)
        missing_groups = frozen_groups - active_groups

        # Every frozen supervised group was classified using at least one
        # authorized signal.  If an entire group disappears here, G6 and G7
        # semantics have diverged and we fail closed.
        if missing_groups:
            preview = sorted(missing_groups)[:5]
            raise ValueError(
                "frozen supervised groups lost all authorized Phase-8 cells: "
                f"count={len(missing_groups)} preview={preview}"
            )

        # Keep BOTH views explicit:
        #   frozen_* = complete Phase-6 group-role population
        #   role_counts/group_count = active optimizer/evaluation population
        self.frozen_role_counts = dict(sorted(frozen_role_counts.items()))
        self.frozen_group_count = len(frozen_groups)

        self.role_counts = dict(sorted(active_role_counts.items()))
        self.group_count = len(active_groups)

        self.skipped_no_signal_counts = dict(
            sorted(skipped_no_signal_counts.items())
        )
        self.skipped_no_signal_contracts = sum(
            skipped_no_signal_counts.values()
        )

    @property
    def group_to_indices(self) -> dict[str, tuple[int, ...]]:
        return {k: tuple(v) for k, v in self._group_to_indices.items()}

    @property
    def contract_ids(self) -> tuple[str, ...]:
        return tuple(str(r["contract_id"]) for r in self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        row = self._rows[idx]
        cid = str(row["contract_id"])
        source = str(row["source"])
        source_dir = self.representations_root / source
        graph_path = source_dir / f"{cid}.pt"
        tokens_path = source_dir / f"{cid}.tokens.pt"
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        if not tokens_path.is_file():
            raise FileNotFoundError(tokens_path)

        graph: Data = torch.load(graph_path, weights_only=False)
        token_payload = torch.load(tokens_path, weights_only=True)
        input_ids = token_payload["input_ids"]
        attention_mask = token_payload["attention_mask"]
        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        supervision = {
            k: v.clone() for k, v in self._supervision[cid].items()
        }
        return (
            graph,
            tokens,
            supervision,
            cid,
            str(row["role"]),
            str(row["group_id"]),
        )


def vnext_collate_fn(batch: list[tuple]):
    """Collate Phase-8 samples while keeping semantic masks explicit."""
    graphs, token_rows, supervision_rows, contract_ids, roles, group_ids = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs), exclude_keys=_EXCLUDE_KEYS)
    tokens = {
        "input_ids": torch.stack([r["input_ids"] for r in token_rows], dim=0),
        "attention_mask": torch.stack([r["attention_mask"] for r in token_rows], dim=0),
    }
    supervision = {
        "targets": torch.stack([r["targets"] for r in supervision_rows], dim=0),
        "effective_loss_mask": torch.stack(
            [r["effective_loss_mask"] for r in supervision_rows], dim=0
        ),
        "outcome_metric_mask": torch.stack(
            [r["outcome_metric_mask"] for r in supervision_rows], dim=0
        ),
        "strength_codes": torch.stack([r["strength_codes"] for r in supervision_rows], dim=0),
    }
    return (
        graph_batch,
        tokens,
        supervision,
        list(contract_ids),
        list(roles),
        list(group_ids),
    )


__all__ = [
    "CANONICAL_G7_BINDING_DIGEST",
    "MODEL_SELECTION_ROLES",
    "TRAIN_ROLES",
    "VNextTrainingDataset",
    "vnext_collate_fn",
]
