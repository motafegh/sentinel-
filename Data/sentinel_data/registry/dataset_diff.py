"""Dataset diff + changelog — Stage 5 Task 5.7.

dataset_diff takes two dataset versions and reports:
  - added/removed contracts (by sha256)
  - label changes (per class, per contract)
  - class distribution deltas (per class, per split)
  - per-class metric projection (per AUDIT_PATCHES 5-P7): for each
    class, show new vs old count, label distribution, confidence tier
    breakdown. The model team uses this to predict "Run 11's per-class
    F1 will likely be X% better than Run 9's because the v2 corpus has
    30% more Reentrancy positives."

changelog.md is the human-readable change log. Updated with every
dataset version registration.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("sentinel_data.registry.dataset_diff")


@dataclass
class PerClassMetric:
    class_name: str
    count_old: int
    count_new: int
    delta_count: int
    delta_pct: float                # (new - old) / old * 100
    tier_breakdown_old: dict[str, int] = field(default_factory=dict)
    tier_breakdown_new: dict[str, int] = field(default_factory=dict)
    predicted_f1_delta_pct: float = 0.0  # coarse heuristic

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetDiff:
    name_old: str
    name_new: str
    added_contracts: list[str] = field(default_factory=list)
    removed_contracts: list[str] = field(default_factory=list)
    common_contracts: list[str] = field(default_factory=list)
    label_changes: list[dict] = field(default_factory=list)
    per_class: list[PerClassMetric] = field(default_factory=list)
    computed_at: str = ""

    def to_dict(self) -> dict:
        d = {
            "name_old": self.name_old,
            "name_new": self.name_new,
            "added_contracts": self.added_contracts,
            "removed_contracts": self.removed_contracts,
            "n_common": len(self.common_contracts),
            "label_changes": self.label_changes,
            "per_class": [pc.to_dict() for pc in self.per_class],
            "computed_at": self.computed_at,
        }
        return d


def _build_label_map(dataset_version_metadata: dict) -> dict[str, dict[str, int]]:
    """Build {sha256 -> {class_name: value}} from a dataset version's metadata.

    The metadata field is expected to be {contract_labels: {sha: {class: val}, ...}}.
    """
    return dataset_version_metadata.get("contract_labels", {})


def diff_dataset_versions(
    metadata_old: dict,
    metadata_new: dict,
    name_old: str = "old",
    name_new: str = "new",
) -> DatasetDiff:
    """Compute the diff between two dataset versions.

    Inputs are the `metadata` dict of each DatasetVersion (which contains
    the per-contract labels and tier info). For large corpora, this is
    an O(N) comparison.

    Returns a DatasetDiff with all the comparison data.
    """
    from datetime import datetime, timezone
    labels_old = _build_label_map(metadata_old)
    labels_new = _build_label_map(metadata_new)
    shas_old = set(labels_old)
    shas_new = set(labels_new)
    common = shas_old & shas_new
    added = sorted(shas_new - shas_old)
    removed = sorted(shas_old - shas_new)

    # Label changes (for common contracts, compare per-class)
    label_changes: list[dict] = []
    for sha in sorted(common):
        old_classes = labels_old[sha]
        new_classes = labels_new[sha]
        diffs = {c: (old_classes.get(c, 0), new_classes.get(c, 0))
                 for c in set(old_classes) | set(new_classes)
                 if old_classes.get(c, 0) != new_classes.get(c, 0)}
        if diffs:
            label_changes.append({"sha256": sha, "changes": diffs})

    # Per-class metric projection
    all_classes = set()
    for sha_classes in list(labels_old.values()) + list(labels_new.values()):
        all_classes.update(sha_classes.keys())

    per_class: list[PerClassMetric] = []
    for cls in sorted(all_classes):
        old_count = sum(1 for s in labels_old if labels_old[s].get(cls, 0) == 1)
        new_count = sum(1 for s in labels_new if labels_new[s].get(cls, 0) == 1)
        delta = new_count - old_count
        delta_pct = (delta / old_count * 100) if old_count > 0 else 0.0
        # Coarse F1 projection: more positives → likely higher recall
        # (assuming the model was recall-bound on this class)
        predicted_f1_delta = min(5.0, abs(delta_pct) * 0.1)
        per_class.append(PerClassMetric(
            class_name=cls,
            count_old=old_count,
            count_new=new_count,
            delta_count=delta,
            delta_pct=delta_pct,
            predicted_f1_delta_pct=predicted_f1_delta,
        ))

    return DatasetDiff(
        name_old=name_old,
        name_new=name_new,
        added_contracts=added,
        removed_contracts=removed,
        common_contracts=sorted(common),
        label_changes=label_changes,
        per_class=per_class,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def update_changelog(changelog_path: Path, version_name: str, summary: str,
                    metrics: dict) -> None:
    """Append a new entry to data/changelog.md."""
    import datetime
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = f"\n## {version_name} ({ts})\n\n{summary}\n"
    if metrics:
        entry += "\n### Per-class metric projection\n\n"
        entry += "| Class | Old | New | Δ Count | Δ % |\n"
        entry += "|---|---|---|---|---|\n"
        for cls, m in sorted(metrics.items()):
            entry += f"| {cls} | {m['count_old']} | {m['count_new']} | {m['delta_count']:+d} | {m['delta_pct']:+.1f}% |\n"

    with changelog_path.open("a") as f:
        f.write(entry)
    log.info(f"Updated changelog: {changelog_path}")
