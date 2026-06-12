"""Stage 6 — balance_viz: per-class / per-source / per-tier counts and bar plot.

D-6.1 (analysis is read-only, DVC-tracked outputs).
Outputs:
  - data/analysis/<run_id>/balance_table.csv
  - data/analysis/<run_id>/balance_plot.png
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sentinel_data.labeling.schema import class_names


@dataclass
class BalanceTable:
    """The per-class / per-source / per-tier count matrix."""
    per_class: dict[str, int] = field(default_factory=dict)         # class -> positive count
    per_source: dict[str, int] = field(default_factory=dict)        # source -> positive count
    per_tier: dict[str, dict[str, int]] = field(default_factory=dict)  # tier -> {class -> count}
    per_class_source: dict[str, dict[str, int]] = field(default_factory=dict)  # class -> {source -> count}
    total_contracts: int = 0
    multi_label_count: int = 0   # contracts with ≥2 positives

    def to_csv_rows(self) -> list[dict]:
        """Flatten to CSV rows: (scope, key_a, key_b, count)."""
        rows = []
        for cls, n in sorted(self.per_class.items()):
            rows.append({"scope": "class", "key_a": cls, "key_b": "", "count": n})
        for src, n in sorted(self.per_source.items()):
            rows.append({"scope": "source", "key_a": src, "key_b": "", "count": n})
        for tier, cls_map in sorted(self.per_tier.items()):
            for cls, n in sorted(cls_map.items()):
                rows.append({"scope": "tier", "key_a": tier, "key_b": cls, "count": n})
        for cls, src_map in sorted(self.per_class_source.items()):
            for src, n in sorted(src_map.items()):
                rows.append({"scope": "class_source", "key_a": cls, "key_b": src, "count": n})
        return rows


def load_merged_labels(labels_dir: Path) -> Iterable[dict]:
    """Yield each merged labels JSON (one per contract)."""
    for p in sorted(Path(labels_dir).glob("*.labels.json")):
        try:
            yield json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue


def build_balance_table(labels_dir: Path) -> BalanceTable:
    """Read all merged labels and build the balance matrix.

    "Positive count" for a class is the number of contracts where
    classes[cls].value == 1. Multi-label contracts count once per positive class.
    """
    table = BalanceTable()
    table.per_class = {cls: 0 for cls in class_names()}
    table.per_source = defaultdict(int)
    table.per_tier = defaultdict(lambda: defaultdict(int))
    table.per_class_source = defaultdict(lambda: defaultdict(int))

    for lj in load_merged_labels(labels_dir):
        table.total_contracts += 1
        sources = lj.get("sources") or ["unknown"]
        source = sources[0] if sources else "unknown"
        positives = 0
        for cls, entry in lj.get("classes", {}).items():
            if entry.get("value") == 1:
                positives += 1
                table.per_class[cls] = table.per_class.get(cls, 0) + 1
                tier = entry.get("tier") or "untiered"
                table.per_tier[tier][cls] = table.per_tier[tier].get(cls, 0) + 1
                per_src = entry.get("source") or source
                table.per_class_source[cls][per_src] = table.per_class_source[cls].get(per_src, 0) + 1
        if positives >= 2:
            table.multi_label_count += 1
        table.per_source[source] += 1  # total contracts per source, NOT positives

    return table


def write_csv(table: BalanceTable, output_path: Path) -> Path:
    """Write the balance table to CSV. Returns the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "key_a", "key_b", "count"])
        w.writeheader()
        for row in table.to_csv_rows():
            w.writerow(row)
    return output_path


def write_plot(table: BalanceTable, output_path: Path) -> Path:
    """Write the per-class bar plot. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted(table.per_class.keys())
    counts = [table.per_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(classes)), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Positive contracts")
    ax.set_title(f"Per-class positive count (n={table.total_contracts} contracts, "
                 f"{table.multi_label_count} multi-label)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def run_balance_viz(labels_dir: Path, output_dir: Path) -> dict:
    """Top-level: build the table, write CSV + plot, return summary."""
    table = build_balance_table(labels_dir)
    csv_path = write_csv(table, output_dir / "balance_table.csv")
    plot_path = write_plot(table, output_dir / "balance_plot.png")
    return {
        "csv": str(csv_path),
        "plot": str(plot_path),
        "total_contracts": table.total_contracts,
        "multi_label_count": table.multi_label_count,
        "per_class": table.per_class,
        "per_source": dict(table.per_source),
    }
