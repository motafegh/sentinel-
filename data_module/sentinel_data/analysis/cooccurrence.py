"""Stage 6 — cooccurrence: directed + conditional co-occurrence matrices.

D-6.3 (multi-label loss design input) + AUDIT_PATCHES 6-P4 (two matrices).

Two matrices per AUDIT_PATCHES 6-P4, C-10:
  - Directed: X→Y means "if X is positive, Y is also positive with probability p"
  - Conditional: P(Y=1 | X=1) — the conditional probability

Note: the directed matrix is the joint P(X=1 ∧ Y=1) (or, equivalently, the
"both are positive" rate per pair of labels), while the conditional matrix is
P(Y=1 | X=1). The BCCC 99% DoS↔Reentrancy co-occurrence is visible as a high
entry in both.

Outputs:
  - data/analysis/<run_id>/cooccurrence_matrix.csv   (directed + conditional)
  - data/analysis/<run_id>/cooccurrence_heatmap.png
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sentinel_data.labeling.schema import class_names


@dataclass
class CooccurrenceMatrices:
    """The directed + conditional co-occurrence matrices.

    `directed[a][b]` = count of contracts where both class a and class b are
    positive (i.e. P(a=1 ∧ b=1) × N). The matrix is symmetric for undirected
    pairs but we store the full directed form for clarity (per 6-P4).

    `conditional[a][b]` = P(b=1 | a=1) = directed[a][b] / count_positive(a).
    """
    classes: list[str] = field(default_factory=list)
    counts_positive: dict[str, int] = field(default_factory=dict)
    directed: dict[str, dict[str, int]] = field(default_factory=dict)
    conditional: dict[str, dict[str, float]] = field(default_factory=dict)
    flagged_pairs: list[dict] = field(default_factory=list)
    multi_label_count: int = 0
    total_contracts: int = 0


def _iter_labels(labels_dir: Path) -> Iterable[dict]:
    for p in sorted(Path(labels_dir).glob("*.labels.json")):
        try:
            yield json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue


def build_cooccurrence_matrices(labels_dir: Path, flag_threshold: float = 0.5) -> CooccurrenceMatrices:
    classes = class_names()
    matrices = CooccurrenceMatrices(classes=list(classes))
    matrices.counts_positive = {c: 0 for c in classes}
    matrices.directed = {a: {b: 0 for b in classes} for a in classes}
    matrices.conditional = {a: {b: 0.0 for b in classes} for a in classes}

    for lj in _iter_labels(labels_dir):
        matrices.total_contracts += 1
        positives = [c for c, e in lj.get("classes", {}).items()
                     if e.get("value") == 1 and c in classes]
        if not positives:
            continue
        if len(positives) >= 2:
            matrices.multi_label_count += 1
        for c in positives:
            matrices.counts_positive[c] += 1
        # Update directed[a][b] for all pairs (a, b) where both positive
        for a in positives:
            for b in positives:
                matrices.directed[a][b] += 1

    # Compute conditional: P(b=1 | a=1) = directed[a][b] / counts_positive[a]
    for a in classes:
        n_a = matrices.counts_positive[a]
        for b in classes:
            if n_a > 0:
                matrices.conditional[a][b] = matrices.directed[a][b] / n_a
            else:
                matrices.conditional[a][b] = 0.0

    # Flag pairs (a < b in name sort) with conditional > threshold.
    # Pairs are undirected; we flag the higher value of {P(b|a), P(a|b)}.
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            p_ab = matrices.conditional[a][b]
            p_ba = matrices.conditional[b][a]
            p_max = max(p_ab, p_ba)
            if p_max > flag_threshold:
                matrices.flagged_pairs.append({
                    "class_a": a, "class_b": b,
                    "p_b_given_a": round(p_ab, 4),
                    "p_a_given_b": round(p_ba, 4),
                    "p_max": round(p_max, 4),
                    "count_joint": matrices.directed[a][b],
                    "count_a": matrices.counts_positive[a],
                    "count_b": matrices.counts_positive[b],
                })
    matrices.flagged_pairs.sort(key=lambda x: -x["p_max"])
    return matrices


def write_csv(matrices: CooccurrenceMatrices, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        # Section 1: directed (joint) counts
        w.writerow(["# DIRECTED CO-OCCURRENCE (joint count: contracts where both are positive)"])
        w.writerow([""] + matrices.classes)
        for a in matrices.classes:
            row = [a] + [matrices.directed[a][b] for b in matrices.classes]
            w.writerow(row)
        w.writerow([])
        # Section 2: conditional P(b=1 | a=1)
        w.writerow(["# CONDITIONAL PROBABILITY: P(b=1 | a=1)"])
        w.writerow([""] + matrices.classes)
        for a in matrices.classes:
            row = [a] + [round(matrices.conditional[a][b], 4) for b in matrices.classes]
            w.writerow(row)
        w.writerow([])
        # Section 3: positive counts (denominators)
        w.writerow(["# POSITIVE COUNTS (denominator for conditional)"])
        for a in matrices.classes:
            w.writerow([a, matrices.counts_positive[a]])
        w.writerow([])
        # Section 4: flagged pairs
        w.writerow(["# FLAGGED PAIRS (max(P(b|a), P(a|b)) > threshold)"])
        w.writerow(["class_a", "class_b", "p_b_given_a", "p_a_given_b", "p_max",
                    "count_joint", "count_a", "count_b"])
        for fp in matrices.flagged_pairs:
            w.writerow([fp["class_a"], fp["class_b"], fp["p_b_given_a"],
                        fp["p_a_given_b"], fp["p_max"], fp["count_joint"],
                        fp["count_a"], fp["count_b"]])
    return output_path


def write_heatmap(matrices: CooccurrenceMatrices, output_path: Path) -> Path:
    """Heatmap of the conditional matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(matrices.classes)
    M = [[matrices.conditional[a][b] for b in matrices.classes] for a in matrices.classes]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrices.classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrices.classes, fontsize=8)
    ax.set_xlabel("class b")
    ax.set_ylabel("class a")
    ax.set_title("Conditional co-occurrence: P(b=1 | a=1)\n"
                 f"(flagged: {len(matrices.flagged_pairs)} pairs)")
    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = M[i][j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    fig.colorbar(im, ax=ax, label="P(b|a)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def run_cooccurrence(labels_dir: Path, output_dir: Path, flag_threshold: float = 0.5) -> dict:
    matrices = build_cooccurrence_matrices(labels_dir, flag_threshold)
    csv_path = write_csv(matrices, output_dir / "cooccurrence_matrix.csv")
    plot_path = write_heatmap(matrices, output_dir / "cooccurrence_heatmap.png")
    return {
        "csv": str(csv_path),
        "plot": str(plot_path),
        "total_contracts": matrices.total_contracts,
        "multi_label_count": matrices.multi_label_count,
        "flagged_count": len(matrices.flagged_pairs),
        "flagged_pairs": matrices.flagged_pairs,
        "per_class_counts": matrices.counts_positive,
    }
