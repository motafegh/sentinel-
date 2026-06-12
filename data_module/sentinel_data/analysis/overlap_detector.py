"""Stage 6 — overlap_detector: pairwise Jaccard similarity between source datasets.

D-6.4 — input to per-source loss weighting in the model training pipeline.

Per AUDIT_PATCHES 6-P5: distinguish EXACT overlap (same contract_id) from
NEAR overlap (AST-similar but different contract_id). The exact overlap is
the more pernicious — the same contract in two sources means double-counting
in the loss. The near overlap is a softer signal.

The v2 corpus uses sha256 as the contract identifier (computed during Stage 1
preprocessing). For exact overlap, we count shared sha256s between sources.
For near overlap, we use the AST similarity threshold from
pipeline.dedup.ast_similarity_threshold (default 0.85) to identify near-dup
groups that span sources.

Outputs:
  - data/analysis/<run_id>/overlap_matrix.csv
  - data/analysis/<run_id>/overlap_heatmap.png
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class OverlapMatrix:
    sources: list[str] = field(default_factory=list)
    exact_jaccard: dict[str, dict[str, float]] = field(default_factory=dict)
    near_jaccard: dict[str, dict[str, float]] = field(default_factory=dict)
    exact_intersection: dict[str, dict[str, int]] = field(default_factory=dict)
    near_intersection: dict[str, dict[str, int]] = field(default_factory=dict)
    source_sizes: dict[str, int] = field(default_factory=dict)


def _index_labels_by_source(labels_root: Path) -> dict[str, set[str]]:
    """Build a map: source -> set of sha256s in that source.

    Reads all `data/labels/<source>/` and the merged labels. A contract
    appears in a source if it's listed in the merged label's `sources` list
    OR if it has a per-source label file.
    """
    by_source: dict[str, set[str]] = defaultdict(set)

    # Merged labels carry the source list per contract
    merged_dir = labels_root / "merged"
    if merged_dir.exists():
        for p in sorted(merged_dir.glob("*.labels.json")):
            try:
                lj = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            sha = lj.get("sha256", "")
            for s in lj.get("sources") or []:
                by_source[s].add(sha)
    # Per-source label files (e.g. data/labels/dive/*.json)
    for d in sorted(Path(labels_root).iterdir()):
        if not d.is_dir() or d.name == "merged":
            continue
        for p in d.glob("*.json"):
            try:
                lj = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            sha = lj.get("sha256", "")
            if not sha:
                continue
            by_source[d.name].add(sha)
    return dict(by_source)


def _index_dedup_groups(preproc_root: Path) -> dict[str, set[str]]:
    """Build a map: dedup_group_id -> set of sha256s in that group.

    Reads the meta.json sidecar which has `dedup_group_id`. The merge logic
    is: same dedup_group_id means near-duplicates.
    """
    by_group: dict[str, set[str]] = defaultdict(set)
    if not preproc_root.exists():
        return dict(by_group)
    for source_dir in sorted(preproc_root.iterdir()):
        if not source_dir.is_dir():
            continue
        for meta in source_dir.glob("*.meta.json"):
            try:
                m = json.loads(meta.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            gid = m.get("dedup_group_id", "")
            sha = m.get("sha256", "")
            if gid and sha:
                by_group[gid].add(sha)
    return dict(by_group)


def _build_dedup_groups_to_sources(by_group: dict[str, set[str]],
                                   sha_to_source: dict[str, str]) -> dict[str, set[str]]:
    """For each dedup_group, the set of sources whose sha256s appear in that group.

    A near-overlap is when a single dedup_group spans multiple sources.
    """
    group_sources: dict[str, set[str]] = {}
    for gid, shas in by_group.items():
        srcs = {sha_to_source.get(s) for s in shas if s in sha_to_source}
        srcs.discard(None)
        if len(srcs) >= 2:
            group_sources[gid] = srcs
    return group_sources


def build_overlap_matrix(labels_root: Path, preproc_root: Path) -> OverlapMatrix:
    by_source = _index_labels_by_source(labels_root)
    by_group = _index_dedup_groups(preproc_root)
    sources = sorted(by_source.keys())

    # sha -> source for "which source does this sha belong to" lookup
    sha_to_source: dict[str, str] = {}
    for s, shas in by_source.items():
        for sha in shas:
            sha_to_source.setdefault(sha, s)

    group_sources = _build_dedup_groups_to_sources(by_group, sha_to_source)
    # For each pair of sources, count near-dup groups that span them
    near_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for gid, srcs in group_sources.items():
        for a in srcs:
            for b in srcs:
                if a != b:
                    key = (min(a, b), max(a, b))
                    near_pairs[key].add(gid)

    matrix = OverlapMatrix(sources=sources, source_sizes={s: len(by_source[s]) for s in sources})
    for a in sources:
        matrix.exact_jaccard[a] = {}
        matrix.near_jaccard[a] = {}
        matrix.exact_intersection[a] = {}
        matrix.near_intersection[a] = {}
        for b in sources:
            if a == b:
                matrix.exact_jaccard[a][b] = 1.0
                matrix.near_jaccard[a][b] = 1.0
                matrix.exact_intersection[a][b] = matrix.source_sizes[a]
                matrix.near_intersection[a][b] = matrix.source_sizes[a]
                continue
            # Exact: |A ∩ B|
            inter = len(by_source[a] & by_source[b])
            union = len(by_source[a] | by_source[b])
            matrix.exact_jaccard[a][b] = inter / union if union else 0.0
            matrix.exact_intersection[a][b] = inter
            # Near: number of dedup_groups spanning (a, b) — this is a count, not a
            # Jaccard (sources have very different sizes). For Jaccard-style
            # comparison, we normalize by the number of groups involving a OR b.
            key = (min(a, b), max(a, b))
            near_inter = len(near_pairs.get(key, set()))
            # Approximate Jaccard: |groups spanning pair| / |groups involving either source|
            groups_in_a = {gid for gid, srcs in group_sources.items() if a in srcs}
            groups_in_b = {gid for gid, srcs in group_sources.items() if b in srcs}
            near_union = groups_in_a | groups_in_b
            matrix.near_jaccard[a][b] = near_inter / len(near_union) if near_union else 0.0
            matrix.near_intersection[a][b] = near_inter
    return matrix


def write_csv(matrix: OverlapMatrix, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# EXACT OVERLAP — Jaccard similarity (same sha256)"])
        w.writerow([""] + matrix.sources)
        for a in matrix.sources:
            row = [a] + [round(matrix.exact_jaccard[a][b], 4) for b in matrix.sources]
            w.writerow(row)
        w.writerow([])
        w.writerow(["# EXACT OVERLAP — intersection count (same sha256)"])
        w.writerow([""] + matrix.sources)
        for a in matrix.sources:
            row = [a] + [matrix.exact_intersection[a][b] for b in matrix.sources]
            w.writerow(row)
        w.writerow([])
        w.writerow(["# NEAR OVERLAP — Jaccard (shared dedup_group, AST-similar)"])
        w.writerow([""] + matrix.sources)
        for a in matrix.sources:
            row = [a] + [round(matrix.near_jaccard[a][b], 4) for b in matrix.sources]
            w.writerow(row)
        w.writerow([])
        w.writerow(["# NEAR OVERLAP — intersection count (shared dedup_groups)"])
        w.writerow([""] + matrix.sources)
        for a in matrix.sources:
            row = [a] + [matrix.near_intersection[a][b] for b in matrix.sources]
            w.writerow(row)
        w.writerow([])
        w.writerow(["# SOURCE SIZES (number of contracts per source)"])
        for s in matrix.sources:
            w.writerow([s, matrix.source_sizes[s]])
    return output_path


def write_heatmap(matrix: OverlapMatrix, output_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(matrix.sources)
    if n == 0:
        # No sources — write a placeholder figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No sources found", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path
    M_exact = [[matrix.exact_jaccard[a][b] for b in matrix.sources] for a in matrix.sources]
    M_near = [[matrix.near_jaccard[a][b] for b in matrix.sources] for a in matrix.sources]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, M, title in [
        (axes[0], M_exact, "Exact overlap (same sha256)"),
        (axes[1], M_near, "Near overlap (shared dedup_group)"),
    ]:
        # Compute vmax robustly for empty/missing rows
        flat_max = max((max(row) for row in M), default=0.0)
        vmax = max(0.01, flat_max)
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(matrix.sources, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(matrix.sources, fontsize=8)
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                v = M[i][j]
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def run_overlap_detector(labels_root: Path, preproc_root: Path, output_dir: Path) -> dict:
    matrix = build_overlap_matrix(labels_root, preproc_root)
    csv_path = write_csv(matrix, output_dir / "overlap_matrix.csv")
    plot_path = write_heatmap(matrix, output_dir / "overlap_heatmap.png")
    # Find the most-overlapping source pair (for the report)
    pairs = []
    for i, a in enumerate(matrix.sources):
        for b in matrix.sources[i + 1:]:
            pairs.append((a, b, matrix.exact_jaccard[a][b], matrix.near_jaccard[a][b]))
    pairs.sort(key=lambda p: -p[2])
    return {
        "csv": str(csv_path),
        "plot": str(plot_path),
        "sources": matrix.sources,
        "source_sizes": matrix.source_sizes,
        "top_overlapping_pairs": [
            {"a": p[0], "b": p[1], "exact_jaccard": round(p[2], 4), "near_jaccard": round(p[3], 4)}
            for p in pairs[:10]
        ],
    }
