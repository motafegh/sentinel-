"""Representation sensitivity profiling for repaired R4 Phase 8.

This module turns representation sidecar metadata into explicit comparison sets
for compatibility-mode and file-union diagnostics. It is read-only and makes
no acceptance or model-quality claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from sentinel_data.representation.r4_compatibility import FULL_ANALYSIS


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(int(value) for value in values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def profile_representation_records(
    records: Iterable[dict[str, Any]],
    *,
    top_n: int = 25,
) -> dict[str, Any]:
    """Summarize representation modes, file unions, and worst-case candidates."""

    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("representation sensitivity profiler received no records")

    mode_counts: Counter[str] = Counter()
    mode_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    mode_provenance_counts: Counter[str] = Counter()
    component_counts: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    window_counts: list[int] = []

    optimizer_compat: list[str] = []
    selection_compat: list[str] = []
    optimizer_union: list[str] = []
    selection_union: list[str] = []

    for row in rows:
        contract_id = str(row["contract_id"])
        role = str(row.get("role") or "")
        raw_mode = row.get("graph_extraction_mode")
        inferred_legacy_standard = bool(
            row.get("graph_extraction_mode_inferred_legacy_standard")
        )
        if inferred_legacy_standard:
            if raw_mode not in (None, "", FULL_ANALYSIS):
                raise ValueError(
                    f"{contract_id} marks legacy standard inference but records "
                    f"non-standard graph_extraction_mode={raw_mode!r}"
                )
            mode = FULL_ANALYSIS
            row["graph_extraction_mode"] = mode
            mode_provenance_counts["inferred_legacy_standard"] += 1
        elif raw_mode:
            mode = str(raw_mode)
            mode_provenance_counts["explicit"] += 1
        else:
            raise ValueError(f"{contract_id} lacks graph_extraction_mode provenance")

        components = int(row.get("graph_component_count", 0))
        nodes = int(row.get("node_count", 0))
        edges = int(row.get("edge_count", 0))
        windows = int(
            row.get("pre_subsampling_window_count", row.get("window_count", 0))
        )
        if components < 1 or nodes < 1 or edges < 0 or windows < 1:
            raise ValueError(
                f"invalid representation telemetry for {contract_id}: "
                f"components={components} nodes={nodes} edges={edges} windows={windows}"
            )
        mode_counts[mode] += 1
        mode_by_role[role][mode] += 1
        component_counts.append(components)
        node_counts.append(nodes)
        edge_counts.append(edges)
        window_counts.append(windows)

        optimizer_active = bool(row.get("optimizer_active"))
        selection_active = bool(row.get("model_selection_active"))
        compat = mode != FULL_ANALYSIS
        file_union = components > 1
        if compat and optimizer_active:
            optimizer_compat.append(contract_id)
        if compat and selection_active:
            selection_compat.append(contract_id)
        if file_union and optimizer_active:
            optimizer_union.append(contract_id)
        if file_union and selection_active:
            selection_union.append(contract_id)

    def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "contract_id": str(row["contract_id"]),
            "source": str(row.get("source") or ""),
            "role": str(row.get("role") or ""),
            "graph_extraction_mode": str(row.get("graph_extraction_mode") or ""),
            "graph_extraction_mode_inferred_legacy_standard": bool(
                row.get("graph_extraction_mode_inferred_legacy_standard")
            ),
            "graph_component_count": int(row.get("graph_component_count", 0)),
            "node_count": int(row.get("node_count", 0)),
            "edge_count": int(row.get("edge_count", 0)),
            "pre_subsampling_window_count": int(
                row.get("pre_subsampling_window_count", row.get("window_count", 0))
            ),
            "optimizer_active": bool(row.get("optimizer_active")),
            "model_selection_active": bool(row.get("model_selection_active")),
        }

    def top(metric: str) -> list[dict[str, Any]]:
        ordered = sorted(
            rows,
            key=lambda row: (
                -int(row.get(metric, 0)),
                str(row["contract_id"]),
            ),
        )[:top_n]
        return [_summary_row(row) for row in ordered]

    active_rows = [
        row
        for row in rows
        if bool(row.get("optimizer_active"))
        or bool(row.get("model_selection_active"))
    ]

    metrics = (
        "node_count",
        "edge_count",
        "graph_component_count",
        "pre_subsampling_window_count",
    )
    active_rankings: dict[str, list[dict[str, Any]]] = {
        metric: sorted(
            active_rows,
            key=lambda row, metric=metric: (
                -int(row.get(metric, 0)),
                str(row["contract_id"]),
            ),
        )[:top_n]
        for metric in metrics
    }

    # Interleave metric rankings by rank so one dimension (usually node count)
    # cannot consume the entire bounded GPU probe budget before edge/component/
    # token-window extremes are represented.
    worst_case_ids: list[str] = []
    for rank in range(top_n):
        for metric in metrics:
            ranking = active_rankings[metric]
            if rank >= len(ranking):
                continue
            contract_id = str(ranking[rank]["contract_id"])
            if contract_id not in worst_case_ids:
                worst_case_ids.append(contract_id)
            if len(worst_case_ids) >= top_n:
                break
        if len(worst_case_ids) >= top_n:
            break

    worst_case_by_metric = {
        metric: [str(row["contract_id"]) for row in ranking]
        for metric, ranking in active_rankings.items()
    }

    return {
        "schema": "sentinel-r4-representation-sensitivity-v1",
        "acceptance_changed": False,
        "model_architecture_changed": False,
        "records": len(rows),
        "mode_counts": dict(sorted(mode_counts.items())),
        "mode_provenance_counts": dict(sorted(mode_provenance_counts.items())),
        "mode_counts_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(mode_by_role.items())
        },
        "telemetry_quantiles": {
            "graph_component_count": _quantiles(component_counts),
            "node_count": _quantiles(node_counts),
            "edge_count": _quantiles(edge_counts),
            "pre_subsampling_window_count": _quantiles(window_counts),
        },
        "comparison_sets": {
            "optimizer_compatibility_contract_ids": sorted(optimizer_compat),
            "model_selection_compatibility_contract_ids": sorted(selection_compat),
            "optimizer_file_union_contract_ids": sorted(optimizer_union),
            "model_selection_file_union_contract_ids": sorted(selection_union),
            "worst_case_gpu_contract_ids": worst_case_ids,
            "worst_case_active_contract_ids_by_metric": worst_case_by_metric,
        },
        "top_by_nodes": top("node_count"),
        "top_by_edges": top("edge_count"),
        "top_by_components": top("graph_component_count"),
        "top_by_windows": top("pre_subsampling_window_count"),
        "decision_boundary": (
            "These sets are for bounded sensitivity/GPU comparisons. Legacy "
            "standard/full-analysis mode inference is reported explicitly and "
            "does not change repaired-v2 acceptance. Excluding or down-weighting "
            "comparison sets in a promoted training lineage requires a new "
            "explicit policy/representation decision."
        ),
    }


__all__ = ["profile_representation_records"]
