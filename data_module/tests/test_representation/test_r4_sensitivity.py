from __future__ import annotations

import pytest

from sentinel_data.representation.r4_sensitivity import profile_representation_records


def _row(
    contract_id,
    *,
    mode="slither_full_analysis",
    components=1,
    nodes=10,
    edges=12,
    windows=5,
    role="TRAIN_STRONG",
    optimizer=False,
    selection=False,
):
    return {
        "contract_id": contract_id,
        "source": "dive",
        "role": role,
        "graph_extraction_mode": mode,
        "graph_component_count": components,
        "node_count": nodes,
        "edge_count": edges,
        "pre_subsampling_window_count": windows,
        "optimizer_active": optimizer,
        "model_selection_active": selection,
    }


def test_profiler_builds_exact_compatibility_and_file_union_sets():
    report = profile_representation_records(
        [
            _row(
                "compat-train",
                mode="slither_parse_only",
                optimizer=True,
            ),
            _row(
                "union-selection",
                components=3,
                role="MODEL_SELECTION",
                selection=True,
            ),
            _row("normal", optimizer=True),
        ]
    )
    sets = report["comparison_sets"]
    assert sets["optimizer_compatibility_contract_ids"] == ["compat-train"]
    assert sets["model_selection_compatibility_contract_ids"] == []
    assert sets["optimizer_file_union_contract_ids"] == []
    assert sets["model_selection_file_union_contract_ids"] == ["union-selection"]
    assert "compat-train" in sets["worst_case_gpu_contract_ids"]


def test_worst_case_gpu_candidates_cover_distinct_active_extremes():
    report = profile_representation_records(
        [
            _row("node-max", nodes=1000, edges=1, windows=1, optimizer=True),
            _row("edge-max", nodes=10, edges=2000, windows=1, optimizer=True),
            _row("component-max", components=20, windows=1, optimizer=True),
            _row("window-max", windows=300, optimizer=True),
        ],
        top_n=4,
    )
    sets = report["comparison_sets"]
    assert sets["worst_case_gpu_contract_ids"] == [
        "node-max",
        "edge-max",
        "component-max",
        "window-max",
    ]
    by_metric = sets["worst_case_active_contract_ids_by_metric"]
    assert by_metric["node_count"][0] == "node-max"
    assert by_metric["edge_count"][0] == "edge-max"
    assert by_metric["graph_component_count"][0] == "component-max"
    assert by_metric["pre_subsampling_window_count"][0] == "window-max"


def test_profiler_accepts_explicitly_marked_legacy_standard_inference():
    row = _row("legacy-reused", mode="")
    row["graph_extraction_mode_inferred_legacy_standard"] = True
    report = profile_representation_records([row])
    assert report["mode_counts"] == {"slither_full_analysis": 1}
    assert report["mode_provenance_counts"] == {"inferred_legacy_standard": 1}
    assert report["top_by_nodes"][0]["graph_extraction_mode"] == (
        "slither_full_analysis"
    )
    assert report["top_by_nodes"][0][
        "graph_extraction_mode_inferred_legacy_standard"
    ] is True


def test_profiler_counts_materialized_legacy_standard_as_inferred():
    row = _row("legacy-materialized", mode="slither_full_analysis")
    row["graph_extraction_mode_inferred_legacy_standard"] = True
    report = profile_representation_records([row])
    assert report["mode_counts"] == {"slither_full_analysis": 1}
    assert report["mode_provenance_counts"] == {"inferred_legacy_standard": 1}


def test_profiler_rejects_inferred_marker_with_nonstandard_mode():
    row = _row("bad-inference", mode="slither_parse_only")
    row["graph_extraction_mode_inferred_legacy_standard"] = True
    with pytest.raises(ValueError, match="legacy standard inference"):
        profile_representation_records([row])


def test_profiler_fails_when_mode_provenance_is_missing():
    row = _row("bad")
    row["graph_extraction_mode"] = ""
    with pytest.raises(ValueError, match="graph_extraction_mode"):
        profile_representation_records([row])
