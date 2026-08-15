from __future__ import annotations

import pytest

from sentinel_data.representation.r4_sensitivity import profile_representation_records


def _row(
    contract_id,
    *,
    mode="slither_full_analysis",
    components=1,
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
        "node_count": 10,
        "edge_count": 12,
        "pre_subsampling_window_count": 5,
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


def test_profiler_fails_when_mode_provenance_is_missing():
    row = _row("bad")
    row["graph_extraction_mode"] = ""
    with pytest.raises(ValueError, match="graph_extraction_mode"):
        profile_representation_records([row])
