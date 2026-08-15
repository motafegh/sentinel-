"""Tests for multi-target bounded-window coverage evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docs/plan/ml-R4/scripts/p8_compare_bounded_window_strategies.py"
)
SPEC = importlib.util.spec_from_file_location("p8_bounded_windows", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_target_char_spans_support_multiple_file_graph_components():
    source = "contract First { function a() public {} } contract Second { uint x; }"

    spans = MODULE._target_char_spans(source, ["First", "Second"])

    assert [source[start:end] for start, end in spans] == [
        "contract First { function a() public {} }",
        "contract Second { uint x; }",
    ]


def test_target_aware_selection_covers_disjoint_target_ranges():
    ranges = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]]
    targets = [[0, 5], [45, 50]]

    selected = MODULE._target_aware_indices(ranges, targets, count=2)

    assert selected == [0, 4]
    assert MODULE._intersect_union_length(
        [ranges[index] for index in selected], targets
    ) == 10


def test_intersection_does_not_double_count_overlapping_target_ranges():
    assert MODULE._intersect_union_length(
        [[0, 20]], [[2, 10], [8, 15]]
    ) == 13
