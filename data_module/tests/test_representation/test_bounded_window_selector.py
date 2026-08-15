from __future__ import annotations

from ml.src.data_extraction.bounded_window_selector import (
    GUARDED_STRATEGY,
    intersect_union_length,
    select_indices,
    target_aware_greedy_indices,
    union_length,
)
from sentinel_data.representation.r4_target_spans import target_contract_char_spans


def test_greedy_selector_covers_disjoint_target_ranges():
    ranges = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]]
    targets = [[0, 5], [45, 50]]
    assert target_aware_greedy_indices(ranges, targets, count=2) == [0, 4]


def test_guarded_selector_never_regresses_target_coverage():
    ranges = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]]
    targets = [[0, 5], [45, 50]]
    result = select_indices(
        ranges,
        targets,
        count=2,
        strategy=GUARDED_STRATEGY,
    )
    assert result.target_coverage_tokens >= result.control_target_coverage_tokens


def test_guarded_selector_uses_control_on_target_coverage_tie():
    ranges = [[0, 10], [10, 20], [20, 30]]
    targets = [[0, 5]]
    result = select_indices(
        ranges,
        targets,
        count=2,
        strategy=GUARDED_STRATEGY,
    )
    assert result.used_control_fallback is True
    assert result.selected_indices == result.control_indices


def test_range_union_and_intersection_do_not_double_count():
    assert union_length([[0, 10], [5, 15]]) == 15
    assert intersect_union_length([[0, 20]], [[2, 10], [8, 15]]) == 13


def test_target_spans_support_multiple_file_components():
    source = (
        "contract First { function a() public {} } "
        "contract Second { uint x; }"
    )
    spans = target_contract_char_spans(source, ["First", "Second"])
    assert [source[start:end] for start, end in spans] == [
        "contract First { function a() public {} }",
        "contract Second { uint x; }",
    ]
