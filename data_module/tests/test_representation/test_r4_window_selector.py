from __future__ import annotations

import pytest

from ml.src.data_extraction.bounded_window_selector import (
    GUARDED_STRATEGY as RESEARCH_GUARDED_STRATEGY,
    intersect_union_length as research_intersect_union_length,
    linspace_indices as research_linspace_indices,
    select_indices as research_select_indices,
    target_aware_greedy_indices as research_greedy_indices,
    union_length as research_union_length,
)
from sentinel_data.representation.r4_window_selector import (
    CONTROL_STRATEGY,
    FALLBACK_REASON_NOT_STRICTLY_GREATER,
    GUARDED_STRATEGY,
    historical_linspace_indices,
    intersect_union_length,
    select_guarded_windows,
    union_length,
)


def test_historical_indices_match_retained_research_across_broad_range() -> None:
    for total_windows in range(0, 2048):
        for max_windows in range(1, 9):
            assert list(
                historical_linspace_indices(total_windows, max_windows)
            ) == research_linspace_indices(total_windows, max_windows)


@pytest.mark.parametrize(
    ("ranges", "targets", "max_windows"),
    [
        ([[0, 10], [10, 20], [20, 30]], [[0, 5]], 2),
        ([[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]], [[12, 18]], 2),
        ([[0, 10], [8, 18], [16, 26], [24, 34], [32, 42]], [[9, 17]], 3),
        (
            [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]],
            [[0, 5], [45, 50]],
            2,
        ),
        (
            [[0, 12], [6, 18], [12, 24], [18, 30], [24, 36], [30, 42]],
            [[7, 17], [28, 40]],
            4,
        ),
    ],
)
def test_guarded_decision_matches_retained_research_semantics(
    ranges: list[list[int]],
    targets: list[list[int]],
    max_windows: int,
) -> None:
    production = select_guarded_windows(
        ranges,
        targets,
        max_windows=max_windows,
    )
    research = research_select_indices(
        ranges,
        targets,
        count=max_windows,
        strategy=RESEARCH_GUARDED_STRATEGY,
    )
    research_candidate = research_greedy_indices(
        ranges,
        targets,
        count=max_windows,
    )
    research_candidate_ranges = [ranges[index] for index in research_candidate]

    assert list(production.selected_indices) == list(research.selected_indices)
    assert list(production.control_indices) == list(research.control_indices)
    assert list(production.candidate_indices) == research_candidate
    assert production.used_control_fallback is research.used_control_fallback
    assert production.target_coverage_tokens == research.target_coverage_tokens
    assert (
        production.control_target_coverage_tokens
        == research.control_target_coverage_tokens
    )
    assert production.retained_tokens == research.retained_tokens
    assert production.control_retained_tokens == research.control_retained_tokens
    assert (
        production.candidate_target_coverage_tokens
        == research_intersect_union_length(research_candidate_ranges, targets)
    )
    assert production.candidate_retained_tokens == research_union_length(
        research_candidate_ranges
    )


def test_under_cap_uses_normal_strict_improvement_guard() -> None:
    decision = select_guarded_windows(
        [[0, 10], [10, 20], [20, 30]],
        [[4, 8]],
        max_windows=4,
    )

    assert decision.control_indices == (0, 1, 2)
    assert decision.candidate_indices == (0, 1, 2)
    assert decision.selected_indices == (0, 1, 2)
    assert decision.effective_strategy == CONTROL_STRATEGY
    assert decision.used_control_fallback is True
    assert decision.fallback_reason == FALLBACK_REASON_NOT_STRICTLY_GREATER


def test_lowest_window_index_wins_equal_greedy_gain() -> None:
    decision = select_guarded_windows(
        [[0, 10], [0, 10], [10, 20]],
        [[2, 8]],
        max_windows=1,
    )
    assert decision.candidate_indices == (0,)


def test_missing_target_evidence_fails_closed() -> None:
    with pytest.raises(ValueError, match="target_ranges must not be empty"):
        select_guarded_windows([[0, 10]], [], max_windows=4)


def test_invalid_window_ranges_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty non-negative"):
        select_guarded_windows([[10, 10]], [[10, 11]], max_windows=4)


def test_range_metrics_match_retained_research_helpers() -> None:
    ranges = [[0, 10], [5, 15], [30, 40]]
    targets = [[2, 12], [8, 14]]
    assert union_length(ranges) == research_union_length(ranges)
    assert intersect_union_length(ranges, targets) == research_intersect_union_length(
        ranges, targets
    )


def test_repeated_selection_is_identical_and_indices_are_valid() -> None:
    ranges = [[index * 7, index * 7 + 12] for index in range(20)]
    targets = [[13, 51], [93, 121]]

    first = select_guarded_windows(ranges, targets, max_windows=4)
    for _ in range(25):
        current = select_guarded_windows(ranges, targets, max_windows=4)
        assert current == first
        for indices in (
            current.selected_indices,
            current.control_indices,
            current.candidate_indices,
        ):
            assert tuple(sorted(set(indices))) == indices
            assert all(0 <= index < len(ranges) for index in indices)
