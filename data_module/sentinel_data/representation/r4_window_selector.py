"""Production R4 bounded-window selector primitives.

This module owns the DATA production implementation of the selector contract
promoted by R4-D-012. Historical/research implementations remain untouched and
serve as control/evidence surfaces.

The selector is intentionally pure: it has no filesystem, tokenizer, torch,
artifact-writing, acceptance, or training responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

CONTROL_STRATEGY = "historical_linspace_v1"
GREEDY_STRATEGY = "target_aware_greedy_v1"
GUARDED_STRATEGY = "target_aware_guarded_v1"
FALLBACK_REASON_NOT_STRICTLY_GREATER = (
    "candidate_target_coverage_not_strictly_greater"
)

TokenRange = tuple[int, int]


def _normalized_ranges(
    ranges: Iterable[Sequence[int]],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[TokenRange, ...]:
    normalized: list[TokenRange] = []
    for index, raw in enumerate(ranges):
        if len(raw) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two integers")
        start, end = raw
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError(f"{name}[{index}] bounds must be integers")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{name}[{index}] bounds must be integers")
        if start < 0 or end <= start:
            raise ValueError(
                f"{name}[{index}] must be a non-empty non-negative half-open range"
            )
        normalized.append((start, end))
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return tuple(normalized)


def union_length(ranges: Iterable[Sequence[int]]) -> int:
    """Return the union length of half-open integer token ranges."""

    normalized = _normalized_ranges(ranges, name="ranges", allow_empty=True)
    if not normalized:
        return 0

    ordered = sorted(normalized)
    total = 0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left)


def intersect_union_length(
    ranges: Iterable[Sequence[int]],
    target_ranges: Iterable[Sequence[int]],
) -> int:
    """Return unique target-token coverage produced by the supplied ranges."""

    normalized_ranges = _normalized_ranges(
        ranges, name="ranges", allow_empty=True
    )
    normalized_targets = _normalized_ranges(
        target_ranges, name="target_ranges", allow_empty=True
    )
    intersections = [
        (max(start, target_start), min(end, target_end))
        for start, end in normalized_ranges
        for target_start, target_end in normalized_targets
        if min(end, target_end) > max(start, target_start)
    ]
    return union_length(intersections)


def _round_fraction_half_even(numerator: int, denominator: int) -> int:
    """Round a non-negative rational number with Python/NumPy tie-to-even."""

    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def historical_linspace_indices(
    total_windows: int,
    max_windows: int = 4,
) -> tuple[int, ...]:
    """Return the exact historical rounded-linspace window indices.

    The historical implementation used ``round(i)`` over NumPy linspace from
    zero through ``total_windows - 1``. Integer rational arithmetic reproduces
    those evenly-spaced tie-to-even decisions without adding NumPy as a DATA
    runtime dependency.
    """

    if isinstance(total_windows, bool) or not isinstance(total_windows, int):
        raise ValueError("total_windows must be an integer")
    if isinstance(max_windows, bool) or not isinstance(max_windows, int):
        raise ValueError("max_windows must be an integer")
    if total_windows < 0:
        raise ValueError("total_windows must be >= 0")
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    if total_windows <= max_windows:
        return tuple(range(total_windows))
    if max_windows == 1:
        return (0,)

    denominator = max_windows - 1
    end = total_windows - 1
    return tuple(
        _round_fraction_half_even(end * index, denominator)
        for index in range(max_windows)
    )


def _target_aware_greedy_indices(
    ranges: tuple[TokenRange, ...],
    target_ranges: tuple[TokenRange, ...],
    *,
    max_windows: int,
) -> tuple[int, ...]:
    """Reproduce the retained greedy marginal-union coverage candidate."""

    selected: list[int] = []
    covered = 0

    while len(selected) < max_windows:
        candidates: list[tuple[int, int]] = []
        for index in range(len(ranges)):
            if index in selected:
                continue
            score = intersect_union_length(
                [ranges[value] for value in (*selected, index)],
                target_ranges,
            )
            candidates.append((score - covered, index))

        if not candidates:
            break

        gain, index = max(candidates, key=lambda item: (item[0], -item[1]))
        if gain <= 0:
            break

        selected.append(index)
        covered += gain

    for index in historical_linspace_indices(len(ranges), max_windows):
        if len(selected) >= max_windows:
            break
        if index not in selected:
            selected.append(index)

    if len(selected) < max_windows:
        for index in range(len(ranges)):
            if len(selected) >= max_windows:
                break
            if index not in selected:
                selected.append(index)

    return tuple(sorted(selected[:max_windows]))


def _validate_indices(
    indices: tuple[int, ...],
    *,
    total_windows: int,
    max_windows: int,
    field: str,
) -> None:
    if len(indices) > max_windows:
        raise ValueError(f"{field} exceeds max_windows")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError(f"{field} must be sorted and unique")
    if any(index < 0 or index >= total_windows for index in indices):
        raise ValueError(f"{field} contains an out-of-range window index")


@dataclass(frozen=True)
class SelectorDecision:
    """Immutable successful guarded-selector decision."""

    requested_strategy: str
    effective_strategy: str
    selected_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    used_control_fallback: bool
    fallback_reason: str | None
    total_windows: int
    max_windows: int
    target_coverage_tokens: int
    control_target_coverage_tokens: int
    candidate_target_coverage_tokens: int
    retained_tokens: int
    control_retained_tokens: int
    candidate_retained_tokens: int

    def __post_init__(self) -> None:
        if self.requested_strategy != GUARDED_STRATEGY:
            raise ValueError("SelectorDecision requires target_aware_guarded_v1")
        if self.total_windows < 1:
            raise ValueError("SelectorDecision requires at least one real window")
        if self.max_windows < 1:
            raise ValueError("SelectorDecision max_windows must be >= 1")

        for field, indices in (
            ("selected_indices", self.selected_indices),
            ("control_indices", self.control_indices),
            ("candidate_indices", self.candidate_indices),
        ):
            _validate_indices(
                indices,
                total_windows=self.total_windows,
                max_windows=self.max_windows,
                field=field,
            )

        counts = (
            self.target_coverage_tokens,
            self.control_target_coverage_tokens,
            self.candidate_target_coverage_tokens,
            self.retained_tokens,
            self.control_retained_tokens,
            self.candidate_retained_tokens,
        )
        if any(value < 0 for value in counts):
            raise ValueError("selector coverage/retention counts must be >= 0")

        if self.used_control_fallback:
            if self.effective_strategy != CONTROL_STRATEGY:
                raise ValueError("fallback decision must use historical control")
            if self.fallback_reason != FALLBACK_REASON_NOT_STRICTLY_GREATER:
                raise ValueError("fallback decision has invalid reason")
            if self.selected_indices != self.control_indices:
                raise ValueError("fallback selected indices must equal control")
            if (
                self.candidate_target_coverage_tokens
                > self.control_target_coverage_tokens
            ):
                raise ValueError("fallback cannot discard a strict coverage improvement")
            if self.target_coverage_tokens != self.control_target_coverage_tokens:
                raise ValueError("fallback target coverage must equal control coverage")
            if self.retained_tokens != self.control_retained_tokens:
                raise ValueError("fallback retained tokens must equal control retention")
        else:
            if self.effective_strategy != GUARDED_STRATEGY:
                raise ValueError("winning guarded decision has invalid effective strategy")
            if self.fallback_reason is not None:
                raise ValueError("winning guarded decision cannot carry fallback reason")
            if self.selected_indices != self.candidate_indices:
                raise ValueError("winning guarded selected indices must equal candidate")
            if (
                self.candidate_target_coverage_tokens
                <= self.control_target_coverage_tokens
            ):
                raise ValueError("guarded candidate must strictly improve target coverage")
            if self.target_coverage_tokens != self.candidate_target_coverage_tokens:
                raise ValueError("guarded target coverage must equal candidate coverage")
            if self.retained_tokens != self.candidate_retained_tokens:
                raise ValueError("guarded retained tokens must equal candidate retention")


def select_guarded_windows(
    ranges: Iterable[Sequence[int]],
    target_ranges: Iterable[Sequence[int]],
    *,
    max_windows: int = 4,
) -> SelectorDecision:
    """Select windows under the exact R4-D-012 guarded contract.

    Valid non-empty target evidence is a precondition. Missing/invalid target
    evidence is a build error and must not silently become historical control.
    """

    normalized_ranges = _normalized_ranges(
        ranges, name="ranges", allow_empty=False
    )
    normalized_targets = _normalized_ranges(
        target_ranges, name="target_ranges", allow_empty=False
    )
    if isinstance(max_windows, bool) or not isinstance(max_windows, int):
        raise ValueError("max_windows must be an integer")
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")

    total_windows = len(normalized_ranges)
    control = historical_linspace_indices(total_windows, max_windows)
    candidate = _target_aware_greedy_indices(
        normalized_ranges,
        normalized_targets,
        max_windows=max_windows,
    )

    control_ranges = [normalized_ranges[index] for index in control]
    candidate_ranges = [normalized_ranges[index] for index in candidate]
    control_target = intersect_union_length(control_ranges, normalized_targets)
    candidate_target = intersect_union_length(candidate_ranges, normalized_targets)
    control_retained = union_length(control_ranges)
    candidate_retained = union_length(candidate_ranges)

    if candidate_target > control_target:
        selected = candidate
        effective = GUARDED_STRATEGY
        used_fallback = False
        fallback_reason = None
        selected_target = candidate_target
        selected_retained = candidate_retained
    else:
        selected = control
        effective = CONTROL_STRATEGY
        used_fallback = True
        fallback_reason = FALLBACK_REASON_NOT_STRICTLY_GREATER
        selected_target = control_target
        selected_retained = control_retained

    return SelectorDecision(
        requested_strategy=GUARDED_STRATEGY,
        effective_strategy=effective,
        selected_indices=selected,
        control_indices=control,
        candidate_indices=candidate,
        used_control_fallback=used_fallback,
        fallback_reason=fallback_reason,
        total_windows=total_windows,
        max_windows=max_windows,
        target_coverage_tokens=selected_target,
        control_target_coverage_tokens=control_target,
        candidate_target_coverage_tokens=candidate_target,
        retained_tokens=selected_retained,
        control_retained_tokens=control_retained,
        candidate_retained_tokens=candidate_retained,
    )


__all__ = [
    "CONTROL_STRATEGY",
    "FALLBACK_REASON_NOT_STRICTLY_GREATER",
    "GREEDY_STRATEGY",
    "GUARDED_STRATEGY",
    "SelectorDecision",
    "historical_linspace_indices",
    "intersect_union_length",
    "select_guarded_windows",
    "union_length",
]
