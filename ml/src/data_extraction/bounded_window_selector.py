"""Versioned research-only bounded-window selectors for R4 Phase 8.

Production repaired-v2 representations remain bound to the historical linspace
selector. This module provides candidate selectors and exact telemetry for
controlled comparison without silently changing the accepted representation
lineage.

The guarded candidate never accepts lower target-contract token coverage than
the historical control for the same source/target spans. Equal target coverage
falls back to the historical control to minimize unnecessary behavioral change.

Research tokenization uses the same comment-stripped, offset-preserving source
view as repaired-v2 production tokenization. Target character spans are computed
against the original preprocessed source and remain valid because lexical comment
removal preserves source length and newline positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from sentinel_data.preprocessing.normalizer import strip_comments_lexically

SELECTOR_RESEARCH_SCHEMA = "sentinel-r4-bounded-window-selector-research-v1"
CONTROL_STRATEGY = "historical_linspace_v1"
GREEDY_STRATEGY = "target_aware_greedy_v1"
GUARDED_STRATEGY = "target_aware_guarded_v1"


def prepare_source_for_tokenization(source_text: str) -> str:
    """Return the repaired-v2 comment-stripped tokenizer input.

    Comment characters are replaced with spaces while newlines and all source
    offsets are preserved. This mirrors the production repaired-v2 tokenizer's
    ``strip_comments=True`` contract without mutating the promoted Solidity.
    """

    if not source_text.strip():
        raise ValueError("source_text must not be empty")
    code, _ = strip_comments_lexically(source_text)
    if not code.strip():
        raise ValueError("source contains no code after comment removal")
    if len(code) != len(source_text):
        raise AssertionError("comment stripping changed source length")
    return code


def union_length(ranges: Iterable[Iterable[int]]) -> int:
    ordered = sorted(
        (int(start), int(end))
        for start, end in ranges
        if int(end) > int(start)
    )
    if not ordered:
        return 0
    total = 0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def intersect_union_length(
    ranges: Iterable[Iterable[int]],
    target_ranges: Iterable[Iterable[int]],
) -> int:
    ranges_list = [list(map(int, value)) for value in ranges]
    targets_list = [list(map(int, value)) for value in target_ranges]
    intersections = [
        [max(start, target_start), min(end, target_end)]
        for start, end in ranges_list
        for target_start, target_end in targets_list
        if min(end, target_end) > max(start, target_start)
    ]
    return union_length(intersections)


def window_ranges(
    total_tokens: int,
    *,
    content_capacity: int,
    stride: int,
) -> list[list[int]]:
    if total_tokens < 0:
        raise ValueError("total_tokens must be >= 0")
    if content_capacity < 1:
        raise ValueError("content_capacity must be >= 1")
    if stride < 0 or stride >= content_capacity:
        raise ValueError("stride must satisfy 0 <= stride < content_capacity")
    if total_tokens == 0:
        return []
    step = content_capacity - stride
    ranges: list[list[int]] = []
    start = 0
    while True:
        end = min(start + content_capacity, total_tokens)
        ranges.append([start, end])
        if end >= total_tokens:
            break
        start += step
    return ranges


def linspace_indices(total_windows: int, count: int = 4) -> list[int]:
    """Exact historical selection rule used by the production tokenizer."""

    if count < 1:
        raise ValueError("count must be >= 1")
    if total_windows < 0:
        raise ValueError("total_windows must be >= 0")
    if total_windows <= count:
        return list(range(total_windows))
    return [round(value) for value in np.linspace(0, total_windows - 1, count)]


def target_aware_greedy_indices(
    ranges: list[list[int]],
    target_ranges: list[list[int]],
    count: int = 4,
) -> list[int]:
    """Greedily maximize union coverage of target-contract token ranges."""

    if count < 1:
        raise ValueError("count must be >= 1")
    if not ranges:
        return []
    if not target_ranges:
        return linspace_indices(len(ranges), count)

    selected: list[int] = []
    covered = 0
    while len(selected) < count:
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

    for index in linspace_indices(len(ranges), count):
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    if len(selected) < count:
        for index in range(len(ranges)):
            if len(selected) >= count:
                break
            if index not in selected:
                selected.append(index)
    return sorted(selected[:count])


@dataclass(frozen=True)
class SelectionResult:
    strategy: str
    selected_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    target_coverage_tokens: int
    control_target_coverage_tokens: int
    retained_tokens: int
    control_retained_tokens: int
    used_control_fallback: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SELECTOR_RESEARCH_SCHEMA,
            "strategy": self.strategy,
            "selected_indices": list(self.selected_indices),
            "control_indices": list(self.control_indices),
            "target_coverage_tokens": self.target_coverage_tokens,
            "control_target_coverage_tokens": self.control_target_coverage_tokens,
            "retained_tokens": self.retained_tokens,
            "control_retained_tokens": self.control_retained_tokens,
            "used_control_fallback": self.used_control_fallback,
        }


def select_indices(
    ranges: list[list[int]],
    target_ranges: list[list[int]],
    *,
    count: int = 4,
    strategy: str = GUARDED_STRATEGY,
) -> SelectionResult:
    """Select windows and return control-relative telemetry."""

    control = linspace_indices(len(ranges), count)
    control_ranges = [ranges[index] for index in control]
    control_target = intersect_union_length(control_ranges, target_ranges)
    control_retained = union_length(control_ranges)

    if strategy == CONTROL_STRATEGY:
        chosen = control
        used_fallback = False
    elif strategy in {GREEDY_STRATEGY, GUARDED_STRATEGY}:
        greedy = target_aware_greedy_indices(ranges, target_ranges, count)
        greedy_ranges = [ranges[index] for index in greedy]
        greedy_target = intersect_union_length(greedy_ranges, target_ranges)
        if strategy == GUARDED_STRATEGY and greedy_target <= control_target:
            chosen = control
            used_fallback = True
        else:
            chosen = greedy
            used_fallback = False
    else:
        raise ValueError(f"unsupported bounded-window strategy {strategy!r}")

    chosen_ranges = [ranges[index] for index in chosen]
    return SelectionResult(
        strategy=strategy,
        selected_indices=tuple(chosen),
        control_indices=tuple(control),
        target_coverage_tokens=intersect_union_length(chosen_ranges, target_ranges),
        control_target_coverage_tokens=control_target,
        retained_tokens=union_length(chosen_ranges),
        control_retained_tokens=control_retained,
        used_control_fallback=used_fallback,
    )


def char_spans_to_token_ranges(
    offsets: Iterable[Iterable[int]],
    char_spans: Iterable[Iterable[int]],
) -> list[list[int]]:
    offsets_list = [tuple(map(int, value)) for value in offsets]
    token_ranges: list[list[int]] = []
    for raw_start, raw_end in char_spans:
        char_start, char_end = int(raw_start), int(raw_end)
        if char_end <= char_start:
            raise ValueError("target character span must be non-empty")
        indices = [
            index
            for index, (start, end) in enumerate(offsets_list)
            if end > char_start and start < char_end
        ]
        if not indices:
            raise ValueError(
                f"target character span [{char_start}, {char_end}) maps to zero tokens"
            )
        token_ranges.append([min(indices), max(indices) + 1])
    return token_ranges


def tokenize_with_selector(
    source_text: str,
    *,
    target_char_spans: list[list[int]] | list[tuple[int, int]],
    tokenizer: Any,
    strategy: str,
    max_windows: int = 4,
    window_size: int = 512,
    stride: int = 256,
) -> dict[str, Any]:
    """Tokenize one repaired source using a research selector.

    The exact production preprocessing contract is retained: Solidity comments
    are lexically replaced with spaces before GraphCodeBERT tokenization while
    source offsets stay unchanged. The output tensor shape matches the frozen
    production contract but is not a promoted representation artifact.
    """

    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    token_source = prepare_source_for_tokenization(source_text)

    raw = tokenizer(
        token_source,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    raw_ids = raw["input_ids"]
    offsets = raw["offset_mapping"]
    if raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]
        offsets = offsets[0]
    total_tokens = len(raw_ids)
    if total_tokens < 1:
        raise ValueError("tokenizer produced zero raw code tokens")

    try:
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special_tokens = 2
    content_capacity = max(1, window_size - special_tokens)
    ranges = window_ranges(
        total_tokens,
        content_capacity=content_capacity,
        stride=stride,
    )
    target_ranges = char_spans_to_token_ranges(offsets, target_char_spans)
    selection = select_indices(
        ranges,
        target_ranges,
        count=max_windows,
        strategy=strategy,
    )

    encoded = tokenizer(
        token_source,
        max_length=window_size,
        padding="max_length",
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    all_ids = encoded["input_ids"]
    all_masks = encoded["attention_mask"]
    if int(all_ids.shape[0]) != len(ranges):
        raise ValueError(
            "research range count diverges from tokenizer overflow windows: "
            f"{len(ranges)} != {int(all_ids.shape[0])}"
        )

    chosen_ids = [all_ids[index].tolist() for index in selection.selected_indices]
    chosen_masks = [all_masks[index].tolist() for index in selection.selected_indices]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    while len(chosen_ids) < max_windows:
        chosen_ids.append([pad_id] * window_size)
        chosen_masks.append([0] * window_size)

    target_tokens = union_length(target_ranges)
    return {
        "input_ids": torch.tensor(chosen_ids, dtype=torch.long),
        "attention_mask": torch.tensor(chosen_masks, dtype=torch.long),
        "selector": selection.as_dict(),
        "total_code_tokens": total_tokens,
        "total_windows": len(ranges),
        "target_token_ranges": target_ranges,
        "target_tokens": target_tokens,
        "target_coverage_ratio": (
            float(selection.target_coverage_tokens) / float(target_tokens)
            if target_tokens
            else 1.0
        ),
        "control_target_coverage_ratio": (
            float(selection.control_target_coverage_tokens) / float(target_tokens)
            if target_tokens
            else 1.0
        ),
        "retained_ratio": float(selection.retained_tokens) / float(total_tokens),
        "control_retained_ratio": (
            float(selection.control_retained_tokens) / float(total_tokens)
        ),
        "promotion_authorized": False,
    }


__all__ = [
    "CONTROL_STRATEGY",
    "GREEDY_STRATEGY",
    "GUARDED_STRATEGY",
    "SELECTOR_RESEARCH_SCHEMA",
    "SelectionResult",
    "char_spans_to_token_ranges",
    "intersect_union_length",
    "linspace_indices",
    "prepare_source_for_tokenization",
    "select_indices",
    "target_aware_greedy_indices",
    "tokenize_with_selector",
    "union_length",
    "window_ranges",
]
