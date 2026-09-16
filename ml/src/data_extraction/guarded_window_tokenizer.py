"""Production payload wrapper for the R4-D-012 guarded four-window selector.

The accepted selector algorithm remains in
:mod:`ml.src.data_extraction.bounded_window_selector`. This module only turns
that exact decision into the standard ``[4, 512]`` token payload and explicit
selector evidence required by a fresh physical lineage.
"""

from __future__ import annotations

from typing import Any, Iterable

from ml.src.data_extraction.bounded_window_selector import (
    CONTROL_STRATEGY,
    GREEDY_STRATEGY,
    GUARDED_STRATEGY,
    intersect_union_length,
    target_aware_greedy_indices,
    tokenize_with_selector,
    union_length,
    window_ranges,
)
from ml.src.data_extraction.windowed_tokenizer import (
    MAX_WINDOWS,
    STRIDE,
    TOKENIZER_MODEL,
    TOKEN_COVERAGE_SCHEMA_VERSION,
    WINDOW_SIZE,
)

GUARDED_SELECTOR_METADATA_SCHEMA_VERSION = "r4-guarded-selector-metadata-v1"
FALLBACK_NOT_STRICTLY_BETTER = "candidate_target_coverage_not_strictly_greater"


class GuardedWindowTokenizationError(RuntimeError):
    """Raised when a guarded token payload cannot be established fail-closed."""


def _content_capacity(tokenizer: Any) -> int:
    try:
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special_tokens = 2
    return max(1, WINDOW_SIZE - special_tokens)


def _normalize_char_spans(
    target_char_spans: Iterable[Iterable[int]],
) -> list[list[int]]:
    spans = [[int(start), int(end)] for start, end in target_char_spans]
    if not spans:
        raise GuardedWindowTokenizationError(
            "target_aware_guarded_v1 requires at least one target character span"
        )
    for start, end in spans:
        if end <= start:
            raise GuardedWindowTokenizationError(
                f"invalid target character span [{start}, {end})"
            )
    return spans


def tokenize_guarded_source(
    source_text: str,
    *,
    target_char_spans: Iterable[Iterable[int]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Build one guarded token payload using the accepted R4-D-012 algorithm.

    Invalid/missing target evidence is an error. Control fallback is reserved
    only for the accepted guard condition: the greedy candidate does not
    strictly improve target-token coverage over ``historical_linspace_v1``.
    """

    spans = _normalize_char_spans(target_char_spans)
    try:
        selected = tokenize_with_selector(
            source_text,
            target_char_spans=spans,
            tokenizer=tokenizer,
            strategy=GUARDED_STRATEGY,
            max_windows=MAX_WINDOWS,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )
    except Exception as exc:
        raise GuardedWindowTokenizationError(
            f"target-aware guarded tokenization failed: {exc}"
        ) from exc

    input_ids = selected["input_ids"]
    attention_mask = selected["attention_mask"]
    expected_shape = (MAX_WINDOWS, WINDOW_SIZE)
    if tuple(input_ids.shape) != expected_shape:
        raise GuardedWindowTokenizationError(
            f"guarded input_ids shape changed: {tuple(input_ids.shape)} != {expected_shape}"
        )
    if tuple(attention_mask.shape) != expected_shape:
        raise GuardedWindowTokenizationError(
            "guarded attention_mask shape changed: "
            f"{tuple(attention_mask.shape)} != {expected_shape}"
        )

    total_code_tokens = int(selected["total_code_tokens"])
    total_windows = int(selected["total_windows"])
    content_capacity = _content_capacity(tokenizer)
    ranges = window_ranges(
        total_code_tokens,
        content_capacity=content_capacity,
        stride=STRIDE,
    )
    if len(ranges) != total_windows:
        raise GuardedWindowTokenizationError(
            "selector range count diverges from encoded overflow windows: "
            f"{len(ranges)} != {total_windows}"
        )

    target_token_ranges = [
        [int(start), int(end)]
        for start, end in selected["target_token_ranges"]
    ]
    candidate_indices = target_aware_greedy_indices(
        ranges,
        target_token_ranges,
        count=MAX_WINDOWS,
    )
    candidate_target_coverage = intersect_union_length(
        [ranges[index] for index in candidate_indices],
        target_token_ranges,
    )
    candidate_retained_tokens = union_length(
        [ranges[index] for index in candidate_indices]
    )

    raw_selector = dict(selected["selector"])
    selected_indices = [int(value) for value in raw_selector["selected_indices"]]
    control_indices = [int(value) for value in raw_selector["control_indices"]]
    used_control_fallback = bool(raw_selector["used_control_fallback"])
    effective_path = (
        CONTROL_STRATEGY if used_control_fallback else GREEDY_STRATEGY
    )
    selected_ranges = [ranges[index] for index in selected_indices]
    retained_unique = min(total_code_tokens, union_length(selected_ranges))
    retained_ratio = (
        float(retained_unique) / float(total_code_tokens)
        if total_code_tokens
        else 1.0
    )

    selector_evidence = {
        "schema": GUARDED_SELECTOR_METADATA_SCHEMA_VERSION,
        "requested_policy": GUARDED_STRATEGY,
        "effective_path": effective_path,
        "used_control_fallback": used_control_fallback,
        "fallback_reason": (
            FALLBACK_NOT_STRICTLY_BETTER if used_control_fallback else None
        ),
        "selected_indices": selected_indices,
        "control_indices": control_indices,
        "candidate_indices": candidate_indices,
        "target_coverage_tokens": int(raw_selector["target_coverage_tokens"]),
        "control_target_coverage_tokens": int(
            raw_selector["control_target_coverage_tokens"]
        ),
        "candidate_target_coverage_tokens": int(candidate_target_coverage),
        "retained_tokens": int(raw_selector["retained_tokens"]),
        "control_retained_tokens": int(raw_selector["control_retained_tokens"]),
        "candidate_retained_tokens": int(candidate_retained_tokens),
    }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_windows": len(selected_indices),
        "stride": STRIDE,
        "num_tokens": int(attention_mask.sum().item()),
        "tokenizer_name": TOKENIZER_MODEL,
        "max_length": WINDOW_SIZE,
        "coverage_schema_version": TOKEN_COVERAGE_SCHEMA_VERSION,
        "pre_subsampling_window_count": total_windows,
        "pre_subsampling_code_tokens": total_code_tokens,
        "selected_window_indices": selected_indices,
        "selected_code_token_ranges": selected_ranges,
        "retained_unique_code_tokens": retained_unique,
        "retained_token_ratio": retained_ratio,
        "content_tokens_per_window": content_capacity,
        "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
        "selector": selector_evidence,
        "target_char_spans": spans,
        "target_token_ranges": target_token_ranges,
        "target_tokens": int(selected["target_tokens"]),
        "target_coverage_ratio": float(selected["target_coverage_ratio"]),
        "control_target_coverage_ratio": float(
            selected["control_target_coverage_ratio"]
        ),
        "retained_ratio": float(selected["retained_ratio"]),
        "control_retained_ratio": float(selected["control_retained_ratio"]),
    }


__all__ = [
    "FALLBACK_NOT_STRICTLY_BETTER",
    "GUARDED_SELECTOR_METADATA_SCHEMA_VERSION",
    "GuardedWindowTokenizationError",
    "tokenize_guarded_source",
]
