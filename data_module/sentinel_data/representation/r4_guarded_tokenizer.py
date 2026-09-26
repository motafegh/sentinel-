"""Guarded GraphCodeBERT tokenization for the R4-D-012 token lineage.

The caller supplies the exact persisted repaired Solidity source and the exact
requested contract names inherited from the accepted R4-D-011 parent sidecar.
This module does not perform a second preprocessing/comment-removal pass.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from sentinel_data.representation.r4_guarded_lineage import (
    MAX_WINDOWS,
    PARENT_BINDING_DIGEST_SHA256,
    PARENT_DECISION_ID,
    SELECTOR_CONFIG_SHA256,
    STRIDE,
    TOKENIZER_NAME,
    TOKEN_LINEAGE_ID,
    WINDOW_SIZE,
    build_selector_metadata,
    build_target_evidence,
)
from sentinel_data.representation.r4_target_spans import target_contract_char_spans
from sentinel_data.representation.r4_window_selector import (
    select_guarded_windows,
    union_length,
)

TOKEN_COVERAGE_SCHEMA_VERSION = "r4-token-coverage-v1"
COVERAGE_INTERPRETATION = "diagnostic_only_no_adequacy_threshold"


class GuardedTokenizationError(RuntimeError):
    """Raised when a guarded token artifact cannot be produced safely."""


@lru_cache(maxsize=1)
def _load_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise GuardedTokenizationError(
            "guarded tokenization requires transformers"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    except Exception as exc:  # pragma: no cover - external model/cache failure
        raise GuardedTokenizationError(
            f"cannot load frozen tokenizer {TOKENIZER_NAME!r}: {exc}"
        ) from exc
    _validate_tokenizer_identity(tokenizer)
    return tokenizer


def _validate_tokenizer_identity(tokenizer: Any) -> None:
    if getattr(tokenizer, "is_fast", True) is False:
        raise GuardedTokenizationError(
            "guarded tokenization requires a fast tokenizer with offset mappings"
        )
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name and name != TOKENIZER_NAME:
        raise GuardedTokenizationError(
            f"tokenizer identity mismatch: {name!r} != {TOKENIZER_NAME!r}"
        )


def _normalize_single_sequence(value: Any, *, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise GuardedTokenizationError(f"{field} must be a tokenizer list")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise GuardedTokenizationError(
                f"{field} unexpectedly contains multiple input sequences"
            )
        value = value[0]
    return value


def _normalize_window_matrix(value: Any, *, field: str) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise GuardedTokenizationError(f"{field} must contain tokenizer windows")
    rows: list[list[int]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list):
            raise GuardedTokenizationError(f"{field}[{index}] is not a list")
        if len(row) != WINDOW_SIZE:
            raise GuardedTokenizationError(
                f"{field}[{index}] shape drift: {len(row)} != {WINDOW_SIZE}"
            )
        try:
            rows.append([int(item) for item in row])
        except (TypeError, ValueError) as exc:
            raise GuardedTokenizationError(
                f"{field}[{index}] contains a non-integer value"
            ) from exc
    return rows


def window_ranges(
    total_tokens: int,
    *,
    content_capacity: int,
    stride: int = STRIDE,
) -> list[list[int]]:
    """Return overflow-window ranges over pre-special-token code positions."""

    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int):
        raise GuardedTokenizationError("total_tokens must be an integer")
    if total_tokens < 1:
        raise GuardedTokenizationError("tokenizer produced zero raw code tokens")
    if (
        isinstance(content_capacity, bool)
        or not isinstance(content_capacity, int)
        or content_capacity < 1
    ):
        raise GuardedTokenizationError("content_capacity must be >= 1")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise GuardedTokenizationError("stride must be an integer")
    if stride < 0 or stride >= content_capacity:
        raise GuardedTokenizationError(
            "stride must satisfy 0 <= stride < content_capacity"
        )

    step = content_capacity - stride
    ranges: list[list[int]] = []
    start = 0
    while True:
        end = min(start + content_capacity, total_tokens)
        ranges.append([start, end])
        if end >= total_tokens:
            return ranges
        start += step


def char_spans_to_token_ranges(
    offsets: Sequence[Sequence[int]],
    char_spans: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Map exact target character spans onto raw GraphCodeBERT token positions."""

    normalized_offsets: list[tuple[int, int]] = []
    for index, raw in enumerate(offsets):
        if len(raw) != 2:
            raise GuardedTokenizationError(
                f"offset_mapping[{index}] must contain exactly two integers"
            )
        start, end = raw
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise GuardedTokenizationError(
                f"offset_mapping[{index}] is not a valid half-open range"
            )
        normalized_offsets.append((start, end))

    token_ranges: list[list[int]] = []
    for index, raw in enumerate(char_spans):
        if len(raw) != 2:
            raise GuardedTokenizationError(
                f"target_char_spans[{index}] must contain exactly two integers"
            )
        char_start, char_end = raw
        if (
            isinstance(char_start, bool)
            or isinstance(char_end, bool)
            or not isinstance(char_start, int)
            or not isinstance(char_end, int)
            or char_start < 0
            or char_end <= char_start
        ):
            raise GuardedTokenizationError(
                f"target_char_spans[{index}] is not a valid half-open range"
            )

        covered = [
            token_index
            for token_index, (start, end) in enumerate(normalized_offsets)
            if end > char_start and start < char_end
        ]
        if not covered:
            raise GuardedTokenizationError(
                f"target character span [{char_start}, {char_end}) maps to zero tokens"
            )
        token_ranges.append([covered[0], covered[-1] + 1])
    return token_ranges


def _requested_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise GuardedTokenizationError(
            "requested_contract_names must be a non-empty sequence"
        )
    names: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise GuardedTokenizationError(
                "requested_contract_names contains an invalid name"
            )
        names.append(value.strip())
    if not names:
        raise GuardedTokenizationError("requested_contract_names must not be empty")
    if len(set(names)) != len(names):
        raise GuardedTokenizationError("requested_contract_names must be unique")
    return tuple(names)


def tokenize_repaired_source_guarded(
    source_text: str,
    *,
    contract_id: str,
    requested_contract_names: Sequence[str],
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Tokenize one repaired source under the exact guarded-selector contract."""

    if not isinstance(source_text, str) or not source_text.strip():
        raise GuardedTokenizationError("source_text must contain repaired Solidity")
    names = _requested_names(requested_contract_names)

    try:
        char_spans = target_contract_char_spans(source_text, names)
    except Exception as exc:
        raise GuardedTokenizationError(
            f"cannot establish requested target spans: {exc}"
        ) from exc

    tokenizer = tokenizer if tokenizer is not None else _load_tokenizer()
    _validate_tokenizer_identity(tokenizer)

    try:
        raw = tokenizer(
            source_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:
        raise GuardedTokenizationError(
            f"raw GraphCodeBERT tokenization failed: {exc}"
        ) from exc

    raw_ids = _normalize_single_sequence(raw.get("input_ids"), field="input_ids")
    raw_offsets = _normalize_single_sequence(
        raw.get("offset_mapping"), field="offset_mapping"
    )
    if len(raw_ids) != len(raw_offsets):
        raise GuardedTokenizationError(
            "raw token IDs and offset mappings have different lengths"
        )
    total_tokens = len(raw_ids)
    if total_tokens < 1:
        raise GuardedTokenizationError("tokenizer produced zero raw code tokens")

    try:
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception as exc:
        raise GuardedTokenizationError(
            f"cannot determine tokenizer special-token count: {exc}"
        ) from exc
    content_capacity = WINDOW_SIZE - special_tokens
    ranges = window_ranges(
        total_tokens,
        content_capacity=content_capacity,
        stride=STRIDE,
    )
    target_ranges = char_spans_to_token_ranges(raw_offsets, char_spans)

    try:
        decision = select_guarded_windows(
            ranges,
            target_ranges,
            max_windows=MAX_WINDOWS,
        )
    except ValueError as exc:
        raise GuardedTokenizationError(
            f"guarded selector rejected token evidence: {exc}"
        ) from exc

    try:
        encoded = tokenizer(
            source_text,
            max_length=WINDOW_SIZE,
            padding="max_length",
            truncation=True,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
    except Exception as exc:
        raise GuardedTokenizationError(
            f"windowed GraphCodeBERT tokenization failed: {exc}"
        ) from exc

    all_ids = _normalize_window_matrix(encoded.get("input_ids"), field="input_ids")
    all_masks = _normalize_window_matrix(
        encoded.get("attention_mask"), field="attention_mask"
    )
    if len(all_ids) != len(all_masks):
        raise GuardedTokenizationError(
            "windowed token IDs and attention masks have different counts"
        )
    if len(all_ids) != len(ranges):
        raise GuardedTokenizationError(
            "computed range count diverges from tokenizer overflow windows: "
            f"{len(ranges)} != {len(all_ids)}"
        )

    selected_ids = [all_ids[index] for index in decision.selected_indices]
    selected_masks = [all_masks[index] for index in decision.selected_indices]
    real_window_count = len(selected_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    try:
        pad_id = int(pad_id)
    except (TypeError, ValueError) as exc:
        raise GuardedTokenizationError("tokenizer pad_token_id must be an integer") from exc

    while len(selected_ids) < MAX_WINDOWS:
        selected_ids.append([pad_id] * WINDOW_SIZE)
        selected_masks.append([0] * WINDOW_SIZE)
    if len(selected_ids) != MAX_WINDOWS:
        raise GuardedTokenizationError("guarded selection exceeded frozen window cap")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise GuardedTokenizationError("guarded tokenization requires torch") from exc

    input_ids = torch.tensor(selected_ids, dtype=torch.long)
    attention_mask = torch.tensor(selected_masks, dtype=torch.long)
    if tuple(input_ids.shape) != (MAX_WINDOWS, WINDOW_SIZE):
        raise GuardedTokenizationError(
            f"frozen input shape drift: {tuple(input_ids.shape)}"
        )
    if tuple(attention_mask.shape) != (MAX_WINDOWS, WINDOW_SIZE):
        raise GuardedTokenizationError(
            f"frozen attention shape drift: {tuple(attention_mask.shape)}"
        )

    target_tokens = union_length(target_ranges)
    try:
        target_evidence = build_target_evidence(
            contract_id=contract_id,
            requested_contract_names=names,
            target_char_spans=char_spans,
            target_token_ranges=target_ranges,
            target_tokens=target_tokens,
        )
        selector_metadata = build_selector_metadata(
            decision,
            target_evidence_sha256=target_evidence["sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise GuardedTokenizationError(
            f"cannot bind guarded selector metadata: {exc}"
        ) from exc

    selected_ranges = [ranges[index] for index in decision.selected_indices]
    retained_unique = decision.retained_tokens
    retained_ratio = float(retained_unique) / float(total_tokens)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_windows": real_window_count,
        "stride": STRIDE,
        "num_tokens": int(attention_mask.sum().item()),
        "tokenizer_name": TOKENIZER_NAME,
        "max_length": WINDOW_SIZE,
        "coverage_schema_version": TOKEN_COVERAGE_SCHEMA_VERSION,
        "pre_subsampling_window_count": len(ranges),
        "pre_subsampling_code_tokens": total_tokens,
        "selected_window_indices": list(decision.selected_indices),
        "selected_code_token_ranges": [list(value) for value in selected_ranges],
        "retained_unique_code_tokens": retained_unique,
        "retained_token_ratio": retained_ratio,
        "content_tokens_per_window": content_capacity,
        "coverage_interpretation": COVERAGE_INTERPRETATION,
        "token_lineage": TOKEN_LINEAGE_ID,
        "token_lineage_parent_decision": PARENT_DECISION_ID,
        "token_lineage_parent_binding_digest_sha256": PARENT_BINDING_DIGEST_SHA256,
        "selector_config_sha256": SELECTOR_CONFIG_SHA256,
        "target_evidence": target_evidence,
        "token_selector": selector_metadata,
    }


__all__ = [
    "COVERAGE_INTERPRETATION",
    "GuardedTokenizationError",
    "TOKEN_COVERAGE_SCHEMA_VERSION",
    "char_spans_to_token_ranges",
    "tokenize_repaired_source_guarded",
    "window_ranges",
]
