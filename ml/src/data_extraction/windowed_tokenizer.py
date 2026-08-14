"""Windowed GraphCodeBERT tokenization for SENTINEL.

The frozen model contract remains exactly ``[max_windows, 512]``.  Phase-8
real-data repair adds *coverage evidence* before subsampling so a successful
shape check cannot be mistaken for adequate long-contract coverage.

Legacy callers may continue using :func:`tokenize_windowed_contract`, which
returns ``None`` on failure.  Repaired build code should use
:func:`tokenize_windowed_contract_strict` so failures are explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

TOKENIZER_MODEL: str = "microsoft/graphcodebert-base"
WINDOW_SIZE: int = 512
STRIDE: int = 256
MAX_WINDOWS: int = 4
TOKEN_COVERAGE_SCHEMA_VERSION: str = "r4-token-coverage-v1"

_tokenizer = None


class TokenizationError(RuntimeError):
    """Raised when repaired tokenization cannot produce a valid artifact."""


def init_worker() -> None:
    """Load GraphCodeBERT tokenizer into the process-level global."""

    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)


def _strip_comments(source: str) -> str:
    """Lexically remove comments while preserving markers inside strings."""

    CODE, SINGLE, DOUBLE, LINE, BLOCK = range(5)
    state = CODE
    out: list[str] = []
    i = 0
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == CODE:
            if ch == "/" and nxt == "/":
                out.extend((" ", " "))
                state = LINE
                i += 2
                continue
            if ch == "/" and nxt == "*":
                out.extend((" ", " "))
                state = BLOCK
                i += 2
                continue
            out.append(ch)
            if ch == "'":
                state = SINGLE
            elif ch == '"':
                state = DOUBLE
            i += 1
            continue
        if state in (SINGLE, DOUBLE):
            quote = "'" if state == SINGLE else '"'
            out.append(ch)
            if ch == "\\" and i + 1 < len(source):
                out.append(source[i + 1])
                i += 2
                continue
            if ch == quote:
                state = CODE
            i += 1
            continue
        if state == LINE:
            if ch == "\n":
                out.append("\n")
                state = CODE
            else:
                out.append(" ")
            i += 1
            continue
        if ch == "*" and nxt == "/":
            out.extend((" ", " "))
            state = CODE
            i += 2
            continue
        out.append("\n" if ch == "\n" else " ")
        i += 1
    return "".join(out)


def _selected_window_indices(total_windows: int, max_windows: int) -> list[int]:
    """Return deterministic linspace-selected window indices."""

    if max_windows <= 0:
        raise ValueError("max_windows must be > 0")
    if total_windows <= max_windows:
        return list(range(total_windows))
    # Preserve the historical selection rule exactly.
    return [round(i) for i in np.linspace(0, total_windows - 1, max_windows)]


def _select_windows(
    all_input_ids: list,
    all_attention_masks: list,
    max_windows: int,
) -> tuple[list, list]:
    indices = _selected_window_indices(len(all_input_ids), max_windows)
    return (
        [all_input_ids[i] for i in indices],
        [all_attention_masks[i] for i in indices],
    )


def _token_ranges(
    *,
    total_code_tokens: int,
    total_windows: int,
    selected_indices: list[int],
    content_capacity: int,
    stride: int,
) -> list[list[int]]:
    """Map selected overflow-window indices to half-open code-token ranges."""

    if total_code_tokens <= 0 or total_windows <= 0:
        return []
    step = max(1, content_capacity - stride)
    ranges: list[list[int]] = []
    for index in selected_indices:
        start = min(index * step, total_code_tokens)
        end = min(start + content_capacity, total_code_tokens)
        ranges.append([start, end])
    return ranges


def _covered_unique_tokens(ranges: list[list[int]]) -> int:
    """Return union length of half-open integer token ranges."""

    if not ranges:
        return 0
    ordered = sorted((int(start), int(end)) for start, end in ranges if end > start)
    if not ordered:
        return 0
    total = 0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    return total + (cur_end - cur_start)


def tokenize_windowed_contract_strict(
    contract_path: str,
    max_windows: int = MAX_WINDOWS,
    strip_comments: bool = True,
) -> dict[str, Any]:
    """Tokenize one file into exactly ``[max_windows, 512]`` plus coverage evidence.

    Coverage is measured over unique pre-special-token code-token positions.
    It is diagnostic evidence only; no threshold in this module declares a
    contract sufficiently covered.
    """

    global _tokenizer
    if _tokenizer is None:
        init_worker()

    path = Path(contract_path)
    try:
        code = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        code = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TokenizationError(f"cannot read {path}: {exc}") from exc

    if not code.strip():
        raise TokenizationError(f"empty Solidity source: {path}")
    if strip_comments:
        code = _strip_comments(code)
        if not code.strip():
            raise TokenizationError(f"source contains no code after comment removal: {path}")

    if max_windows <= 0:
        raise TokenizationError(f"max_windows must be > 0, got {max_windows}")

    try:
        raw_tokens = _tokenizer(
            code,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        # Fast tokenizers normally return list[int] for a single string.  Some
        # test doubles return one nested list; normalize both forms.
        if raw_tokens and isinstance(raw_tokens[0], list):
            raw_tokens = raw_tokens[0]
        total_code_tokens = len(raw_tokens)

        encoded = _tokenizer(
            code,
            max_length=WINDOW_SIZE,
            padding="max_length",
            truncation=True,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
    except Exception as exc:
        raise TokenizationError(f"GraphCodeBERT tokenization failed for {path}: {exc}") from exc

    all_ids: list = encoded["input_ids"].tolist()
    all_masks: list = encoded["attention_mask"].tolist()
    if not all_ids:
        raise TokenizationError(f"tokenizer produced zero windows for {path}")

    total_windows = len(all_ids)
    selected_indices = _selected_window_indices(total_windows, max_windows)
    selected_ids = [all_ids[i] for i in selected_indices]
    selected_masks = [all_masks[i] for i in selected_indices]
    num_real_windows = len(selected_ids)

    pad_id = _tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else 0
    while len(selected_ids) < max_windows:
        selected_ids.append([pad_id] * WINDOW_SIZE)
        selected_masks.append([0] * WINDOW_SIZE)

    input_ids = torch.tensor(selected_ids, dtype=torch.long)
    attention_mask = torch.tensor(selected_masks, dtype=torch.long)
    num_real_attention_tokens = int(attention_mask.sum().item())

    try:
        special_tokens = int(_tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special_tokens = 2
    content_capacity = max(1, WINDOW_SIZE - special_tokens)
    selected_ranges = _token_ranges(
        total_code_tokens=total_code_tokens,
        total_windows=total_windows,
        selected_indices=selected_indices,
        content_capacity=content_capacity,
        stride=STRIDE,
    )
    retained_unique = min(total_code_tokens, _covered_unique_tokens(selected_ranges))
    retained_ratio = (
        float(retained_unique) / float(total_code_tokens)
        if total_code_tokens > 0
        else 1.0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_windows": num_real_windows,
        "stride": STRIDE,
        "num_tokens": num_real_attention_tokens,
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
    }


def tokenize_windowed_contract(
    contract_path: str,
    max_windows: int = MAX_WINDOWS,
    strip_comments: bool = True,
) -> Optional[dict[str, Any]]:
    """Backward-compatible wrapper returning ``None`` on tokenization failure."""

    try:
        return tokenize_windowed_contract_strict(
            contract_path,
            max_windows=max_windows,
            strip_comments=strip_comments,
        )
    except TokenizationError:
        return None
