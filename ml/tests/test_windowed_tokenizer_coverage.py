"""Repository-safe tests for GraphCodeBERT coverage telemetry.

A fake tokenizer is used: no model download or protected DATA is required.
"""

from __future__ import annotations

import torch

from ml.src.data_extraction import windowed_tokenizer as wt


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, *, total_tokens: int = 5000, total_windows: int = 20):
        self.total_tokens = total_tokens
        self.total_windows = total_windows

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, code, **kwargs):
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": list(range(self.total_tokens))}
        ids = torch.arange(self.total_windows * wt.WINDOW_SIZE).reshape(
            self.total_windows, wt.WINDOW_SIZE
        )
        masks = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": masks}


def test_selected_window_indices_preserve_historical_linspace_rule():
    assert wt._selected_window_indices(20, 4) == [0, 6, 13, 19]
    assert wt._selected_window_indices(3, 4) == [0, 1, 2]


def test_coverage_union_does_not_double_count_overlap():
    assert wt._covered_unique_tokens([[0, 10], [8, 12], [20, 25]]) == 17


def test_strict_tokenizer_keeps_frozen_shape_and_reports_omission(tmp_path, monkeypatch):
    sol = tmp_path / "C.sol"
    sol.write_text("pragma solidity ^0.8.0; contract C {}\n")
    monkeypatch.setattr(wt, "_tokenizer", FakeTokenizer())

    result = wt.tokenize_windowed_contract_strict(str(sol), strip_comments=False)

    assert tuple(result["input_ids"].shape) == (4, 512)
    assert tuple(result["attention_mask"].shape) == (4, 512)
    assert result["pre_subsampling_window_count"] == 20
    assert result["pre_subsampling_code_tokens"] == 5000
    assert result["selected_window_indices"] == [0, 6, 13, 19]
    assert 0.0 < result["retained_token_ratio"] < 1.0
    assert result["coverage_interpretation"] == "diagnostic_only_no_adequacy_threshold"


def test_comment_markers_in_string_survive_safe_tokenizer_stripping():
    source = 'contract C { string s = "https://x/a//b/*c*/"; } // gone\n'
    stripped = wt._strip_comments(source)
    assert 'https://x/a//b/*c*/' in stripped
    assert 'gone' not in stripped


def test_strict_empty_source_raises_but_legacy_wrapper_returns_none(tmp_path, monkeypatch):
    sol = tmp_path / "empty.sol"
    sol.write_text("   \n")
    monkeypatch.setattr(wt, "_tokenizer", FakeTokenizer())

    try:
        wt.tokenize_windowed_contract_strict(str(sol))
    except wt.TokenizationError:
        pass
    else:
        raise AssertionError("strict tokenizer must fail on empty source")

    assert wt.tokenize_windowed_contract(str(sol)) is None
