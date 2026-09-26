from __future__ import annotations

import torch
import pytest

from sentinel_data.representation.r4_guarded_lineage import TOKENIZER_NAME
from sentinel_data.representation.r4_guarded_tokenizer import (
    GuardedTokenizationError,
    char_spans_to_token_ranges,
    tokenize_repaired_source_guarded,
    window_ranges,
)


class _FakeFastTokenizer:
    name_or_path = TOKENIZER_NAME
    is_fast = True
    pad_token_id = 0

    def __init__(self) -> None:
        self.seen_source_texts: list[str] = []

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    def __call__(self, text: str, **kwargs):
        self.seen_source_texts.append(text)
        raw_ids = list(range(1, len(text) + 1))

        if kwargs.get("return_offsets_mapping"):
            return {
                "input_ids": raw_ids,
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }

        max_length = int(kwargs["max_length"])
        stride = int(kwargs["stride"])
        content_capacity = max_length - 2
        step = content_capacity - stride

        rows: list[list[int]] = []
        masks: list[list[int]] = []
        start = 0
        while True:
            end = min(start + content_capacity, len(raw_ids))
            payload = raw_ids[start:end]
            row = [101, *payload, 102]
            mask = [1] * len(row)
            row.extend([self.pad_token_id] * (max_length - len(row)))
            mask.extend([0] * (max_length - len(mask)))
            rows.append(row)
            masks.append(mask)
            if end >= len(raw_ids):
                break
            start += step

        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def test_short_under_cap_source_falls_back_and_pads_to_frozen_shape() -> None:
    tokenizer = _FakeFastTokenizer()
    source = 'contract Vault { string constant URL = "https://a//b"; uint x; }'

    result = tokenize_repaired_source_guarded(
        source,
        contract_id="a" * 64,
        requested_contract_names=["Vault"],
        tokenizer=tokenizer,
    )

    assert tuple(result["input_ids"].shape) == (4, 512)
    assert tuple(result["attention_mask"].shape) == (4, 512)
    assert result["num_windows"] == 1
    assert result["token_selector"]["used_control_fallback"] is True
    assert (
        result["token_selector"]["fallback_reason"]
        == "candidate_target_coverage_not_strictly_greater"
    )
    assert result["selected_window_indices"] == [0]
    assert int(result["attention_mask"][1:].sum().item()) == 0

    # The persisted repaired source is the exact tokenizer source view.
    assert tokenizer.seen_source_texts == [source, source]
    assert '"https://a//b"' in tokenizer.seen_source_texts[0]


def test_over_cap_target_gap_uses_guarded_candidate_on_strict_improvement() -> None:
    tokenizer = _FakeFastTokenizer()
    source = " " * 1400 + "contract Target { uint x; }" + " " * 1600

    result = tokenize_repaired_source_guarded(
        source,
        contract_id="b" * 64,
        requested_contract_names=["Target"],
        tokenizer=tokenizer,
    )

    selector = result["token_selector"]
    assert result["pre_subsampling_window_count"] > 4
    assert selector["used_control_fallback"] is False
    assert selector["effective_strategy"] == "target_aware_guarded_v1"
    assert (
        selector["candidate_target_coverage_tokens"]
        > selector["control_target_coverage_tokens"]
    )
    assert result["selected_window_indices"] == selector["selected_indices"]
    assert result["retained_unique_code_tokens"] == selector["retained_tokens"]


def test_target_evidence_and_selector_metadata_are_cross_bound() -> None:
    result = tokenize_repaired_source_guarded(
        " " * 1400 + "contract Target { uint x; }" + " " * 1600,
        contract_id="c" * 64,
        requested_contract_names=["Target"],
        tokenizer=_FakeFastTokenizer(),
    )

    evidence = result["target_evidence"]
    selector = result["token_selector"]

    assert evidence["requested_contract_names"] == ["Target"]
    assert len(evidence["target_char_spans"]) == 1
    assert len(evidence["target_token_ranges"]) == 1
    assert evidence["sha256"] == selector["target_evidence_sha256"]
    assert result["selector_config_sha256"] == selector["selector_config_sha256"]
    assert result["selected_window_indices"] == selector["selected_indices"]


def test_missing_requested_target_fails_closed_before_successful_artifact() -> None:
    with pytest.raises(
        GuardedTokenizationError,
        match="cannot establish requested target spans",
    ):
        tokenize_repaired_source_guarded(
            "contract Existing { uint x; }",
            contract_id="d" * 64,
            requested_contract_names=["Missing"],
            tokenizer=_FakeFastTokenizer(),
        )


def test_empty_requested_target_evidence_fails_closed() -> None:
    with pytest.raises(
        GuardedTokenizationError,
        match="requested_contract_names must not be empty",
    ):
        tokenize_repaired_source_guarded(
            "contract Existing { uint x; }",
            contract_id="e" * 64,
            requested_contract_names=[],
            tokenizer=_FakeFastTokenizer(),
        )


def test_wrong_tokenizer_identity_is_rejected() -> None:
    tokenizer = _FakeFastTokenizer()
    tokenizer.name_or_path = "different/tokenizer"

    with pytest.raises(GuardedTokenizationError, match="tokenizer identity mismatch"):
        tokenize_repaired_source_guarded(
            "contract Existing { uint x; }",
            contract_id="f" * 64,
            requested_contract_names=["Existing"],
            tokenizer=tokenizer,
        )


def test_repeated_guarded_tokenization_is_deterministic() -> None:
    source = " " * 1400 + "contract Target { uint x; }" + " " * 1600
    kwargs = {
        "contract_id": "1" * 64,
        "requested_contract_names": ["Target"],
    }

    first = tokenize_repaired_source_guarded(
        source,
        tokenizer=_FakeFastTokenizer(),
        **kwargs,
    )
    second = tokenize_repaired_source_guarded(
        source,
        tokenizer=_FakeFastTokenizer(),
        **kwargs,
    )

    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["attention_mask"], second["attention_mask"])
    for key in first:
        if key not in {"input_ids", "attention_mask"}:
            assert first[key] == second[key]


def test_character_spans_map_to_raw_token_ranges_without_offset_shift() -> None:
    offsets = [(index, index + 1) for index in range(20)]
    assert char_spans_to_token_ranges(offsets, [(3, 8), (12, 15)]) == [
        [3, 8],
        [12, 15],
    ]


def test_window_ranges_match_frozen_overlap_contract() -> None:
    assert window_ranges(1200, content_capacity=510, stride=256) == [
        [0, 510],
        [254, 764],
        [508, 1018],
        [762, 1200],
    ]
