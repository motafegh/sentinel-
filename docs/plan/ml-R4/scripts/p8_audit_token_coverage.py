#!/usr/bin/env python3
"""Measure pre-subsampling GraphCodeBERT coverage for every Phase-8 contract.

Saved token payloads contain at most four selected windows and do not retain
the original overflow-window count.  This read-only profiler re-tokenizes the
stored normalized Solidity with the pinned local GraphCodeBERT tokenizer,
reconstructs the exact fixed-window selection, and reports how many source
tokens are represented or omitted.  JSON is printed to stdout; progress goes
to stderr.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
EXPORT_ROOT = DATA_ROOT / "exports/sentinel-r4-vnext-v1"
PREPROCESSED_ROOT = DATA_ROOT / "preprocessed"
REPRESENTATIONS_ROOT = DATA_ROOT / "representations"
TOKENIZER_NAME = "microsoft/graphcodebert-base"
WINDOW_SIZE = 512
SPECIAL_TOKENS_PER_WINDOW = 2
CONTENT_WINDOW = WINDOW_SIZE - SPECIAL_TOKENS_PER_WINDOW
STRIDE = 256
WINDOW_ADVANCE = CONTENT_WINDOW - STRIDE
MAX_WINDOWS = 4
CLASS_NAMES = (
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
)


def _quantiles(values: list[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)

    def at(fraction: float) -> int | float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "min": at(0.0),
        "p05": at(0.05),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": at(1.0),
    }


def _full_window_count(code_tokens: int) -> int:
    if code_tokens <= CONTENT_WINDOW:
        return 1
    return 1 + math.ceil((code_tokens - CONTENT_WINDOW) / WINDOW_ADVANCE)


def _selected_token_coverage(code_tokens: int, window_count: int) -> int:
    if code_tokens == 0:
        return 0
    if window_count <= MAX_WINDOWS:
        selected = list(range(window_count))
    else:
        selected = [
            round(index)
            for index in np.linspace(0, window_count - 1, MAX_WINDOWS)
        ]
    intervals: list[list[int]] = []
    for window_index in selected:
        start = window_index * WINDOW_ADVANCE
        end = min(code_tokens, start + CONTENT_WINDOW)
        if end <= start:
            continue
        if not intervals or start > intervals[-1][1]:
            intervals.append([start, end])
        else:
            intervals[-1][1] = max(intervals[-1][1], end)
    return sum(end - start for start, end in intervals)


def _new_accumulator() -> dict[str, Any]:
    return {
        "contracts": 0,
        "full_code_tokens": [],
        "pre_subsampling_windows": [],
        "retained_code_token_coverage_ratio": [],
        "omitted_code_tokens": [],
        "contracts_over_four_windows": 0,
        "omitted_code_tokens_total": 0,
    }


def _finalize(acc: dict[str, Any]) -> dict[str, Any]:
    return {
        "contracts": acc["contracts"],
        "full_code_tokens": _quantiles(acc["full_code_tokens"]),
        "pre_subsampling_windows": _quantiles(acc["pre_subsampling_windows"]),
        "contracts_over_four_windows": acc["contracts_over_four_windows"],
        "retained_code_token_coverage_ratio": _quantiles(
            acc["retained_code_token_coverage_ratio"]
        ),
        "omitted_code_tokens": _quantiles(acc["omitted_code_tokens"]),
        "omitted_code_tokens_total": acc["omitted_code_tokens_total"],
    }


def main() -> int:
    rows = [
        row
        for row in pq.read_table(EXPORT_ROOT / "ml_targets.parquet").to_pylist()
        if bool(row["representation_required"])
    ]
    rows.sort(key=lambda row: (str(row["source"]), str(row["contract_id"])))
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        local_files_only=True,
    )
    # The model limit is applied only later by fixed-window construction.  A
    # very high value suppresses a misleading tokenizer warning during this
    # intentional full-length measurement.
    tokenizer.model_max_length = 10**30

    by_source: dict[str, dict[str, Any]] = defaultdict(_new_accumulator)
    total = _new_accumulator()
    over_cap_by_role: Counter[str] = Counter()
    over_cap_target_cells: Counter[tuple[str, str, str, str]] = Counter()
    over_cap_effective_cells: Counter[tuple[str, str]] = Counter()
    over_cap_metric_cells: Counter[tuple[str, str]] = Counter()
    saved_window_count_mismatches = 0

    for ordinal, row in enumerate(rows, start=1):
        source = str(row["source"])
        contract_id = str(row["contract_id"])
        role = str(row["role"])
        code = (PREPROCESSED_ROOT / source / f"{contract_id}.sol").read_text(
            errors="replace"
        )
        code_tokens = len(
            tokenizer(
                code,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        window_count = _full_window_count(code_tokens)
        covered_tokens = _selected_token_coverage(code_tokens, window_count)
        omitted_tokens = code_tokens - covered_tokens
        coverage_ratio = covered_tokens / code_tokens if code_tokens else 1.0
        saved_sidecar = json.loads(
            (
                REPRESENTATIONS_ROOT
                / source
                / f"{contract_id}.rep.json"
            ).read_text()
        )
        if int(saved_sidecar["window_count"]) != min(window_count, MAX_WINDOWS):
            saved_window_count_mismatches += 1

        for acc in (by_source[source], total):
            acc["contracts"] += 1
            acc["full_code_tokens"].append(code_tokens)
            acc["pre_subsampling_windows"].append(window_count)
            acc["retained_code_token_coverage_ratio"].append(coverage_ratio)
            acc["omitted_code_tokens"].append(omitted_tokens)
            acc["omitted_code_tokens_total"] += omitted_tokens
            acc["contracts_over_four_windows"] += window_count > MAX_WINDOWS

        if window_count > MAX_WINDOWS:
            over_cap_by_role[role] += 1
            for index, class_name in enumerate(CLASS_NAMES):
                if row[f"target_{index}"] != 1:
                    continue
                strength = str(row[f"strength_{index}"])
                over_cap_target_cells[(source, class_name, strength, role)] += 1
                if bool(row[f"effective_loss_mask_{index}"]):
                    over_cap_effective_cells[(class_name, strength)] += 1
                if bool(row[f"outcome_metric_mask_{index}"]):
                    over_cap_metric_cells[(class_name, strength)] += 1

        if ordinal % 2000 == 0 or ordinal == len(rows):
            print(f"tokenized {ordinal}/{len(rows)} contracts", file=sys.stderr)

    report = {
        "schema": "sentinel-r4-phase8-token-coverage-audit-v1",
        "read_only": True,
        "tokenizer": TOKENIZER_NAME,
        "window_config": {
            "window_size": WINDOW_SIZE,
            "content_tokens_per_window": CONTENT_WINDOW,
            "stride": STRIDE,
            "window_advance": WINDOW_ADVANCE,
            "max_selected_windows": MAX_WINDOWS,
            "selection": "round(linspace(0, full_window_count - 1, 4))",
        },
        "saved_window_count_mismatches": saved_window_count_mismatches,
        "all_sources": _finalize(total),
        "by_source": {
            source: _finalize(acc) for source, acc in sorted(by_source.items())
        },
        "contracts_over_four_windows_by_role": dict(sorted(over_cap_by_role.items())),
        "over_cap_target_cells": {
            f"{source}|{class_name}|{strength}|{role}": count
            for (source, class_name, strength, role), count in sorted(
                over_cap_target_cells.items()
            )
        },
        "over_cap_effective_loss_cells": {
            f"{class_name}|{strength}": count
            for (class_name, strength), count in sorted(over_cap_effective_cells.items())
        },
        "over_cap_outcome_metric_cells": {
            f"{class_name}|{strength}": count
            for (class_name, strength), count in sorted(over_cap_metric_cells.items())
        },
        "interpretation": (
            "A contract over four pre-subsampling windows omits code tokens from the "
            "saved token branch. Coverage is the exact union of original tokenizer "
            "positions included by the fixed window-selection algorithm; the graph "
            "branch is evaluated separately."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if saved_window_count_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
