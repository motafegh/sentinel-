#!/usr/bin/env python3
"""Compare bounded four-window strategies on repaired local contracts.

This is a read-only experiment.  It does not regenerate representation files or
change the frozen `[4,512]` model input.  It compares the historical global
linspace control with a target-contract-aware four-window candidate using the
same GraphCodeBERT tokenization and repaired graph target recorded in sidecars.

Run only after repaired preprocessing, representations, and publication exist.
Use TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 so the experiment resolves the
already-accepted local GraphCodeBERT cache instead of network state.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_REPRESENTATIONS = DATA_ROOT / "representations-r4-v2"
DEFAULT_PUBLICATION = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
ROLES = ("TRAIN_STRONG", "TRAIN_WEAK", "MODEL_SELECTION")


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _mask_strings(source: str) -> str:
    chars = list(source)
    quote: str | None = None
    i = 0
    while i < len(chars):
        ch = chars[i]
        if quote is None:
            if ch in {"'", '"'}:
                quote = ch
                chars[i] = " "
            i += 1
            continue
        if ch == "\\" and i + 1 < len(chars):
            chars[i] = " "
            chars[i + 1] = " "
            i += 2
            continue
        chars[i] = "\n" if ch == "\n" else " "
        if ch == quote:
            quote = None
        i += 1
    return "".join(chars)


def _target_char_spans(
    source: str, targets: list[str]
) -> list[tuple[int, int]]:
    from sentinel_data.representation.target_selector import declarations

    masked = _mask_strings(source)
    items = declarations(source)
    spans: list[tuple[int, int]] = []
    for target in targets:
        matches = [item for item in items if item.name == target]
        if len(matches) != 1:
            raise ValueError(
                f"target declaration count for {target!r} is {len(matches)}"
            )
        start = matches[0].source_offset
        open_brace = masked.find("{", start)
        if open_brace < 0:
            raise ValueError(f"target {target!r} has no opening brace")
        depth = 0
        for index in range(open_brace, len(masked)):
            ch = masked[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1))
                    break
        else:
            raise ValueError(f"target {target!r} has no matching closing brace")
    return spans


def _window_ranges(total_tokens: int, content_capacity: int, stride: int) -> list[list[int]]:
    if total_tokens <= 0:
        return []
    step = max(1, content_capacity - stride)
    total_windows = 1 if total_tokens <= content_capacity else (
        math.ceil((total_tokens - content_capacity) / step) + 1
    )
    return [
        [index * step, min(index * step + content_capacity, total_tokens)]
        for index in range(total_windows)
    ]


def _union_length(ranges: list[list[int]]) -> int:
    ordered = sorted((start, end) for start, end in ranges if end > start)
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


def _intersect_union_length(
    ranges: list[list[int]], target_ranges: list[list[int]]
) -> int:
    intersections = [
        [max(start, target_start), min(end, target_end)]
        for start, end in ranges
        for target_start, target_end in target_ranges
        if min(end, target_end) > max(start, target_start)
    ]
    return _union_length(intersections)


def _linspace_indices(total: int, count: int = 4) -> list[int]:
    if total <= count:
        return list(range(total))
    # Exact rule retained by the production tokenizer.
    import numpy as np

    return [round(value) for value in np.linspace(0, total - 1, count)]


def _target_aware_indices(
    ranges: list[list[int]], target_ranges: list[list[int]], count: int = 4
) -> list[int]:
    selected: list[int] = []
    covered = 0
    while len(selected) < count:
        candidates: list[tuple[int, int]] = []
        for index in range(len(ranges)):
            if index in selected:
                continue
            score = _intersect_union_length(
                [ranges[value] for value in (*selected, index)], target_ranges
            )
            candidates.append((score - covered, index))
        if not candidates:
            break
        gain, index = max(candidates, key=lambda item: (item[0], -item[1]))
        if gain <= 0:
            break
        selected.append(index)
        covered += gain
    for index in _linspace_indices(len(ranges), count):
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


def _load_rows(publication: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(publication / "ml_targets.parquet").to_pylist()
    return [row for row in rows if str(row["role"]) in ROLES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument("--representations-root", type=Path, default=DEFAULT_REPRESENTATIONS)
    parser.add_argument("--publication-root", type=Path, default=DEFAULT_PUBLICATION)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    import transformers

    from ml.src.data_extraction.windowed_tokenizer import (
        STRIDE,
        TOKENIZER_MODEL,
        WINDOW_SIZE,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=True,
        local_files_only=True,
    )
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    capacity = WINDOW_SIZE - special_tokens

    rows = sorted(_load_rows(args.publication_root), key=lambda row: str(row["contract_id"]))
    if args.limit is not None:
        rows = rows[: args.limit]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        contract_id = str(row["contract_id"])
        source = str(row["source"])
        sol = args.preprocessed_root / source / f"{contract_id}.sol"
        sidecar_path = args.representations_root / source / f"{contract_id}.rep.json"
        try:
            source_text = sol.read_text(encoding="utf-8")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            targets = [
                str(value)
                for value in (sidecar.get("requested_contract_names") or ())
            ]
            if not targets:
                raise ValueError("representation sidecar has no requested targets")
            char_spans = _target_char_spans(source_text, targets)
            encoded = tokenizer(
                source_text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            token_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
                offsets = offsets[0]
            total_tokens = len(token_ids)
            target_ranges: list[list[int]] = []
            for target, char_span in zip(targets, char_spans):
                target_token_indices = [
                    index
                    for index, (start, end) in enumerate(offsets)
                    if end > char_span[0] and start < char_span[1]
                ]
                if not target_token_indices:
                    raise ValueError(
                        f"target declaration {target!r} maps to zero code tokens"
                    )
                target_ranges.append(
                    [min(target_token_indices), max(target_token_indices) + 1]
                )
            ranges = _window_ranges(total_tokens, capacity, STRIDE)
            control_indices = _linspace_indices(len(ranges), 4)
            target_indices = _target_aware_indices(ranges, target_ranges, 4)
            control_ranges = [ranges[index] for index in control_indices]
            candidate_ranges = [ranges[index] for index in target_indices]
            target_tokens = _union_length(target_ranges)
            records.append(
                {
                    "contract_id": contract_id,
                    "source": source,
                    "role": str(row["role"]),
                    "target_contract_names": targets,
                    "total_code_tokens": total_tokens,
                    "total_windows": len(ranges),
                    "target_contract_tokens": target_tokens,
                    "control_indices": control_indices,
                    "target_aware_indices": target_indices,
                    "control_retained_ratio": _union_length(control_ranges) / total_tokens,
                    "target_aware_retained_ratio": _union_length(candidate_ranges) / total_tokens,
                    "control_target_coverage_ratio": _intersect_union_length(control_ranges, target_ranges) / target_tokens,
                    "target_aware_target_coverage_ratio": _intersect_union_length(candidate_ranges, target_ranges) / target_tokens,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "contract_id": contract_id,
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    over_cap = [row for row in records if row["total_windows"] > 4]
    control_target = [row["control_target_coverage_ratio"] for row in over_cap]
    candidate_target = [row["target_aware_target_coverage_ratio"] for row in over_cap]
    control_global = [row["control_retained_ratio"] for row in over_cap]
    candidate_global = [row["target_aware_retained_ratio"] for row in over_cap]
    result = {
        "schema": "sentinel-r4-phase8-bounded-window-experiment-v1",
        "experiment_only": True,
        "changes_representations": False,
        "changes_model_shape": False,
        "tokenizer_name": TOKENIZER_MODEL,
        "transformers_version": transformers.__version__,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_windows": 4,
        "roles": list(ROLES),
        "records_requested": len(rows),
        "records_analyzed": len(records),
        "failures": failures[:200],
        "failures_total": len(failures),
        "over_four_window_records": len(over_cap),
        "control_global_retained_ratio": _quantiles(control_global),
        "target_aware_global_retained_ratio": _quantiles(candidate_global),
        "control_target_contract_coverage_ratio": _quantiles(control_target),
        "target_aware_target_contract_coverage_ratio": _quantiles(candidate_target),
        "target_coverage_improved_records": sum(
            candidate > control
            for candidate, control in zip(candidate_target, control_target)
        ),
        "target_coverage_regressed_records": sum(
            candidate < control
            for candidate, control in zip(candidate_target, control_target)
        ),
        "decision_boundary": (
            "This profiler alone cannot change the production selector. Review coverage "
            "with the bounded GPU diagnostic and record a new extractor decision/version "
            "before adopting target-aware windows."
        ),
        "records": records,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
