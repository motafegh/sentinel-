#!/usr/bin/env python3
"""Compare historical, greedy, and guarded four-window selectors on repaired DATA.

This is a read-only CPU/tokenizer experiment. It does not rewrite bound token
artifacts and does not promote a selector. The guarded candidate must never
reduce target-contract token coverage relative to the historical control.

All strategies operate on the same comment-stripped, offset-preserving code view
used by repaired-v2 production tokenization. The report binds itself to the
publication manifest, physical representation digest, and source commit so it
cannot be mixed silently with another logical population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from ml.src.data_extraction.bounded_window_selector import (
    CONTROL_STRATEGY,
    GREEDY_STRATEGY,
    GUARDED_STRATEGY,
    char_spans_to_token_ranges,
    prepare_source_for_tokenization,
    select_indices,
    union_length,
    window_ranges,
)
from ml.src.data_extraction.windowed_tokenizer import STRIDE, TOKENIZER_MODEL, WINDOW_SIZE
from sentinel_data.representation.r4_target_spans import target_contract_char_spans

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_REPRESENTATIONS = DATA_ROOT / "representations-r4-v2"
DEFAULT_PUBLICATION = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
DEFAULT_OUTPUT = DATA_ROOT / "r4-v2-build/bounded_window_selector_v1.json"
ROLES = ("TRAIN_STRONG", "TRAIN_WEAK", "MODEL_SELECTION")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    import transformers

    manifest_path = args.publication_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding_digest = str(
        (manifest.get("representation_binding_report") or {}).get(
            "binding_digest_sha256"
        )
        or ""
    )
    if not binding_digest:
        raise ValueError("selector publication lacks representation binding digest")
    ml_targets_path = args.publication_root / "ml_targets.parquet"
    expected_ml_sha = ((manifest.get("artifacts") or {}).get("ml_targets") or {}).get(
        "sha256"
    )
    if not expected_ml_sha or _sha256(ml_targets_path) != expected_ml_sha:
        raise ValueError("selector publication ml_targets.parquet hash mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=True,
        local_files_only=True,
    )
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    capacity = WINDOW_SIZE - special_tokens

    rows = sorted(
        _load_rows(args.publication_root),
        key=lambda row: (
            str(row["role"]),
            str(row["group_id"]),
            str(row["contract_id"]),
        ),
    )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        rows = rows[: args.limit]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    strategies = (CONTROL_STRATEGY, GREEDY_STRATEGY, GUARDED_STRATEGY)

    for row in rows:
        contract_id = str(row["contract_id"])
        source = str(row["source"])
        sol_path = args.preprocessed_root / source / f"{contract_id}.sol"
        sidecar_path = args.representations_root / source / f"{contract_id}.rep.json"
        try:
            source_text = sol_path.read_text(encoding="utf-8")
            token_source = prepare_source_for_tokenization(source_text)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            targets = [
                str(value)
                for value in (sidecar.get("requested_contract_names") or ())
            ]
            if not targets:
                raise ValueError("representation sidecar has no requested_contract_names")
            char_spans = target_contract_char_spans(source_text, targets)

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
            ranges = window_ranges(
                total_tokens,
                content_capacity=capacity,
                stride=STRIDE,
            )
            target_ranges = char_spans_to_token_ranges(offsets, char_spans)
            target_tokens = union_length(target_ranges)
            result_by_strategy: dict[str, Any] = {}
            for strategy in strategies:
                selection = select_indices(
                    ranges,
                    target_ranges,
                    count=4,
                    strategy=strategy,
                )
                result_by_strategy[strategy] = {
                    **selection.as_dict(),
                    "target_coverage_ratio": (
                        selection.target_coverage_tokens / target_tokens
                        if target_tokens
                        else 1.0
                    ),
                    "retained_ratio": (
                        selection.retained_tokens / total_tokens
                        if total_tokens
                        else 1.0
                    ),
                }

            control_coverage = result_by_strategy[CONTROL_STRATEGY][
                "target_coverage_ratio"
            ]
            guarded_coverage = result_by_strategy[GUARDED_STRATEGY][
                "target_coverage_ratio"
            ]
            if guarded_coverage + 1e-12 < control_coverage:
                raise AssertionError(
                    "guarded selector regressed target coverage: "
                    f"{guarded_coverage} < {control_coverage}"
                )

            records.append(
                {
                    "contract_id": contract_id,
                    "source": source,
                    "role": str(row["role"]),
                    "group_id": str(row["group_id"]),
                    "graph_component_count": int(
                        sidecar.get("graph_component_count", 0)
                    ),
                    "graph_extraction_mode": str(
                        sidecar.get("graph_extraction_mode") or ""
                    ),
                    "total_code_tokens": total_tokens,
                    "total_windows": len(ranges),
                    "target_contract_names": targets,
                    "target_tokens": target_tokens,
                    "strategies": result_by_strategy,
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

    over_cap = [record for record in records if record["total_windows"] > 4]
    summary: dict[str, Any] = {}
    for strategy in strategies:
        coverages = [
            record["strategies"][strategy]["target_coverage_ratio"]
            for record in over_cap
        ]
        retained = [
            record["strategies"][strategy]["retained_ratio"]
            for record in over_cap
        ]
        summary[strategy] = {
            "target_coverage_ratio": _quantiles(coverages),
            "retained_ratio": _quantiles(retained),
        }

    improved = sum(
        record["strategies"][GUARDED_STRATEGY]["target_coverage_ratio"]
        > record["strategies"][CONTROL_STRATEGY]["target_coverage_ratio"]
        for record in over_cap
    )
    equal = sum(
        record["strategies"][GUARDED_STRATEGY]["target_coverage_ratio"]
        == record["strategies"][CONTROL_STRATEGY]["target_coverage_ratio"]
        for record in over_cap
    )
    fallback = sum(
        bool(record["strategies"][GUARDED_STRATEGY]["used_control_fallback"])
        for record in over_cap
    )

    report = {
        "schema": "sentinel-r4-phase8-bounded-window-selector-experiment-v1",
        "experiment_only": True,
        "promotion_authorized": False,
        "changes_bound_representations": False,
        "lineage": {
            "dataset_version": manifest.get("dataset_version"),
            "grouping_version": manifest.get("grouping_version"),
            "partition_version": manifest.get("partition_version"),
            "publication_manifest_sha256": _sha256(manifest_path),
            "representation_binding_digest_sha256": binding_digest,
            "source_commit": _source_commit(),
        },
        "token_source_semantics": "repaired_v2_comment_stripped_offset_preserving",
        "tokenizer_name": TOKENIZER_MODEL,
        "transformers_version": transformers.__version__,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_windows": 4,
        "roles": list(ROLES),
        "records_requested": len(rows),
        "records_analyzed": len(records),
        "failures_total": len(failures),
        "failures": failures[:200],
        "over_four_window_records": len(over_cap),
        "strategy_summary_over_cap": summary,
        "guarded_target_coverage_improved_records": improved,
        "guarded_target_coverage_equal_records": equal,
        "guarded_control_fallback_records": fallback,
        "guarded_target_coverage_regressed_records": 0,
        "decision_boundary": (
            "CPU coverage evidence cannot promote a selector. Promotion requires "
            "identical-initialization bounded CUDA comparison, regression review, "
            "worst-case graph/token diagnostics, bound-token equivalence verification, "
            "and an explicit new extractor decision."
        ),
        "records": records,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
