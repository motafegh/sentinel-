#!/usr/bin/env python3
"""Freeze the R4-GAP-002 DIVE population and initial blind-review sample.

This script operates only on the committed Phase-3 evidence ledger. It does not
read model predictions, tool votes, thresholds, calibration, or downstream
policy. The initial sample is a *screening* batch, not a fixed final review size.
High-authority source roles require adaptive expansion and, where appropriate,
a second review.

Sampling design
---------------
* five approved DIVE-positive strata only;
* historical TRAIN groups only;
* groups touching historical val/test are excluded from the initial review;
* one representative contract per project/dedup/contract group;
* no group is reused across the five initial strata;
* deterministic SHA-256 ranking bound to the gap ID and committed ledger;
* 20 contracts per stratum initially (100 total), with later expansion only when
  the role decision remains ambiguous or a high-authority role is plausible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

GAP_ID = "R4-GAP-002"
LEDGER_VERSION = "r4-ledger-v1"
EXPECTED_LEDGER_SHA256 = "3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7"
EXPECTED_ROWS = 224_930
INITIAL_PER_STRATUM = 20
SAMPLE_VERSION = "r4-gap-002-sample-v1"

TARGETS: tuple[tuple[str, int, str], ...] = (
    ("DenialOfService", 1, "DoS"),
    ("IntegerUO", 4, "Arithmetic"),
    ("Timestamp", 7, "Time manipulation"),
    ("TransactionOrderDependence", 8, "Front Running"),
    ("UnusedReturn", 9, "Unchecked Return Values"),
)
TARGET_BY_NAME = {name: (idx, native) for name, idx, native in TARGETS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_group_key(row: dict[str, Any]) -> str:
    project = row.get("project_group_id")
    if project not in (None, ""):
        return f"project:{project}"
    dedup = row.get("dedup_group_id")
    if dedup not in (None, ""):
        return f"dedup:{dedup}"
    return f"contract:{row['contract_id']}"


def rank_key(class_name: str, group_key: str, contract_id: str, ledger_sha: str) -> str:
    payload = f"{SAMPLE_VERSION}|{GAP_ID}|{ledger_sha}|{class_name}|{group_key}|{contract_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_row(row: dict[str, Any]) -> None:
    if row.get("ledger_version") != LEDGER_VERSION:
        raise ValueError(f"unexpected ledger_version: {row.get('ledger_version')!r}")
    class_name = row.get("class_name")
    if class_name in TARGET_BY_NAME:
        expected_idx = TARGET_BY_NAME[class_name][0]
        if int(row.get("class_index")) != expected_idx:
            raise ValueError(
                f"class order mismatch for {class_name}: {row.get('class_index')} != {expected_idx}"
            )


def build_population_and_sample(
    rows: Iterable[dict[str, Any]],
    *,
    ledger_sha: str,
    per_stratum: int = INITIAL_PER_STRATUM,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")

    rows = [dict(row) for row in rows]
    for row in rows:
        _validate_row(row)

    # A review group is eligible only if all of its DIVE occurrences remain in
    # the historical training split. This preserves val/test groups for later
    # independent role assignment rather than exposing them during Phase 4.
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("primary_source") != "dive":
            continue
        split = row.get("historical_split")
        if split is not None:
            group_splits[canonical_group_key(row)].add(str(split))

    populations: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in TARGETS}
    for row in rows:
        if row.get("primary_source") != "dive":
            continue
        class_name = row.get("class_name")
        if class_name not in populations:
            continue
        if row.get("historical_state") != "HISTORICAL_POSITIVE" or int(row.get("historical_target")) != 1:
            continue
        populations[str(class_name)].append(row)

    # Reduce each stratum to one deterministic representative per review group.
    candidates: dict[str, list[dict[str, Any]]] = {}
    population_report: dict[str, Any] = {}
    for class_name, class_index, native_label in TARGETS:
        source_rows = populations[class_name]
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        split_counts = Counter(str(row.get("historical_split")) for row in source_rows)
        for row in source_rows:
            by_group[canonical_group_key(row)].append(row)

        eligible_reps: list[dict[str, Any]] = []
        excluded_cross_split_groups = 0
        for group_key, group_rows in by_group.items():
            if group_splits.get(group_key, set()) != {"train"}:
                excluded_cross_split_groups += 1
                continue
            representative = min(
                group_rows,
                key=lambda row: rank_key(class_name, group_key, str(row["contract_id"]), ledger_sha),
            )
            candidate = dict(representative)
            candidate["_group_key"] = group_key
            candidate["_rank"] = rank_key(
                class_name, group_key, str(representative["contract_id"]), ledger_sha
            )
            eligible_reps.append(candidate)

        eligible_reps.sort(key=lambda row: (row["_rank"], str(row["contract_id"])))
        candidates[class_name] = eligible_reps
        population_report[class_name] = {
            "class_index": class_index,
            "source_native_label": native_label,
            "positive_contract_rows": len(source_rows),
            "unique_review_groups": len(by_group),
            "eligible_train_only_groups": len(eligible_reps),
            "excluded_groups_touching_val_or_test": excluded_cross_split_groups,
            "historical_split_counts": dict(sorted(split_counts.items())),
        }

    # Round-robin gives each stratum equal opportunity while enforcing a global
    # no-group-reuse rule across multi-label DIVE contracts.
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in TARGETS}
    pointers = {name: 0 for name, _, _ in TARGETS}
    used_groups: set[str] = set()

    for _round in range(per_stratum):
        for class_name, _, _ in TARGETS:
            pool = candidates[class_name]
            pointer = pointers[class_name]
            while pointer < len(pool) and pool[pointer]["_group_key"] in used_groups:
                pointer += 1
            if pointer >= len(pool):
                raise RuntimeError(
                    f"insufficient globally-disjoint TRAIN-only groups for {class_name}: "
                    f"needed {per_stratum}, selected {len(selected[class_name])}"
                )
            row = pool[pointer]
            pointers[class_name] = pointer + 1
            used_groups.add(row["_group_key"])
            selected[class_name].append(row)

    sample_rows: list[dict[str, Any]] = []
    for class_name, class_index, native_label in TARGETS:
        for ordinal, row in enumerate(selected[class_name], start=1):
            sample_rows.append(
                {
                    "sample_version": SAMPLE_VERSION,
                    "gap_id": GAP_ID,
                    "batch_id": "P4-WP2-INITIAL-BLIND",
                    "stratum_ordinal": ordinal,
                    "class_index": class_index,
                    "class_name": class_name,
                    "source_native_label": native_label,
                    "contract_id": str(row["contract_id"]),
                    "review_group_id": row["_group_key"],
                    "dedup_group_id": row.get("dedup_group_id"),
                    "project_group_id": row.get("project_group_id"),
                    "historical_split": row.get("historical_split"),
                    "representation_available": bool(row.get("representation_available")),
                    "selection_rank_sha256": row["_rank"],
                    "blind_fields_excluded": [
                        "model_probability",
                        "model_tier",
                        "tool_votes",
                        "downstream_merger_outcome",
                        "non_target_historical_labels",
                    ],
                }
            )

    if len(sample_rows) != per_stratum * len(TARGETS):
        raise AssertionError("unexpected sample size")
    if len({row["review_group_id"] for row in sample_rows}) != len(sample_rows):
        raise AssertionError("review group reuse detected across initial sample")
    if any(row["historical_split"] != "train" for row in sample_rows):
        raise AssertionError("non-train row entered the initial review sample")

    sample_payload = "\n".join(json.dumps(row, sort_keys=True) for row in sample_rows) + "\n"
    sample_sha = hashlib.sha256(sample_payload.encode("utf-8")).hexdigest()
    manifest = {
        "schema": "r4-gap-population-and-sample-manifest-v1",
        "gap_id": GAP_ID,
        "sample_version": SAMPLE_VERSION,
        "ledger_sha256": ledger_sha,
        "initial_per_stratum": per_stratum,
        "initial_total": len(sample_rows),
        "selection_policy": {
            "source": "dive",
            "historical_state": "HISTORICAL_POSITIVE",
            "eligible_split_policy": "TRAIN_ONLY_AND_GROUP_MUST_NOT_TOUCH_VAL_OR_TEST",
            "group_precedence": ["project_group_id", "dedup_group_id", "contract_id"],
            "cross_stratum_group_reuse": "FORBIDDEN",
            "ranking": "SHA256(sample_version|gap_id|ledger_sha|class_name|group_key|contract_id)",
            "adaptive_review": True,
            "note": "20 per stratum is an initial screening batch, not a fixed final sample size or authority threshold.",
        },
        "strata": population_report,
        "sample_sha256": sample_sha,
    }
    return manifest, sample_rows


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required to read the committed Phase-3 ledger") from exc
    return pq


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("docs/plan/ml-R4/ledger/evidence_ledger_v1.parquet"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("docs/plan/ml-R4/manifests/p4_gap002_population_manifest.json"),
    )
    parser.add_argument(
        "--output-sample",
        type=Path,
        default=Path("docs/plan/ml-R4/manifests/p4_gap002_initial_sample.jsonl"),
    )
    parser.add_argument("--per-stratum", type=int, default=INITIAL_PER_STRATUM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger_sha = sha256_file(args.ledger)
    if ledger_sha != EXPECTED_LEDGER_SHA256:
        raise RuntimeError(
            f"Phase-3 ledger hash mismatch: {ledger_sha} != {EXPECTED_LEDGER_SHA256}"
        )
    pq = require_pyarrow()
    table = pq.read_table(args.ledger)
    if table.num_rows != EXPECTED_ROWS:
        raise RuntimeError(f"ledger row count mismatch: {table.num_rows} != {EXPECTED_ROWS}")
    rows = table.to_pylist()
    manifest, sample_rows = build_population_and_sample(
        rows, ledger_sha=ledger_sha, per_stratum=args.per_stratum
    )
    write_json(args.output_manifest, manifest)
    write_jsonl(args.output_sample, sample_rows)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
