#!/usr/bin/env python3
"""Read-only Phase-8 audit of the physical Solidity training corpus.

The script profiles the three active DATA vNext sources from their ingestion
manifests through preprocessed Solidity and the vNext ML projection.  It does
not write or repair any dataset artifact.  JSON is printed to stdout so a
reviewer can preserve or compare the evidence independently.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
ACTIVE_SOURCES = ("dive", "smartbugs_curated", "solidifi")
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
_DECLARATION_RE = re.compile(r"\b(?:contract|library|interface)\s+[A-Za-z_$]")
_IMPORT_RE = re.compile(r"^\s*import\b", re.MULTILINE)
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_WHITESPACE_RE = re.compile(r"\s+")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pipeline_text_sha(path: Path) -> str:
    """Match preprocessing's read_text(errors='replace') plus UTF-8 encode."""
    return _sha256_bytes(path.read_text(errors="replace").encode())


def _normalized_dedup_hash(text: str) -> str:
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return _sha256_bytes(text.encode())


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _category(source: str, original_path: str) -> str:
    parts = Path(original_path).parts
    if source == "smartbugs_curated":
        return parts[1]
    if source == "solidifi":
        return parts[2]
    return "__source__"


def _read_dropped(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _duplicate_level(
    *,
    raw_text: str,
    raw_sha: str,
    duplicate_of: str,
    canonical_text: str | None,
) -> str:
    if raw_sha == duplicate_of:
        return "exact"
    if canonical_text is None:
        return "canonical_missing"
    if _normalized_dedup_hash(raw_text) == _normalized_dedup_hash(canonical_text):
        return "normalized_text"
    if set(_ADDRESS_RE.findall(raw_text)) & set(_ADDRESS_RE.findall(canonical_text)):
        return "address_only"
    return "unexplained"


def profile_source(source: str) -> tuple[dict[str, Any], set[str]]:
    raw_dir = DATA_ROOT / "raw" / source
    preprocessed_dir = DATA_ROOT / "preprocessed" / source
    manifest_path = raw_dir / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    raw_records: list[dict[str, Any]] = []
    manifest_mismatches: list[str] = []
    for entry in manifest["files"]:
        path = raw_dir / entry["path"]
        data = path.read_bytes()
        byte_sha = _sha256_bytes(data)
        if len(data) != entry["size_bytes"] or byte_sha != entry["sha256"]:
            manifest_mismatches.append(entry["path"])
        text = path.read_text(errors="replace")
        raw_records.append(
            {
                "original_path": entry["path"],
                "path": path,
                "byte_sha": byte_sha,
                "pipeline_sha": _sha256_bytes(text.encode()),
                "text": text,
                "bytes": len(data),
                "replacement_characters": text.count("\ufffd"),
            }
        )

    pipeline_sha_counts = Counter(row["pipeline_sha"] for row in raw_records)
    raw_by_path = {row["original_path"]: row for row in raw_records}

    meta_rows: list[dict[str, Any]] = []
    malformed_meta: list[str] = []
    for path in sorted(preprocessed_dir.glob("*.meta.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            malformed_meta.append(path.name)
            continue
        row["_meta_path"] = path
        meta_rows.append(row)

    meta_by_original = {row["original_path"]: row for row in meta_rows}
    meta_by_sha = {row["sha256"]: row for row in meta_rows}
    metadata_failures: Counter[str] = Counter()
    output_sizes: list[int] = []
    raw_lines: list[int] = []
    normalized_lines: list[int] = []
    declarationless = 0
    import_outputs = 0
    zero_contract_names = 0

    # Import locally so the audit checks the current implementation's exact
    # normalization contract rather than maintaining a second copy here.
    sys.path.insert(0, str(REPO_ROOT / "data_module"))
    from sentinel_data.preprocessing.normalizer import normalize  # noqa: PLC0415

    for meta in meta_rows:
        meta_path = meta["_meta_path"]
        contract_id = meta["sha256"]
        output_path = preprocessed_dir / f"{contract_id}.sol"
        expected_stem = meta_path.name.removesuffix(".meta.json")
        if expected_stem != contract_id:
            metadata_failures["filename_sha_field_mismatch"] += 1
        if meta.get("source_name") != source:
            metadata_failures["source_name_mismatch"] += 1
        if meta.get("compile_status") != "ok":
            metadata_failures["compile_status_not_ok"] += 1
        if meta.get("meta_schema_version") != "1":
            metadata_failures["meta_schema_not_1"] += 1
        if not output_path.is_file():
            metadata_failures["preprocessed_sol_missing"] += 1
            continue
        raw = raw_by_path.get(meta["original_path"])
        if raw is None:
            metadata_failures["original_path_not_in_manifest"] += 1
            continue
        if meta.get("n_raw_lines") != raw["text"].count("\n") + 1:
            metadata_failures["n_raw_lines_mismatch"] += 1
        if meta.get("flatten_status") == "skipped_no_imports":
            if contract_id != raw["pipeline_sha"]:
                metadata_failures["pipeline_text_sha_mismatch"] += 1
            expected = normalize(raw["text"])
            if output_path.read_text(errors="replace") != expected.content:
                metadata_failures["normalized_content_mismatch"] += 1
            if meta.get("n_normalized_lines") != expected.n_lines_after:
                metadata_failures["n_normalized_lines_mismatch"] += 1

        output_text = output_path.read_text(errors="replace")
        output_sizes.append(output_path.stat().st_size)
        raw_lines.append(int(meta.get("n_raw_lines") or 0))
        normalized_lines.append(int(meta.get("n_normalized_lines") or 0))
        declarationless += not bool(_DECLARATION_RE.search(output_text))
        import_outputs += bool(_IMPORT_RE.search(output_text))
        zero_contract_names += not bool(meta.get("contract_names"))

    dropped = _read_dropped(preprocessed_dir / "dropped.csv")
    dropped_by_path = {row.get("original_path", ""): row for row in dropped}
    missing_paths = sorted(set(raw_by_path) - set(meta_by_original))
    missing_classification: Counter[str] = Counter()
    for original_path in missing_paths:
        if original_path in dropped_by_path:
            missing_classification[
                f"recorded_{dropped_by_path[original_path].get('reason', 'unknown')}"
            ] += 1
        elif raw_by_path[original_path]["pipeline_sha"] in meta_by_sha:
            missing_classification["silent_exact_content_collapse"] += 1
        else:
            missing_classification["unexplained"] += 1

    duplicate_levels: Counter[str] = Counter()
    for dropped_row in dropped:
        if dropped_row.get("reason") != "duplicate":
            continue
        original_path = dropped_row["original_path"]
        raw = raw_by_path[original_path]
        duplicate_of = dropped_row.get("duplicate_of", "")
        canonical_meta = meta_by_sha.get(duplicate_of)
        canonical_text = None
        if canonical_meta is not None:
            canonical_text = raw_by_path[canonical_meta["original_path"]]["text"]
        duplicate_levels[
            _duplicate_level(
                raw_text=raw["text"],
                raw_sha=raw["pipeline_sha"],
                duplicate_of=duplicate_of,
                canonical_text=canonical_text,
            )
        ] += 1

    raw_category_counts = Counter(
        _category(source, row["original_path"]) for row in raw_records
    )
    retained_category_counts = Counter(
        _category(source, row["original_path"]) for row in meta_rows
    )
    dropped_category_reason_counts = Counter(
        (_category(source, row["original_path"]), row.get("reason", "unknown"))
        for row in dropped
    )

    profile = {
        "manifest": {
            "path": manifest_path.relative_to(REPO_ROOT).as_posix(),
            "connector": manifest.get("connector"),
            "pin": manifest.get("pin"),
            "resolved_pin": manifest.get("resolved_pin"),
            "records": len(raw_records),
            "declared_contract_count": manifest.get("contract_count"),
            "byte_or_size_mismatches": len(manifest_mismatches),
        },
        "raw": {
            "unique_pipeline_text_sha": len(pipeline_sha_counts),
            "exact_duplicate_groups": sum(count > 1 for count in pipeline_sha_counts.values()),
            "exact_duplicate_extra_records": sum(
                count - 1 for count in pipeline_sha_counts.values() if count > 1
            ),
            "files_with_replacement_characters": sum(
                row["replacement_characters"] > 0 for row in raw_records
            ),
            "replacement_characters": sum(
                row["replacement_characters"] for row in raw_records
            ),
            "bytes": _quantiles([row["bytes"] for row in raw_records]),
            "category_counts": dict(sorted(raw_category_counts.items())),
        },
        "preprocessed": {
            "contracts": len(meta_rows),
            "sol_files": len(list(preprocessed_dir.glob("*.sol"))),
            "malformed_meta_json": len(malformed_meta),
            "metadata_or_content_failures": dict(sorted(metadata_failures.items())),
            "flatten_status": dict(sorted(Counter(row.get("flatten_status") for row in meta_rows).items())),
            "version_bucket": dict(sorted(Counter(row.get("version_bucket") for row in meta_rows).items())),
            "solc_version": dict(sorted(Counter(row.get("solc_version") for row in meta_rows).items())),
            "output_bytes": _quantiles(output_sizes),
            "raw_lines": _quantiles(raw_lines),
            "normalized_lines": _quantiles(normalized_lines),
            "declarationless_outputs": declarationless,
            "outputs_with_import_directive": import_outputs,
            "zero_contract_names_metadata": zero_contract_names,
            "category_counts": dict(sorted(retained_category_counts.items())),
        },
        "attrition": {
            "missing_from_preprocessed": len(missing_paths),
            "classification": dict(sorted(missing_classification.items())),
            "dropped_csv_rows": len(dropped),
            "dropped_reasons": dict(sorted(Counter(row.get("reason") for row in dropped).items())),
            "duplicate_levels": dict(sorted(duplicate_levels.items())),
            "category_reason_counts": {
                f"{category}|{reason}": count
                for (category, reason), count in sorted(dropped_category_reason_counts.items())
            },
        },
    }
    return profile, set(meta_by_sha)


def profile_dive_labels() -> dict[str, Any]:
    labels_path = DATA_ROOT / "raw_staging/dive_labels/DIVE_Labels.csv"
    with labels_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    classes = [name for name in rows[0] if name != "contractID"]
    by_id = {row["contractID"]: row for row in rows}
    source_dir = DATA_ROOT / "raw/dive/repo/__source__"
    source_paths = list(source_dir.glob("*.sol"))

    folder_checks: dict[str, Any] = {}
    for class_name in classes:
        entries = list((DATA_ROOT / "raw/dive/repo" / class_name).iterdir())
        expected_ids = {row["contractID"] for row in rows if row[class_name] == "1"}
        actual_ids = {entry.stem for entry in entries}
        folder_checks[class_name] = {
            "entries": len(entries),
            "symlinks": sum(entry.is_symlink() for entry in entries),
            "broken_symlinks": sum(entry.is_symlink() and not entry.exists() for entry in entries),
            "matches_csv_positive_ids": actual_ids == expected_ids,
        }

    groups: dict[str, list[str]] = defaultdict(list)
    for path in source_paths:
        groups[_pipeline_text_sha(path)].append(path.stem)
    duplicate_groups = {sha: ids for sha, ids in groups.items() if len(ids) > 1}
    conflict_by_class: Counter[str] = Counter()
    conflict_groups = 0
    for ids in duplicate_groups.values():
        differences = [
            class_name
            for class_name in classes
            if len({by_id[contract_id][class_name] for contract_id in ids}) > 1
        ]
        if differences:
            conflict_groups += 1
            conflict_by_class.update(differences)

    return {
        "labels_path": labels_path.relative_to(REPO_ROOT).as_posix(),
        "rows": len(rows),
        "unique_contract_ids": len(by_id),
        "contract_id_range": [min(map(int, by_id)), max(map(int, by_id))],
        "invalid_binary_cells": sum(
            row[class_name] not in {"0", "1"} for row in rows for class_name in classes
        ),
        "positive_counts": {
            class_name: sum(row[class_name] == "1" for row in rows)
            for class_name in classes
        },
        "source_file_ids_match_csv_ids": {path.stem for path in source_paths} == set(by_id),
        "folder_checks": folder_checks,
        "exact_content_duplicate_groups": len(duplicate_groups),
        "exact_content_duplicate_members": sum(len(ids) for ids in duplicate_groups.values()),
        "exact_content_duplicate_extra_records": sum(
            len(ids) - 1 for ids in duplicate_groups.values()
        ),
        "duplicate_groups_with_label_conflict": conflict_groups,
        "conflict_groups_by_class": dict(sorted(conflict_by_class.items())),
    }


def profile_vnext_supervision() -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("pyarrow is required to inspect the vNext projection") from exc

    path = DATA_ROOT / "exports/sentinel-r4-vnext-v1/ml_targets.parquet"
    rows = pq.read_table(path).to_pylist()
    result: dict[str, Any] = {}
    for source in ACTIVE_SOURCES:
        source_rows = [row for row in rows if row["source"] == source]
        by_class: dict[str, Any] = {}
        for index, class_name in enumerate(CLASS_NAMES):
            counts = {
                "targets": sum(row[f"target_{index}"] == 1 for row in source_rows),
                "effective_loss": sum(bool(row[f"effective_loss_mask_{index}"]) for row in source_rows),
                "outcome_metric": sum(bool(row[f"outcome_metric_mask_{index}"]) for row in source_rows),
            }
            if any(counts.values()):
                by_class[class_name] = counts
        result[source] = {
            "contracts": len(source_rows),
            "roles": dict(sorted(Counter(row["role"] for row in source_rows).items())),
            "class_cells": by_class,
        }
    return result


def profile_normalized_output_duplicates() -> dict[str, Any]:
    """Find identical stored normalized code that survived as distinct IDs."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("pyarrow is required to inspect vNext roles") from exc

    target_path = DATA_ROOT / "exports/sentinel-r4-vnext-v1/ml_targets.parquet"
    rows = pq.read_table(target_path).to_pylist()
    by_identity = {
        (str(row["source"]), str(row["contract_id"])): row for row in rows
    }
    by_normalized_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in ACTIVE_SOURCES:
        for path in sorted((DATA_ROOT / "preprocessed" / source).glob("*.sol")):
            contract_id = path.stem
            row = by_identity[(source, contract_id)]
            by_normalized_sha[_sha256_bytes(path.read_bytes())].append(row)

    duplicate_groups = [
        (digest, members)
        for digest, members in sorted(by_normalized_sha.items())
        if len(members) > 1
    ]
    cross_source = 0
    cross_role = 0
    multiple_group_ids = 0
    target_conflicts = 0
    source_group_counts: Counter[str] = Counter()
    cross_role_details: list[dict[str, Any]] = []
    for digest, members in duplicate_groups:
        sources = {str(row["source"]) for row in members}
        roles = {str(row["role"]) for row in members}
        group_ids = {str(row["group_id"]) for row in members}
        target_vectors = {
            tuple(row[f"target_{index}"] for index in range(len(CLASS_NAMES)))
            for row in members
        }
        cross_source += len(sources) > 1
        cross_role += len(roles) > 1
        multiple_group_ids += len(group_ids) > 1
        target_conflicts += len(target_vectors) > 1
        source_group_counts.update(sources)
        if len(roles) > 1:
            cross_role_details.append(
                {
                    "normalized_sha256": digest,
                    "members": [
                        {
                            "source": row["source"],
                            "contract_id": row["contract_id"],
                            "role": row["role"],
                            "group_id": row["group_id"],
                            "targets": [
                                row[f"target_{index}"]
                                for index in range(len(CLASS_NAMES))
                            ],
                        }
                        for row in members
                    ],
                }
            )

    return {
        "groups": len(duplicate_groups),
        "members": sum(len(members) for _, members in duplicate_groups),
        "extra_contract_records": sum(
            len(members) - 1 for _, members in duplicate_groups
        ),
        "groups_by_source_membership": dict(sorted(source_group_counts.items())),
        "cross_source_groups": cross_source,
        "cross_role_groups": cross_role,
        "groups_with_multiple_frozen_group_ids": multiple_group_ids,
        "groups_with_target_state_conflict": target_conflicts,
        "cross_role_details": cross_role_details,
    }


def main() -> int:
    source_profiles: dict[str, Any] = {}
    retained_ids: dict[str, set[str]] = {}
    for source in ACTIVE_SOURCES:
        source_profiles[source], retained_ids[source] = profile_source(source)

    exact_cross_source_overlap: dict[str, int] = {}
    for index, left in enumerate(ACTIVE_SOURCES):
        for right in ACTIVE_SOURCES[index + 1 :]:
            exact_cross_source_overlap[f"{left}|{right}"] = len(
                retained_ids[left] & retained_ids[right]
            )

    report = {
        "schema": "sentinel-r4-phase8-real-data-audit-v1",
        "scope": "active raw Solidity manifests through DATA vNext ML targets",
        "read_only": True,
        "source_profiles": source_profiles,
        "dive_labels_and_folderization": profile_dive_labels(),
        "exact_cross_source_contract_id_overlap": exact_cross_source_overlap,
        "normalized_output_duplicates": profile_normalized_output_duplicates(),
        "vnext_supervision": profile_vnext_supervision(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
