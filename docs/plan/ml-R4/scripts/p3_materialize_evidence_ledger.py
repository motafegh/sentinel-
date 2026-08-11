#!/usr/bin/env python3
"""Materialize the R4 Phase-3 contract×class evidence ledger.

This script deliberately does **not** invoke DVC. It reads the already-existing
protected v3 split/labels artifacts directly, verifies them against the frozen
Phase-0 hashes/counts, expands 22,493 contracts into 224,930 class rows, writes
Parquet, writes an updated manifest + validation report, and runs the Phase-3
semantic validator.

It is fail-closed: a wrong/missing split, wrong labels.parquet, population drift,
class-count drift, or representation-count drift aborts before promotion.

Example from repository root:

    python docs/plan/ml-R4/scripts/p3_materialize_evidence_ledger.py \
      --representations-root <LOCAL_REP_ROOT>

No historical DATA artifact is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import p3_validate_evidence_ledger as ledger_validator

CLASS_NAMES = [
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
]

EXPECTED_SPLIT_SHA256 = {
    "train": "03f2a2376f630165d89615ef47a796ea01a015375313208b556d921dd7d6409b",
    "val": "cf9a7b45fabbad2e3581282f69d5adf4fa4d09eb88bce3721544956a01b7506f",
    "test": "b9bb4649283cc7ec1d39b6e4cee980140b1752aea1c1df69e4b17a498d6fd20c",
}
EXPECTED_SPLIT_COUNTS = {"train": 18596, "val": 1983, "test": 1914}
EXPECTED_LABELS_SHA256 = "26e739b5d82ba512e5a1830817d09609216e2184b79cf4ca7ec2d62ef34e32b5"
EXPECTED_CONTRACTS = 22493
EXPECTED_ROWS = 224930
EXPECTED_REPRESENTED = 21657
EXPECTED_SOURCE_COUNTS = {
    "dive": 22073,
    "solidifi": 283,
    "smartbugs_curated": 137,
}
EXPECTED_CLASS_POSITIVES = {
    "CallToUnknown": 87,
    "DenialOfService": 1101,
    "ExternalBug": 16638,
    "GasException": 0,
    "IntegerUO": 9452,
    "MishandledException": 39,
    "Reentrancy": 11399,
    "Timestamp": 6324,
    "TransactionOrderDependence": 647,
    "UnusedReturn": 5859,
}
EXPORT_ARTIFACT_ID = "R4-P0-EXP-002"
SPLIT_ARTIFACT_IDS = {
    "train": "R4-P0-SPL-002",
    "val": "R4-P0-SPL-003",
    "test": "R4-P0-SPL-004",
}

DIVE_UNSUPPORTED = {0, 3, 5}
SOLIDIFI_SUPPORTED = {0, 2, 4, 5, 6, 7, 8}
SOLIDIFI_LOSSY = {0, 2}
DIVE_LOSSY_POSITIVE = {2, 9}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected object")
            rows.append(value)
    return rows


def require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Phase-3 production materialization requires pyarrow. Use the existing "
            "SENTINEL ML/data environment that already reads the protected Parquet; "
            "do not regenerate/substitute the historical data."
        ) from exc
    return pa, pq


def load_and_verify_splits(splits_dir: Path) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()

    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Protected split missing: {path}")
        actual_hash = sha256_file(path)
        expected_hash = EXPECTED_SPLIT_SHA256[split]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Protected split hash mismatch for {split}: {actual_hash} != {expected_hash}"
            )

        rows = read_jsonl(path)
        if len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError(
                f"Protected split count mismatch for {split}: {len(rows)} != "
                f"{EXPECTED_SPLIT_COUNTS[split]}"
            )

        for row in rows:
            sha = row.get("sha256")
            if not isinstance(sha, str) or not sha:
                raise ValueError(f"{path}: row missing sha256")
            if sha in seen:
                raise RuntimeError(f"contract_id appears in multiple protected split rows: {sha}")
            seen.add(sha)

            classes = row.get("classes")
            if not isinstance(classes, dict):
                raise ValueError(f"{path}: {sha}: classes must be object")
            unknown = sorted(set(classes) - set(CLASS_NAMES))
            if unknown:
                raise ValueError(f"{path}: {sha}: unknown class keys {unknown}")
            for name in CLASS_NAMES:
                value = int(classes.get(name, 0))
                if value not in (0, 1):
                    raise ValueError(f"{path}: {sha}: {name} must be 0/1, got {value}")

            enriched = dict(row)
            enriched["_split"] = split
            combined.append(enriched)

    if len(combined) != EXPECTED_CONTRACTS:
        raise RuntimeError(f"protected population mismatch: {len(combined)} != {EXPECTED_CONTRACTS}")
    return combined


def verify_frozen_counts(contracts: list[dict[str, Any]]) -> None:
    source_counts = Counter(str(c.get("source", "unknown")) for c in contracts)
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(
            f"source-count mismatch: actual={dict(source_counts)} expected={EXPECTED_SOURCE_COUNTS}"
        )

    positives = Counter()
    for contract in contracts:
        classes = contract.get("classes") or {}
        for name in CLASS_NAMES:
            positives[name] += int(classes.get(name, 0))
    if dict(positives) != EXPECTED_CLASS_POSITIVES:
        raise RuntimeError(
            f"class-positive mismatch: actual={dict(positives)} expected={EXPECTED_CLASS_POSITIVES}"
        )


def verify_labels_parquet(
    labels_path: Path,
    contracts: list[dict[str, Any]],
    pq: Any,
) -> None:
    if not labels_path.exists():
        raise FileNotFoundError(f"Protected labels.parquet missing: {labels_path}")
    actual_hash = sha256_file(labels_path)
    if actual_hash != EXPECTED_LABELS_SHA256:
        raise RuntimeError(
            f"protected labels.parquet hash mismatch: {actual_hash} != {EXPECTED_LABELS_SHA256}"
        )

    table = pq.read_table(labels_path)
    if table.num_rows != EXPECTED_CONTRACTS:
        raise RuntimeError(
            f"labels.parquet row count mismatch: {table.num_rows} != {EXPECTED_CONTRACTS}"
        )

    expected_by_id: dict[str, dict[str, Any]] = {str(c["sha256"]): c for c in contracts}
    seen: set[str] = set()
    required_cols = {"contract_id", "source", "split", "confidence_tier"} | {
        f"class_{i}" for i in range(10)
    }
    missing_cols = sorted(required_cols - set(table.column_names))
    if missing_cols:
        raise RuntimeError(f"labels.parquet missing required columns: {missing_cols}")

    for row in table.to_pylist():
        contract_id = str(row["contract_id"])
        if contract_id in seen:
            raise RuntimeError(f"duplicate contract_id in labels.parquet: {contract_id}")
        seen.add(contract_id)
        expected = expected_by_id.get(contract_id)
        if expected is None:
            raise RuntimeError(f"labels.parquet contains contract absent from protected splits: {contract_id}")
        if str(row["source"]) != str(expected.get("source", "unknown")):
            raise RuntimeError(f"source mismatch for {contract_id}")
        if str(row["split"]) != str(expected["_split"]):
            raise RuntimeError(f"split mismatch for {contract_id}")
        classes = expected.get("classes") or {}
        for idx, name in enumerate(CLASS_NAMES):
            if int(row[f"class_{idx}"]) != int(classes.get(name, 0)):
                raise RuntimeError(f"target mismatch for {contract_id}:{idx}:{name}")

    if seen != set(expected_by_id):
        raise RuntimeError("labels.parquet contract population differs from protected split population")


def representation_available(rep_root: Path, source: str, contract_id: str) -> bool:
    return (rep_root / source / f"{contract_id}.pt").exists()


def _base_zero_state(source: str, class_index: int, all_zero: bool) -> tuple[str, str, list[str], list[str], list[str]]:
    """Return source_native_state, crosswalk_action, zero_origins, evidence_ids, independence_groups."""
    origins: list[str] = []
    evidence: list[str] = []
    groups: list[str] = []

    if source == "dive":
        if class_index in DIVE_UNSUPPORTED:
            state = "UNSUPPORTED"
            crosswalk = "UNSUPPORTED"
            origins.append("CLASS_UNSUPPORTED")
            eid = {
                0: "P3-EVID-DIVE-UNSUPPORTED-CTU",
                3: "P3-EVID-DIVE-UNSUPPORTED-GAS",
                5: "P3-EVID-DIVE-UNSUPPORTED-MISH",
            }[class_index]
            evidence.append(eid)
            groups.append("phase2-dive-crosswalk-authority")
        else:
            state = "NOT_RECONSTRUCTED"
            crosswalk = "UNKNOWN"
            origins.append("UNRESOLVED_WITHIN_KNOWN_MECHANISMS")
        if all_zero:
            origins.append("SYNTHETIC_NONVULNERABLE")
        return state, crosswalk, origins, evidence, groups

    if source == "solidifi":
        if class_index not in SOLIDIFI_SUPPORTED:
            state = "UNSUPPORTED"
            crosswalk = "UNSUPPORTED"
            origins.append("CLASS_UNSUPPORTED")
        else:
            state = "ABSENT"
            crosswalk = "NONE"
            origins.append("SOURCE_ABSENCE")
        evidence.append("P3-EVID-TRANS-SOLIDIFI-NONTARGET")
        groups.append("phase2-solidifi-parser-semantics")
        return state, crosswalk, origins, evidence, groups

    if source == "smartbugs_curated":
        if all_zero:
            state = "MAPPED_NONVULNERABLE"
            crosswalk = "MAP_NONVULNERABLE"
            origins.extend(["MAPPED_NONVULNERABLE", "SYNTHETIC_NONVULNERABLE"])
        else:
            state = "ABSENT"
            crosswalk = "NONE"
            origins.append("SOURCE_ABSENCE")
        evidence.append("P3-EVID-TRANS-SMARTBUGS-NONVULNERABLE")
        groups.append("phase2-smartbugs-crosswalk-semantics")
        return state, crosswalk, origins, evidence, groups

    return (
        "NOT_RECONSTRUCTED",
        "UNKNOWN",
        ["UNRESOLVED_WITHIN_KNOWN_MECHANISMS"],
        [],
        [],
    )


def build_ledger_rows(
    contracts: list[dict[str, Any]],
    rep_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    represented = 0

    for contract in contracts:
        contract_id = str(contract["sha256"])
        source = str(contract.get("source", "unknown"))
        split = str(contract["_split"])
        classes = contract.get("classes") or {}
        n_pos = sum(int(classes.get(name, 0)) for name in CLASS_NAMES)
        all_zero = n_pos == 0
        rep_ok = representation_available(rep_root, source, contract_id)
        represented += int(rep_ok)
        tier = contract.get("tier") if n_pos > 0 else None
        dedup_group = contract.get("dedup_group")
        project_id = contract.get("project_id")

        for class_index, class_name in enumerate(CLASS_NAMES):
            target = int(classes.get(class_name, 0))
            if target == 1:
                source_native_state = "EXPLICIT_POSITIVE"
                if source == "solidifi" and class_index in SOLIDIFI_LOSSY:
                    crosswalk_action = "LOSSY_MAP"
                elif source == "dive" and class_index in DIVE_LOSSY_POSITIVE:
                    crosswalk_action = "LOSSY_MAP"
                elif source == "smartbugs_curated":
                    # The protected split does not retain the original SmartBugs
                    # category, so direct vs bad_randomness→Timestamp cannot be
                    # distinguished safely per row.
                    crosswalk_action = "UNKNOWN"
                else:
                    crosswalk_action = "DIRECT"
                zero_origins = ["NONE"]
                evidence_ids: list[str] = []
                independence_groups: list[str] = []
                outcome_state = "NOT_REVIEWED"
            else:
                (
                    source_native_state,
                    crosswalk_action,
                    zero_origins,
                    evidence_ids,
                    independence_groups,
                ) = _base_zero_state(source, class_index, all_zero)
                outcome_state = "UNKNOWN"

            limitations = [
                "Phase-3 initialization preserves historical target but does not infer a confirmed outcome without direct evidence."
            ]
            if target == 0 and "UNRESOLVED_WITHIN_KNOWN_MECHANISMS" in zero_origins:
                limitations.append(
                    "Exact per-row zero origin is unresolved; Phase 2 bounds the allowed mechanism category set."
                )

            row = {
                "ledger_version": "r4-ledger-v1",
                "contract_id": contract_id,
                "class_index": class_index,
                "class_name": class_name,
                "primary_source": source,
                "source_record_id": contract_id,
                "source_tier": tier,
                "dedup_group_id": str(dedup_group) if dedup_group is not None else None,
                "project_group_id": str(project_id) if project_id is not None else None,
                "leakage_group_id": None,
                "historical_state": "HISTORICAL_POSITIVE" if target else "HISTORICAL_ZERO",
                "historical_target": target,
                "historical_export_artifact_id": EXPORT_ARTIFACT_ID,
                "historical_export_sha256": EXPECTED_LABELS_SHA256,
                "historical_split": split,
                "representation_available": rep_ok,
                "source_native_state": source_native_state,
                "source_native_label": None,
                "parser_id": f"{source}.py" if source in {"dive", "solidifi", "smartbugs_curated"} else None,
                "crosswalk_id": {
                    "dive": "R4-P0-XWK-001",
                    "solidifi": "R4-P0-XWK-002",
                    "smartbugs_curated": "R4-P0-XWK-003",
                }.get(source),
                "crosswalk_action": crosswalk_action,
                "merger_action": "SINGLE_SOURCE",
                "verification_action": "GATE_ONLY",
                "zero_origin_categories": zero_origins,
                "phase2_trace_ids": [],
                "evidence_ids": evidence_ids,
                "independence_groups": independence_groups,
                "prior_review_state": "NONE",
                "outcome_state": outcome_state,
                "limitations": limitations,
                "supervised_loss_masked": True,
                "outcome_metrics_masked": True,
                "role_eligibility": ["TRAIN_UNLABELED", "EXCLUDE_OUTCOME_METRICS"],
                "partition": "UNASSIGNED",
                "artifact_ids": [EXPORT_ARTIFACT_ID, SPLIT_ARTIFACT_IDS[split]],
                "provenance_kind": "TRANSFORMATION_RECONSTRUCTION",
            }
            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"ledger row count mismatch: {len(rows)} != {EXPECTED_ROWS}")
    return rows, represented


def write_parquet(rows: list[dict[str, Any]], path: Path, pa: Any, pq: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data_module/data/splits/v3"))
    parser.add_argument(
        "--labels-parquet",
        type=Path,
        default=Path("data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/labels.parquet"),
    )
    parser.add_argument(
        "--representations-root",
        type=Path,
        required=True,
        help="Directory used by graph_writer: <root>/<source>/<sha256>.pt",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("docs/plan/ml-R4/manifests/evidence_items_v1.jsonl"),
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("docs/plan/ml-R4/ledger/evidence_ledger_v1.parquet"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("docs/plan/ml-R4/manifests/evidence_ledger_v1.materialized.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("docs/plan/ml-R4/findings/04_evidence_ledger_validation_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pa, pq = require_pyarrow()
        contracts = load_and_verify_splits(args.splits_dir)
        verify_frozen_counts(contracts)
        verify_labels_parquet(args.labels_parquet, contracts, pq)

        if not args.representations_root.exists():
            raise FileNotFoundError(
                f"representations root does not exist: {args.representations_root}"
            )

        rows, represented = build_ledger_rows(contracts, args.representations_root)
        if represented != EXPECTED_REPRESENTED:
            raise RuntimeError(
                f"representation-count mismatch: {represented} != {EXPECTED_REPRESENTED}. "
                "The --representations-root is wrong or the protected representation population changed."
            )

        write_parquet(rows, args.output_parquet, pa, pq)
        ledger_sha = sha256_file(args.output_parquet)
        evidence_sha = sha256_file(args.evidence)

        source_row_counts = Counter(row["primary_source"] for row in rows)
        historical_counts = Counter(row["historical_state"] for row in rows)
        outcome_counts = Counter(row["outcome_state"] for row in rows)
        role_counts: Counter[str] = Counter()
        for row in rows:
            role_counts.update(row["role_eligibility"])

        manifest = {
            "ledger_version": "r4-ledger-v1",
            "schema_version": "1",
            "created_at": None,
            "source_export_artifact_id": EXPORT_ARTIFACT_ID,
            "source_export_sha256": EXPECTED_LABELS_SHA256,
            "expected_contracts": EXPECTED_CONTRACTS,
            "expected_classes": 10,
            "expected_rows": EXPECTED_ROWS,
            "actual_contracts": EXPECTED_CONTRACTS,
            "actual_rows": len(rows),
            "ledger_parquet": {"path": str(args.output_parquet), "sha256": ledger_sha},
            "evidence_jsonl": {"path": str(args.evidence), "sha256": evidence_sha},
            "validation_report": {"path": str(args.output_report), "sha256": None},
            "class_names": CLASS_NAMES,
            "source_counts": dict(sorted(source_row_counts.items())),
            "outcome_state_counts": dict(sorted(outcome_counts.items())),
            "historical_state_counts": dict(sorted(historical_counts.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "generation_commit": "r4/phase3-evidence-ledger",
            "status": "MATERIALIZED",
            "limitations": [
                "Initial Phase-3 ledger is deliberately conservative: historical positives are not automatically confirmed positives, and historical zeros are not automatically confirmed negatives.",
                "Exact DIVE per-row zero origin remains unresolved where protected split rows do not retain source-native zero/unknown distinction.",
                "leakage_group_id remains null in Phase 3; dedup_group_id and project_group_id are preserved for later role/partition construction."
            ],
        }
        write_json(args.output_manifest, manifest)

        evidence_items = read_jsonl(args.evidence)
        report = ledger_validator.validate_ledger(rows, evidence_items, manifest)
        report["protected_split_hashes_verified"] = True
        report["protected_labels_hash_verified"] = True
        report["protected_source_counts_verified"] = True
        report["protected_class_counts_verified"] = True
        report["represented_contracts"] = represented
        write_json(args.output_report, report)

        # Bind final report identity and rewrite the materialized manifest.
        manifest["validation_report"]["sha256"] = sha256_file(args.output_report)
        manifest["status"] = "VALIDATED" if report["passed"] else "FAILED"
        write_json(args.output_manifest, manifest)

        print(json.dumps({
            "passed": report["passed"],
            "contracts": EXPECTED_CONTRACTS,
            "rows": len(rows),
            "represented_contracts": represented,
            "ledger_sha256": ledger_sha,
            "manifest": str(args.output_manifest),
            "report": str(args.output_report),
        }, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"MATERIALIZATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
