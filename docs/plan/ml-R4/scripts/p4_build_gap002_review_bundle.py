#!/usr/bin/env python3
"""Build a deterministic local blind-review bundle for approved R4-GAP-002.

The bundle contains only sampled preprocessed DIVE Solidity sources and minimal
review metadata. It deliberately excludes model outputs, tool votes, merger
outcomes, non-target historical labels, and raw dataset folders.

The resulting ZIP is intended for reviewer handoff (for example, upload into a
review session). It is not automatically committed to Git because the source
files are protected/local dataset material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_GAP_ID = "R4-GAP-002"
EXPECTED_SAMPLE_VERSION = "r4-gap-002-sample-v1"
EXPECTED_INITIAL_TOTAL = 100


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sample(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_INITIAL_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_INITIAL_TOTAL} sample rows, got {len(rows)}")
    if any(row.get("gap_id") != EXPECTED_GAP_ID for row in rows):
        raise RuntimeError("sample contains an unexpected gap_id")
    if any(row.get("sample_version") != EXPECTED_SAMPLE_VERSION for row in rows):
        raise RuntimeError("sample contains an unexpected sample_version")
    if len({row["contract_id"] for row in rows}) != len(rows):
        raise RuntimeError("sample contract_id reuse detected")
    if len({row["review_group_id"] for row in rows}) != len(rows):
        raise RuntimeError("sample review_group_id reuse detected")
    if any(row.get("historical_split") != "train" for row in rows):
        raise RuntimeError("non-train sample row detected")
    return rows


def safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def source_binding(
    row: dict[str, Any],
    *,
    preprocessed_dir: Path,
) -> tuple[bytes, dict[str, Any], str, str]:
    contract_id = str(row["contract_id"])
    sol_path = preprocessed_dir / f"{contract_id}.sol"
    meta_path = preprocessed_dir / f"{contract_id}.meta.json"
    if not sol_path.is_file():
        raise FileNotFoundError(f"sampled preprocessed source missing: {sol_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"sampled metadata missing: {meta_path}")

    source_bytes = sol_path.read_bytes()
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    if str(meta.get("sha256")) != contract_id:
        raise RuntimeError(f"metadata contract identity mismatch for {contract_id}")
    if str(meta.get("source_name")) != "dive":
        raise RuntimeError(f"metadata source mismatch for {contract_id}: {meta.get('source_name')!r}")

    return source_bytes, meta, sha256_bytes(source_bytes), sha256_bytes(meta_bytes)


def deterministic_zip_write(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def build_bundle(sample_path: Path, preprocessed_dir: Path, output_zip: Path) -> dict[str, Any]:
    rows = load_sample(sample_path)
    sample_sha = sha256_file(sample_path)
    tasks: list[dict[str, Any]] = []
    files: list[tuple[str, bytes]] = []

    for row in rows:
        source_bytes, meta, source_sha, meta_sha = source_binding(row, preprocessed_dir=preprocessed_dir)
        class_name = str(row["class_name"])
        ordinal = int(row["stratum_ordinal"])
        contract_id = str(row["contract_id"])
        source_rel = (
            f"sources/{int(row['class_index']):02d}_{safe_component(class_name)}/"
            f"{ordinal:02d}_{contract_id}.sol"
        )
        files.append((source_rel, source_bytes))
        tasks.append(
            {
                "gap_id": EXPECTED_GAP_ID,
                "sample_version": EXPECTED_SAMPLE_VERSION,
                "batch_id": row["batch_id"],
                "class_index": int(row["class_index"]),
                "class_name": class_name,
                "stratum_ordinal": ordinal,
                "contract_id": contract_id,
                "review_group_id": row["review_group_id"],
                "source_file": source_rel,
                "source_file_sha256": source_sha,
                "meta_file_sha256": meta_sha,
                "compiler_pragma": meta.get("pragma"),
                "solc_version": meta.get("solc_version"),
                "flatten_status": meta.get("flatten_status"),
                "version_bucket": meta.get("version_bucket"),
                "review_state": None,
                "review_rationale": None,
                "boundary_notes": None,
            }
        )

    counts = Counter(task["class_name"] for task in tasks)
    if sorted(counts.values()) != [20, 20, 20, 20, 20]:
        raise RuntimeError(f"unexpected stratum counts: {dict(counts)}")

    task_payload = "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks).encode("utf-8")
    manifest = {
        "schema": "r4-gap-review-bundle-v1",
        "gap_id": EXPECTED_GAP_ID,
        "sample_version": EXPECTED_SAMPLE_VERSION,
        "sample_file_sha256": sample_sha,
        "task_count": len(tasks),
        "class_counts": dict(sorted(counts.items())),
        "blind_exclusions": [
            "model predictions/probabilities/tiers",
            "Slither/Aderyn/tool votes",
            "merger outcome",
            "non-target historical labels",
        ],
        "source_material": "local preprocessed DIVE normalized/flattened Solidity",
        "note": "No semantic verdicts are pre-populated. Source bundle is reviewer handoff material and is not automatically committed.",
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        deterministic_zip_write(zf, "bundle_manifest.json", manifest_payload)
        deterministic_zip_write(zf, "review_tasks.jsonl", task_payload)
        for arcname, data in sorted(files):
            deterministic_zip_write(zf, arcname, data)

    report = dict(manifest)
    report["bundle_zip"] = str(output_zip)
    report["bundle_sha256"] = sha256_file(output_zip)
    report["bundle_size_bytes"] = output_zip.stat().st_size
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("docs/plan/ml-R4/manifests/p4_gap002_initial_sample.jsonl"),
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data_module/data/preprocessed/dive"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/r4_gap002_blind_review_bundle.zip"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_bundle(args.sample, args.preprocessed_dir, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
