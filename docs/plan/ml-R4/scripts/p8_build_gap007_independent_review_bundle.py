#!/usr/bin/env python3
"""Build a deterministic blind source bundle for one R4-GAP-007 verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_write(zf: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, payload)


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    matches = [
        row
        for row in (queue.get("candidates") or [])
        if row.get("candidate_id") == args.candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one queue candidate, got {len(matches)}")
    row = matches[0]
    if row.get("candidate_status") != "PENDING_REVIEW":
        raise RuntimeError("candidate is not PENDING_REVIEW")
    if row.get("current_target_value") is not None or row.get("negative_truth_claim"):
        raise RuntimeError("candidate already carries forbidden negative truth")

    source_path = (
        args.preprocessed_root / row["source"] / f"{row['contract_id']}.sol"
    )
    meta_path = source_path.with_suffix(".meta.json")
    source_bytes = source_path.read_bytes()
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes)
    if sha256_bytes(source_bytes) != row["contract_id"]:
        raise RuntimeError("source hash does not match queued contract identity")
    if meta.get("sha256") != row["contract_id"]:
        raise RuntimeError("metadata identity mismatch")

    source_name = f"source/{row['contract_id']}.sol"
    task = {
        "schema": "sentinel-r4-gap007-independent-review-task-v1",
        "candidate_id": row["candidate_id"],
        "contract_id": row["contract_id"],
        "group_id": row["group_id"],
        "class_index": row["class_index"],
        "class_name": row["class_name"],
        "negative_scope": "CLASS_SPECIFIC_ONLY",
        "source_file": source_name,
        "source_sha256": sha256_bytes(source_bytes),
        "compiler_pragma": meta.get("pragma"),
        "solc_version": meta.get("solc_version"),
        "allowed_status": ["AGREES", "DISAGREES", "INSUFFICIENT_EVIDENCE"],
        "review_requirements": [
            "review the complete source independently",
            "decide only whether CallToUnknown is absent",
            "do not infer negative truth from source labels, queue membership, or tool silence",
            "report other-class concerns separately",
            "provide a distinct reviewer identity, ISO-8601 time, rationale, and evidence",
        ],
    }
    manifest = {
        "schema": "sentinel-r4-gap007-independent-review-bundle-v1",
        "gap_id": "R4-GAP-007",
        "task_count": 1,
        "queue_sha256": sha256_file(args.queue),
        "source_file_sha256": sha256_bytes(source_bytes),
        "metadata_file_sha256": sha256_bytes(meta_bytes),
        "blind_exclusions": [
            "primary reviewer verdict and rationale",
            "model predictions and historical training labels",
            "static-tool verdicts",
            "graph and token-derived class signals",
        ],
        "note": "Queue membership is review reservation only and is not negative truth.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        deterministic_write(
            zf,
            "bundle_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        deterministic_write(
            zf,
            "review_task.json",
            (json.dumps(task, indent=2, sort_keys=True) + "\n").encode(),
        )
        deterministic_write(zf, source_name, source_bytes)

    bundle_hash = sha256_file(args.output)
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(f"{bundle_hash}  {args.output.name}\n", encoding="utf-8")
    return {
        **manifest,
        "bundle": str(args.output),
        "bundle_sha256": bundle_hash,
        "bundle_size_bytes": args.output.stat().st_size,
        "checksum": str(checksum_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/"
            "confirmed_negative_review_queue_v1.json"
        ),
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data_module/data/sentinel-preprocessed-r4-v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = build_bundle(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
