#!/usr/bin/env python3
"""Create a Git-safe snapshot of local logical-V3 Phase-8 evidence.

The V3 logical build and research reports remain under Git-ignored DATA roots.
This helper copies only decision-level JSON evidence into docs, sanitizes the
local repository path, reduces the large selector report to its top-level
summary, and binds every snapshot file by SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
SOURCE_ROOT = DATA_ROOT / "r4-v3-logical-build"
PUBLICATION_ROOT = DATA_ROOT / "exports/sentinel-r4-vnext-v3"
OUTPUT_ROOT = REPO_ROOT / "docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3"

SMALL_REPORTS = (
    "logical_v3_summary.json",
    "logical_v3_acceptance.json",
    "grouping_breadth_audit_v1.json",
    "representation_sensitivity_v1.json",
    "confirmed_negative_review_queue_v1.json",
    "selector_gpu_compare_v1.json",
)
PUBLICATION_REPORTS = {
    "logical_v3_manifest.json": "manifest.json",
    "logical_v3_partition_manifest.json": "partition_manifest.json",
    "logical_v3_representation_binding_report.json": "representation_binding_report.json",
}
LARGE_REPORT = "bounded_window_selector_v1.json"
LARGE_SUMMARY = "bounded_window_selector_v1.summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        root = str(REPO_ROOT)
        if value == root:
            return "<REPO_ROOT>"
        if value.startswith(root + "/"):
            return "<REPO_ROOT>/" + value[len(root) + 1 :]
    return value


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing logical-v3 evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"logical-v3 evidence is not a JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_sanitize(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"logical-v3 snapshot already contains files: {OUTPUT_ROOT}; "
            "do not overwrite durable evidence"
        )

    for name in SMALL_REPORTS:
        _write(OUTPUT_ROOT / name, _load(SOURCE_ROOT / name))

    for output_name, source_name in PUBLICATION_REPORTS.items():
        _write(OUTPUT_ROOT / output_name, _load(PUBLICATION_ROOT / source_name))

    large_path = SOURCE_ROOT / LARGE_REPORT
    large = _load(large_path)
    records = large.pop("records", [])
    large["source_report_sha256"] = _sha256(large_path)
    large["source_report_bytes"] = large_path.stat().st_size
    large["records_omitted_from_git_snapshot"] = len(records)
    large["snapshot_scope"] = (
        "Decision-level top-level summary only; per-contract records remain "
        "local and are bound by source_report_sha256."
    )
    _write(OUTPUT_ROOT / LARGE_SUMMARY, large)

    hashes = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(OUTPUT_ROOT.glob("*.json"))
    ]
    (OUTPUT_ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(hashes) + "\n", encoding="utf-8"
    )

    print(f"snapshot={OUTPUT_ROOT.relative_to(REPO_ROOT)}")
    for path in sorted(OUTPUT_ROOT.iterdir()):
        print(f"{path.name}\t{path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
