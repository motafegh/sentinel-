#!/usr/bin/env python3
"""Verify the committed hardened logical-V3 snapshot from a fresh clone.

Unlike snapshot generation, this verifier requires no ignored/local DATA roots.
It validates the committed SHA256SUMS manifest, re-runs semantic coherence over
the committed JSON reports using the strengthened queue contract, and confirms
that the historical build-stage summary is explicitly contextualized by the
snapshot index/addendum.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT_ROOT = REPO_ROOT / "docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3"
SNAPSHOT_HELPER = REPO_ROOT / "docs/plan/ml-R4/scripts/p8_snapshot_logical_v3_evidence.py"
INDEX_PATH = SNAPSHOT_ROOT / "SNAPSHOT_INDEX.md"
CHECKSUM_PATH = SNAPSHOT_ROOT / "SHA256SUMS.txt"

spec = importlib.util.spec_from_file_location(
    "p8_snapshot_logical_v3_evidence",
    SNAPSHOT_HELPER,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import snapshot helper: {SNAPSHOT_HELPER}")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_checksums() -> dict[str, str]:
    if not CHECKSUM_PATH.is_file():
        raise FileNotFoundError(CHECKSUM_PATH)
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        CHECKSUM_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"invalid SHA256SUMS line {line_number}: {raw!r}")
        digest, filename = parts
        filename = filename.lstrip("*").strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid SHA-256 on line {line_number}")
        if not filename or "/" in filename or filename in result:
            raise ValueError(f"invalid/duplicate snapshot filename on line {line_number}")
        result[filename] = digest
    return result


def main() -> int:
    expected_json = {
        *snapshot.SMALL_REPORTS,
        *snapshot.PUBLICATION_REPORTS.keys(),
        snapshot.LARGE_SUMMARY,
        snapshot.COHERENCE_REPORT,
    }
    listed = _parse_checksums()
    actual_json = {path.name for path in SNAPSHOT_ROOT.glob("*.json")}

    if set(listed) != expected_json:
        raise ValueError(
            "SHA256SUMS JSON inventory mismatch: "
            f"listed_only={sorted(set(listed) - expected_json)} "
            f"missing={sorted(expected_json - set(listed))}"
        )
    if actual_json != expected_json:
        raise ValueError(
            "committed snapshot JSON inventory mismatch: "
            f"extra={sorted(actual_json - expected_json)} "
            f"missing={sorted(expected_json - actual_json)}"
        )

    for filename, expected_digest in sorted(listed.items()):
        actual_digest = snapshot._sha256(SNAPSHOT_ROOT / filename)
        if actual_digest != expected_digest:
            raise ValueError(
                f"snapshot checksum mismatch for {filename}: "
                f"expected={expected_digest} actual={actual_digest}"
            )

    summary = _load(SNAPSHOT_ROOT / "logical_v3_summary.json")
    acceptance = _load(SNAPSHOT_ROOT / "logical_v3_acceptance.json")
    grouping_audit = _load(SNAPSHOT_ROOT / "grouping_breadth_audit_v1.json")
    sensitivity_path = SNAPSHOT_ROOT / "representation_sensitivity_v1.json"
    sensitivity = _load(sensitivity_path)
    queue = _load(SNAPSHOT_ROOT / "confirmed_negative_review_queue_v1.json")
    selector = _load(SNAPSHOT_ROOT / snapshot.LARGE_SUMMARY)
    gpu = _load(SNAPSHOT_ROOT / "selector_gpu_compare_v1.json")
    manifest_path = SNAPSHOT_ROOT / "logical_v3_manifest.json"
    manifest = _load(manifest_path)
    partition = _load(SNAPSHOT_ROOT / "logical_v3_partition_manifest.json")
    binding_path = SNAPSHOT_ROOT / "logical_v3_representation_binding_report.json"
    binding = _load(binding_path)
    committed_coherence = _load(SNAPSHOT_ROOT / snapshot.COHERENCE_REPORT)

    if committed_coherence.get("status") != "PASS":
        raise ValueError("committed snapshot_coherence_v1.json is not PASS")
    if committed_coherence.get("failures") != []:
        raise ValueError("committed snapshot coherence records failures")
    committed_checks = committed_coherence.get("checks")
    if not isinstance(committed_checks, dict) or not committed_checks:
        raise ValueError("committed snapshot coherence checks are missing")
    if not all(value is True for value in committed_checks.values()):
        raise ValueError("committed snapshot coherence contains a failed check")

    committed_lineage = committed_coherence.get("lineage") or {}
    source_commit = str(committed_lineage.get("source_commit") or "")
    selector_source_sha = str(selector.get("source_report_sha256") or "")
    if not selector_source_sha:
        raise ValueError("bounded selector summary lacks source_report_sha256")

    recomputed = snapshot.validate_snapshot_coherence(
        manifest=manifest,
        manifest_sha256=snapshot._sha256(manifest_path),
        partition=partition,
        binding=binding,
        binding_sha256=snapshot._sha256(binding_path),
        summary=summary,
        acceptance=acceptance,
        grouping_audit=grouping_audit,
        sensitivity=sensitivity,
        sensitivity_sha256=snapshot._sha256(sensitivity_path),
        queue=queue,
        selector=selector,
        selector_sha256=selector_source_sha,
        gpu=gpu,
        current_source_commit=source_commit,
    )

    if recomputed.get("status") != "PASS" or recomputed.get("failures"):
        raise ValueError("strengthened committed-snapshot coherence did not pass")

    for field in (
        "dataset_version",
        "grouping_version",
        "partition_version",
        "publication_manifest_sha256",
        "representation_binding_digest_sha256",
        "representation_binding_report_sha256",
        "representation_sensitivity_sha256",
        "bounded_selector_source_report_sha256",
        "source_commit",
        "report_source_commits",
    ):
        if recomputed["lineage"].get(field) != committed_lineage.get(field):
            raise ValueError(f"committed snapshot lineage mismatch for {field}")

    index_text = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.is_file() else ""
    required_index_markers = (
        "HISTORICAL_BUILD_STAGE_SUMMARY",
        "logical_v3_summary.json",
        "RESEARCH_REGENERATION_PENDING",
        "44fbb9c1d2033be8002fe404d650cf09f08b0f29",
    )
    missing_markers = [marker for marker in required_index_markers if marker not in index_text]
    if missing_markers:
        raise ValueError(f"snapshot index/addendum missing markers: {missing_markers}")

    if summary.get("status") != "LOGICAL_V3_REBUILD_COMPLETE_RESEARCH_REGENERATION_PENDING":
        raise ValueError("historical logical_v3_summary status changed unexpectedly")

    queue_checks = snapshot.validate_queue_coherence(
        queue,
        manifest_sha256=snapshot._sha256(manifest_path),
    )
    if not all(queue_checks.values()):
        failed = sorted(name for name, passed in queue_checks.items() if not passed)
        raise ValueError("committed queue semantic validation failed: " + ", ".join(failed))

    outcomes: dict[str, int] = {}
    for row in queue["candidates"]:
        state = str(row["current_outcome_state"])
        outcomes[state] = outcomes.get(state, 0) + 1

    print("committed_snapshot=PASS")
    print(f"json_checksums_verified={len(listed)}")
    print(f"coherence_checks_recomputed={len(recomputed['checks'])}")
    print(f"queue_cells={queue['queued_cells']}")
    print(f"queue_groups={len(queue['reserved_group_ids'])}")
    print(f"queue_outcomes={json.dumps(outcomes, sort_keys=True)}")
    print(f"evidence_source_commit={source_commit}")
    print("training_authorized=false")
    print("selector_promotion_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
