#!/usr/bin/env python3
"""Rebuild only the corrected Phase-8 logical lineage V3.

This driver is intentionally cheaper and narrower than the repaired-v2 physical
rebuild.  It reuses the accepted repaired preprocessing, source claims,
role-independent evidence ledger, and all graph/token/sidecar files.  It writes
new V3 grouping, roles, publication, and binding artifacts without changing any
physical representation bytes and without launching training.

Recommended order:

    ... p8_rebuild_logical_v3.py prerequisites
    ... p8_rebuild_logical_v3.py grouping
    ... p8_rebuild_logical_v3.py publish
    ... p8_rebuild_logical_v3.py bind
    ... p8_rebuild_logical_v3.py audit
    ... p8_rebuild_logical_v3.py summarize
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

DATA_ROOT = REPO_ROOT / "data_module/data"
POLICY_PATH = REPO_ROOT / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json"
ACTIVE_SOURCES = ("dive", "smartbugs_curated", "solidifi")

PREPROCESSED_ROOT = DATA_ROOT / "sentinel-preprocessed-r4-v2"
REPRESENTATIONS_ROOT = DATA_ROOT / "representations-r4-v2"
V2_BUILD_ROOT = DATA_ROOT / "r4-v2-build"
V2_PUBLICATION_ROOT = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
V3_BUILD_ROOT = DATA_ROOT / "r4-v3-logical-build"
V3_PUBLICATION_ROOT = DATA_ROOT / "exports/sentinel-r4-vnext-v3"

V2_CLAIMS = V2_BUILD_ROOT / "source_claims.jsonl"
V2_LEDGER = V2_BUILD_ROOT / "evidence_ledger_v2.parquet"
V2_LEDGER_MANIFEST = V2_BUILD_ROOT / "evidence_ledger_v2_manifest.json"
V2_MANIFEST = V2_PUBLICATION_ROOT / "manifest.json"
V3_GROUPING = V3_BUILD_ROOT / "grouping.json"
V3_GROUPING_AUDIT = V3_BUILD_ROOT / "grouping_breadth_audit_v1.json"


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _tracked_clean() -> tuple[bool, str]:
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    return not bool(status), status


def _source_dirs() -> dict[str, Path]:
    return {source: PREPROCESSED_ROOT / source for source in ACTIVE_SOURCES}


def _require_fresh_file(path: Path, label: str) -> None:
    if path.exists():
        raise FileExistsError(
            f"{label} already exists: {path}. V3 rebuild outputs are immutable; "
            "archive/remove only the V3 attempt or choose a different path."
        )


def _require_fresh_dir(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{label} already contains files: {path}. V3 outputs are immutable."
        )


def cmd_prerequisites(_: argparse.Namespace) -> int:
    from sentinel_data.preprocessing.r4_completeness import (
        require_complete_preprocessed_sources,
        require_complete_representation_sources,
    )

    clean, detail = _tracked_clean()
    checks: list[dict[str, Any]] = [
        {"check": "tracked_worktree_clean", "passed": clean, "detail": detail},
        {"check": "git_head", "passed": True, "detail": _git_head()},
    ]
    required = (
        POLICY_PATH,
        V2_CLAIMS,
        V2_LEDGER,
        V2_LEDGER_MANIFEST,
        V2_MANIFEST,
    )
    for path in required:
        checks.append(
            {"check": f"required:{path.name}", "passed": path.is_file(), "detail": str(path)}
        )

    try:
        preprocessing = require_complete_preprocessed_sources(_source_dirs())
        require_complete_representation_sources(REPRESENTATIONS_ROOT, preprocessing)
        checks.append(
            {
                "check": "accepted_physical_v2_complete",
                "passed": True,
                "detail": {source: value["artifacts_written"] for source, value in preprocessing.items()},
            }
        )
    except Exception as exc:
        checks.append(
            {"check": "accepted_physical_v2_complete", "passed": False, "detail": str(exc)}
        )

    if V2_MANIFEST.is_file():
        parent = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        binding = parent.get("representation_binding_report") or {}
        checks.extend(
            [
                {
                    "check": "parent_dataset_version",
                    "passed": parent.get("dataset_version") == "sentinel-r4-vnext-v2",
                    "detail": parent.get("dataset_version"),
                },
                {
                    "check": "parent_binding_digest",
                    "passed": bool(binding.get("binding_digest_sha256")),
                    "detail": binding.get("binding_digest_sha256"),
                },
                {
                    "check": "parent_confirmed_negative_rows_zero",
                    "passed": parent.get("confirmed_negative_rows") == 0,
                    "detail": parent.get("confirmed_negative_rows"),
                },
            ]
        )

    passed = all(item["passed"] for item in checks)
    _emit({"passed": passed, "checks": checks})
    return 0 if passed else 2


def cmd_grouping(_: argparse.Namespace) -> int:
    from sentinel_data.preprocessing.r4_grouping_v3 import build_grouping_v3

    _require_fresh_file(V3_GROUPING, "logical-v3 grouping")
    result = build_grouping_v3(_source_dirs(), V3_GROUPING)
    if result.address_edges != 0:
        raise AssertionError("logical-v3 grouping created address-authority edges")
    _emit(result.__dict__)
    return 0


def cmd_publish(_: argparse.Namespace) -> int:
    from sentinel_data.vnext.r4_logical_v3 import build_logical_v3_publication

    if not V3_GROUPING.is_file():
        raise FileNotFoundError(V3_GROUPING)
    _require_fresh_dir(V3_PUBLICATION_ROOT, "logical-v3 publication")
    manifest = build_logical_v3_publication(
        claims_path=V2_CLAIMS,
        grouping_path=V3_GROUPING,
        policy_path=POLICY_PATH,
        representation_root=REPRESENTATIONS_ROOT,
        source_ledger_path=V2_LEDGER,
        source_ledger_manifest_path=V2_LEDGER_MANIFEST,
        source_v2_manifest_path=V2_MANIFEST,
        output_dir=V3_PUBLICATION_ROOT,
    )
    _emit(manifest)
    return 0


def cmd_bind(_: argparse.Namespace) -> int:
    from sentinel_data.vnext.r4_logical_v3 import bind_logical_v3_publication

    report = bind_logical_v3_publication(
        publication_dir=V3_PUBLICATION_ROOT,
        representations_root=REPRESENTATIONS_ROOT,
    )
    parent = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    parent_digest = (parent.get("representation_binding_report") or {}).get(
        "binding_digest_sha256"
    )
    same_physical_digest = report.get("binding_digest_sha256") == parent_digest
    if report.get("passed") is not True:
        _emit({"passed": False, "report": report})
        return 1
    if not same_physical_digest:
        raise AssertionError(
            "logical-v3 physical binding digest changed despite byte-reused representations"
        )
    _emit(
        {
            "passed": True,
            "binding_digest_sha256": report["binding_digest_sha256"],
            "matches_parent_v2_physical_binding": True,
            "required_contracts": report["required_contracts"],
            "checked_files": report["checked_files"],
        }
    )
    return 0


def cmd_audit(_: argparse.Namespace) -> int:
    from sentinel_data.preprocessing.r4_grouping_audit import audit_grouping_payload

    payload = json.loads(V3_GROUPING.read_text(encoding="utf-8"))
    report = audit_grouping_payload(payload)
    report["v3_policy_check"] = {
        "address_literal_grouping_authority": False,
        "address_edge_count": sum(
            1
            for edge in payload.get("evidence_edges") or []
            if edge.get("reason") == "same_source_shared_address_candidate"
        ),
        "passed": not any(
            edge.get("reason") == "same_source_shared_address_candidate"
            for edge in payload.get("evidence_edges") or []
        ),
    }
    V3_GROUPING_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    V3_GROUPING_AUDIT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _emit(report)
    return 0 if report["v3_policy_check"]["passed"] else 1


def cmd_summarize(_: argparse.Namespace) -> int:
    v2 = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    v3_manifest_path = V3_PUBLICATION_ROOT / "manifest.json"
    v3_partition_path = V3_PUBLICATION_ROOT / "partition_manifest.json"
    v3_binding_path = V3_PUBLICATION_ROOT / "representation_binding_report.json"
    for path in (V3_GROUPING, v3_manifest_path, v3_partition_path, v3_binding_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    grouping = json.loads(V3_GROUPING.read_text(encoding="utf-8"))
    v3 = json.loads(v3_manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(v3_partition_path.read_text(encoding="utf-8"))
    binding = json.loads(v3_binding_path.read_text(encoding="utf-8"))
    group_sizes = sorted(len(group["members"]) for group in grouping["groups"])
    summary = {
        "status": "LOGICAL_V3_REBUILD_COMPLETE_RESEARCH_REGENERATION_PENDING",
        "git_head": _git_head(),
        "physical_rebuild_performed": False,
        "training_authorized": False,
        "versions": {
            "dataset": v3["dataset_version"],
            "grouping": v3["grouping_version"],
            "partition": v3["partition_version"],
            "source_evidence_ledger": v3["ledger_version"],
        },
        "population": v3["population"],
        "grouping": {
            "groups": len(group_sizes),
            "max_group_members": max(group_sizes),
            "multi_member_groups": sum(value > 1 for value in group_sizes),
            "address_literal_grouping_authority": False,
            "evidence_edge_counts": dict(
                Counter(edge["reason"] for edge in grouping.get("evidence_edges") or [])
            ),
        },
        "partition": {
            "role_group_counts": partition["role_group_counts"],
            "role_contract_counts": partition["role_contract_counts"],
            "effective_loss_cells": v3["effective_loss_cells"],
            "outcome_metric_cells": v3["outcome_metric_cells"],
        },
        "physical_binding": {
            "passed": binding["passed"],
            "binding_digest_sha256": binding["binding_digest_sha256"],
            "matches_parent_v2": binding["binding_digest_sha256"]
            == (v2.get("representation_binding_report") or {}).get(
                "binding_digest_sha256"
            ),
            "checked_contracts": binding["checked_contracts"],
            "checked_files": binding["checked_files"],
        },
        "semantic_invariants": {
            "target_counts_unchanged_from_v2": v3["target_counts"] == v2["target_counts"],
            "training_strength_counts_unchanged_from_v2": v3[
                "training_strength_counts"
            ]
            == v2["training_strength_counts"],
            "confirmed_negative_rows": v3["confirmed_negative_rows"],
        },
        "next": [
            "regenerate V3 confirmed-negative pilot queue; V2 queue is obsolete",
            "regenerate selector coverage statistics against V3 roles",
            "regenerate representation sensitivity comparison sets against V3 roles",
            "rerun selector CUDA comparison with worst-case probes present",
            "review evidence before selector promotion or PU-objective design",
        ],
    }
    # Keep the summary local/generated; it is suitable for later sanitized snapshotting.
    out = V3_BUILD_ROOT / "logical_v3_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(summary)
    required_bools = (
        summary["physical_binding"]["passed"],
        summary["physical_binding"]["matches_parent_v2"],
        summary["semantic_invariants"]["target_counts_unchanged_from_v2"],
        summary["semantic_invariants"]["training_strength_counts_unchanged_from_v2"],
        summary["semantic_invariants"]["confirmed_negative_rows"] == 0,
    )
    return 0 if all(required_bools) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_subparsers(dest="command", required=True)
    for name in ("prerequisites", "grouping", "publish", "bind", "audit", "summarize"):
        parser.add_subparsers if False else None
    # argparse requires parsers to be registered on the single subparser action.
    sub = parser._subparsers._group_actions[0]
    for name in ("prerequisites", "grouping", "publish", "bind", "audit", "summarize"):
        sub.add_parser(name)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands = {
        "prerequisites": cmd_prerequisites,
        "grouping": cmd_grouping,
        "publish": cmd_publish,
        "bind": cmd_bind,
        "audit": cmd_audit,
        "summarize": cmd_summarize,
    }
    try:
        return commands[args.command](args)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"R4 LOGICAL V3 ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
