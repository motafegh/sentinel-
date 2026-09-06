#!/usr/bin/env python3
"""Validate the handbook's current R4 authority contract.

This validator is intentionally separate from ``verify_handbook.py``. The older
handbook metadata still records the historical G6/G7/runtime compatibility
baseline. This file verifies the later current DATA/ML authority chain without
rewriting that historical contract.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "docs" / "handbook" / "_meta" / "current_r4.json"
STATUS_MATRIX = ROOT / "docs" / "plan" / "ml-R4" / "PLAN_STATUS_MATRIX.md"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _relative(raw: str) -> Path:
    path = (ROOT / raw).resolve()
    path.relative_to(ROOT.resolve())
    return path


def _equal(checks: list[Check], name: str, actual: Any, expected: Any) -> None:
    checks.append(Check(name, actual == expected, f"actual={actual!r}, expected={expected!r}"))


def validate() -> list[Check]:
    contract = _json(CONTRACT_PATH)
    checks: list[Check] = []

    _equal(checks, "contract schema", contract.get("schema"), "sentinel-handbook-current-r4-v1")

    logical_cfg = contract["logical_v3"]
    logical = _json(_relative(logical_cfg["evidence_path"]))
    _equal(checks, "logical V3 status", logical.get("status"), logical_cfg["status"])
    _equal(checks, "logical V3 dataset", logical.get("versions", {}).get("dataset"), logical_cfg["dataset_version"])
    _equal(checks, "logical V3 grouping", logical.get("versions", {}).get("grouping"), logical_cfg["grouping_version"])
    _equal(checks, "logical V3 partition", logical.get("versions", {}).get("partition"), logical_cfg["partition_version"])
    _equal(checks, "logical V3 groups", logical.get("grouping_comparison", {}).get("v3", {}).get("groups"), logical_cfg["groups"])
    _equal(checks, "logical V3 max group", logical.get("grouping_comparison", {}).get("v3", {}).get("max"), logical_cfg["max_group_size"])
    _equal(checks, "logical V3 address edges", logical.get("grouping_comparison", {}).get("v3_address_edges"), logical_cfg["address_authority_edges"])
    _equal(checks, "logical V3 training authorization", logical.get("training_authorized"), logical_cfg["training_authorized"])
    _equal(checks, "logical V3 confirmed negatives", logical.get("checks", {}).get("confirmed_negative_rows_zero"), True)

    physical_cfg = contract["physical_v10_v26"]
    physical = _json(_relative(physical_cfg["evidence_path"]))
    lineage = physical.get("accepted_lineage", {})
    structural = physical.get("structural_evidence", {})
    _equal(checks, "D-011 decision id", physical.get("decision_id"), physical_cfg["decision_id"])
    _equal(checks, "D-011 status", physical.get("status"), physical_cfg["status"])
    _equal(checks, "D-011 decision", physical.get("decision"), physical_cfg["decision"])
    _equal(checks, "D-011 schema", lineage.get("graph_schema_version"), physical_cfg["graph_schema_version"])
    _equal(checks, "D-011 extractor", lineage.get("extractor_version"), physical_cfg["extractor_version"])
    _equal(checks, "D-011 contracts", lineage.get("contracts"), physical_cfg["contracts"])
    _equal(checks, "D-011 files", lineage.get("files"), physical_cfg["files"])
    _equal(checks, "D-011 digest", lineage.get("binding_digest_sha256"), physical_cfg["binding_digest_sha256"])
    _equal(checks, "D-011 unexplained drift", structural.get("unexplained_drift_identities"), physical_cfg["unexplained_structural_drift"])
    _equal(checks, "D-011 physical acceptance", physical.get("physical_acceptance"), True)
    _equal(checks, "D-011 training authorization", physical.get("training_authorized"), physical_cfg["training_authorized"])

    selector_cfg = contract["selector_successor"]
    selector_summary = _json(_relative(selector_cfg["evidence_path"]))
    selector_adr = _text(_relative(selector_cfg["adr_path"]))
    _equal(checks, "selector records analyzed", selector_summary.get("records_analyzed"), selector_cfg["records_analyzed"])
    _equal(checks, "selector over-cap records", selector_summary.get("over_four_window_records"), selector_cfg["over_four_window_records"])
    _equal(checks, "selector improved records", selector_summary.get("guarded_target_coverage_improved_records"), selector_cfg["improved_records"])
    _equal(checks, "selector fallback records", selector_summary.get("guarded_control_fallback_records"), selector_cfg["control_fallback_records"])
    _equal(checks, "selector regressions", selector_summary.get("guarded_target_coverage_regressed_records"), selector_cfg["regressed_records"])

    adr_requirements = [
        f"Decision ID: {selector_cfg['decision_id']}",
        f"Promote `{selector_cfg['strategy']}`",
        "new versioned token/representation candidate",
        "does not alter the R4-D-011 root or digest",
        "Physical acceptance, model-quality claims",
        "training remain unauthorized",
    ]
    missing_adr = [phrase for phrase in adr_requirements if phrase not in selector_adr]
    checks.append(Check("D-012 promotion boundary", not missing_adr, "ok" if not missing_adr else f"missing={missing_adr}"))

    matrix = _text(STATUS_MATRIX)
    phase_cfg = contract["phase8"]
    phase8_row = "| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS |"
    checks.append(Check("Phase 8 matrix state", phase8_row in matrix and phase_cfg["status"] == "IN_PROGRESS", "Phase 8 remains IN_PROGRESS"))
    checks.append(Check("full training hold", "Full training / G8 | HOLD" in matrix and phase_cfg["full_training_authorized"] is False, "G8/full training remains on hold"))
    checks.append(Check("confirmed negative boundary", "confirmed negatives remain zero" in matrix.lower() and phase_cfg["confirmed_negatives"] == 0, "confirmed negatives remain zero"))

    current_docs: dict[str, list[str]] = {
        "README.md": ["R4-D-011", "R4-D-012", "Run12", "full repaired training"],
        "docs/handbook/01_architecture.md": ["R4-D-011", "R4-D-012", "Run12"],
        "docs/handbook/03_data_pipeline.md": ["r4-leakage-groups-v3", "V10 V2.6", "R4-D-012"],
        "docs/handbook/04_data_artifacts.md": ["r4-vnext-roles-v3", "R4-D-011", "R4-D-012"],
        "docs/handbook/05_ml_model_inference.md": ["Run12", "R4-D-011", "R4-D-012"],
        "docs/handbook/06_ml_training_quality.md": ["R4-D-011", "R4-D-012", "full training"],
        "docs/handbook/11_cross_module_contracts.md": ["r4-vnext-roles-v3", "R4-D-011", "R4-D-012"],
        "docs/handbook/13_evaluation.md": ["positive-only", "confirmed negatives", "Run12"],
        "docs/handbook/16_current_status.md": ["R4-D-011", "R4-D-012", "d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd"],
        "docs/handbook/17_reference.md": ["r4-vnext-roles-v3", "V10 V2.6", "R4-D-012"],
    }
    for raw_path, phrases in current_docs.items():
        body = _text(_relative(raw_path))
        missing = [phrase for phrase in phrases if phrase not in body]
        checks.append(Check("current-R4 documentation", not missing, f"{raw_path}: " + ("ok" if not missing else f"missing={missing}")))

    no_training_claim = re.compile(r"(?:full|100-epoch).*training.*(?:authorized|launched|complete)", re.IGNORECASE)
    for raw_path in ("README.md", "docs/handbook/16_current_status.md"):
        body = _text(_relative(raw_path))
        unsafe_matches = [m.group(0) for m in no_training_claim.finditer(body) if "not authorized" not in m.group(0).lower()]
        checks.append(Check("no unsupported training claim", not unsafe_matches, f"{raw_path}: " + ("ok" if not unsafe_matches else f"matches={unsafe_matches[:3]}")))

    return checks


def main() -> int:
    checks = validate()
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    failures = [check for check in checks if not check.passed]
    print(f"\ncurrent R4: {len(checks) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
