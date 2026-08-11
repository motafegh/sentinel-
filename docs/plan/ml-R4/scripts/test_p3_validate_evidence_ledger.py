#!/usr/bin/env python3
"""Deterministic self-tests for the Phase-3 evidence-ledger validator."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import p3_validate_evidence_ledger as validator

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_valid() -> tuple[list[dict], list[dict], dict]:
    rows = read_jsonl(FIXTURES / "p3_valid_ledger_fixture.jsonl")
    evidence = read_jsonl(FIXTURES / "p3_valid_evidence_fixture.jsonl")
    manifest = json.loads((FIXTURES / "p3_valid_manifest_fixture.json").read_text(encoding="utf-8"))
    return rows, evidence, manifest


def apply_case(case_id: str, rows: list[dict], evidence: list[dict], manifest: dict) -> None:
    if case_id == "duplicate_key":
        rows.append(copy.deepcopy(rows[0]))
        return

    if case_id == "class_order_mismatch":
        rows[0]["class_name"] = "Reentrancy"
        return

    if case_id == "confirmed_without_evidence":
        rows[0]["outcome_state"] = "CONFIRMED_NEGATIVE"
        rows[0]["supervised_loss_masked"] = False
        rows[0]["outcome_metrics_masked"] = False
        rows[0]["evidence_ids"] = []
        return

    if case_id == "unknown_not_masked":
        rows[0]["supervised_loss_masked"] = False
        return

    if case_id == "unresolved_evidence":
        rows[6]["evidence_ids"] = ["DOES-NOT-EXIST"]
        return

    if case_id == "evidence_scope_mismatch":
        evidence[0]["class_index"] = 5
        return

    if case_id == "acceptance_tool_only":
        rows[6]["role_eligibility"].append("UNTOUCHED_ACCEPTANCE")
        evidence[0]["tool_only"] = True
        return

    if case_id == "leakage_partition_crossing":
        # Two complete contracts share one leakage group but are assigned to
        # incompatible partitions. The second contract is intentionally left
        # UNKNOWN so no unrelated evidence-scope failure is required.
        for row in rows:
            row["partition"] = "TRAIN"
        second = copy.deepcopy(rows)
        second_id = "b" * 64
        for row in second:
            row["contract_id"] = second_id
            row["partition"] = "UNTOUCHED_ACCEPTANCE"
            row["outcome_state"] = "UNKNOWN"
            row["prior_review_state"] = "NONE"
            row["evidence_ids"] = []
            row["independence_groups"] = []
            row["supervised_loss_masked"] = True
            row["outcome_metrics_masked"] = True
            row["role_eligibility"] = ["TRAIN_UNLABELED", "EXCLUDE_OUTCOME_METRICS"]
        rows.extend(second)
        return

    if case_id == "invalid_export_hash":
        rows[0]["historical_export_sha256"] = "bad"
        return

    if case_id == "historical_zero_without_origin":
        rows[0]["zero_origin_categories"] = []
        return

    raise AssertionError(f"Unknown invalid fixture case: {case_id}")


class Phase3LedgerValidatorTests(unittest.TestCase):
    def test_valid_fixture_passes(self) -> None:
        rows, evidence, manifest = load_valid()
        report = validator.validate_ledger(rows, evidence, manifest)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["actual_contracts"], 1)
        self.assertEqual(report["actual_rows"], 10)
        self.assertEqual(report["unique_keys"], 10)

    def test_every_declared_invalid_case_fails_for_expected_reason(self) -> None:
        cases = read_jsonl(FIXTURES / "p3_invalid_ledger_cases.jsonl")
        self.assertGreaterEqual(len(cases), 10)

        for case in cases:
            with self.subTest(case_id=case["case_id"]):
                rows, evidence, manifest = load_valid()
                apply_case(case["case_id"], rows, evidence, manifest)
                report = validator.validate_ledger(
                    rows,
                    evidence,
                    manifest,
                    allow_partial_population=True,
                )
                self.assertFalse(report["passed"])
                rendered = "\n".join(report["errors"])
                self.assertIn(case["expected_error_fragment"], rendered, rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
