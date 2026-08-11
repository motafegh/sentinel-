#!/usr/bin/env python3
"""Dataset-independent tests for the strict Phase-3 ledger validator."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
FIXTURES = ROOT / "docs" / "plan" / "ml-R4" / "fixtures"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p3_validate_evidence_ledger as semantic
import p3_validate_evidence_ledger_strict as strict


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class StrictLedgerValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(FIXTURES / "p3_valid_ledger_fixture.jsonl")
        cls.evidence = load_jsonl(FIXTURES / "p3_valid_evidence_fixture.jsonl")
        cls.manifest = json.loads((FIXTURES / "p3_valid_manifest_fixture.json").read_text(encoding="utf-8"))

    def report(self, rows=None, evidence=None, manifest=None):
        return strict.validate_strict(
            copy.deepcopy(self.rows if rows is None else rows),
            copy.deepcopy(self.evidence if evidence is None else evidence),
            copy.deepcopy(self.manifest if manifest is None else manifest),
        )

    def assert_surface_failure(self, mutate):
        rows = copy.deepcopy(self.rows)
        evidence = copy.deepcopy(self.evidence)
        manifest = copy.deepcopy(self.manifest)
        mutate(rows, evidence, manifest)
        report = strict.validate_strict(rows, evidence, manifest)
        self.assertFalse(report["passed"])
        self.assertTrue(report["surface_errors"], report)

    def test_valid_fixture_passes_strict_validation(self):
        report = self.report()
        self.assertTrue(report["passed"], report)
        self.assertEqual([], report["surface_errors"])
        self.assertEqual([], report["semantic_errors"])

    def test_rejects_unknown_row_property(self):
        self.assert_surface_failure(lambda rows, _e, _m: rows[0].__setitem__("mystery", 1))

    def test_rejects_invalid_source_native_enum(self):
        self.assert_surface_failure(
            lambda rows, _e, _m: rows[0].__setitem__("source_native_state", "SAFE_BY_MAGIC")
        )

    def test_rejects_non_boolean_representation_available(self):
        self.assert_surface_failure(
            lambda rows, _e, _m: rows[0].__setitem__("representation_available", 1)
        )

    def test_rejects_duplicate_role_values(self):
        def mutate(rows, _e, _m):
            rows[0]["role_eligibility"] = ["TRAIN_WEAK", "TRAIN_WEAK"]
        self.assert_surface_failure(mutate)

    def test_rejects_invalid_zero_origin_enum(self):
        def mutate(rows, _e, _m):
            zero_row = next(r for r in rows if r["historical_state"] == "HISTORICAL_ZERO")
            zero_row["zero_origin_categories"] = ["MAGIC_ZERO"]
        self.assert_surface_failure(mutate)

    def test_rejects_unknown_evidence_property(self):
        self.assert_surface_failure(lambda _r, evidence, _m: evidence[0].__setitem__("extra", True))

    def test_rejects_invalid_evidence_polarity(self):
        self.assert_surface_failure(
            lambda _r, evidence, _m: evidence[0].__setitem__("polarity", "MAYBE")
        )

    def test_rejects_non_boolean_tool_only(self):
        self.assert_surface_failure(
            lambda _r, evidence, _m: evidence[0].__setitem__("tool_only", 0)
        )

    def test_rejects_manifest_extra_property(self):
        self.assert_surface_failure(lambda _r, _e, manifest: manifest.__setitem__("surprise", "x"))

    def test_rejects_manifest_invalid_artifact_hash(self):
        def mutate(_r, _e, manifest):
            manifest["ledger_parquet"]["sha256"] = "not-a-sha"
        self.assert_surface_failure(mutate)

    def test_rejects_manifest_status_enum(self):
        self.assert_surface_failure(
            lambda _r, _e, manifest: manifest.__setitem__("status", "DONE")
        )

    def test_semantic_failure_is_preserved(self):
        rows = copy.deepcopy(self.rows)
        confirmed = next(r for r in rows if r["outcome_state"] == "CONFIRMED_POSITIVE")
        confirmed["evidence_ids"] = []
        confirmed["independence_groups"] = []
        report = strict.validate_strict(rows, copy.deepcopy(self.evidence), copy.deepcopy(self.manifest))
        self.assertFalse(report["passed"])
        self.assertTrue(report["semantic_errors"], report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
