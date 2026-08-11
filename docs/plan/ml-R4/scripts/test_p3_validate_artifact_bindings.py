#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import p3_validate_artifact_bindings as bindings


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ledger.parquet").write_bytes(b"ledger")
        (self.root / "evidence.jsonl").write_bytes(b"{}\n")
        (self.root / "strict.json").write_text(
            json.dumps({"passed": True}) + "\n", encoding="utf-8"
        )
        self.manifest = {
            "status": "VALIDATED",
            "generation_commit": "a" * 40,
            "ledger_parquet": {"path": "ledger.parquet", "sha256": sha(b"ledger")},
            "evidence_jsonl": {"path": "evidence.jsonl", "sha256": sha(b"{}\n")},
            "validation_report": {
                "path": "strict.json",
                "sha256": sha((json.dumps({"passed": True}) + "\n").encode()),
            },
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_validated_manifest_binds_all_artifacts(self) -> None:
        report = bindings.validate_bindings(self.manifest, self.root)
        self.assertTrue(report["passed"], report["errors"])

    def test_hash_mismatch_fails(self) -> None:
        self.manifest["ledger_parquet"]["sha256"] = "0" * 64
        report = bindings.validate_bindings(self.manifest, self.root)
        self.assertFalse(report["passed"])
        self.assertTrue(any("SHA-256 mismatch" in e for e in report["errors"]))

    def test_missing_file_fails(self) -> None:
        (self.root / "ledger.parquet").unlink()
        report = bindings.validate_bindings(self.manifest, self.root)
        self.assertFalse(report["passed"])

    def test_path_escape_fails(self) -> None:
        self.manifest["ledger_parquet"]["path"] = "../outside.parquet"
        report = bindings.validate_bindings(self.manifest, self.root)
        self.assertFalse(report["passed"])
        self.assertTrue(any("escapes repository root" in e for e in report["errors"]))

    def test_absolute_path_fails(self) -> None:
        self.manifest["ledger_parquet"]["path"] = "/tmp/ledger.parquet"
        report = bindings.validate_bindings(self.manifest, self.root)
        self.assertFalse(report["passed"])

    def test_validated_manifest_requires_exact_commit(self) -> None:
        self.manifest["generation_commit"] = "r4/phase3-evidence-ledger"
        report = bindings.validate_bindings(self.manifest, self.root)
        self.assertFalse(report["passed"])
        self.assertTrue(any("40-hex Git commit" in e for e in report["errors"]))

    def test_validated_manifest_requires_passing_report(self) -> None:
        report_path = self.root / "strict.json"
        report_path.write_text(json.dumps({"passed": False}) + "\n", encoding="utf-8")
        self.manifest["validation_report"]["sha256"] = bindings.sha256_file(report_path)
        report = bindings.validate_bindings(self.manifest, self.root)
        self.assertFalse(report["passed"])
        self.assertTrue(any("did not pass" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
