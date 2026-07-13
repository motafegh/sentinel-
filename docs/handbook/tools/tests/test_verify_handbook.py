from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "verify_handbook.py"
SPEC = importlib.util.spec_from_file_location("verify_handbook", SCRIPT)
assert SPEC and SPEC.loader
vh = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vh
SPEC.loader.exec_module(vh)


class SymbolTests(unittest.TestCase):
    def test_python_top_level_and_method_symbols(self) -> None:
        self.assertTrue(vh._symbol_exists("ml/src/models/sentinel_model.py::SentinelModel.forward")[0])
        self.assertTrue(vh._symbol_exists("agents/src/orchestration/graph.py::build_graph")[0])

    def test_solidity_contract_and_function_symbols(self) -> None:
        self.assertTrue(vh._symbol_exists("contracts/src/AuditRegistry.sol::AuditRegistry")[0])
        self.assertTrue(vh._symbol_exists("contracts/src/AuditRegistry.sol::submitAuditV2")[0])

    def test_missing_symbol_fails(self) -> None:
        passed, detail = vh._symbol_exists("contracts/src/AuditRegistry.sol::notARealMethod")
        self.assertFalse(passed)
        self.assertIn("missing symbol", detail)


class DocumentSafetyTests(unittest.TestCase):
    def test_missing_lab_section_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lab.md"
            path.write_text("## Learning objective\n", encoding="utf-8")
            self.assertEqual(vh._missing_sections(path, ["Learning objective", "Verification"]), ["Verification"])

    def test_broken_local_link_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(dir=vh.ROOT) as tmp:
            page = Path(tmp) / "page.md"
            page.write_text("[missing](nope.md)\n", encoding="utf-8")
            checks = vh._check_links([page])
            self.assertTrue(any(not check.passed and "missing" in check.detail for check in checks))

    def test_secret_shapes_are_detected_without_real_secret(self) -> None:
        fake = "operator_key=" + "0x" + ("a" * 64)
        self.assertIn("private-key assignment", vh._secret_leaks(fake))
        self.assertEqual(vh._secret_leaks("OPERATOR_KEY must be supplied at runtime"), [])

    def test_volatile_counts_are_rejected_outside_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "guide.md"
            page.write_text("123 passed\n", encoding="utf-8")
            self.assertEqual(vh._volatile_count_pages([page]), ["guide.md"])


class ArtifactTests(unittest.TestCase):
    def test_tracked_artifact_must_be_tracked_and_fresh(self) -> None:
        item = {"classification": "tracked", "tracked": False, "fresh_clone": True}
        self.assertFalse(vh._artifact_classification_ok(item))
        item["tracked"] = True
        self.assertTrue(vh._artifact_classification_ok(item))

    def test_local_artifact_cannot_claim_fresh_clone(self) -> None:
        item = {"classification": "ignored-local", "tracked": False, "fresh_clone": True}
        self.assertFalse(vh._artifact_classification_ok(item))


class RegistryTests(unittest.TestCase):
    def test_ten_guides_and_labs_registered(self) -> None:
        meta = vh._meta()
        self.assertEqual(len(meta["technical_guide"]), 10)
        self.assertEqual(len(meta["lab"]), 10)
        self.assertEqual({lab["guide"] for lab in meta["lab"]}, {guide["id"] for guide in meta["technical_guide"]})


if __name__ == "__main__":
    unittest.main()
