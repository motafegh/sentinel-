from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "verify_current_r4.py"
SPEC = importlib.util.spec_from_file_location("verify_current_r4", SCRIPT)
assert SPEC and SPEC.loader
vr4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vr4
SPEC.loader.exec_module(vr4)


class CurrentR4AuthorityTests(unittest.TestCase):
    def test_current_r4_contract_and_evidence_validate(self) -> None:
        checks = vr4.validate()
        failures = [check for check in checks if not check.passed]
        self.assertEqual(failures, [], "\n".join(f"{c.name}: {c.detail}" for c in failures))

    def test_validator_checks_later_authority_not_only_g7(self) -> None:
        names = {check.name for check in vr4.validate()}
        self.assertIn("D-011 digest", names)
        self.assertIn("D-012 promotion boundary", names)
        self.assertIn("logical V3 groups", names)
        self.assertIn("full training hold", names)


if __name__ == "__main__":
    unittest.main()
