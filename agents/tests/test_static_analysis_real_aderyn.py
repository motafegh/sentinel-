"""
Real (non-mocked) Aderyn integration test for static_analysis / quick_screen.

Why this file exists (2026-06-21): the Aderyn integration had THREE compounding
bugs, none caught by the (fully-mocked) existing test suite:
  1. `_run_aderyn_on_file` passed a FILE path as Aderyn's [ROOT] argument.
     Aderyn 0.6.8 requires a DIRECTORY — "Error making context: Not a directory
     (os error 20)", exit code 1.
  2. `--output json` passed the literal string "json" as the output FILE PATH
     (Aderyn's `-o/--output` flag takes a real path like report.json, not a
     format selector) — even with bug #1 fixed, this would silently write a
     markdown-formatted file named "json" instead of valid JSON.
  3. The JSON parser assumed a schema (`{"high": [...], "low": [...]}` with
     `id`/`line`/`function_name` fields) that does not exist. Real schema
     (verified by manually running `aderyn --output report.json <dir>` and
     inspecting the file): `{"high_issues": {"issues": [{"detector_name",
     "title", "description", "instances": [{"line_no", ...}]}]}, "low_issues": {...}}`.

All three combined meant exit code 1 was hit first and silently swallowed by the
`returncode != 0: return []` early-return — Aderyn has NEVER produced a single
finding through the agents module, on any contract, ever. These tests use the
REAL Aderyn binary (skipped if not installed) against a fixture with a known,
unambiguous vulnerability — they would have caught all three bugs.
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.nodes import _run_aderyn_on_file, quick_screen

aderyn_available = pytest.mark.skipif(
    shutil.which("aderyn") is None, reason="aderyn binary not installed",
)

_VAULT_PATH = Path(__file__).resolve().parent.parent / "test_contracts" / "vulnerable_reentrant.sol"


@aderyn_available
class TestAderynRealBinary:
    def test_finds_reentrancy_state_change(self):
        """
        vulnerable_reentrant.sol's withdraw() changes state after an external
        call — Aderyn's reentrancy-state-change detector must fire (verified
        manually: `aderyn --output report.json <dir>` produces exactly this
        finding as a High issue). If this fails, the directory/output-path/
        schema fix has regressed.
        """
        contract_code = _VAULT_PATH.read_text()
        findings = _run_aderyn_on_file(contract_code)

        assert findings, "Aderyn produced zero findings on a textbook reentrant contract"
        detectors = {f["detector"] for f in findings}
        assert "reentrancy-state-change" in detectors, f"got detectors: {detectors}"

        hit = next(f for f in findings if f["detector"] == "reentrancy-state-change")
        assert hit["impact"] == "High"
        assert hit["tool"] == "aderyn"
        assert hit["lines"], "expected at least one line number"

    def test_quick_screen_escalates_on_aderyn_high_finding(self):
        """quick_screen must escalate when Aderyn reports a High-impact finding."""
        state = {"contract_code": _VAULT_PATH.read_text(), "contract_address": "0xTEST"}
        import asyncio
        result = asyncio.run(quick_screen(state))

        aderyn_hits = result["quick_screen_hits"]["aderyn"]
        assert aderyn_hits, "quick_screen found zero Aderyn hits on a textbook reentrant contract"
        assert "reentrancy-state-change" in aderyn_hits

    def test_safe_contract_produces_no_high_findings(self):
        """Negative control: a contract with no external calls should have no High issues."""
        safe_code = """
            pragma solidity ^0.8.0;
            contract Pure {
                uint256 public value;
                function setValue(uint256 v) external { value = v; }
            }
        """
        findings = _run_aderyn_on_file(safe_code)
        high = [f for f in findings if f["impact"] == "High"]
        assert high == [], f"unexpected High findings on a trivial safe contract: {high}"
