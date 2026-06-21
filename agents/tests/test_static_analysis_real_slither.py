"""
Real (non-mocked) Slither integration test for static_analysis / quick_screen.

Why this file exists (2026-06-21): every existing test that touches static_analysis
or quick_screen mocks `slither.Slither` entirely (see test_smoke_e2e.py's
`mock_slither_cls`), so NONE of them exercise the real Slither Python API. This let a
critical bug ship invisibly: `Slither(tmp_path)` registers ZERO detectors on
construction — `sl._detectors` starts empty, so every "Slither found 0 findings" log
line across the agents module's entire history was silently meaningless, not a sign of
a safe contract. Found via manual verification: running `slither contract.sol` from the
CLI on a textbook reentrant Vault found `reentrancy-eth`; the in-process node found
nothing on the identical file, because the CLI explicitly calls
`slither.register_detector()` for every detector class and the node didn't.

These tests use the REAL Slither library (skipped if not installed) against a fixture
with a known, unambiguous vulnerability — they would have caught the registration bug.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.nodes import static_analysis, quick_screen

slither_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("slither"),
    reason="slither not installed",
)

_VAULT_PATH = Path(__file__).resolve().parent.parent / "test_contracts" / "vulnerable_reentrant.sol"


@slither_available
class TestStaticAnalysisRealSlither:
    @pytest.mark.asyncio
    async def test_finds_textbook_reentrancy(self):
        """
        vulnerable_reentrant.sol's withdraw() makes an external call BEFORE
        decrementing the caller's balance — Slither's reentrancy-eth detector
        must fire on this. If this test fails, detector registration is broken
        again (see module docstring).
        """
        contract_code = _VAULT_PATH.read_text()
        state = {
            "contract_code": contract_code,
            "contract_address": "0xTEST",
            "ml_result": {"probabilities": {"Reentrancy": 0.9}},
        }
        result = await static_analysis(state)

        assert "error" not in result, f"static_analysis errored: {result.get('error')}"
        findings = result["static_findings"]
        slither_findings = [f for f in findings if f.get("tool") == "slither"]
        assert slither_findings, "Slither produced zero findings on a textbook reentrant contract"

        detectors_found = {f["detector"] for f in slither_findings}
        assert "reentrancy-eth" in detectors_found, (
            f"reentrancy-eth not detected; got {detectors_found}"
        )

    @pytest.mark.asyncio
    async def test_quick_screen_escalates_on_reentrancy(self):
        """quick_screen's Tier-0 Slither scan must also register detectors."""
        contract_code = _VAULT_PATH.read_text()
        state = {"contract_code": contract_code, "contract_address": "0xTEST"}
        result = await quick_screen(state)

        hits = result["quick_screen_hits"]["slither"]
        assert hits, "quick_screen found zero Slither hits on a textbook reentrant contract"
        assert "reentrancy-eth" in hits

    @pytest.mark.asyncio
    async def test_safe_contract_produces_no_reentrancy_finding(self):
        """Negative control: a contract with no external calls must not flag reentrancy."""
        safe_code = """
            pragma solidity ^0.8.0;
            contract Pure {
                uint256 public value;
                function setValue(uint256 v) external { value = v; }
            }
        """
        state = {"contract_code": safe_code, "contract_address": "0xTEST"}
        result = await static_analysis(state)
        detectors_found = {f["detector"] for f in result["static_findings"] if f.get("tool") == "slither"}
        assert "reentrancy-eth" not in detectors_found
