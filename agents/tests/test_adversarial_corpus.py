"""Tests for P4 adversarial corpus — injection detection on real contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.security.injection_detect import detect_injections


_ADVERSARIAL_DIR = Path(__file__).parent / "fixtures" / "adversarial"


def _load_contract(name: str) -> str:
    path = _ADVERSARIAL_DIR / name
    return path.read_text()


class TestAdversarialCorpus:
    def test_01_comment_injection_detected(self):
        src = _load_contract("adversarial_01_comment_injection.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "comment" in patterns

    def test_02_string_injection_detected(self):
        src = _load_contract("adversarial_02_string_injection.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "string" in patterns

    def test_03_role_swap_detected(self):
        src = _load_contract("adversarial_03_role_swap.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "role-swap" in patterns

    def test_04_extraction_detected(self):
        src = _load_contract("adversarial_04_extraction.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "extraction" in patterns

    def test_05_identifier_injection_detected(self):
        src = _load_contract("adversarial_05_identifier_injection.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "identifier" in patterns

    def test_06_natspec_injection_detected(self):
        src = _load_contract("adversarial_06_natspec_injection.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "NatSpec" in patterns

    def test_07_multi_pattern_detected(self):
        src = _load_contract("adversarial_07_multi_pattern.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "multi" in patterns
        assert "comment" in patterns
        assert "import" in patterns

    def test_08_import_injection_detected(self):
        src = _load_contract("adversarial_08_import_injection.sol")
        matches = detect_injections(src)
        patterns = {m.pattern for m in matches}
        assert "import" in patterns

    def test_all_contracts_have_expect_header(self):
        for sol_file in _ADVERSARIAL_DIR.glob("adversarial_*.sol"):
            content = sol_file.read_text()
            assert "// expect:" in content, f"{sol_file.name} missing // expect: header"
