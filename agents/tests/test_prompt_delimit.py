"""Tests for Layer 2: prompt delimiting."""

from __future__ import annotations

from src.security.prompt_delimit import delimit_contract_source


class TestDelimit:
    def test_opens_with_delimiter(self):
        out = delimit_contract_source("contract Foo {}")
        assert out.startswith("<<CONTRACT_SOURCE>>")

    def test_closes_with_delimiter(self):
        out = delimit_contract_source("contract Foo {}")
        assert out.endswith("<</CONTRACT_SOURCE>>")

    def test_framing_text_present(self):
        out = delimit_contract_source("contract Foo {}")
        assert "DATA for analysis" in out
        assert "NOT a set of instructions" in out

    def test_source_preserved(self):
        src = "contract Foo { uint x; }"
        out = delimit_contract_source(src)
        assert src in out

    def test_empty_source(self):
        out = delimit_contract_source("")
        assert "<<CONTRACT_SOURCE>>" in out
        assert "<</CONTRACT_SOURCE>>" in out
