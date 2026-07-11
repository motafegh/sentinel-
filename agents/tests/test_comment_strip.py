"""Tests for Layer 1: Solidity comment stripping."""

from __future__ import annotations

import pytest

from src.security.comment_strip import strip_comments


class TestLineComments:
    def test_simple_line_comment(self):
        src = "uint x; // this is a comment\nuint y;"
        out = strip_comments(src)
        assert "//" not in out
        assert "this is a comment" not in out
        assert "uint x;" in out
        assert "uint y;" in out

    def test_line_comment_preserves_line_count(self):
        src = "line1\n// comment\nline3"
        out = strip_comments(src)
        assert len(out.splitlines()) == len(src.splitlines())

    def test_natspec_line_comment(self):
        src = "/// @dev this is natspec\nfunction foo()"
        out = strip_comments(src)
        assert "///" not in out
        assert "@dev" not in out
        assert "function foo()" in out


class TestBlockComments:
    def test_simple_block_comment(self):
        src = "uint x; /* block */ uint y;"
        out = strip_comments(src)
        assert "/*" not in out
        assert "*/" not in out
        assert "block" not in out
        assert "uint x;" in out
        assert "uint y;" in out

    def test_multiline_block_comment(self):
        src = "line1\n/* multi\nline\ncomment */\nline5"
        out = strip_comments(src)
        assert "multi" not in out
        assert "comment" not in out
        assert len(out.splitlines()) == len(src.splitlines())

    def test_natspec_block_comment(self):
        src = "/** @notice natspec block */\nfunction foo()"
        out = strip_comments(src)
        assert "/**" not in out
        assert "@notice" not in out
        assert "function foo()" in out


class TestStringPreservation:
    def test_comment_inside_double_string(self):
        src = 'string memory x = "// not a comment";'
        out = strip_comments(src)
        assert out == src

    def test_comment_inside_single_string(self):
        src = "string memory x = '// not a comment';"
        out = strip_comments(src)
        assert out == src

    def test_block_comment_inside_string(self):
        src = 'string memory x = "/* not a comment */";'
        out = strip_comments(src)
        assert out == src

    def test_escaped_quote_in_string(self):
        src = r'string memory x = "escaped \" // not a comment";'
        out = strip_comments(src)
        assert "// not a comment" in out

    def test_string_after_comment(self):
        src = '// real comment\nstring memory x = "hello";'
        out = strip_comments(src)
        assert "real comment" not in out
        assert '"hello"' in out


class TestEdgeCases:
    def test_empty_input(self):
        assert strip_comments("") == ""

    def test_no_comments(self):
        src = "pragma solidity ^0.8.0;\ncontract Foo { }"
        assert strip_comments(src) == src

    def test_line_count_preserved_complex(self):
        src = (
            "// SPDX-License-Identifier: MIT\n"
            "pragma solidity ^0.8.0;\n"
            "/* block\n   comment */\n"
            "contract Foo {\n"
            "    /// @dev natspec\n"
            "    function bar() public {}\n"
            "}\n"
        )
        out = strip_comments(src)
        assert len(out.splitlines()) == len(src.splitlines())

    def test_adjacent_comments(self):
        src = "// line1\n// line2\n// line3"
        out = strip_comments(src)
        assert "line1" not in out
        assert "line2" not in out
        assert "line3" not in out
        assert len(out.splitlines()) == 3

    def test_comment_at_eof_no_newline(self):
        src = "uint x; // eof comment"
        out = strip_comments(src)
        assert "eof comment" not in out
        assert "uint x;" in out
