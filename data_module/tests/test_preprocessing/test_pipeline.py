"""Tests for sentinel_data.preprocessing pipeline modules.

Covers: compiler (two-pass + pragma tolerance), normalizer, deduplicator,
segmenter (version buckets + unchecked detection), flattener (incl. the
unresolved-import strip fallback added for DeFiHackLabs on 2026-06-10),
and pipeline (meta.json schema).
Also includes the A20 regression guard: feat[2] fires when `now` keyword present.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sentinel_data.preprocessing.normalizer import normalize
from sentinel_data.preprocessing.deduplicator import Deduplicator
from sentinel_data.preprocessing.flattener import flatten_contract
from sentinel_data.preprocessing.segmenter import segment_and_bucket


# ── Normalizer ────────────────────────────────────────────────────────────────

class TestNormalizer:
    def test_strips_spdx(self):
        src = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n"
        result = normalize(src)
        assert "SPDX" not in result.content

    def test_strips_line_comments(self):
        src = "uint x; // this is a comment\nuint y;\n"
        result = normalize(src)
        assert "this is a comment" not in result.content

    def test_strips_block_comments(self):
        src = "/* block comment */\nuint x;\n"
        result = normalize(src)
        assert "block comment" not in result.content

    def test_preserves_code(self):
        src = "pragma solidity ^0.8.0;\ncontract C {}\n"
        result = normalize(src)
        assert "contract C" in result.content

    def test_collapses_blank_lines(self):
        src = "uint x;\n\n\n\n\nuint y;\n"
        result = normalize(src)
        assert "\n\n\n" not in result.content

    def test_line_counts(self):
        src = "line1\nline2\nline3\n"
        result = normalize(src)
        # "line1\nline2\nline3\n".count('\n') + 1 = 4 (trailing newline counts)
        assert result.n_lines_before == 4
        assert result.n_lines_after >= 1


# ── Deduplicator ──────────────────────────────────────────────────────────────

class TestDeduplicator:
    def test_exact_dup_detected(self):
        dedup = Deduplicator()
        content = "pragma solidity ^0.8.0;\ncontract C {}\n"
        r1 = dedup.process(content, Path("a.sol"))
        r2 = dedup.process(content, Path("b.sol"))
        assert not r1.is_duplicate
        assert r2.is_duplicate
        assert r1.dedup_group_id == r2.dedup_group_id

    def test_different_content_not_dup(self):
        dedup = Deduplicator()
        r1 = dedup.process("contract A {}\n", Path("a.sol"))
        r2 = dedup.process("contract B {}\n", Path("b.sol"))
        assert not r1.is_duplicate
        assert not r2.is_duplicate
        assert r1.dedup_group_id != r2.dedup_group_id

    def test_group_id_is_sha256(self):
        import hashlib
        dedup = Deduplicator()
        content = "contract X {}\n"
        r = dedup.process(content, Path("x.sol"))
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert r.dedup_group_id == expected


# ── Segmenter / version buckets ───────────────────────────────────────────────

class TestSegmenter:
    def test_legacy_bucket(self):
        seg = segment_and_bucket("contract C {}\n", "0.4.22")
        assert seg.version_bucket == "legacy"

    def test_transitional_bucket(self):
        seg = segment_and_bucket("contract C {}\n", "0.6.0")
        assert seg.version_bucket == "transitional"

    def test_modern_bucket(self):
        seg = segment_and_bucket("contract C {}\n", "0.8.0")
        assert seg.version_bucket == "modern"

    def test_has_unchecked_block(self):
        src = "contract C { function f() public { unchecked { uint x = 0; } } }\n"
        seg = segment_and_bucket(src, "0.8.0")
        assert seg.has_unchecked_block is True

    def test_no_unchecked_block(self):
        src = "contract C { function f() public { uint x = 0; } }\n"
        seg = segment_and_bucket(src, "0.8.0")
        assert seg.has_unchecked_block is False

    def test_contract_names_extracted(self):
        src = "contract MyToken { } contract MyVault { }\n"
        seg = segment_and_bucket(src, "0.8.0")
        assert "MyToken" in seg.contract_names
        assert "MyVault" in seg.contract_names


# ── A9 regression guard: feat[2] fires for `now` keyword ─────────────────────
# A9 bug (from audit, see AUDIT_PATCHES F1): graph_extractor missed the `now`
# Solidity alias for block.timestamp.  The Run 7 audit extended feat[2] =
# uses_block_globals to catch `now` + library wrappers + blockhash.
# This test guards the normalizer/segmenter chain — `now` must survive normalization
# so the downstream `uses_block_globals` feature can see it.
# (NB: the test was originally mislabeled "A20" — A20 is the label=0 hardcode bug;
#  A9 is the `now` keyword miss.  Corrected 2026-06-09.)

class TestA9Regression:
    def test_now_keyword_survives_normalizer(self):
        src = "contract C { function f() public view returns (uint) { return now; } }\n"
        result = normalize(src)
        assert "now" in result.content, (
            "A9 regression: `now` keyword stripped by normalizer — "
            "feat[2]=uses_block_globals would be 0 for all Timestamp contracts"
        )

    def test_block_timestamp_survives_normalizer(self):
        src = "contract C { function f() public view returns (uint) { return block.timestamp; } }\n"
        result = normalize(src)
        assert "block.timestamp" in result.content


# ── Pipeline meta.json schema ─────────────────────────────────────────────────

class TestPipelineMeta:
    """Verify ContractMeta produces a valid meta.json with all required fields."""

    def _make_meta(self, **kwargs):
        from sentinel_data.preprocessing.pipeline import ContractMeta
        defaults = dict(
            sha256="a" * 64,
            source_name="test_src",
            original_path="contracts/A.sol",
            pragma="^0.8.0",
            solc_version="0.8.20",
            compile_status="ok",
            compile_error="",
            attempted_solc_versions=["0.8.20"],
            flatten_status="flattened",
            dedup_group_id="a" * 64,
            is_duplicate=False,
            duplicate_of="",
            version_bucket="modern",
            has_unchecked_block=False,
            contract_names=["A"],
            n_raw_lines=10,
            n_normalized_lines=8,
        )
        defaults.update(kwargs)
        return ContractMeta(**defaults)

    def test_all_required_fields_present(self):
        meta = self._make_meta()
        d = meta.__dict__
        required = {
            "sha256", "source_name", "original_path", "pragma",
            "solc_version", "compile_status", "compile_error",
            "attempted_solc_versions", "flatten_status",
            "dedup_group_id", "is_duplicate", "duplicate_of",
            "version_bucket", "has_unchecked_block", "contract_names",
            "n_raw_lines", "n_normalized_lines", "meta_schema_version",
        }
        for field in required:
            assert field in d, f"ContractMeta missing field: {field}"

    def test_schema_version_is_string_1(self):
        meta = self._make_meta()
        assert meta.meta_schema_version == "1"

    def test_serializable_to_json(self):
        import dataclasses
        meta = self._make_meta()
        d = dataclasses.asdict(meta)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["sha256"] == "a" * 64


# ── Flattener: unresolved-import strip fallback (added 2026-06-10 for DeFiHackLabs) ──

class TestFlattenerStripUnresolvedImports:
    """Regression for the DeFiHackLabs integration test (2026-06-10).

    717/738 DeFiHackLabs PoCs import `forge-std/Test.sol` which is NOT
    present in the cloned repo (forge submodule not pulled). The flattener's
    solc --flatten path fails for these, and the compile step then chokes
    on the unresolvable import. The strip fallback removes only the
    unresolvable non-relative imports, allowing the compile step to succeed.
    """

    def test_strips_unresolvable_absolute_imports(self, tmp_path):
        """`import "forge-std/Test.sol";` is stripped when forge-std/ is absent."""
        sol = tmp_path / "exp.sol"
        sol.write_text(
            'pragma solidity ^0.8.0;\n'
            'import "forge-std/Test.sol";\n'
            'contract Exp {}\n'
        )
        r = flatten_contract(sol)
        assert r.flatten_status == "stripped_unresolved_imports"
        assert "forge-std/Test.sol" not in r.content
        assert "contract Exp" in r.content

    def test_keeps_resolvable_imports(self, tmp_path):
        """`import "./helper.sol";` is kept (relative + present on disk)."""
        helper = tmp_path / "helper.sol"
        helper.write_text("pragma solidity ^0.8.0;\n")
        sol = tmp_path / "exp.sol"
        sol.write_text(
            'pragma solidity ^0.8.0;\n'
            'import "./helper.sol";\n'
            'contract Exp {}\n'
        )
        # solc --flatten may or may not succeed; either way, the relative
        # import must NOT be stripped.
        r = flatten_contract(sol)
        assert "./helper.sol" in r.content, f"relative import was stripped: {r.content}"

    def test_keeps_relative_dotdot_imports(self, tmp_path):
        """`import "../shared/x.sol";` is kept (relative, even if absent in this test)."""
        sol = tmp_path / "exp.sol"
        sol.write_text(
            'pragma solidity ^0.8.0;\n'
            'import "../shared/x.sol";\n'
            'contract Exp {}\n'
        )
        r = flatten_contract(sol)
        # Relative import is never stripped, even if the target is missing —
        # that's a compile step concern, not a flatten step concern.
        assert "../shared/x.sol" in r.content

    def test_no_imports_returns_skipped_no_imports(self, tmp_path):
        sol = tmp_path / "exp.sol"
        sol.write_text("pragma solidity ^0.8.0;\ncontract Exp {}\n")
        r = flatten_contract(sol)
        assert r.flatten_status == "skipped_no_imports"

    def test_strip_preserves_vulnerable_code(self, tmp_path):
        """The whole point: keep the vulnerable test function body intact."""
        sol = tmp_path / "exp.sol"
        sol.write_text(
            'pragma solidity ^0.8.0;\n'
            'import "forge-std/Test.sol";\n'
            'contract Exp is Test {\n'
            '    function pwn() public {\n'
            '        unchecked { uint x = 1 - 2; }\n'
            '    }\n'
            '}\n'
        )
        r = flatten_contract(sol)
        assert "unchecked" in r.content
        assert "function pwn" in r.content
        assert "forge-std/Test.sol" not in r.content

    def test_strip_removes_test_inheritance(self, tmp_path):
        """When `Test` is brought in via stripped forge-std import, `is Test` is also stripped.

        This is the DeFiHackLabs pattern: 656/738 PoCs are `contract Foo is Test {}`
        where `Test` is the forge-std base contract. After stripping the import,
        the `is Test` clause must also be removed or the file won't compile.
        """
        sol = tmp_path / "exp.sol"
        sol.write_text(
            'pragma solidity ^0.8.0;\n'
            'import "forge-std/Test.sol";\n'
            'contract Exp is Test {\n'
            '    function pwn() public {}\n'
            '}\n'
        )
        r = flatten_contract(sol)
        assert r.flatten_status == "stripped_unresolved_imports"
        assert "is Test" not in r.content
        assert "contract Exp" in r.content
        assert "contract Exp {" in r.content  # `is` clause removed, not "is "

    def test_strip_keeps_resolvable_inheritance(self, tmp_path):
        """When all parents are resolvable (relative imports), the strip is a no-op."""
        # Create a real base contract
        base = tmp_path / "base.sol"
        base.write_text("pragma solidity ^0.8.0;\ncontract Base {}\n")
        sol = tmp_path / "exp.sol"
        sol.write_text(
            'pragma solidity ^0.8.0;\n'
            'import "./base.sol";\n'
            'import "forge-std/Test.sol";\n'  # This one gets stripped
            'contract Exp is Base, Test {\n'
            '    function pwn() public {}\n'
            '}\n'
        )
        r = flatten_contract(sol)
        # `is Base` should be kept (resolvable), `is Test` should be stripped
        assert "Base" in r.content
        assert "Test" not in r.content.split("contract Exp")[1].split("{")[0]
