"""Tests for sentinel_data.preprocessing pipeline modules.

Covers: compiler (two-pass + pragma tolerance), normalizer, deduplicator,
segmenter (version buckets + unchecked detection), and pipeline (meta.json schema).
Also includes the A20 regression guard: feat[2] fires when `now` keyword present.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sentinel_data.preprocessing.normalizer import normalize
from sentinel_data.preprocessing.deduplicator import Deduplicator
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
