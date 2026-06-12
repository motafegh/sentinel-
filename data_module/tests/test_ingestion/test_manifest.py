"""Tests for sentinel_data.ingestion.manifest"""

import json
import tempfile
from pathlib import Path

import pytest

from sentinel_data.ingestion.manifest import (
    FileRecord,
    IngestionManifest,
    build_file_records,
    load_manifest,
    verify_manifest,
)


def _write_sol(path: Path, content: str = "// test\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestFileRecord:
    def test_fields_present(self):
        rec = FileRecord(path="a.sol", sha256="abc123", size_bytes=42)
        assert rec.path == "a.sol"
        assert rec.sha256 == "abc123"
        assert rec.size_bytes == 42


class TestIngestionManifest:
    def _make_manifest(self):
        return IngestionManifest(
            source="test_src",
            connector="git",
            url="https://example.com/repo.git",
            pin="abc1234",
            resolved_pin="abc1234abc1234abc1234abc1234abc1234abc1234",
            fetched_at="2026-06-09T00:00:00Z",
            duration_s=1.5,
            contract_count=2,
            files=[
                FileRecord("contracts/A.sol", "sha_a", 100),
                FileRecord("contracts/B.sol", "sha_b", 200),
            ],
            extra={},
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        m = self._make_manifest()
        dest = tmp_path / "manifest.json"
        m.save(dest)
        loaded = IngestionManifest.load(dest)

        assert loaded.source == "test_src"
        assert loaded.contract_count == 2
        assert len(loaded.files) == 2
        assert loaded.files[0].sha256 == "sha_a"

    def test_save_creates_parent_dirs(self, tmp_path):
        m = self._make_manifest()
        dest = tmp_path / "a" / "b" / "manifest.json"
        m.save(dest)
        assert dest.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            IngestionManifest.load(tmp_path / "nope.json")

    def test_verify_ok(self, tmp_path):
        raw_dir = tmp_path / "raw"
        a = raw_dir / "contracts" / "A.sol"
        b = raw_dir / "contracts" / "B.sol"
        _write_sol(a, "// A\n")
        _write_sol(b, "// B\n")

        import hashlib
        sha_a = hashlib.sha256(a.read_bytes()).hexdigest()
        sha_b = hashlib.sha256(b.read_bytes()).hexdigest()

        m = IngestionManifest(
            source="test_src", connector="git", url="", pin="", resolved_pin="",
            fetched_at="", duration_s=0, contract_count=2,
            files=[
                FileRecord("contracts/A.sol", sha_a, a.stat().st_size),
                FileRecord("contracts/B.sol", sha_b, b.stat().st_size),
            ],
            extra={},
        )
        ok, errors = m.verify(raw_dir)
        assert ok
        assert errors == []

    def test_verify_detects_tamper(self, tmp_path):
        raw_dir = tmp_path / "raw"
        a = raw_dir / "contracts" / "A.sol"
        _write_sol(a, "// A\n")

        m = IngestionManifest(
            source="test_src", connector="git", url="", pin="", resolved_pin="",
            fetched_at="", duration_s=0, contract_count=1,
            files=[FileRecord("contracts/A.sol", "deadbeef" * 8, a.stat().st_size)],
            extra={},
        )
        ok, errors = m.verify(raw_dir)
        assert not ok
        assert len(errors) == 1
        assert "CHANGED" in errors[0]  # format: "CHANGED  path  expected=...  got=..."

    def test_verify_missing_file(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        m = IngestionManifest(
            source="test_src", connector="git", url="", pin="", resolved_pin="",
            fetched_at="", duration_s=0, contract_count=1,
            files=[FileRecord("missing.sol", "abc", 0)],
            extra={},
        )
        ok, errors = m.verify(raw_dir)
        assert not ok
        assert any("MISSING" in e for e in errors)


class TestBuildFileRecords:
    def test_basic(self, tmp_path):
        a = tmp_path / "A.sol"
        b = tmp_path / "sub" / "B.sol"
        _write_sol(a, "// A\n")
        _write_sol(b, "// B\n")

        records = build_file_records([a, b], base_dir=tmp_path)
        assert len(records) == 2
        paths = {r.path for r in records}
        assert "A.sol" in paths
        assert "sub/B.sol" in paths

    def test_sha256_correctness(self, tmp_path):
        import hashlib
        a = tmp_path / "A.sol"
        content = b"pragma solidity ^0.8.0;\n"
        a.write_bytes(content)
        records = build_file_records([a], base_dir=tmp_path)
        expected = hashlib.sha256(content).hexdigest()
        assert records[0].sha256 == expected

    def test_size_bytes(self, tmp_path):
        a = tmp_path / "A.sol"
        a.write_bytes(b"x" * 77)
        records = build_file_records([a], base_dir=tmp_path)
        assert records[0].size_bytes == 77
