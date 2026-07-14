"""Tests for R0.2 safe ZIP extraction (archive_safety)."""

from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_module"))

from sentinel_data.ingestion.archive_safety import (
    ArchiveBadNameError,
    ArchiveLimitError,
    ArchiveLimits,
    ArchiveSafetyError,
    ArchiveSymlinkError,
    ArchiveTraversalError,
    extract_zip_safe,
)


def _make_zip(path: Path, members: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            if isinstance(content, str):
                zf.writestr(name, content)
            else:
                zf.writestr(name, content)


def _make_zip_with_attr(
    path: Path, members: list[tuple[str, bytes, int]]
) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content, attr in members:
            info = zipfile.ZipInfo(name)
            info.external_attr = attr
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, content)


class TestTraversalRejection:
    def test_dotdot_traversal_rejected(self, tmp_path):
        archive = tmp_path / "bad.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"../evil.txt": "escaped"})
        with pytest.raises(ArchiveTraversalError, match="path traversal"):
            extract_zip_safe(archive, dest)
        assert not (tmp_path / "evil.txt").exists()

    def test_sibling_prefix_traversal_rejected(self, tmp_path):
        archive = tmp_path / "bad.zip"
        dest = tmp_path / "repo"
        dest.mkdir()
        _make_zip(archive, {"../repo_evil/pwned.txt": "escaped"})
        with pytest.raises((ArchiveTraversalError, ArchiveSafetyError)):
            extract_zip_safe(archive, dest)
        assert not (tmp_path / "repo_evil").exists()

    def test_deep_traversal_rejected(self, tmp_path):
        archive = tmp_path / "bad.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"legit/../../../evil.txt": "escaped"})
        with pytest.raises((ArchiveTraversalError, ArchiveSafetyError)):
            extract_zip_safe(archive, dest)

    def test_no_outside_write_on_failure(self, tmp_path):
        archive = tmp_path / "bad.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"../evil1.txt": "a", "../evil2.txt": "b"})
        try:
            extract_zip_safe(archive, dest)
        except ArchiveSafetyError:
            pass
        assert not (tmp_path / "evil1.txt").exists()
        assert not (tmp_path / "evil2.txt").exists()


class TestSymlinkRejection:
    def test_symlink_member_rejected(self, tmp_path):
        archive = tmp_path / "sym.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        symlink_attr = (stat.S_IFLNK | 0o777) << 16
        _make_zip_with_attr(
            archive,
            [("link.txt", b"/etc/passwd", symlink_attr)],
        )
        with pytest.raises(ArchiveSymlinkError, match="symlink"):
            extract_zip_safe(archive, dest)


class TestSpecialFileRejection:
    def test_fifo_rejected(self, tmp_path):
        archive = tmp_path / "fifo.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        fifo_attr = (stat.S_IFIFO | 0o644) << 16
        _make_zip_with_attr(archive, [("myfifo", b"", fifo_attr)])
        with pytest.raises(ArchiveSafetyError, match="special file"):
            extract_zip_safe(archive, dest)


class TestBadNameRejection:
    def test_nul_in_name_rejected(self, tmp_path):
        from sentinel_data.ingestion.archive_safety import _check_member_name

        with pytest.raises(ArchiveBadNameError, match="NUL"):
            _check_member_name("ev\x00il.txt")

    def test_control_char_in_name_rejected(self, tmp_path):
        archive = tmp_path / "ctrl.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"ev\x01il.txt": "bad"})
        with pytest.raises(ArchiveBadNameError, match="control character"):
            extract_zip_safe(archive, dest)

    def test_windows_drive_letter_rejected(self, tmp_path):
        archive = tmp_path / "drive.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"C:/evil.txt": "bad"})
        with pytest.raises(ArchiveBadNameError, match="Windows drive"):
            extract_zip_safe(archive, dest)


class TestLimitEnforcement:
    def test_member_count_limit(self, tmp_path):
        archive = tmp_path / "many.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        members = {f"file_{i}.txt": str(i) for i in range(10)}
        _make_zip(archive, members)
        limits = ArchiveLimits(max_members=5)
        with pytest.raises(ArchiveLimitError, match="members"):
            extract_zip_safe(archive, dest, limits=limits)

    def test_per_file_size_limit(self, tmp_path):
        archive = tmp_path / "big.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"big.txt": "x" * 1000})
        limits = ArchiveLimits(max_per_file_bytes=100)
        with pytest.raises(ArchiveLimitError, match="per-file"):
            extract_zip_safe(archive, dest, limits=limits)

    def test_path_depth_limit(self, tmp_path):
        archive = tmp_path / "deep.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        deep_name = "/".join(f"d{i}" for i in range(20)) + "/file.txt"
        _make_zip(archive, {deep_name: "deep"})
        limits = ArchiveLimits(max_path_depth=10)
        with pytest.raises(ArchiveLimitError, match="path depth"):
            extract_zip_safe(archive, dest, limits=limits)


class TestNormalExtraction:
    def test_valid_zip_extracts_correctly(self, tmp_path):
        archive = tmp_path / "good.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"a.sol": "// a", "sub/b.sol": "// b"})
        result = extract_zip_safe(archive, dest)
        assert (result / "a.sol").read_text() == "// a"
        assert (result / "sub" / "b.sol").read_text() == "// b"

    def test_macos_metadata_skipped(self, tmp_path):
        archive = tmp_path / "dive.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(
            archive,
            {
                "Raw/a.sol": "// a",
                "__MACOSX/Raw/._a.sol": "noise",
                "Raw/.DS_Store": "noise",
            },
        )
        result = extract_zip_safe(archive, dest)
        assert (result / "Raw" / "a.sol").exists()
        assert not (result / "Raw" / ".DS_Store").exists()
        assert not (result / "__MACOSX").exists()

    def test_empty_directories_created(self, tmp_path):
        archive = tmp_path / "dirs.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"empty_dir/": ""})
        result = extract_zip_safe(archive, dest)
        assert (result / "empty_dir").is_dir()

    def test_bad_zip_rejected(self, tmp_path):
        archive = tmp_path / "notazip.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        archive.write_bytes(b"not a zip file")
        with pytest.raises(ArchiveSafetyError, match="not a valid zip"):
            extract_zip_safe(archive, dest)

    def test_atomic_promotion_no_partial_state(self, tmp_path):
        archive = tmp_path / "good.zip"
        dest = tmp_path / "dest"
        dest.mkdir()
        _make_zip(archive, {"a.sol": "// a", "b.sol": "// b"})
        extract_zip_safe(archive, dest)
        assert (dest / "a.sol").exists()
        assert (dest / "b.sol").exists()
        assert not list(dest.parent.glob(".*__extraction_tmp"))
