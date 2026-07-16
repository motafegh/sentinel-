"""Tests for R0.2 persistence path validation and containment."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.persistence.paths import (
    assert_contained,
    is_valid_job_id,
    job_report_dir,
    job_report_path,
    validate_address,
    validate_job_id,
)


class TestValidateAddress:
    def test_valid_checksummed_address(self):
        addr = "0x5B38Da6a701c568545dCfcB03FcB875f56beddC4"
        assert validate_address(addr) == addr

    def test_valid_lowercase_address(self):
        addr = "0x5b38da6a701c568545dcfcb03fcb875f56beddc4"
        assert validate_address(addr) == addr

    def test_strips_whitespace(self):
        addr = "  0x5B38Da6a701c568545dCfcB03FcB875f56beddC4  "
        assert validate_address(addr) == "0x5B38Da6a701c568545dCfcB03FcB875f56beddC4"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            validate_address("")

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            validate_address(None)  # type: ignore[arg-type]

    def test_rejects_traversal(self):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            validate_address("../../etc/passwd")

    def test_rejects_short(self):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            validate_address("0x1234")

    def test_rejects_no_prefix(self):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            validate_address("5B38Da6a701c568545dCfcB03FcB875f56beddC4")


class TestValidateJobId:
    def test_valid_uuid(self):
        jid = str(uuid.uuid4())
        assert validate_job_id(jid) == jid.lower()

    def test_valid_uuid_uppercase(self):
        jid = str(uuid.uuid4()).upper()
        assert is_valid_job_id(jid)

    def test_rejects_empty(self):
        assert not is_valid_job_id("")
        with pytest.raises(ValueError):
            validate_job_id("")

    def test_rejects_traversal(self):
        assert not is_valid_job_id("../../etc/passwd")
        with pytest.raises(ValueError):
            validate_job_id("../../etc/passwd")

    def test_rejects_address(self):
        assert not is_valid_job_id("0x5B38Da6a701c568545dCfcB03FcB875f56beddC4")


class TestJobReportDir:
    def test_resolves_job_dir(self, tmp_path):
        jid = str(uuid.uuid4())
        result = job_report_dir(tmp_path, jid)
        assert result == tmp_path.resolve() / jid

    def test_rejects_invalid_job_id(self, tmp_path):
        with pytest.raises(ValueError, match="invalid job_id"):
            job_report_dir(tmp_path, "../../etc")

    def test_containment_check(self, tmp_path):
        jid = str(uuid.uuid4())
        d = job_report_dir(tmp_path, jid)
        assert d.is_relative_to(tmp_path.resolve())


class TestJobReportPath:
    def test_resolves_report_json(self, tmp_path):
        jid = str(uuid.uuid4())
        p = job_report_path(tmp_path, jid, "report.json")
        assert p.name == "report.json"
        assert p.parent == tmp_path.resolve() / jid

    def test_rejects_path_separator_in_filename(self, tmp_path):
        jid = str(uuid.uuid4())
        with pytest.raises(ValueError, match="unsafe report filename"):
            job_report_path(tmp_path, jid, "../../etc/passwd")

    def test_rejects_dotdot_in_filename(self, tmp_path):
        jid = str(uuid.uuid4())
        with pytest.raises(ValueError, match="unsafe report filename"):
            job_report_path(tmp_path, jid, "..")

    def test_rejects_backslash_in_filename(self, tmp_path):
        jid = str(uuid.uuid4())
        with pytest.raises(ValueError, match="unsafe report filename"):
            job_report_path(tmp_path, jid, "evil\\..\\etc")


class TestAssertContained:
    def test_contained_path_passes(self, tmp_path):
        p = tmp_path / "subdir" / "file.json"
        p.parent.mkdir()
        p.touch()
        result = assert_contained(p, tmp_path)
        assert result == p.resolve()

    def test_escaped_path_raises(self, tmp_path):
        p = tmp_path.parent / "evil" / "file.json"
        with pytest.raises(ValueError, match="path containment violation"):
            assert_contained(p, tmp_path)
