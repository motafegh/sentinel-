"""Tests for R0.2 legacy report adapter (read-only lookup)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.persistence.legacy_adapter import find_legacy_hotspot, find_legacy_report


_VALID_ADDR = "0x5B38Da6a701c568545dCfcB03FcB875f56beddC4"


class TestFindLegacyReport:
    def test_finds_existing_report(self, tmp_path):
        (tmp_path / f"{_VALID_ADDR}.json").write_text(
            json.dumps({"top_vulnerability": "Reentrancy"})
        )
        report = find_legacy_report(tmp_path, _VALID_ADDR)
        assert report is not None
        assert report["top_vulnerability"] == "Reentrancy"

    def test_returns_none_for_missing(self, tmp_path):
        assert find_legacy_report(tmp_path, _VALID_ADDR) is None

    def test_rejects_malformed_address(self, tmp_path):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            find_legacy_report(tmp_path, "../../etc/passwd")

    def test_rejects_empty_address(self, tmp_path):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            find_legacy_report(tmp_path, "")

    def test_returns_none_on_corrupt_json(self, tmp_path):
        (tmp_path / f"{_VALID_ADDR}.json").write_text("{not valid json")
        assert find_legacy_report(tmp_path, _VALID_ADDR) is None


class TestFindLegacyHotspot:
    def test_finds_existing_hotspot(self, tmp_path):
        (tmp_path / f"{_VALID_ADDR}_hotspot.html").write_text("<html/>")
        result = find_legacy_hotspot(tmp_path, _VALID_ADDR)
        assert result is not None
        assert result.exists()

    def test_returns_none_for_missing(self, tmp_path):
        assert find_legacy_hotspot(tmp_path, _VALID_ADDR) is None

    def test_rejects_malformed_address(self, tmp_path):
        with pytest.raises(ValueError, match="invalid Ethereum address"):
            find_legacy_hotspot(tmp_path, "../../etc/passwd")
