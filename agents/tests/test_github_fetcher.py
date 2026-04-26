"""
tests/test_github_fetcher.py

Unit tests for DeFiHackLabsFetcher — no LM Studio, no network, no FAISS.
Tests parsing logic, loss extraction, vuln type inference, and the
fixes applied in 2026-04-11.

Run:
  cd ~/projects/sentinel/agents
  poetry run pytest tests/test_github_fetcher.py -v
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from src.rag.fetchers.github_fetcher import DeFiHackLabsFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a minimal DeFiHackLabs directory structure for testing."""
    repo  = tmp_path / "defihacklabs"
    cache = tmp_path / "exploits"
    src   = repo / "src" / "test"
    src.mkdir(parents=True)
    cache.mkdir()
    return DeFiHackLabsFetcher(repo_path=repo, data_dir=cache)


def _write_sol(directory: Path, filename: str, content: str) -> Path:
    """Helper: write a .sol file inside a date-named subdirectory."""
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


class TestExtractLoss:
    def test_millions(self, fetcher):
        assert fetcher._extract_loss("// @KeyInfo - Total Lost : ~$197M") == 197_000_000

    def test_thousands(self, fetcher):
        assert fetcher._extract_loss("// @KeyInfo - Total Lost : 15k") == 15_000

    def test_raw_usd(self, fetcher):
        assert fetcher._extract_loss("// @KeyInfo -- Total Lost : ~59643 USD") == 59_643

    def test_decimal_millions(self, fetcher):
        assert fetcher._extract_loss("// @KeyInfo - Total Lost : 1.4M") == 1_400_000

    def test_billions(self, fetcher):
        assert fetcher._extract_loss("// @KeyInfo - Total Lost : 1.2B") == 1_200_000_000

    def test_old_format(self, fetcher):
        assert fetcher._extract_loss("// Loss ~$50M") == 50_000_000

    def test_no_loss_returns_none(self, fetcher):
        assert fetcher._extract_loss("// pragma solidity ^0.8.0;") is None

    def test_malformed_returns_none(self, fetcher):
        assert fetcher._extract_loss("// @KeyInfo - Total Lost : abc") is None


class TestInferVulnType:
    """FIX-22b: infer from structured fields only — no raw content[:1000]."""

    def test_reentrancy_from_root_cause(self, fetcher):
        vt = fetcher._infer_vuln_type("reentrancy in withdraw()", None, {})
        assert vt == "reentrancy"

    def test_flash_loan_from_summary(self, fetcher):
        vt = fetcher._infer_vuln_type(None, "1) Flash loan 30M DAI\n2) Swap", {})
        assert vt == "flash_loan"

    def test_oracle_from_keyinfo(self, fetcher):
        vt = fetcher._infer_vuln_type(None, None, {"keyinfo_line": "price manipulation attack"})
        assert vt == "oracle_manipulation"

    def test_access_control(self, fetcher):
        vt = fetcher._infer_vuln_type("unauthorized access to admin", None, {})
        assert vt == "access_control"

    def test_integer_overflow(self, fetcher):
        vt = fetcher._infer_vuln_type("integer overflow in balance", None, {})
        assert vt == "integer_overflow"

    def test_fallback_to_other(self, fetcher):
        vt = fetcher._infer_vuln_type(None, None, {})
        assert vt == "other"

    def test_no_content_slice_used(self, fetcher):
        """
        FIX-22b: SPDX/pragma content should NOT influence type inference.
        Even if we pass Solidity boilerplate as if it were root_cause,
        the meaningful patterns won't match → returns 'other'.
        """
        boilerplate = "SPDX-License-Identifier: MIT pragma solidity import Test"
        vt = fetcher._infer_vuln_type(boilerplate, None, {})
        # boilerplate doesn't contain any vuln keywords
        assert vt == "other"


class TestExtractDate:
    def test_extracts_yyyy_mm_from_path(self, tmp_path, fetcher):
        sol = tmp_path / "2023-03" / "Euler_exp.sol"
        sol.parent.mkdir(parents=True)
        sol.write_text("// test")
        assert fetcher._extract_date(sol) == "2023-03-01"

    def test_returns_empty_string_when_no_date(self, tmp_path, fetcher):
        sol = tmp_path / "misc" / "OldExploit.sol"
        sol.parent.mkdir(parents=True)
        sol.write_text("// test")
        assert fetcher._extract_date(sol) == ""


class TestFetchSince:
    """FIX-21: undated files must always be included in fetch_since()."""

    def test_includes_undated_files(self, tmp_path):
        """
        FIX-21: Old code dropped docs with no date via `if doc_date_str:`.
        Undated files must now pass through unconditionally.
        """
        repo  = tmp_path / "repo"
        cache = tmp_path / "cache"
        src   = repo / "src" / "test"
        src.mkdir(parents=True)
        cache.mkdir()

        # File without a date directory (no YYYY-MM in path)
        undated = src / "OldExploit_exp.sol"
        undated.write_text(
            "// @KeyInfo - Total Lost : ~$5M\n"
            "// @Analysis\n"
            "// https://example.com/analysis\n"
        )

        fetcher = DeFiHackLabsFetcher(repo_path=repo, data_dir=cache)
        results = fetcher.fetch_since(since=datetime(2024, 1, 1))

        # The undated file must be present despite `since` being recent
        assert len(results) >= 1
        undated_docs = [d for d in results if "OldExploit" in d.metadata.get("protocol", "")]
        assert len(undated_docs) == 1

    def test_excludes_old_dated_files(self, tmp_path):
        repo  = tmp_path / "repo"
        cache = tmp_path / "cache"
        dated = repo / "src" / "test" / "2020-01"
        dated.mkdir(parents=True)
        cache.mkdir()

        old = dated / "OldHack_exp.sol"
        old.write_text("// @KeyInfo - Total Lost : ~$1M\n")

        fetcher = DeFiHackLabsFetcher(repo_path=repo, data_dir=cache)
        results = fetcher.fetch_since(since=datetime(2023, 1, 1))

        old_docs = [d for d in results if "OldHack" in d.metadata.get("protocol", "")]
        assert len(old_docs) == 0

    def test_includes_new_dated_files(self, tmp_path):
        repo  = tmp_path / "repo"
        cache = tmp_path / "cache"
        dated = repo / "src" / "test" / "2024-06"
        dated.mkdir(parents=True)
        cache.mkdir()

        new = dated / "NewHack_exp.sol"
        new.write_text("// @KeyInfo - Total Lost : ~$10M\n")

        fetcher = DeFiHackLabsFetcher(repo_path=repo, data_dir=cache)
        results = fetcher.fetch_since(since=datetime(2023, 1, 1))

        new_docs = [d for d in results if "NewHack" in d.metadata.get("protocol", "")]
        assert len(new_docs) == 1


class TestPastDirectoryScanning:
    """FIX-20: past/ directory must be scanned by fetch()."""

    def test_past_dir_files_are_fetched(self, tmp_path):
        repo  = tmp_path / "repo"
        cache = tmp_path / "cache"
        past  = repo / "past" / "2019-01"
        past.mkdir(parents=True)
        (repo / "src" / "test").mkdir(parents=True)
        cache.mkdir()

        past_file = past / "HistoricalHack_exp.sol"
        past_file.write_text("// @KeyInfo - Total Lost : ~$2M\n")

        fetcher = DeFiHackLabsFetcher(repo_path=repo, data_dir=cache)
        docs    = fetcher.fetch()

        historical = [d for d in docs if "HistoricalHack" in d.metadata.get("protocol", "")]
        assert len(historical) == 1, "past/ directory files must be included in fetch()"

    def test_no_past_dir_does_not_crash(self, tmp_path):
        """If past/ doesn't exist, fetch() should still work."""
        repo  = tmp_path / "repo"
        cache = tmp_path / "cache"
        (repo / "src" / "test").mkdir(parents=True)
        cache.mkdir()
        # No past/ directory

        fetcher = DeFiHackLabsFetcher(repo_path=repo, data_dir=cache)
        docs    = fetcher.fetch()   # should not raise
        assert isinstance(docs, list)
