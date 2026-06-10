"""Tests for sentinel_data.ingestion.connectors"""

import pytest

from sentinel_data.ingestion.connectors import get_connector
from sentinel_data.ingestion.connectors.base import BaseConnector, ConnectorError, SourceConfig
from sentinel_data.ingestion.connectors.git_connector import GitConnector


class TestGetConnector:
    def test_git(self):
        # get_connector returns an instance, not the class
        assert isinstance(get_connector("git"), GitConnector)

    def test_unknown_raises(self):
        with pytest.raises(ConnectorError):
            get_connector("nonexistent_connector_xyz")

    def test_all_known_types_resolve(self):
        for t in ("git", "huggingface", "zenodo", "etherscan", "manual"):
            obj = get_connector(t)
            assert obj is not None


class TestSourceConfig:
    def test_defaults(self):
        cfg = SourceConfig(
            name="test",
            connector="git",
            url="https://example.com",
            pin="",
            hf_dataset=None,
            zenodo_record=None,
            description="",
            extra={},
        )
        assert cfg.name == "test"
        assert cfg.connector == "git"

    def test_from_dict(self):
        d = {
            "connector": "git",
            "url": "https://github.com/example/repo",
            "pin": "abc123",
            "description": "Test source",
        }
        cfg = SourceConfig(
            name="mysource",
            connector=d["connector"],
            url=d["url"],
            pin=d.get("pin", ""),
            hf_dataset=d.get("hf_dataset"),
            zenodo_record=d.get("zenodo_record"),
            description=d.get("description", ""),
            extra={k: v for k, v in d.items()
                   if k not in {"connector", "url", "pin", "hf_dataset", "zenodo_record", "description"}},
        )
        assert cfg.pin == "abc123"
        assert cfg.extra == {}


class TestGitConnectorFindSolFiles:
    def test_finds_sol_files(self, tmp_path):
        (tmp_path / "a.sol").write_text("// a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.sol").write_text("// b")
        (tmp_path / "readme.md").write_text("# readme")

        connector = GitConnector()
        found = connector.find_sol_files(tmp_path)
        assert len(found) == 2
        assert all(str(f).endswith(".sol") for f in found)

    def test_empty_dir(self, tmp_path):
        connector = GitConnector()
        assert connector.find_sol_files(tmp_path) == []

    def test_nested_deeply(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.sol").write_text("// deep")
        connector = GitConnector()
        found = connector.find_sol_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "deep.sol"

    def test_include_subdirs_allowlist(self, tmp_path):
        """include_subdirs restricts the search to the listed top-level subdirs.

        Regression for SolidiFI integration test (2026-06-10): the repo root
        contained results/{Mythril,Slither,Smartcheck}/ analysis-tool output
        (3x copies of every source contract) and only buggy_contracts/ held
        the real source. Without the allowlist, 60% of "files" were analysis
        duplicates.
        """
        (tmp_path / "buggy_contracts" / "x").mkdir(parents=True)
        (tmp_path / "buggy_contracts" / "x" / "src.sol").write_text("// src")
        (tmp_path / "results" / "Slither").mkdir(parents=True)
        (tmp_path / "results" / "Slither" / "src.sol").write_text("// src")
        (tmp_path / "results" / "Mythril").mkdir(parents=True)
        (tmp_path / "results" / "Mythril" / "src.sol").write_text("// src")

        connector = GitConnector()
        all_found = connector.find_sol_files(tmp_path)
        assert len(all_found) == 3  # all 3 copies

        scoped = connector.find_sol_files(tmp_path, include_subdirs=["buggy_contracts"])
        assert len(scoped) == 1
        assert "buggy_contracts" in str(scoped[0])

    def test_exclude_subdirs_blocklist(self, tmp_path):
        """exclude_subdirs blocks the listed top-level subdirs from the search."""
        (tmp_path / "src.sol").write_text("// src")
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "analysis.sol").write_text("// analysis")

        connector = GitConnector()
        all_found = connector.find_sol_files(tmp_path)
        assert len(all_found) == 2

        filtered = connector.find_sol_files(tmp_path, exclude_subdirs=["results"])
        assert len(filtered) == 1
        assert "src.sol" in str(filtered[0])

    def test_source_config_carries_subdirs(self):
        """SourceConfig + _source_config flow include_subdirs / exclude_subdirs."""
        from sentinel_data.ingestion.ingest import _source_config
        entry = {
            "connector": "git",
            "url": "https://example.com",
            "pin": "abc",
            "include_subdirs": ["buggy_contracts"],
            "exclude_subdirs": ["results"],
        }
        cfg = _source_config("solidifi", entry)
        assert cfg.include_subdirs == ["buggy_contracts"]
        assert cfg.exclude_subdirs == ["results"]
        # Make sure include_subdirs is NOT in extra (would be a leak)
        assert "include_subdirs" not in cfg.extra
        assert "exclude_subdirs" not in cfg.extra


# ── ManualConnector (added 2026-06-10 for DIVE / manual sources) ──────────────

class TestManualConnector:
    """Tests for the manual-source connector (DIVE, SmartBugs Wild, Zenodo, etc.).

    Regression context: DIVE is distributed as Nature Sci. Data zip downloads,
    not a git repo. SmartBugs Wild is a tarball, FORGE is a Zenodo record.
    """

    def test_missing_staging_path_raises(self, tmp_path):
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        c = ManualConnector()
        cfg = SourceConfig(
            name="dive", connector="manual",
            url="", pin="v1", description="",
            extra={},  # no staging_path
        )
        with pytest.raises(ConnectorError, match="staging_path"):
            c._pull(cfg, tmp_path)

    def test_symlink_staging_dir(self, tmp_path):
        """materialize='symlink' creates a symlink from dest/repo → staging dir."""
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        # Create staging with .sol files
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "a.sol").write_text("pragma solidity ^0.8.0;\n")
        (staging / "b.sol").write_text("pragma solidity ^0.8.0;\n")
        # Add noise file
        (staging / "README.md").write_text("# readme")
        # Set up dest
        dest = tmp_path / "dest"
        cfg = SourceConfig(
            name="dive", connector="manual", url="", pin="v1", description="",
            extra={"staging_path": str(staging), "materialize": "symlink"},
        )
        c = ManualConnector()
        result = c._pull(cfg, dest)
        assert result.resolved_pin == "v1"
        assert len(result.sol_files) == 2
        # The symlink must resolve back to staging
        repo = dest / "repo"
        assert repo.is_symlink()
        assert (repo / "a.sol").read_text().startswith("pragma")

    def test_copy_staging_dir(self, tmp_path):
        """materialize='copy' duplicates the staging tree into dest/repo/."""
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "a.sol").write_text("pragma solidity ^0.8.0;\n")
        dest = tmp_path / "dest"
        cfg = SourceConfig(
            name="dive", connector="manual", url="", pin="v1", description="",
            extra={"staging_path": str(staging), "materialize": "copy"},
        )
        c = ManualConnector()
        c._pull(cfg, dest)
        repo = dest / "repo"
        assert not repo.is_symlink()
        assert (repo / "a.sol").exists()

    def test_zip_extraction_strips_macos_metadata(self, tmp_path):
        """The DIVE zip was full of __MACOSX/ and .DS_Store noise. Strips them.

        Regression for the DIVE integration test (2026-06-10): 44,687 files
        in the zip but only 22,332 were actual .sol; the rest were macOS
        resource forks. Without the strip, find_sol_files would report 44K
        paths, all the actual .sol's duplicates.
        """
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        # Build a fake zip with macOS noise
        import zipfile
        z = tmp_path / "fake.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("Raw/a.sol", "// a")
            zf.writestr("Raw/b.sol", "// b")
            zf.writestr("__MACOSX/Raw/._a.sol", "macos noise")
            zf.writestr("Raw/.DS_Store", "macos noise")
        dest = tmp_path / "dest"
        cfg = SourceConfig(
            name="dive", connector="manual", url="", pin="v1", description="",
            extra={"staging_path": str(z), "materialize": "symlink"},
        )
        c = ManualConnector()
        result = c._pull(cfg, dest)
        # 2 .sol files found, none of the macOS noise
        sol_paths = [str(p) for p in result.sol_files]
        assert len(result.sol_files) == 2
        assert all(".DS_Store" not in p for p in sol_paths)
        assert all("__MACOSX" not in p for p in sol_paths)

    def test_nonexistent_staging_raises(self, tmp_path):
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        c = ManualConnector()
        cfg = SourceConfig(
            name="dive", connector="manual", url="", pin="v1", description="",
            extra={"staging_path": "/nonexistent/path/that/does/not/exist"},
        )
        with pytest.raises(ConnectorError, match="does not exist"):
            c._pull(cfg, tmp_path / "dest")

    def test_bad_materialize_mode_raises(self, tmp_path):
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        staging = tmp_path / "staging"
        staging.mkdir()
        c = ManualConnector()
        cfg = SourceConfig(
            name="dive", connector="manual", url="", pin="v1", description="",
            extra={"staging_path": str(staging), "materialize": "magic"},
        )
        with pytest.raises(ConnectorError, match="materialize"):
            c._pull(cfg, tmp_path / "dest")

    def test_glob_staging_resolves_single_match(self, tmp_path):
        """Glob patterns in staging_path are resolved to a single directory."""
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        staging = tmp_path / "my_staging"
        staging.mkdir()
        (staging / "a.sol").write_text("// a")
        # Use a glob
        glob_str = str(tmp_path / "my_stag*")
        dest = tmp_path / "dest"
        cfg = SourceConfig(
            name="dive", connector="manual", url="", pin="v1", description="",
            extra={"staging_path": glob_str},
        )
        c = ManualConnector()
        result = c._pull(cfg, dest)
        assert len(result.sol_files) == 1

    def test_include_subdirs_applies_to_manual_source(self, tmp_path):
        """include_subdirs flows through manual connector like git connector."""
        from sentinel_data.ingestion.connectors.manual_connector import ManualConnector
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "real").mkdir()
        (staging / "real" / "x.sol").write_text("// x")
        (staging / "noise").mkdir()
        (staging / "noise" / "x.sol").write_text("// x")
        dest = tmp_path / "dest"
        cfg = SourceConfig(
            name="dive", connector="manual", url="", pin="v1", description="",
            extra={"staging_path": str(staging)},
            include_subdirs=["real"],
        )
        c = ManualConnector()
        result = c._pull(cfg, dest)
        assert len(result.sol_files) == 1
        assert "real" in str(result.sol_files[0])
