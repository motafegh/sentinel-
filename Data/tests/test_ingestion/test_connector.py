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
