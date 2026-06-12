"""Tests for the v2 representation orchestrator."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from sentinel_data.representation.orchestrator import _extract_one, represent_source


_MODULE_ROOT = Path(__file__).resolve().parents[2]  # data_module/


@pytest.fixture(scope="module")
def test_config():
    """Load test config pointing to solidifi preprocessed data."""
    with open(_MODULE_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def temp_data_dir(tmp_path: Path):
    """Create a temporary data directory with symlinked preprocessed solidifi."""
    data_dir = tmp_path / "data"
    preproc_src = _MODULE_ROOT / "data" / "preprocessed" / "solidifi"
    preproc_dest = data_dir / "preprocessed" / "solidifi"
    preproc_dest.parent.mkdir(parents=True, exist_ok=True)
    preproc_dest.symlink_to(preproc_src, target_is_directory=True)
    return data_dir


@pytest.fixture
def temp_output_dir():
    """Temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestOrchestrator:
    """Test the v2 representation orchestrator."""

    def test_represent_source_works(self, test_config, temp_data_dir):
        """Smoke test: orchestrator runs on 5 contracts without errors in a fresh dir."""
        import torch
        result = represent_source("solidifi", test_config, temp_data_dir, limit=5)
        assert result.contracts_seen == 5
        assert result.graphs_written == 5
        assert result.graphs_failed == 0
        assert result.tokens_written == 5
        assert result.tokens_failed == 0
        assert result.schema_version == "v9"
        assert result.extractor_version.startswith("v2.")

        # Verify token shape: [4, 512] (windowed graphcodebert, not (512,) codebert)
        out_dir = temp_data_dir / "representations" / "solidifi"
        token_files = list(out_dir.glob("*.tokens.pt"))
        assert len(token_files) >= 1
        tok = torch.load(token_files[0], weights_only=False)
        assert tok["input_ids"].shape == (4, 512), (
            f"Expected [4, 512] windowed tokens, got {tok['input_ids'].shape}"
        )
        assert tok["attention_mask"].shape == (4, 512)
        assert "sha256" in tok
        assert "num_windows" in tok

    def test_represent_source_cache_hit(self, test_config, data_dir, temp_output_dir):
        """Second run should be cache hit (0 written, all cached)."""
        r1 = represent_source("solidifi", test_config, data_dir, limit=3, output_dir=temp_output_dir)
        assert r1.graphs_written == 3
        assert r1.graphs_cached == 0

        r2 = represent_source("solidifi", test_config, data_dir, limit=3, output_dir=temp_output_dir)
        assert r2.graphs_written == 0
        assert r2.graphs_cached == 3
        assert r2.tokens_written == 0
        assert r2.tokens_cached == 3

    def test_force_flag_bypasses_cache(self, test_config, data_dir, temp_output_dir):
        """force=True should bypass cache and re-extract."""
        r1 = represent_source("solidifi", test_config, data_dir, limit=2, output_dir=temp_output_dir)
        assert r1.graphs_written == 2

        r2 = represent_source("solidifi", test_config, data_dir, limit=2, output_dir=temp_output_dir, force=True)
        assert r2.graphs_written == 2
        assert r2.graphs_cached == 0

    def test_idempotency(self, test_config, data_dir, tmp_path):
        """Multiple runs produce identical output files."""
        out_dir = tmp_path / "idempotency_out"
        r1 = represent_source("solidifi", test_config, data_dir, limit=2, output_dir=out_dir)
        r2 = represent_source("solidifi", test_config, data_dir, limit=2, output_dir=out_dir)

        rep_files = sorted(out_dir.glob("*.rep.json"))
        assert len(rep_files) >= 2

        with open(rep_files[0]) as f:
            j1 = json.load(f)
        with open(rep_files[1]) as f:
            j2 = json.load(f)

        assert j1["schema_version"] == j2["schema_version"] == "v9"
        assert j1["extractor_version"] == j2["extractor_version"]


class TestExtractOne:
    """Test the internal _extract_one function."""

    def test_extract_one_returns_correct_counts(self, data_dir, temp_output_dir):
        """_extract_one returns (g_written, g_cached, t_written, t_cached)."""
        prep_dir = data_dir / "preprocessed" / "solidifi"
        sol_files = list(prep_dir.glob("*.sol"))
        assert len(sol_files) > 0

        sol_path = sol_files[0]
        meta_path = prep_dir / f"{sol_path.stem}.meta.json"
        meta = json.loads(meta_path.read_text())
        sha256 = meta["sha256"]

        g_w, g_c, t_w, t_c = _extract_one(
            source="solidifi",
            sol_path=sol_path,
            sha256=sha256,
            output_dir=temp_output_dir,
            force=False,
        )
        assert g_w == 1
        assert g_c == 0
        assert t_w == 1
        assert t_c == 0

        g_w2, g_c2, t_w2, t_c2 = _extract_one(
            source="solidifi",
            sol_path=sol_path,
            sha256=sha256,
            output_dir=temp_output_dir,
            force=False,
        )
        assert g_w2 == 0
        assert g_c2 == 1
        assert t_w2 == 0
        assert t_c2 == 1


class TestOutputFiles:
    """Test that output files are created correctly."""

    def test_three_files_per_contract(self, test_config, data_dir, temp_output_dir):
        """Each successful contract produces .pt, .tokens.pt, .rep.json"""
        result = represent_source("solidifi", test_config, data_dir, limit=3, output_dir=temp_output_dir)

        graph_pts = sorted(temp_output_dir.glob("*.pt"))
        graph_files = [p for p in graph_pts if not p.name.endswith(".tokens.pt")]
        assert len(graph_files) >= 3
        for gf in graph_files[:3]:
            sha = gf.stem
            for ext in (".pt", ".tokens.pt", ".rep.json"):
                p = temp_output_dir / f"{sha}{ext}"
                assert p.exists(), f"Missing {p.name}"
                assert p.stat().st_size > 0, f"Empty {p.name}"

    def test_rep_json_structure(self, test_config, data_dir, tmp_path):
        """Verify .rep.json has expected structure."""
        out_dir = tmp_path / "rep_json_out"
        result = represent_source("solidifi", test_config, data_dir, limit=1, output_dir=out_dir)

        rep_files = list(out_dir.glob("*.rep.json"))
        assert len(rep_files) >= 1

        with open(rep_files[0]) as f:
            rep = json.load(f)

        assert "sha256" in rep
        assert "source" in rep
        assert rep["source"] == "solidifi"
        assert "original_path" in rep
        assert "schema_version" in rep
        assert rep["schema_version"] == "v9"
        assert "extractor_version" in rep
        assert "node_count" in rep
        assert "edge_count" in rep
        assert "compute_time_ms" in rep
        assert "pragma" in rep
        assert "solc_version" in rep
        assert "window_count" in rep
        assert 1 <= rep["window_count"] <= 4  # windowed tokenizer: 1–4 real windows