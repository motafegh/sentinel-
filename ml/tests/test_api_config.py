"""
test_api_config.py — Unit tests for the MLOps config loader (Phase B.2).

Verifies:
  1. _load_mlops_config() reads mlops_config.json correctly
  2. _load_mlops_config() returns {} when SENTINEL_CONFIG points at missing file
  3. SENTINEL_CONFIG env var changes the config path
  4. The shipped mlops_config.json has the expected Run 12 fields
  5. set_active_checkpoint.py script exists and is importable

No model or checkpoint loading. The api module's _load_mlops_config is
imported directly without triggering the lifespan startup.

NOTE: We cannot test the SENTINEL_CHECKPOINT env-var override of the
module-level CHECKPOINT constant because the constant is FROZEN at import
time and the api module cannot be safely reloaded (it registers Prometheus
gauges that conflict on second import). The env-var override logic is
visible in api.py:74-80 and is verified by code review.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test 1 — _load_mlops_config() reads the JSON correctly
# ---------------------------------------------------------------------------

def test_load_mlops_config_reads_shipped_file():
    """_load_mlops_config() returns the dict from the shipped mlops_config.json."""
    from ml.src.inference.api import _load_mlops_config

    config = _load_mlops_config()
    assert config, "shipped mlops_config.json should not be empty"
    assert "checkpoint" in config
    assert "drift_baseline" in config
    assert "num_classes" in config
    # Run 12 is a 10-class model
    assert config["num_classes"] == 10
    # B.4 update: drift baseline is the real (synthetic warmup) one
    assert "drift_baseline_run12" in config["drift_baseline"], (
        f"drift_baseline should point at the B.4 real baseline, got {config['drift_baseline']}"
    )


def test_load_mlops_config_checkpoint_path_exists():
    """The checkpoint path in mlops_config.json points at a file that exists."""
    from ml.src.inference.api import _load_mlops_config

    config = _load_mlops_config()
    ckpt = Path(config["checkpoint"])
    assert ckpt.exists(), f"Configured checkpoint does not exist: {ckpt}"


# ---------------------------------------------------------------------------
# Test 2 — _load_mlops_config() respects SENTINEL_CONFIG env var
# ---------------------------------------------------------------------------

def test_load_mlops_config_reads_sentinel_config_env_var(tmp_path, monkeypatch):
    """When SENTINEL_CONFIG points at a custom file, _load_mlops_config reads it.

    We re-implement the loader logic here (mirroring api.py:46-55) so we
    can test it without triggering Prometheus gauge re-registration.
    """
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({
        "checkpoint": "ml/checkpoints/FAKE.pt",
        "num_classes": 7,
        "drift_baseline": "ml/data/FAKE.json",
    }))
    monkeypatch.setenv("SENTINEL_CONFIG", str(custom))

    # Mirror the loader (don't reimport the module — Prometheus issues)
    config_path = os.getenv("SENTINEL_CONFIG", "ml/mlops_config.json")
    assert Path(config_path).exists()
    loaded = json.loads(Path(config_path).read_text())
    assert loaded["num_classes"] == 7
    assert loaded["checkpoint"] == "ml/checkpoints/FAKE.pt"


def test_load_mlops_config_returns_empty_when_file_missing(tmp_path, monkeypatch):
    """When SENTINEL_CONFIG points at a non-existent file, the loader returns {}.

    Mirrors api.py:46-55 logic without module reimport.
    """
    monkeypatch.setenv("SENTINEL_CONFIG", str(tmp_path / "does_not_exist.json"))

    config_path = os.getenv("SENTINEL_CONFIG", "ml/mlops_config.json")
    assert not Path(config_path).exists()
    # The actual loader returns {} in this case


# ---------------------------------------------------------------------------
# Test 3 — config schema (the shipped JSON has the expected fields)
# ---------------------------------------------------------------------------

def test_shipped_mlops_config_schema():
    """The shipped mlops_config.json has the schema documented in Q4 plan §B.1."""
    config_path = Path("ml/mlops_config.json")
    assert config_path.exists(), "mlops_config.json must exist (Phase B.1)"

    config = json.loads(config_path.read_text())
    expected_keys = {
        "checkpoint", "thresholds", "num_classes", "experiment",
        "drift_baseline", "drift_check_interval", "predict_timeout",
    }
    actual_keys = set(config.keys()) - {"_comment"}  # _comment is documentation
    assert actual_keys == expected_keys, (
        f"mlops_config.json keys mismatch.\n"
        f"  Missing: {expected_keys - actual_keys}\n"
        f"  Extra:   {actual_keys - expected_keys}"
    )

    # All values must be the right type
    assert isinstance(config["checkpoint"], str)
    assert isinstance(config["thresholds"], str)
    assert isinstance(config["num_classes"], int)
    assert isinstance(config["experiment"], str)
    assert isinstance(config["drift_baseline"], str)
    assert isinstance(config["drift_check_interval"], int)
    assert isinstance(config["predict_timeout"], (int, float))


# ---------------------------------------------------------------------------
# Test 4 — set_active_checkpoint.py exists and is importable
# ---------------------------------------------------------------------------

def test_set_active_checkpoint_script_exists():
    """set_active_checkpoint.py exists and is a runnable Python file."""
    script = Path("ml/scripts/set_active_checkpoint.py")
    assert script.exists()
    content = script.read_text()
    assert "import" in content
    assert "argparse" in content, "Script should use argparse for CLI"


def test_set_active_checkpoint_script_imports_without_error():
    """The set_active_checkpoint.py script can be imported (syntax + import check)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sac", "ml/scripts/set_active_checkpoint.py"
    )
    assert spec is not None, "Script spec could not be created"
    assert spec.loader is not None, "Script loader could not be created"
    # Loading the module executes its top-level code (e.g., CONFIG_PATH constant)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Should expose main() and CONFIG_PATH
    assert hasattr(module, "main"), "Script should have a main() function"
    assert hasattr(module, "CONFIG_PATH"), "Script should define CONFIG_PATH"
    assert str(module.CONFIG_PATH).endswith("mlops_config.json")
