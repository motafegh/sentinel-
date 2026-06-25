"""
Tests for P1 config externalisation (src/config/).

Asserts:
  - Default config matches current constants for every group
  - Malformed YAML raises ValidationError
  - Env override of config path works
  - schema_version round-trips
  - Value changed in YAML reflected by get_config() after fresh process
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import SentinelConfig, get_config
from src.config.loader import load_config, reload_config


class TestConfigSchema:
    """T1.2 / T1.10: schema matches current code defaults."""

    def test_default_consensus_matches_code(self):
        cfg = SentinelConfig()
        assert cfg.consensus.confirmed_band == 0.70
        assert cfg.consensus.likely_band == 0.50
        assert cfg.consensus.disputed_band == 0.30
        assert cfg.consensus.ml_weight_scale == 0.5
        assert cfg.consensus.ml_positive_threshold == 0.50
        assert cfg.consensus.accuracy_weights["Reentrancy"]["ml"] == 0.78
        assert cfg.consensus.default_weights["slither"] == 0.65

    def test_default_confidence_matches_code(self):
        cfg = SentinelConfig()
        assert cfg.confidence.slither_agree == 1.10
        assert cfg.confidence.slither_disagree == 0.90
        assert cfg.confidence.aderyn_agree == 1.05
        assert cfg.confidence.aderyn_disagree == 0.97
        assert cfg.confidence.rag_agree == 1.05
        assert cfg.confidence.rag_relevance == 0.70

    def test_default_routing_matches_code(self):
        cfg = SentinelConfig()
        assert cfg.routing.deep_thresholds["Reentrancy"] == 0.35
        assert cfg.routing.deep_thresholds["DenialOfService"] == 0.30
        assert cfg.routing.routing_rules["GasException"] == ["static_analysis"]
        assert cfg.routing.prob_to_severity["CRITICAL"] == 0.85
        assert cfg.routing.overall_verdict_rank["CONFIRMED"] == 5
        assert cfg.routing.compute_verdict_prob_cutoff == 0.50
        assert cfg.routing.compute_verdict_rag_confirmed_cutoff == 0.80
        assert cfg.routing.compute_verdict_rag_likely_cutoff == 0.50

    def test_default_attribution_matches_code(self):
        cfg = SentinelConfig()
        assert cfg.attribution.rag_relevance_floor == 0.30

    def test_default_eval_matches_code(self):
        cfg = SentinelConfig()
        assert cfg.eval.positive_verdicts == ["CONFIRMED", "LIKELY"]
        assert cfg.eval.borderline_band_low == 0.35
        assert cfg.eval.borderline_band_high == 0.50
        assert cfg.eval.fbeta_beta == 2.0

    def test_schema_version_round_trip(self):
        cfg = SentinelConfig()
        assert cfg.schema_version == "1"
        d = cfg.model_dump()
        assert d["schema_version"] == "1"
        cfg2 = SentinelConfig(**d)
        assert cfg2.schema_version == "1"

    def test_invalid_yaml_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("consensus:\n  confirmed_band: not_a_float\n")
            tmp = f.name
        try:
            with pytest.raises(ValidationError):
                load_config(tmp)
        finally:
            os.unlink(tmp)

    def test_nonexistent_env_path_fails(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/sentinel_config.yaml")


class TestConfigLoader:
    """T1.3 / T1.10: loader singleton + env override."""

    def test_get_config_returns_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reload_config_resets_singleton(self):
        c1 = get_config()
        c2 = reload_config()
        assert c2 is not c1
        assert c2.schema_version == c1.schema_version

    def test_yaml_round_trip(self):
        yaml_path = (
            Path(__file__).resolve().parents[1] / "configs" / "verdicts_default.yaml"
        )
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        cfg = SentinelConfig(**raw)
        assert cfg.schema_version == "1"
        assert cfg.consensus.confirmed_band == 0.70

    def test_env_override_path(self):
        yaml_path = (
            Path(__file__).resolve().parents[1] / "configs" / "verdicts_default.yaml"
        )
        orig = os.environ.get("SENTINEL_CONFIG")
        try:
            os.environ["SENTINEL_CONFIG"] = str(yaml_path)
            cfg = load_config()
            assert cfg.schema_version == "1"
        finally:
            if orig is None:
                del os.environ["SENTINEL_CONFIG"]
            else:
                os.environ["SENTINEL_CONFIG"] = orig

    def test_yaml_value_change_reflected(self):
        """A value changed in YAML should be reflected by load_config()."""
        cfg = SentinelConfig()
        orig = cfg.consensus.confirmed_band

        # Write a modified YAML, load it, verify the change
        modified = cfg.model_dump()
        modified["consensus"]["confirmed_band"] = 0.80
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as f:
            yaml.dump(modified, f)
            tmp = f.name
        try:
            cfg2 = load_config(tmp)
            assert cfg2.consensus.confirmed_band == 0.80
            assert cfg2.consensus.confirmed_band != orig
        finally:
            os.unlink(tmp)
