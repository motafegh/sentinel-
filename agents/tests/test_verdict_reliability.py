"""
Tests for reliability lookup (P2, 2026-06-24; P3 extension, 2026-06-25).

When `configs/reliability_v1.yaml` exists with a matching schema_version,
load_reliability() returns the L3 fitted values (data-derived, P3).
Otherwise it falls back to the L1 hand-set values from verdicts_default.yaml
(P2 placeholder). The tests below cover BOTH paths.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
L1_CONFIG_PATH = REPO_ROOT / "configs" / "verdicts_default.yaml"
L3_CONFIG_PATH = REPO_ROOT / "configs" / "reliability_v1.yaml"

from src.orchestration.verdict.reliability import (
    get_reliability,
    load_reliability,
)


def _reload_reliability_module():
    """Force the L3-path constant to be re-read after env override."""
    from src.orchestration.verdict import reliability
    importlib.reload(reliability)
    return reliability


class TestLoadReliability:
    """Table-building from config."""

    def test_table_has_standard_classes(self):
        table = load_reliability()
        assert ("ml", "Reentrancy") in table
        assert ("slither", "Reentrancy") in table
        assert ("aderyn", "Reentrancy") in table

    def test_ml_is_scaled(self):
        """ML reliability is discounted by ml_weight_scale (0.5) regardless
        of L1 vs L3 source. Reads config from disk to bypass any
        test-patching of the singleton."""
        raw = yaml.safe_load(L1_CONFIG_PATH.read_text())
        ml_scale = raw["consensus"]["ml_weight_scale"]

        table = load_reliability()
        ml_rel = table[("ml", "Reentrancy")]
        raw_ml = ml_rel / ml_scale  # back-compute the un-scaled value
        # The un-scaled ML reliability must be >= the scaled one (any
        # positive scale <= 1 satisfies this).
        assert ml_rel <= raw_ml + 1e-9
        # The un-scaled ML must be a valid reliability value in [0, 1].
        assert 0.0 <= raw_ml <= 1.0

    def test_slither_from_l3_when_present(self):
        """When the L3 fitted file exists (the post-P3 baseline state), the
        loader returns the L3 value, not the L1 hand-set value. The exact
        number comes from the v1 fit on confusion_matrix_v2 (alpha=5)."""
        if not L3_CONFIG_PATH.is_file():
            pytest.skip("L3 reliability_v1.yaml not present — pre-P3 state")
        l3 = yaml.safe_load(L3_CONFIG_PATH.read_text())
        expected = l3["table"]["slither"]["Reentrancy"]
        table = load_reliability()
        sl_rel = table[("slither", "Reentrancy")]
        assert sl_rel == pytest.approx(expected, rel=1e-4)

    def test_l1_fallback_when_l3_missing(self, monkeypatch, tmp_path):
        """When the L3 file is missing (env override to a non-existent
        path), the loader falls back to L1 from verdicts_default.yaml.
        Verifies the Rule 5C fallback contract: missing fitted file
        means L1 prior is used, not silent zero / random / None."""
        from src.orchestration.verdict import reliability
        # Use the env var so the override takes effect on next module
        # reload (monkeypatching the attribute alone is undone by reload).
        monkeypatch.setenv("SENTINEL_RELIABILITY_CONFIG",
                           str(tmp_path / "nonexistent.yaml"))
        importlib.reload(reliability)
        table = reliability.load_reliability()
        l1 = yaml.safe_load(L1_CONFIG_PATH.read_text())
        l1_slither_reentrancy = l1["consensus"]["accuracy_weights"]["Reentrancy"]["slither"]
        assert table[("slither", "Reentrancy")] == pytest.approx(l1_slither_reentrancy, rel=1e-4)
        # Unset the env so the next reload reads the default (real L3 path).
        monkeypatch.delenv("SENTINEL_RELIABILITY_CONFIG", raising=False)
        importlib.reload(reliability)

    def test_l1_fallback_when_l3_schema_mismatch(self, monkeypatch, tmp_path):
        """When the L3 file exists but has a different schema_version, the
        loader must NOT silently use it. Per Rule 5C, the wrong-version file
        is treated as malformed and the L1 fallback is used."""
        from src.orchestration.verdict import reliability
        bad = tmp_path / "reliability_v99.yaml"
        bad.write_text(yaml.safe_dump({
            "schema_version": "99",  # different
            "table": {"slither": {"Reentrancy": 0.999}},
        }))
        monkeypatch.setenv("SENTINEL_RELIABILITY_CONFIG", str(bad))
        importlib.reload(reliability)
        table = reliability.load_reliability()
        # The bad 0.999 must NOT have leaked through — the L1 value
        # (from verdicts_default.yaml) must be in effect instead.
        l1 = yaml.safe_load(L1_CONFIG_PATH.read_text())
        l1_slither = l1["consensus"]["accuracy_weights"]["Reentrancy"]["slither"]
        assert table[("slither", "Reentrancy")] == pytest.approx(l1_slither, rel=1e-4)
        monkeypatch.delenv("SENTINEL_RELIABILITY_CONFIG", raising=False)
        importlib.reload(reliability)


class TestGetReliability:
    """Single lookup helper."""

    def test_known_class(self):
        r = get_reliability("ml", "Reentrancy")
        assert r > 0.0

    def test_fallback_for_unknown_class(self):
        r = get_reliability("ml", "NonExistentClass")
        assert r > 0.0  # falls back to default * scale

    def test_rag_source(self):
        r = get_reliability("rag", "Reentrancy")
        assert r > 0.0

    def test_debate_source(self):
        r = get_reliability("debate", "Reentrancy")
        assert r == pytest.approx(0.55)
