"""Tests for the 10 per-class pattern YAMLs (Stage 4)."""
from pathlib import Path

import pytest
import yaml

_PATTERNS_DIR = Path("Data/sentinel_data/verification/patterns")
_EXPECTED_CLASSES = {
    "Reentrancy", "CallToUnknown", "Timestamp", "IntegerUO",
    "UnusedReturn", "MishandledException", "ExternalBug",
    "DenialOfService", "GasException", "TransactionOrderDependence",
}
_REQUIRED_KEYS = {"class", "description", "v9_signal", "positive_example", "negative_example"}


class TestPatternYAMLs:
    def test_all_10_patterns_exist(self):
        existing = {f.stem for f in _PATTERNS_DIR.glob("*.yaml")}
        assert existing == _EXPECTED_CLASSES, (
            f"Missing: {_EXPECTED_CLASSES - existing}, Extra: {existing - _EXPECTED_CLASSES}"
        )

    @pytest.mark.parametrize("cls", sorted(_EXPECTED_CLASSES))
    def test_pattern_loads_as_valid_yaml(self, cls):
        path = _PATTERNS_DIR / f"{cls}.yaml"
        if not path.exists():
            pytest.skip(f"Pattern file missing: {cls}.yaml")
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)

    @pytest.mark.parametrize("cls", sorted(_EXPECTED_CLASSES))
    def test_pattern_has_required_keys(self, cls):
        path = _PATTERNS_DIR / f"{cls}.yaml"
        if not path.exists():
            pytest.skip(f"Pattern file missing: {cls}.yaml")
        data = yaml.safe_load(path.read_text())
        missing = _REQUIRED_KEYS - set(data.keys())
        assert not missing, f"{cls}.yaml missing keys: {missing}"

    @pytest.mark.parametrize("cls", sorted(_EXPECTED_CLASSES))
    def test_class_field_matches_filename(self, cls):
        path = _PATTERNS_DIR / f"{cls}.yaml"
        if not path.exists():
            pytest.skip(f"Pattern file missing: {cls}.yaml")
        data = yaml.safe_load(path.read_text())
        assert data.get("class") == cls

    @pytest.mark.parametrize("cls", sorted(_EXPECTED_CLASSES))
    def test_v9_signal_has_method(self, cls):
        path = _PATTERNS_DIR / f"{cls}.yaml"
        if not path.exists():
            pytest.skip(f"Pattern file missing: {cls}.yaml")
        data = yaml.safe_load(path.read_text())
        signal = data.get("v9_signal", {})
        assert "method" in signal, f"{cls}: v9_signal missing 'method'"

    def test_not_extractable_classes(self):
        """DoS, GasException, TOD have no v9 feature — must be NOT_EXTRACTABLE."""
        not_extractable = ["DenialOfService", "GasException", "TransactionOrderDependence"]
        for cls in not_extractable:
            path = _PATTERNS_DIR / f"{cls}.yaml"
            if not path.exists():
                continue
            data = yaml.safe_load(path.read_text())
            assert data["v9_signal"]["method"] == "NOT_EXTRACTABLE", (
                f"{cls} should be NOT_EXTRACTABLE but got {data['v9_signal']['method']}"
            )
