"""Tests for the Go/No-Go minimum-viable-corpus gate (Task 3.11)."""
import json
from pathlib import Path

import pytest
import yaml

from sentinel_data.labeling.gate import run_gate, GateResult
from sentinel_data.labeling.schema import class_names

_HERE = Path(__file__).resolve()
_DATA_DIR = _HERE.parents[2] / "data"
_CONFIG_PATH = _HERE.parents[2] / "config.yaml"


def _load_config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _skip_if_no_merged():
    if not (_DATA_DIR / "labels" / "merged").exists():
        pytest.skip("Merged labels not found — run merger first")


class TestGateReport:
    def test_runs_without_error(self):
        _skip_if_no_merged()
        cfg = _load_config()
        result = run_gate(_DATA_DIR, cfg)
        assert isinstance(result, GateResult)

    def test_str_representation_works(self):
        _skip_if_no_merged()
        cfg = _load_config()
        result = run_gate(_DATA_DIR, cfg)
        report = str(result)
        assert "Go/No-Go Gate Report" in report
        assert ("PASS" in report or "FAIL" in report)

    def test_all_10_classes_have_criteria(self):
        _skip_if_no_merged()
        cfg = _load_config()
        result = run_gate(_DATA_DIR, cfg)
        class_criteria = [c.name for c in result.criteria if c.name.startswith("class_")]
        for cls in class_names():
            assert f"class_{cls}" in class_criteria, f"Missing criterion for {cls}"

    def test_total_contracts_criterion_present(self):
        _skip_if_no_merged()
        cfg = _load_config()
        result = run_gate(_DATA_DIR, cfg)
        names = [c.name for c in result.criteria]
        assert "total_contracts" in names

    def test_empty_merged_dir_fails_gate(self, tmp_path):
        cfg = _load_config()
        fake_data = tmp_path
        (fake_data / "labels" / "merged").mkdir(parents=True)
        result = run_gate(fake_data, cfg)
        assert result.gate_passed is False

    def test_gate_result_for_current_corpus(self):
        """Gate PASSES with SolidiFI (283) + DIVE full (22,073) — 2026-06-11."""
        _skip_if_no_merged()
        cfg = _load_config()
        result = run_gate(_DATA_DIR, cfg)
        # With full DIVE (22,073) + SolidiFI (283), all blocking criteria pass.
        # Remaining gaps: GasException=0, MishandledException≈39, CallToUnknown≈39
        # (non-blocking minor classes — need SmartBugs Curated for coverage)
        total_criterion = next(c for c in result.criteria if c.name == "total_contracts")
        print(f"\nCurrent corpus: {total_criterion.actual} contracts "
              f"(threshold={total_criterion.threshold})")
        print(str(result))
        assert total_criterion.actual >= 22000
        assert result.gate_passed is True
