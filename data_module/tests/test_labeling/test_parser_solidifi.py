"""Tests for the SolidiFI label parser (Task 3.5)."""
import json
import tempfile
from pathlib import Path

import pytest

from sentinel_data.labeling.parsers.solidifi import label_source, _extract_folder
from sentinel_data.labeling.schema import class_names

_PREP_DIR = Path("data_module/data/preprocessed/solidifi")
_DATA_DIR = Path("data_module/data")


def _skip_if_no_data():
    if not _PREP_DIR.exists():
        pytest.skip("SolidiFI preprocessed data not found")


class TestExtractFolder:
    def test_standard_path(self):
        assert _extract_folder("repo/buggy_contracts/Re-entrancy/buggy_3.sol") == "Re-entrancy"

    def test_tx_origin(self):
        assert _extract_folder("repo/buggy_contracts/tx.origin/buggy_1.sol") == "tx.origin"

    def test_overflow_underflow(self):
        assert _extract_folder("repo/buggy_contracts/Overflow-Underflow/buggy_5.sol") == "Overflow-Underflow"

    def test_unknown_structure_returns_none(self):
        assert _extract_folder("repo/__source__/12345.sol") is None

    def test_too_short_path_returns_none(self):
        assert _extract_folder("buggy_1.sol") is None


class TestLabelSourceSolidiFI:
    def test_smoke_5_contracts(self, tmp_path):
        _skip_if_no_data()
        result = label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        assert result.contracts_seen == 5
        assert result.labels_written == 5
        assert result.labels_failed == 0
        assert result.labels_cached == 0

    def test_labels_json_structure(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=1, output_dir=tmp_path)

        label_files = list(tmp_path.glob("*.labels.json"))
        assert len(label_files) == 1

        lj = json.loads(label_files[0].read_text())
        assert "sha256" in lj
        assert "source" in lj and lj["source"] == "solidifi"
        assert "injection_folder" in lj
        assert "classes" in lj
        assert "n_pos" in lj and lj["n_pos"] == 1

        # All 10 canonical classes present
        assert set(lj["classes"].keys()) == set(class_names())

    def test_exactly_one_positive_class(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=10, output_dir=tmp_path)
        for lf in tmp_path.glob("*.labels.json"):
            lj = json.loads(lf.read_text())
            positives = [c for c, v in lj["classes"].items() if v["value"] == 1]
            assert len(positives) == 1, f"{lf.name}: expected 1 positive, got {positives}"

    def test_positive_class_has_T0_tier(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=10, output_dir=tmp_path)
        for lf in tmp_path.glob("*.labels.json"):
            lj = json.loads(lf.read_text())
            for cls, entry in lj["classes"].items():
                if entry["value"] == 1:
                    assert entry["tier"] == "T0", f"{lf.name}: positive class {cls} has tier {entry['tier']}"
                else:
                    assert entry["tier"] is None

    def test_cache_hit_on_second_run(self, tmp_path):
        _skip_if_no_data()
        r1 = label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        assert r1.labels_written == 5
        r2 = label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        assert r2.labels_written == 0
        assert r2.labels_cached == 5

    def test_force_overwrites_cache(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=3, output_dir=tmp_path)
        r2 = label_source(_DATA_DIR, limit=3, output_dir=tmp_path, force=True)
        assert r2.labels_written == 3
        assert r2.labels_cached == 0

    def test_full_run_matches_expected_counts(self, tmp_path):
        """Full 283-contract run: ~276 written, ~7 failed (buggy_35.sol artifacts)."""
        _skip_if_no_data()
        result = label_source(_DATA_DIR, output_dir=tmp_path)
        assert result.contracts_seen == 283
        # 7 buggy_35.sol files have lone '/' — they preprocess but have valid meta
        # The parser should handle all 283 (folder extraction always works)
        assert result.labels_failed == 0
        assert result.labels_written == 283

    def test_sha256_filename_matches_content(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        for lf in tmp_path.glob("*.labels.json"):
            lj = json.loads(lf.read_text())
            file_sha = lf.name.removesuffix(".labels.json")
            assert file_sha == lj["sha256"], f"Filename {file_sha} != sha256 {lj['sha256']}"
