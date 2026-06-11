"""Tests for the DIVE label parser (Task 3.6)."""
import json
from pathlib import Path

import pytest

from sentinel_data.labeling.parsers.dive import label_source, _build_folder_index, _load_crosswalk
from sentinel_data.labeling.schema import class_names

_DATA_DIR = Path("Data/data")
_RAW_REPO = _DATA_DIR / "raw" / "dive" / "repo"
_PREP_DIR = _DATA_DIR / "preprocessed" / "dive"


def _skip_if_no_data():
    if not _PREP_DIR.exists():
        pytest.skip("DIVE preprocessed data not found")


class TestFolderIndex:
    def test_builds_without_error(self):
        if not _RAW_REPO.exists():
            pytest.skip("DIVE raw repo not found")
        cw = _load_crosswalk()
        index = _build_folder_index(_RAW_REPO, cw["class_map"])
        assert len(index) > 0

    def test_bad_randomness_not_in_index(self):
        if not _RAW_REPO.exists():
            pytest.skip("DIVE raw repo not found")
        cw = _load_crosswalk()
        index = _build_folder_index(_RAW_REPO, cw["class_map"])
        # No file should have BadRandomness mapped (it's dropped from class_map)
        for filename, classes in index.items():
            assert "BadRandomness" not in classes

    def test_multi_label_files_exist(self):
        if not _RAW_REPO.exists():
            pytest.skip("DIVE raw repo not found")
        cw = _load_crosswalk()
        index = _build_folder_index(_RAW_REPO, cw["class_map"])
        multi = [f for f, c in index.items() if len(c) > 1]
        assert len(multi) > 0, "Expected multi-label files in DIVE"

    def test_nonvulnerable_files_not_in_index(self):
        """Files in __source__ but no folder have no entry in the index."""
        if not _RAW_REPO.exists():
            pytest.skip("DIVE raw repo not found")
        cw = _load_crosswalk()
        index = _build_folder_index(_RAW_REPO, cw["class_map"])
        source_files = set(s.name for s in (_RAW_REPO / "__source__").glob("*.sol"))
        nonvulnerable = source_files - set(index.keys())
        assert len(nonvulnerable) > 0, "Expected some NonVulnerable files in DIVE"


class TestLabelSourceDIVE:
    def test_smoke_5_contracts(self, tmp_path):
        _skip_if_no_data()
        result = label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        assert result.contracts_seen == 5
        assert result.labels_written == 5
        assert result.labels_failed == 0

    def test_labels_json_structure(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=1, output_dir=tmp_path)
        lf = next(tmp_path.glob("*.labels.json"))
        lj = json.loads(lf.read_text())

        assert "sha256" in lj
        assert lj["source"] == "dive"
        assert "source_filename" in lj
        assert "classes" in lj
        assert "n_pos" in lj
        assert set(lj["classes"].keys()) == set(class_names())

    def test_all_classes_have_value_and_tier(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=20, output_dir=tmp_path)
        for lf in tmp_path.glob("*.labels.json"):
            lj = json.loads(lf.read_text())
            for cls, entry in lj["classes"].items():
                assert "value" in entry and entry["value"] in (0, 1)
                assert "tier" in entry
                if entry["value"] == 1:
                    assert entry["tier"] == "T2"
                else:
                    assert entry["tier"] is None

    def test_n_pos_matches_positive_count(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=50, output_dir=tmp_path)
        for lf in tmp_path.glob("*.labels.json"):
            lj = json.loads(lf.read_text())
            positives = sum(v["value"] for v in lj["classes"].values())
            assert lj["n_pos"] == positives

    def test_nonvulnerable_contracts_have_n_pos_zero(self, tmp_path):
        _skip_if_no_data()
        result = label_source(_DATA_DIR, limit=100, output_dir=tmp_path)
        nonvuln = [
            lf for lf in tmp_path.glob("*.labels.json")
            if json.loads(lf.read_text())["n_pos"] == 0
        ]
        assert result.nonvulnerable_written == len(nonvuln)

    def test_bad_randomness_never_appears_as_class(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=50, output_dir=tmp_path)
        for lf in tmp_path.glob("*.labels.json"):
            lj = json.loads(lf.read_text())
            assert "BadRandomness" not in lj["classes"]
            assert "Bad Randomness" not in lj["classes"]

    def test_cache_hit_on_second_run(self, tmp_path):
        _skip_if_no_data()
        r1 = label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        assert r1.labels_written == 5
        r2 = label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        assert r2.labels_written == 0
        assert r2.labels_cached == 5

    def test_sha256_filename_matches_content(self, tmp_path):
        _skip_if_no_data()
        label_source(_DATA_DIR, limit=5, output_dir=tmp_path)
        for lf in tmp_path.glob("*.labels.json"):
            lj = json.loads(lf.read_text())
            file_sha = lf.name.removesuffix(".labels.json")
            assert file_sha == lj["sha256"]
