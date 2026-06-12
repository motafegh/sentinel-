"""Tests for the multi-source label merger (Task 3.10)."""
import json
import tempfile
from pathlib import Path

import pytest

from sentinel_data.labeling.merger import run_merger, _merge_class_entries, _check_co_occurrence_flag
from sentinel_data.labeling.schema import class_names

_DATA_DIR = Path("data_module/data")
_LABELS_DIR = _DATA_DIR / "labels"


def _skip_if_no_labels():
    if not (_LABELS_DIR / "solidifi").exists():
        pytest.skip("SolidiFI labels not found — run parsers first")


# ── Unit tests for merge helpers ─────────────────────────────────────────────

class TestMergeClassEntries:
    def test_single_positive_wins(self):
        result = _merge_class_entries([("solidifi", {"value": 1, "tier": "T0"})])
        assert result["value"] == 1 and result["tier"] == "T0"

    def test_higher_confidence_positive_wins(self):
        result = _merge_class_entries([
            ("dive",     {"value": 1, "tier": "T2"}),
            ("solidifi", {"value": 1, "tier": "T0"}),
        ])
        assert result["tier"] == "T0" and result["source"] == "solidifi"

    def test_positive_beats_negative_same_tier(self):
        result = _merge_class_entries([
            ("dive",     {"value": 0, "tier": None}),
            ("solidifi", {"value": 1, "tier": "T0"}),
        ])
        assert result["value"] == 1

    def test_all_negative_returns_negative(self):
        result = _merge_class_entries([
            ("solidifi", {"value": 0, "tier": None}),
            ("dive",     {"value": 0, "tier": None}),
        ])
        assert result["value"] == 0


class TestCoOccurrenceFlag:
    def _make_classes(self, dos: int, reen: int) -> dict:
        return {
            cls: {"value": 1 if cls == "DenialOfService" and dos
                  else (1 if cls == "Reentrancy" and reen else 0), "tier": "T3"}
            for cls in class_names()
        }

    def test_no_flag_when_not_cooccurring(self):
        classes = self._make_classes(dos=1, reen=0)
        assert not _check_co_occurrence_flag(classes, ["some_t3_src"], {"some_t3_src": 0.9})

    def test_no_flag_for_high_confidence_source(self):
        # T2 source (DIVE) should NOT be flagged even at high co-occurrence rate
        classes = self._make_classes(dos=1, reen=1)
        assert not _check_co_occurrence_flag(classes, ["dive"], {"dive": 0.99})

    def test_no_flag_when_independently_attested(self):
        classes = self._make_classes(dos=1, reen=1)
        # Two sources → independent attestation → no flag
        assert not _check_co_occurrence_flag(
            classes, ["src_a", "src_b"], {"src_a": 0.99, "src_b": 0.99}
        )

    def test_flag_for_low_confidence_single_source_high_rate(self):
        classes = self._make_classes(dos=1, reen=1)
        # T3/T4 source with 80% co-occurrence rate → flag
        assert _check_co_occurrence_flag(
            classes, ["noisy_t3"], {"noisy_t3": 0.80}
        )


# ── Integration tests ─────────────────────────────────────────────────────────

class TestRunMerger:
    def test_smoke_solidifi_only(self, tmp_path):
        _skip_if_no_labels()
        result = run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path)
        assert result.contracts_merged == 283
        assert result.single_source == 283
        assert result.multi_source == 0
        assert result.failed == 0

    def test_merged_json_structure(self, tmp_path):
        _skip_if_no_labels()
        run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path)
        lf = next(tmp_path.glob("*.labels.json"))
        lj = json.loads(lf.read_text())

        assert "sha256" in lj
        assert "sources" in lj and isinstance(lj["sources"], list)
        assert "classes" in lj
        assert "n_pos" in lj
        assert "flags" in lj and isinstance(lj["flags"], list)
        assert set(lj["classes"].keys()) == set(class_names())

    def test_each_class_has_source_field(self, tmp_path):
        _skip_if_no_labels()
        run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path)
        for lf in list(tmp_path.glob("*.labels.json"))[:10]:
            lj = json.loads(lf.read_text())
            for cls, entry in lj["classes"].items():
                assert "source" in entry, f"{lf.name}: class {cls} missing 'source'"

    def test_n_pos_consistent_with_classes(self, tmp_path):
        _skip_if_no_labels()
        run_merger(_DATA_DIR, ["solidifi", "dive"], output_dir=tmp_path)
        for lf in list(tmp_path.glob("*.labels.json"))[:20]:
            lj = json.loads(lf.read_text())
            assert lj["n_pos"] == sum(v["value"] for v in lj["classes"].values())

    def test_no_cross_source_overlap_solidifi_dive(self, tmp_path):
        """SolidiFI and DIVE are disjoint — all contracts single-source."""
        if not (_LABELS_DIR / "dive").exists():
            pytest.skip("DIVE labels not found")
        _skip_if_no_labels()
        result = run_merger(_DATA_DIR, ["solidifi", "dive"], output_dir=tmp_path)
        assert result.multi_source == 0

    def test_dive_no_dos_reentrancy_flag(self, tmp_path):
        """DIVE T2 co-occurrence must NOT be flagged — it's legitimate multi-label."""
        if not (_LABELS_DIR / "dive").exists():
            pytest.skip("DIVE labels not found")
        _skip_if_no_labels()
        run_merger(_DATA_DIR, ["solidifi", "dive"], output_dir=tmp_path)
        flagged = [
            lf for lf in tmp_path.glob("*.labels.json")
            if "dos_reentrancy_cooccur_suspect" in json.loads(lf.read_text())["flags"]
        ]
        # DIVE is T2 → should produce zero flags
        assert len(flagged) == 0

    def test_cache_hit_on_second_run(self, tmp_path):
        _skip_if_no_labels()
        r1 = run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path)
        assert r1.contracts_merged == 283
        r2 = run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path)
        assert r2.contracts_merged == 0
        assert r2.cached == 283

    def test_force_overwrites(self, tmp_path):
        _skip_if_no_labels()
        run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path)
        r2 = run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path, force=True)
        assert r2.contracts_merged == 283
        assert r2.cached == 0

    def test_sha256_filename_matches_content(self, tmp_path):
        _skip_if_no_labels()
        run_merger(_DATA_DIR, ["solidifi"], output_dir=tmp_path)
        for lf in list(tmp_path.glob("*.labels.json"))[:5]:
            lj = json.loads(lf.read_text())
            assert lf.name.removesuffix(".labels.json") == lj["sha256"]
