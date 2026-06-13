"""Tests for class_auditor (Stage 4 Task 4.A)."""
import json
import tempfile
from pathlib import Path

import pytest

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.class_auditor import run_audit, AuditResult, CO_OCCUR_FLAG_THRESHOLD

_DATA_DIR = Path("data_module/data")
_MERGED_DIR = _DATA_DIR / "labels" / "merged"


def _skip_if_no_merged():
    if not _MERGED_DIR.exists() or not any(_MERGED_DIR.glob("*.labels.json")):
        pytest.skip("Merged labels not found — run merger first")


def _make_fake_merged(tmp_path: Path, contracts: list[dict]) -> Path:
    """Write fake merged labels to tmp_path and return it."""
    import hashlib
    merged = tmp_path / "labels" / "merged"
    merged.mkdir(parents=True)
    for i, c in enumerate(contracts):
        sha = hashlib.sha256(f"contract_{i}".encode()).hexdigest()
        classes = {
            cls: {"value": c.get(cls, 0), "tier": c.get(f"{cls}_tier"), "source": c.get("source", "test")}
            for cls in class_names()
        }
        n_pos = sum(v["value"] for v in classes.values())
        lj = {"sha256": sha, "sources": [c.get("source", "test")], "classes": classes, "n_pos": n_pos, "flags": []}
        (merged / f"{sha}.labels.json").write_text(json.dumps(lj))
    # Return the parent dir (not merged) since run_audit expects data_dir
    return tmp_path


class TestClassAuditorUnit:
    def test_empty_corpus_raises(self, tmp_path):
        (tmp_path / "labels" / "merged").mkdir(parents=True)
        result = run_audit(tmp_path)
        assert result.total_contracts == 0

    def test_counts_positives(self, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [
            {"Reentrancy": 1, "source": "dive"},
            {"Reentrancy": 1, "IntegerUO": 1, "source": "solidifi"},
            {"IntegerUO": 0, "source": "solidifi"},
        ])
        result = run_audit(data_dir)
        assert result.total_contracts == 3
        assert result.per_class["Reentrancy"].total_positives == 2
        assert result.per_class["IntegerUO"].total_positives == 1
        assert result.per_class["Timestamp"].total_positives == 0

    def test_per_source_breakdown(self, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [
            {"Reentrancy": 1, "source": "solidifi"},
            {"Reentrancy": 1, "source": "dive"},
            {"Reentrancy": 1, "source": "dive"},
        ])
        result = run_audit(data_dir)
        reen = result.per_class["Reentrancy"]
        assert reen.by_source.get("solidifi", 0) == 1
        assert reen.by_source.get("dive", 0) == 2

    def test_co_occurrence_matrix_computed(self, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [
            {"DenialOfService": 1, "Reentrancy": 1},
            {"DenialOfService": 1, "Reentrancy": 1},
            {"DenialOfService": 1, "Reentrancy": 0},
        ])
        result = run_audit(data_dir)
        # Find P(Reentrancy=1 | DenialOfService=1)
        pair = next((p for p in result.co_occurrence if p.class_a == "DenialOfService" and p.class_b == "Reentrancy"), None)
        assert pair is not None
        assert abs(pair.rate - 2/3) < 0.01

    def test_co_occurrence_flag_above_threshold(self, tmp_path):
        # 90% co-occurrence → should be flagged
        contracts = [{"DenialOfService": 1, "Reentrancy": 1}] * 9 + [{"DenialOfService": 1, "Reentrancy": 0}]
        data_dir = _make_fake_merged(tmp_path, contracts)
        result = run_audit(data_dir)
        flagged = [p for p in result.flagged_pairs if p.class_a == "DenialOfService" and p.class_b == "Reentrancy"]
        assert len(flagged) == 1
        assert flagged[0].rate > CO_OCCUR_FLAG_THRESHOLD

    def test_low_co_occurrence_not_flagged(self, tmp_path):
        # 12% DIVE-style co-occurrence → NOT flagged
        contracts = [{"DenialOfService": 1, "Reentrancy": 1}] * 12 + [{"DenialOfService": 1, "Reentrancy": 0}] * 88
        data_dir = _make_fake_merged(tmp_path, contracts)
        result = run_audit(data_dir)
        flagged = [p for p in result.flagged_pairs if p.class_a == "DenialOfService" and p.class_b == "Reentrancy"]
        assert len(flagged) == 0

    def test_all_10_classes_in_result(self, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [{"Reentrancy": 1}])
        result = run_audit(data_dir)
        for cls in class_names():
            assert cls in result.per_class

    def test_str_representation_works(self, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [{"Reentrancy": 1}])
        result = run_audit(data_dir)
        report = str(result)
        assert "Class Audit Report" in report
        assert "Reentrancy" in report

    def test_prevalence_calculated(self, tmp_path):
        contracts = [{"Reentrancy": 1}] * 3 + [{}] * 7
        data_dir = _make_fake_merged(tmp_path, contracts)
        result = run_audit(data_dir)
        assert abs(result.per_class["Reentrancy"].prevalence - 0.3) < 0.01


class TestClassAuditorIntegration:
    def test_smoke_on_real_corpus(self):
        _skip_if_no_merged()
        result = run_audit(_DATA_DIR)
        assert isinstance(result, AuditResult)
        assert result.total_contracts >= 22000

    def test_dive_dos_reentrancy_cooccurrence_finding(self):
        """DoS+Reentrancy co-occurrence was patched out on 2026-06-13.

        Pre-patch: P(Reentrancy=1 | DoS=1) ~ 70% in DIVE (T2) — the legacy
        DIVE labels had multi-label noise where many contracts were tagged
        both DoS AND Reentrancy even when the DoS was a reentrancy-style
        attack, not a true DoS. The 2,655 such contracts were zeroed
        in data/labels/dive by the DoS patch (see pre-run12-fixes-2026-06-13.md
        §Item 3 + §Step A decision).

        Post-patch: 0 DoS+Reentrancy overlap. P(Reentrancy=1 | DoS=1) is
        now ~ DoS_ree_overlap / total_DoS ≈ 0/1101 = 0% — well below the
        auditor's 50% flag threshold.

        This test verifies the patch is in effect, not the co-occurrence.
        If the patch is reverted, this test will fail at len(flagged_dos) == 0.
        """
        _skip_if_no_merged()
        result = run_audit(_DATA_DIR)
        # P(Reentrancy|DoS) is now ~0% after the DoS patch — NOT flagged
        flagged_dos = [p for p in result.flagged_pairs
                       if p.class_a == "DenialOfService" and p.class_b == "Reentrancy"]
        assert len(flagged_dos) == 0, (
            f"DoS+Reentrancy co-occurrence is still flagged: "
            f"{[(p.class_a, p.class_b, p.rate) for p in flagged_dos]}. "
            f"The DoS patch should have zeroed these in DIVE source labels. "
            f"See data_module/temp/patch_dos_v3.py"
        )
        # P(DoS|Reentrancy) is also ~0% — NOT flagged
        flagged_reen = [p for p in result.flagged_pairs
                        if p.class_a == "Reentrancy" and p.class_b == "DenialOfService"]
        assert len(flagged_reen) == 0

    def test_all_class_positives_counted(self):
        _skip_if_no_merged()
        result = run_audit(_DATA_DIR)
        # Known from gate run: Reentrancy ≥ 11000, IntegerUO ≥ 9000
        assert result.per_class["Reentrancy"].total_positives >= 11000
        assert result.per_class["IntegerUO"].total_positives >= 9000
