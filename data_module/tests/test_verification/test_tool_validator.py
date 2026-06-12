"""Tests for tool_validator (Stage 4 Task 4.4)."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.slither_runner import SlitherFindings
from sentinel_data.verification.tool_validator import (
    AgreementVerdict, ClassAgreementStats, ContractAgreement,
    ToolValidationResult, run_tool_validation,
)

_DATA_DIR = Path("data_module/data")
_MERGED_DIR = _DATA_DIR / "labels" / "merged"
_PREP_DIR = _DATA_DIR / "preprocessed"


def _skip_if_no_merged():
    if not _MERGED_DIR.exists() or not any(_MERGED_DIR.glob("*.labels.json")):
        pytest.skip("Merged labels not found — run merger first")


def _make_fake_merged(tmp_path: Path, contracts: list[dict]) -> Path:
    """Write fake merged labels to tmp_path and return the data_dir."""
    import hashlib
    merged = tmp_path / "labels" / "merged"
    merged.mkdir(parents=True)
    for i, c in enumerate(contracts):
        sha = hashlib.sha256(f"contract_{i}".encode()).hexdigest()
        classes = {
            cls: {"value": c.get(cls, 0), "tier": c.get(f"{cls}_tier"),
                  "source": c.get("source", "test")}
            for cls in class_names()
        }
        merged_lj = {
            "sha256": sha, "sources": [c.get("source", "test")],
            "classes": classes, "n_pos": sum(v["value"] for v in classes.values()),
            "flags": [],
        }
        (merged / f"{sha}.labels.json").write_text(json.dumps(merged_lj))
    return tmp_path


def _make_fake_findings(sha: str, source: str, agreed: bool, detectors: list[str]):
    """Build a SlitherFindings object that agrees or disagrees."""
    det_list = ["reentrancy-eth", "low-level-calls"]  # representative
    fired = det_list if agreed else []
    return SlitherFindings(
        sha256=sha, source=source,
        detectors_run=det_list,
        findings=[{"check": c, "impact": "high", "confidence": "medium"} for c in fired],
    )


class TestDataclasses:
    def test_agreement_rate_when_no_checkable(self):
        s = ClassAgreementStats(class_name="X")
        assert s.agreement_rate is None
        assert s.coverage == 0.0

    def test_agreement_rate_basic(self):
        s = ClassAgreementStats(class_name="X", positives_total=20, agree=7, disagree=3, checkable=10)
        assert s.agreement_rate == 0.7
        assert s.coverage == 0.5  # 10/20 positives_total checked


class TestRunToolValidation:
    def test_empty_corpus(self, tmp_path):
        (tmp_path / "labels" / "merged").mkdir(parents=True)
        result = run_tool_validation(tmp_path)
        assert result.total_positives == 0
        assert all(s.positives_total == 0 for s in result.by_class.values())

    def test_missing_merged_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Merged labels dir"):
            run_tool_validation(tmp_path)

    def test_no_detector_class_marks_no_detector(self, tmp_path):
        """IntegerUO has no Slither detector in v0.10 — all positives become NO_DETECTOR."""
        data_dir = _make_fake_merged(tmp_path, [
            {"IntegerUO": 1, "source": "solidifi"},
            {"IntegerUO": 1, "source": "solidifi"},
        ])
        result = run_tool_validation(data_dir)
        intuo = result.by_class["IntegerUO"]
        assert intuo.positives_total == 2
        assert intuo.no_detector == 2
        assert intuo.checkable == 0
        assert intuo.agreement_rate is None
        for d in intuo.details:
            assert d.verdict == AgreementVerdict.NO_DETECTOR

    def test_only_classes_filter(self, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [
            {"Reentrancy": 1, "IntegerUO": 1, "source": "solidifi"},
        ])
        result = run_tool_validation(data_dir, only_classes=["Reentrancy"])
        assert result.by_class["Reentrancy"].positives_total == 1
        # other classes are not even present in the result
        assert "IntegerUO" not in result.by_class

    @patch("sentinel_data.verification.tool_validator.run_on_contract")
    def test_agreement_counted_correctly(self, mock_run, tmp_path):
        """Mock Slither: 3 Reentrancy positives, Slither agrees on 2, disagrees on 1."""
        data_dir = _make_fake_merged(tmp_path, [
            {"Reentrancy": 1, "source": "solidifi"},
            {"Reentrancy": 1, "source": "solidifi"},
            {"Reentrancy": 1, "source": "solidifi"},
        ])
        shas = sorted(p.name.removesuffix(".labels.json") for p in (data_dir / "labels" / "merged").glob("*.labels.json"))
        # shas[0] → agree, shas[1] → agree, shas[2] → disagree
        responses = {
            shas[0]: _make_fake_findings(shas[0], "solidifi", agreed=True,  detectors=[]),
            shas[1]: _make_fake_findings(shas[1], "solidifi", agreed=True,  detectors=[]),
            shas[2]: _make_fake_findings(shas[2], "solidifi", agreed=False, detectors=[]),
        }
        mock_run.side_effect = lambda sha, src, *_a, **_kw: responses.get(sha)
        result = run_tool_validation(data_dir)
        reen = result.by_class["Reentrancy"]
        assert reen.positives_total == 3
        assert reen.agree == 2
        assert reen.disagree == 1
        assert reen.checkable == 3
        assert reen.agreement_rate == pytest.approx(2/3)
        assert result.total_agrees == 2
        assert result.total_checkable == 3

    @patch("sentinel_data.verification.tool_validator.run_on_contract")
    def test_slither_error_marked(self, mock_run, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [{"Reentrancy": 1, "source": "solidifi"}])
        mock_run.return_value = SlitherFindings(
            sha256="x", source="solidifi", detectors_run=["reentrancy-eth"],
            findings=[], error="compile failed",
        )
        result = run_tool_validation(data_dir)
        reen = result.by_class["Reentrancy"]
        assert reen.errored == 1
        assert reen.checkable == 0
        assert reen.agreement_rate is None

    @patch("sentinel_data.verification.tool_validator.run_on_contract")
    def test_no_sol_file_marked_skip(self, mock_run, tmp_path):
        """run_on_contract returns None when .sol is not found."""
        data_dir = _make_fake_merged(tmp_path, [{"Reentrancy": 1, "source": "solidifi"}])
        mock_run.return_value = None
        result = run_tool_validation(data_dir)
        reen = result.by_class["Reentrancy"]
        assert reen.skipped == 1
        assert reen.checkable == 0

    def test_str_representation_works(self, tmp_path):
        (tmp_path / "labels" / "merged").mkdir(parents=True)
        result = run_tool_validation(tmp_path)
        s = str(result)
        assert "Tool Validation Report" in s
        assert "Per-class Slither agreement" in s


class TestRunToolValidationIntegration:
    def test_smoke_limit1(self):
        _skip_if_no_merged()
        # Patch run_on_contract so the smoke test doesn't actually invoke Slither
        with patch("sentinel_data.verification.tool_validator.run_on_contract") as mock_run:
            mock_run.return_value = None  # everything skipped
            result = run_tool_validation(_DATA_DIR, limit_per_class=1)
            assert isinstance(result, ToolValidationResult)
            assert len(result.by_class) == 10
