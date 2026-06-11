"""Tests for negative_checker (Stage 4 Task 4.5)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.slither_runner import SlitherFindings
from sentinel_data.verification.negative_checker import (
    DEFAULT_FAIL_THRESHOLD, DEFAULT_WARN_THRESHOLD,
    NonVulnContractCheck, NonVulnResult, NonVulnSourceStats,
    _is_nonvulnerable, run_negative_check,
)


def _make_fake_merged(tmp_path: Path, contracts: list[dict]) -> Path:
    """Write fake merged labels. `contracts` is a list of {class: value, source: ...} dicts."""
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
        n_pos = sum(v["value"] for v in classes.values())
        lj = {
            "sha256": sha, "sources": [c.get("source", "test")],
            "classes": classes, "n_pos": n_pos, "flags": [],
        }
        (merged / f"{sha}.labels.json").write_text(json.dumps(lj))
    return tmp_path


def _make_findings(sha, source, with_hits: bool, detectors: list[str] = None):
    detectors = detectors or ["reentrancy-eth"]
    return SlitherFindings(
        sha256=sha, source=source, detectors_run=detectors,
        findings=([{"check": d, "impact": "high", "confidence": "medium"}
                   for d in detectors] if with_hits else []),
    )


class TestIsNonVulnerable:
    def test_empty_classes(self):
        lj = {"classes": {c: {"value": 0} for c in class_names()}}
        assert _is_nonvulnerable(lj) is True

    def test_one_positive(self):
        lj = {"classes": {**{c: {"value": 0} for c in class_names()}, "Reentrancy": {"value": 1}}}
        assert _is_nonvulnerable(lj) is False

    def test_no_classes_at_all(self):
        assert _is_nonvulnerable({"classes": {}}) is True


class TestDataclasses:
    def test_source_hit_rate(self):
        s = NonVulnSourceStats(source="disl", total=10, hits=2, errored=0)
        assert s.hit_rate == 0.2

    def test_result_status_fail(self):
        r = NonVulnResult(total_checked=20, total_hits=4, total_errored=0)
        assert r.hit_rate == 0.2
        assert r.status == "FAIL"

    def test_result_status_warn(self):
        r = NonVulnResult(total_checked=100, total_hits=7, total_errored=0)
        assert r.hit_rate == 0.07
        assert r.status == "WARN"

    def test_result_status_ok(self):
        r = NonVulnResult(total_checked=100, total_hits=3, total_errored=0)
        assert r.hit_rate == 0.03
        assert r.status == "OK"

    def test_result_status_no_data(self):
        r = NonVulnResult()
        assert r.status == "OK"


class TestRunNegativeCheck:
    def test_empty_corpus(self, tmp_path):
        (tmp_path / "labels" / "merged").mkdir(parents=True)
        result = run_negative_check(tmp_path)
        assert result.total_checked == 0
        assert result.hit_rate is None

    def test_missing_merged_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_negative_check(tmp_path)

    def test_no_nonvulnerable_contracts(self, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [
            {"Reentrancy": 1, "source": "dive"},
            {"IntegerUO": 1, "source": "solidifi"},
        ])
        with patch("sentinel_data.verification.negative_checker.run_on_contract") as mock:
            mock.return_value = _make_findings("x", "dive", with_hits=False)
            result = run_negative_check(data_dir)
            assert result.total_checked == 0

    @patch("sentinel_data.verification.negative_checker.run_on_contract")
    def test_hit_counted(self, mock_run, tmp_path):
        """5 NonVuln contracts: 1 hit, 4 clean."""
        data_dir = _make_fake_merged(tmp_path, [{"source": "disl"} for _ in range(5)])
        shas = sorted(p.name.removesuffix(".labels.json") for p in (data_dir / "labels" / "merged").glob("*.labels.json"))
        responses = {shas[0]: _make_findings(shas[0], "disl", with_hits=True),
                     **{s: _make_findings(s, "disl", with_hits=False) for s in shas[1:]}}
        mock_run.side_effect = lambda sha, src, *a, **kw: responses.get(sha)

        result = run_negative_check(data_dir)
        assert result.total_checked == 5
        assert result.total_hits == 1
        assert result.hit_rate == 0.2
        assert result.status == "FAIL"  # 20% > 10%

    @patch("sentinel_data.verification.negative_checker.run_on_contract")
    def test_status_warn_between_thresholds(self, mock_run, tmp_path):
        """7% hit rate → WARN (between 5% warn and 10% fail)."""
        data_dir = _make_fake_merged(tmp_path, [{"source": "disl"} for _ in range(100)])
        shas = sorted(p.name.removesuffix(".labels.json") for p in (data_dir / "labels" / "merged").glob("*.labels.json"))
        # 7 of 100 hit
        hit_shas = set(shas[:7])
        responses = {
            s: (_make_findings(s, "disl", with_hits=True) if s in hit_shas
                else _make_findings(s, "disl", with_hits=False))
            for s in shas
        }
        mock_run.side_effect = lambda sha, src, *a, **kw: responses.get(sha)

        result = run_negative_check(data_dir)
        assert result.hit_rate == pytest.approx(0.07)
        assert result.status == "WARN"

    @patch("sentinel_data.verification.negative_checker.run_on_contract")
    def test_status_ok_below_warn(self, mock_run, tmp_path):
        """3% hit rate → OK."""
        data_dir = _make_fake_merged(tmp_path, [{"source": "disl"} for _ in range(100)])
        shas = sorted(p.name.removesuffix(".labels.json") for p in (data_dir / "labels" / "merged").glob("*.labels.json"))
        hit_shas = set(shas[:3])
        responses = {
            s: (_make_findings(s, "disl", with_hits=True) if s in hit_shas
                else _make_findings(s, "disl", with_hits=False))
            for s in shas
        }
        mock_run.side_effect = lambda sha, src, *a, **kw: responses.get(sha)

        result = run_negative_check(data_dir)
        assert result.hit_rate == pytest.approx(0.03)
        assert result.status == "OK"

    @patch("sentinel_data.verification.negative_checker.run_on_contract")
    def test_per_source_breakdown(self, mock_run, tmp_path):
        """2 sources: 3 dive (all clean), 5 disl (3 of 5 hit)."""
        data_dir = _make_fake_merged(tmp_path, [
            {"source": "dive"}, {"source": "dive"}, {"source": "dive"},
            {"source": "disl"}, {"source": "disl"}, {"source": "disl"},
            {"source": "disl"}, {"source": "disl"},
        ])
        # Build a (sha → source) map by reading the actual labels
        sha_to_source: dict[str, str] = {}
        for p in (data_dir / "labels" / "merged").glob("*.labels.json"):
            sha = p.name.removesuffix(".labels.json")
            lj = json.loads(p.read_text())
            sha_to_source[sha] = lj["sources"][0]
        disl_shas = [s for s, src in sha_to_source.items() if src == "disl"]
        disl_hit = set(disl_shas[:3])

        def side_effect(sha, src, *a, **kw):
            if src == "disl" and sha in disl_hit:
                return _make_findings(sha, src, with_hits=True)
            return _make_findings(sha, src, with_hits=False)
        mock_run.side_effect = side_effect

        result = run_negative_check(data_dir)
        assert "dive" in result.by_source
        assert "disl" in result.by_source
        assert result.by_source["dive"].hits == 0
        assert result.by_source["disl"].hits == 3
        assert result.by_source["disl"].hit_rate == 0.6

    @patch("sentinel_data.verification.negative_checker.run_on_contract")
    def test_slither_error_counted(self, mock_run, tmp_path):
        data_dir = _make_fake_merged(tmp_path, [{"source": "disl"} for _ in range(5)])
        mock_run.return_value = SlitherFindings(
            sha256="x", source="disl", detectors_run=["reentrancy-eth"],
            findings=[], error="compile failed",
        )
        result = run_negative_check(data_dir)
        assert result.total_errored == 5
        assert result.hit_rate is None  # all errored, denom=0

    def test_str_representation_works(self, tmp_path):
        (tmp_path / "labels" / "merged").mkdir(parents=True)
        result = run_negative_check(tmp_path)
        s = str(result)
        assert "Negative Checker Report" in s
        assert "warn>" in s
        assert "fail>" in s
