"""Tests for fp_estimator (Stage 4 Task 4.5)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.slither_runner import SlitherFindings
from sentinel_data.verification.fp_estimator import (
    FP_RATE_FAIL_THRESHOLD, ClassFPStats, FPRecord, FPEstimationResult,
    StratumStats, _stratified_sample, run_fp_estimation,
)


def _make_fake_merged(tmp_path: Path, contracts: list[dict]) -> Path:
    """Write fake merged labels. `contracts` is a list of class+source+tier dicts."""
    import hashlib
    merged = tmp_path / "labels" / "merged"
    merged.mkdir(parents=True)
    for i, c in enumerate(contracts):
        sha = hashlib.sha256(f"contract_{i}".encode()).hexdigest()
        classes = {
            cls: {
                "value": c.get(cls, 0),
                "tier": c.get(f"{cls}_tier"),
                "source": c.get("source", "test"),
            }
            for cls in class_names()
        }
        n_pos = sum(v["value"] for v in classes.values())
        lj = {
            "sha256": sha, "sources": [c.get("source", "test")],
            "classes": classes, "n_pos": n_pos, "flags": [],
        }
        (merged / f"{sha}.labels.json").write_text(json.dumps(lj))
    return tmp_path


def _make_findings(sha, source, agreed: bool):
    return SlitherFindings(
        sha256=sha, source=source,
        detectors_run=["reentrancy-eth"],
        findings=[{"check": "reentrancy-eth", "impact": "high", "confidence": "medium"}] if agreed else [],
    )


class TestStratifiedSample:
    def test_empty(self):
        assert _stratified_sample([], 10, seed=0) == []

    def test_n_zero(self):
        positives = [("a", {"classes": {"X": {"value": 1}}})]
        assert _stratified_sample(positives, 0, seed=0) == []

    def test_proportional_allocation(self):
        # 60 from (src_a, T0), 40 from (src_b, T0); sample 10
        positives = []
        for i in range(60):
            positives.append((f"a_{i}", {"sources": ["src_a"], "classes": {"X": {"value": 1, "tier": "T0"}}}))
        for i in range(40):
            positives.append((f"b_{i}", {"sources": ["src_b"], "classes": {"X": {"value": 1, "tier": "T0"}}}))
        sample = _stratified_sample(positives, 10, seed=42)
        assert len(sample) == 10
        a_count = sum(1 for s, _ in sample if s.startswith("a_"))
        b_count = sum(1 for s, _ in sample if s.startswith("b_"))
        # ~6 from src_a (60% of 10), ~4 from src_b
        assert 5 <= a_count <= 7
        assert 3 <= b_count <= 5

    def test_deterministic_with_seed(self):
        positives = [
            (f"x_{i}", {"sources": ["src"], "classes": {"X": {"value": 1, "tier": "T0"}}})
            for i in range(20)
        ]
        s1 = _stratified_sample(positives, 5, seed=42)
        s2 = _stratified_sample(positives, 5, seed=42)
        assert [s for s, _ in s1] == [s for s, _ in s2]

    def test_no_replacement_within_sample(self):
        positives = [
            (f"x_{i}", {"sources": ["src"], "classes": {"X": {"value": 1, "tier": "T0"}}})
            for i in range(20)
        ]
        sample = _stratified_sample(positives, 5, seed=42)
        shas = [s for s, _ in sample]
        assert len(shas) == len(set(shas))  # all unique


class TestDataclasses:
    def test_stratum_fp_rate(self):
        st = StratumStats(class_name="X", source="dive", tier="T2",
                          sampled=10, likely_fp=3, errored=0)
        assert st.fp_rate == 0.3

    def test_class_fp_rate_aggregate(self):
        s = ClassFPStats(class_name="X", sampled=20, likely_fp=8, errored=2)
        # denom = 20 - 2 = 18, rate = 8/18
        assert abs(s.fp_rate - 8/18) < 0.001

    def test_class_failed_above_threshold(self):
        s = ClassFPStats(class_name="X", sampled=10, likely_fp=5, errored=0)
        assert s.failed is True  # 50% > 30%
        s.likely_fp = 2
        assert s.failed is False  # 20% < 30%


class TestRunFPEstimation:
    def test_empty_corpus(self, tmp_path):
        (tmp_path / "labels" / "merged").mkdir(parents=True)
        result = run_fp_estimation(tmp_path, sample_size=10)
        assert result.total_sampled == 0

    def test_missing_merged_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_fp_estimation(tmp_path)

    def test_no_detector_class_skipped(self, tmp_path):
        """IntegerUO has no Slither detector — FP estimation skipped."""
        data_dir = _make_fake_merged(tmp_path, [
            {"IntegerUO": 1, "IntegerUO_tier": "T0", "source": "solidifi"},
            {"IntegerUO": 1, "IntegerUO_tier": "T0", "source": "solidifi"},
        ])
        result = run_fp_estimation(data_dir, sample_size=2)
        # IntegerUO has no detector, so sampled=0
        assert result.by_class["IntegerUO"].sampled == 0

    @patch("sentinel_data.verification.fp_estimator.run_on_contract")
    def test_aggregation_across_strata(self, mock_run, tmp_path):
        """3 contracts: 2 from (dive, T2), 1 from (solidifi, T0).
        Slither agrees on 1/2 dive + 1/1 solidifi."""
        data_dir = _make_fake_merged(tmp_path, [
            {"Reentrancy": 1, "Reentrancy_tier": "T2", "source": "dive"},        # agree
            {"Reentrancy": 1, "Reentrancy_tier": "T2", "source": "dive"},        # disagree
            {"Reentrancy": 1, "Reentrancy_tier": "T0", "source": "solidifi"},   # agree
        ])
        shas = sorted(p.name.removesuffix(".labels.json") for p in (data_dir / "labels" / "merged").glob("*.labels.json"))
        # We need to map shas → mock response. shas[0] and shas[1] are the 2 dives;
        # shas[2] is the solidifi. Make one dive disagree.
        # Use the contracts list order (not sorted) to assign which dive disagrees.
        # contract_0 = first dive (will be agree); contract_1 = second dive (will be disagree).
        # Find the shas for contract_0 and contract_1.
        import hashlib
        sha_0 = hashlib.sha256(b"contract_0").hexdigest()
        sha_1 = hashlib.sha256(b"contract_1").hexdigest()
        sha_2 = hashlib.sha256(b"contract_2").hexdigest()
        responses = {
            sha_0: _make_findings(sha_0, "dive", agreed=True),       # dive agree
            sha_1: _make_findings(sha_1, "dive", agreed=False),      # dive disagree (likely FP)
            sha_2: _make_findings(sha_2, "solidifi", agreed=True),   # solidifi agree
        }
        mock_run.side_effect = lambda sha, src, *a, **kw: responses.get(sha)

        result = run_fp_estimation(data_dir, sample_size=10)
        reen = result.by_class["Reentrancy"]
        assert reen.sampled == 3
        assert reen.likely_fp == 1  # 1 of 3 disagreed
        # 1 agree + 1 disagree from (dive, T2); 1 agree from (solidifi, T0)
        assert ("dive", "T2") in reen.strata
        assert ("solidifi", "T0") in reen.strata
        assert reen.strata[("dive", "T2")].likely_fp == 1
        assert reen.strata[("solidifi", "T0")].likely_fp == 0
        assert reen.strata[("solidifi", "T0")].sampled == 1

    @patch("sentinel_data.verification.fp_estimator.run_on_contract")
    def test_stratified_sampling_respects_proportions(self, mock_run, tmp_path):
        """10 contracts in 90/10 (dive T2 / solidifi T0) split; sample 10; expect ~9 from dive, ~1 from solidifi."""
        contracts = [{"Reentrancy": 1, "Reentrancy_tier": "T2", "source": "dive"} for _ in range(9)]
        contracts += [{"Reentrancy": 1, "Reentrancy_tier": "T0", "source": "solidifi"} for _ in range(1)]
        data_dir = _make_fake_merged(tmp_path, contracts)
        shas = sorted(p.name.removesuffix(".labels.json") for p in (data_dir / "labels" / "merged").glob("*.labels.json"))
        responses = {sha: _make_findings(sha, "test", agreed=True) for sha in shas}
        mock_run.side_effect = lambda sha, src, *a, **kw: responses.get(sha)

        result = run_fp_estimation(data_dir, sample_size=10)
        reen = result.by_class["Reentrancy"]
        assert reen.sampled == 10
        assert reen.strata[("dive", "T2")].sampled == 9
        assert reen.strata[("solidifi", "T0")].sampled == 1

    @patch("sentinel_data.verification.fp_estimator.run_on_contract")
    def test_failed_class_above_threshold(self, mock_run, tmp_path):
        """All 5 sampled contracts have Slither disagreement → failed=True."""
        contracts = [{"Reentrancy": 1, "Reentrancy_tier": "T0", "source": "solidifi"} for _ in range(5)]
        data_dir = _make_fake_merged(tmp_path, contracts)
        shas = sorted(p.name.removesuffix(".labels.json") for p in (data_dir / "labels" / "merged").glob("*.labels.json"))
        responses = {sha: _make_findings(sha, "solidifi", agreed=False) for sha in shas}
        mock_run.side_effect = lambda sha, src, *a, **kw: responses.get(sha)

        result = run_fp_estimation(data_dir, sample_size=5)
        assert result.by_class["Reentrancy"].failed is True

    def test_str_representation_works(self, tmp_path):
        (tmp_path / "labels" / "merged").mkdir(parents=True)
        result = run_fp_estimation(tmp_path, sample_size=10)
        s = str(result)
        assert "FP Estimation Report" in s
        assert "FAIL threshold" in s
