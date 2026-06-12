"""Stage 6 — analysis tests.

The 5 tools (balance_viz, feature_dist, cooccurrence, overlap_detector,
drift_monitor) + probe_dataset re-export are tested with synthetic
fixtures that don't require the v2 corpus.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────
# Synthetic fixtures (in-memory, no real corpus)
# ─────────────────────────────────────────────────────────────────────

def make_label(sha: str, positives: dict[str, int], source: str = "dive",
               tier_map: dict[str, str] | None = None) -> dict:
    """Build a merged label dict."""
    classes = {}
    all_classes = [
        "Reentrancy", "Timestamp", "IntegerUO", "UnusedReturn", "CallToUnknown",
        "ExternalBug", "DenialOfService", "GasException", "MishandledException",
        "TransactionOrderDependence",
    ]
    for c in all_classes:
        v = positives.get(c, 0)
        if v == 1:
            tier = (tier_map or {}).get(c, "T2")
        else:
            tier = None
        classes[c] = {"value": v, "tier": tier, "source": source}
    return {
        "sha256": sha,
        "sources": [source],
        "classes": classes,
    }


@pytest.fixture
def tmp_labels_dir(tmp_path: Path) -> Path:
    """A directory with a few synthetic merged labels."""
    d = tmp_path / "labels" / "merged"
    d.mkdir(parents=True)
    # 10 contracts with varying class distributions
    fixtures = [
        ("a" * 64, {"Reentrancy": 1, "ExternalBug": 1, "IntegerUO": 1}),  # multi
        ("b" * 64, {"Reentrancy": 1, "ExternalBug": 1}),  # multi
        ("c" * 64, {"Reentrancy": 1, "DenialOfService": 1}),  # multi
        ("d" * 64, {"Timestamp": 1}),
        ("e" * 64, {"Timestamp": 1}),
        ("f" * 64, {"Timestamp": 1}),
        ("g" * 64, {"IntegerUO": 1, "MishandledException": 1}),  # multi
        ("h" * 64, {}),  # NonVulnerable
        ("i" * 64, {}),  # NonVulnerable
        ("j" * 64, {}),  # NonVulnerable
    ]
    for sha, positives in fixtures:
        lj = make_label(sha, positives, source="synth")
        (d / f"{sha}.labels.json").write_text(json.dumps(lj))
    return d


# ─────────────────────────────────────────────────────────────────────
# balance_viz
# ─────────────────────────────────────────────────────────────────────

class TestBalanceViz:

    def test_build_table_counts(self, tmp_labels_dir: Path):
        from sentinel_data.analysis.balance_viz import build_balance_table
        table = build_balance_table(tmp_labels_dir)
        assert table.total_contracts == 10
        assert table.multi_label_count == 4
        # 3 Reentrancy, 3 Timestamp, 2 IntegerUO, 2 ExternalBug, 1 DoS, 1 MishandledException
        assert table.per_class["Reentrancy"] == 3
        assert table.per_class["Timestamp"] == 3
        assert table.per_class["IntegerUO"] == 2
        assert table.per_class["ExternalBug"] == 2
        assert table.per_class["DenialOfService"] == 1
        assert table.per_class["MishandledException"] == 1
        assert table.per_class["CallToUnknown"] == 0
        assert table.per_source["synth"] == 10

    def test_write_csv(self, tmp_labels_dir: Path, tmp_path: Path):
        from sentinel_data.analysis.balance_viz import build_balance_table, write_csv
        table = build_balance_table(tmp_labels_dir)
        out = write_csv(table, tmp_path / "b.csv")
        assert out.exists()
        text = out.read_text()
        assert "scope" in text
        assert "Reentrancy" in text

    def test_write_plot(self, tmp_labels_dir: Path, tmp_path: Path):
        from sentinel_data.analysis.balance_viz import build_balance_table, write_plot
        table = build_balance_table(tmp_labels_dir)
        out = write_plot(table, tmp_path / "b.png")
        assert out.exists()
        assert out.stat().st_size > 100  # non-trivial PNG


# ─────────────────────────────────────────────────────────────────────
# feature_dist (the headline — Run-9-failure catcher)
# ─────────────────────────────────────────────────────────────────────

def _write_rep(rep_root: Path, source: str, sha: str, node_count: int, edge_count: int):
    """Write a synthetic .rep.json sidecar."""
    d = rep_root / source
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sha}.rep.json").write_text(json.dumps({
        "sha256": sha, "source": source,
        "node_count": node_count, "edge_count": edge_count,
        "schema_version": "v9", "extractor_version": "v2.1",
    }))


def _write_sol(preproc_root: Path, source: str, sha: str, content: str):
    d = preproc_root / source
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sha}.sol").write_text(content)


class TestFeatureDist:

    def test_features_for_contract(self, tmp_path: Path):
        from sentinel_data.analysis.feature_dist import _features_for_contract
        rep_root = tmp_path / "rep"
        preproc_root = tmp_path / "preproc"
        sha = "a" * 64
        _write_rep(rep_root, "synth", sha, 100, 200)
        _write_sol(preproc_root, "synth", sha,
                   "pragma solidity ^0.8.0;\n"
                   "contract C {\n"
                   "  function f() public { if (x) { g(); } }\n"
                   "  function g() public {}\n"
                   "}\n")
        feats = _features_for_contract(sha, "synth", rep_root, preproc_root)
        assert feats["node_count"] == 100
        assert feats["edge_count"] == 200
        assert feats["loc"] >= 4
        assert feats["function_count"] >= 2
        assert "cyclomatic_complexity" in feats
        assert "call_depth" in feats

    def test_find_high_risk_pairs_synthetic(self, tmp_path: Path):
        """One class has 2σ higher node_count than another → must be flagged."""
        from sentinel_data.analysis.feature_dist import (
            build_per_class_stats, find_high_risk_pairs,
        )
        rep_root = tmp_path / "rep"
        preproc_root = tmp_path / "preproc"
        labels_dir = tmp_path / "labels" / "merged"
        labels_dir.mkdir(parents=True)
        # Create 10 contracts for class A (Reentrancy) with high node_count
        # and 10 for class B (Timestamp) with low node_count
        for i in range(10):
            sha_a = f"a{i:062d}"
            _write_rep(rep_root, "synth", sha_a, 200 + i, 400 + i)
            _write_sol(preproc_root, "synth", sha_a, "pragma solidity ^0.8.0;\ncontract C {}\n")
            (labels_dir / f"{sha_a}.labels.json").write_text(json.dumps(
                make_label(sha_a, {"Reentrancy": 1}, source="synth",
                           tier_map={"Reentrancy": "T2"})
            ))
            sha_b = f"b{i:062d}"
            _write_rep(rep_root, "synth", sha_b, 50 + i, 100 + i)  # ~150/50 std
            _write_sol(preproc_root, "synth", sha_b, "pragma solidity ^0.8.0;\ncontract C {}\n")
            (labels_dir / f"{sha_b}.labels.json").write_text(json.dumps(
                make_label(sha_b, {"Timestamp": 1}, source="synth",
                           tier_map={"Timestamp": "T2"})
            ))
        by_class = build_per_class_stats(labels_dir, rep_root, preproc_root)
        pairs = find_high_risk_pairs(by_class, sigma_threshold=1.5)
        # At least one Reentrancy↔Timestamp pair should be flagged
        flag_nodes = [p for p in pairs if p.feature == "node_count"]
        assert any(p.class_a == "Reentrancy" and p.class_b == "Timestamp" for p in flag_nodes), \
            f"Expected Reentrancy↔Timestamp node_count to be flagged, got: {[(p.class_a, p.class_b, p.feature, p.sigma_diff) for p in flag_nodes]}"

    def test_report_contains_required_sections(self, tmp_path: Path):
        from sentinel_data.analysis.feature_dist import (
            build_per_class_stats, find_high_risk_pairs,
            write_complexity_proxy_risk, DEFAULT_FEATURES,
        )
        by_class = {c: type("S", (), {
            "feature_stats": {f: {"mean": 1.0, "std": 0.1, "n": 5, "min": 0.9, "max": 1.1, "median": 1.0}
                              for f in DEFAULT_FEATURES},
            "label_conditional": {f: {"positive": {"mean": 1.0, "std": 0.0, "n": 5, "min": 1.0, "max": 1.0, "median": 1.0},
                                     "negative": {"mean": 0.0, "std": 0.0, "n": 0, "min": 0, "max": 0, "median": 0}}
                                  for f in DEFAULT_FEATURES},
        })() for c in ["Reentrancy", "Timestamp"]}
        # Need real dataclass — use the actual one
        from sentinel_data.analysis.feature_dist import PerClassStats
        by_class = {c: PerClassStats(c) for c in ["Reentrancy", "Timestamp"]}
        for c in by_class.values():
            for f in DEFAULT_FEATURES:
                c.feature_stats[f] = {"mean": 1.0, "std": 0.1, "n": 5, "min": 0.9, "max": 1.1, "median": 1.0}
                c.label_conditional[f] = {
                    "positive": {"mean": 1.0, "std": 0.1, "n": 5, "min": 0.9, "max": 1.1, "median": 1.0},
                    "negative": {"mean": 0.5, "std": 0.1, "n": 5, "min": 0.4, "max": 0.6, "median": 0.5},
                }
        pairs = find_high_risk_pairs(by_class, sigma_threshold=1.5)
        report = write_complexity_proxy_risk(by_class, pairs, 1.5, tmp_path / "report.md")
        text = report.read_text()
        assert "Complexity Proxy Risk Report" in text
        assert "HIGH-RISK Pairs" in text
        assert "Per-Class Feature Stats" in text
        assert "Label-Conditional Feature Distribution" in text
        assert "Recommendation" in text


# ─────────────────────────────────────────────────────────────────────
# cooccurrence
# ─────────────────────────────────────────────────────────────────────

class TestCooccurrence:

    def test_directed_and_conditional(self, tmp_labels_dir: Path):
        from sentinel_data.analysis.cooccurrence import build_cooccurrence_matrices
        m = build_cooccurrence_matrices(tmp_labels_dir, flag_threshold=0.5)
        # From the fixture: Reentrancy positive in 3 contracts, ExternalBug in 2
        # Of the 3 Reentrancy positives, 2 are also ExternalBug → P(External|Reentrancy) = 2/3
        assert m.counts_positive["Reentrancy"] == 3
        assert m.counts_positive["ExternalBug"] == 2
        # directed[Reentrancy][ExternalBug] = 2 (both contracts a, b)
        assert m.directed["Reentrancy"]["ExternalBug"] == 2
        # conditional[Reentrancy][ExternalBug] = 2/3 ≈ 0.667
        assert abs(m.conditional["Reentrancy"]["ExternalBug"] - 2 / 3) < 1e-6
        # Flagged: Reentrancy↔ExternalBug is 0.667 (above 0.5) AND ExternalBug↔Reentrancy = 2/2 = 1.0
        # So max(P(a|b), P(b|a)) > 0.5 → flagged (keys are sorted alphabetically)
        flagged = {(fp["class_a"], fp["class_b"]): fp for fp in m.flagged_pairs}
        key = ("ExternalBug", "Reentrancy")  # alphabetical sort
        assert key in flagged
        assert flagged[key]["p_max"] == 1.0  # P(Reentrancy|ExternalBug) = 2/2

    def test_csv_sections(self, tmp_labels_dir: Path, tmp_path: Path):
        from sentinel_data.analysis.cooccurrence import (
            build_cooccurrence_matrices, write_csv,
        )
        m = build_cooccurrence_matrices(tmp_labels_dir)
        out = write_csv(m, tmp_path / "cooc.csv")
        text = out.read_text()
        assert "DIRECTED CO-OCCURRENCE" in text
        assert "CONDITIONAL PROBABILITY" in text
        assert "FLAGGED PAIRS" in text

    def test_heatmap(self, tmp_labels_dir: Path, tmp_path: Path):
        from sentinel_data.analysis.cooccurrence import (
            build_cooccurrence_matrices, write_heatmap,
        )
        m = build_cooccurrence_matrices(tmp_labels_dir)
        out = write_heatmap(m, tmp_path / "cooc.png")
        assert out.exists()
        assert out.stat().st_size > 100


# ─────────────────────────────────────────────────────────────────────
# overlap_detector
# ─────────────────────────────────────────────────────────────────────

class TestOverlapDetector:

    def test_exact_jaccard_no_overlap(self, tmp_path: Path):
        from sentinel_data.analysis.overlap_detector import build_overlap_matrix
        labels_root = tmp_path / "labels"
        (labels_root / "merged").mkdir(parents=True)
        (labels_root / "src_a").mkdir(parents=True)
        (labels_root / "src_b").mkdir(parents=True)
        # 3 contracts in src_a, 2 in src_b, no overlap
        for i in range(3):
            sha = f"a{i:062d}"
            (labels_root / "merged" / f"{sha}.labels.json").write_text(json.dumps(
                {"sha256": sha, "sources": ["src_a"], "classes": {}}
            ))
            (labels_root / "src_a" / f"{sha}.json").write_text(json.dumps(
                {"sha256": sha}
            ))
        for i in range(2):
            sha = f"b{i:062d}"
            (labels_root / "merged" / f"{sha}.labels.json").write_text(json.dumps(
                {"sha256": sha, "sources": ["src_b"], "classes": {}}
            ))
            (labels_root / "src_b" / f"{sha}.json").write_text(json.dumps(
                {"sha256": sha}
            ))
        m = build_overlap_matrix(labels_root, tmp_path / "preproc")
        assert m.source_sizes["src_a"] == 3
        assert m.source_sizes["src_b"] == 2
        assert m.exact_jaccard["src_a"]["src_b"] == 0.0
        assert m.exact_jaccard["src_a"]["src_a"] == 1.0

    def test_exact_jaccard_full_overlap(self, tmp_path: Path):
        from sentinel_data.analysis.overlap_detector import build_overlap_matrix
        labels_root = tmp_path / "labels"
        (labels_root / "merged").mkdir(parents=True)
        (labels_root / "src_a").mkdir(parents=True)
        (labels_root / "src_b").mkdir(parents=True)
        # 2 contracts, both in src_a AND src_b
        for i in range(2):
            sha = f"a{i:062d}"
            (labels_root / "merged" / f"{sha}.labels.json").write_text(json.dumps(
                {"sha256": sha, "sources": ["src_a", "src_b"], "classes": {}}
            ))
            (labels_root / "src_a" / f"{sha}.json").write_text(json.dumps({"sha256": sha}))
            (labels_root / "src_b" / f"{sha}.json").write_text(json.dumps({"sha256": sha}))
        m = build_overlap_matrix(labels_root, tmp_path / "preproc")
        assert m.exact_jaccard["src_a"]["src_b"] == 1.0
        assert m.exact_intersection["src_a"]["src_b"] == 2

    def test_heatmap(self, tmp_path: Path):
        from sentinel_data.analysis.overlap_detector import build_overlap_matrix, write_heatmap
        labels_root = tmp_path / "labels"
        (labels_root / "merged").mkdir(parents=True)
        m = build_overlap_matrix(labels_root, tmp_path / "preproc")
        out = write_heatmap(m, tmp_path / "overlap.png")
        assert out.exists()


# ─────────────────────────────────────────────────────────────────────
# drift_monitor
# ─────────────────────────────────────────────────────────────────────

class TestDriftMonitor:

    def test_feature_drift_flags_intentional_drift(self, tmp_path: Path):
        from sentinel_data.analysis.drift_monitor import compute_feature_drift
        # Use the path structure _features_for_contract expects: rep_root/<source>/<sha>.rep.json
        baseline_labels = tmp_path / "base" / "labels" / "merged"
        baseline_rep = tmp_path / "base" / "rep"
        baseline_labels.mkdir(parents=True)
        baseline_rep.mkdir(parents=True)
        # Baseline: 50 contracts with node_count ~50 (above min_sample=30)
        for i in range(50):
            sha = f"b{i:062d}"
            (baseline_labels / f"{sha}.labels.json").write_text(json.dumps(
                {"sha256": sha, "sources": ["synth"], "classes": {"Reentrancy": {"value": 1, "tier": "T2", "source": "synth"}}}
            ))
            (baseline_rep / "synth" / f"{sha}.rep.json").parent.mkdir(parents=True, exist_ok=True)
            (baseline_rep / "synth" / f"{sha}.rep.json").write_text(json.dumps({
                "sha256": sha, "source": "synth",
                "node_count": 50 + (i % 5), "edge_count": 100 + (i % 5),
                "schema_version": "v9",
            }))
        # New: 50 contracts with node_count ~500 (10x shift)
        new_labels = tmp_path / "new" / "labels" / "merged"
        new_rep = tmp_path / "new" / "rep"
        new_labels.mkdir(parents=True)
        new_rep.mkdir(parents=True)
        for i in range(50):
            sha = f"n{i:062d}"
            (new_labels / f"{sha}.labels.json").write_text(json.dumps(
                {"sha256": sha, "sources": ["synth"], "classes": {"Reentrancy": {"value": 1, "tier": "T2", "source": "synth"}}}
            ))
            (new_rep / "synth" / f"{sha}.rep.json").parent.mkdir(parents=True, exist_ok=True)
            (new_rep / "synth" / f"{sha}.rep.json").write_text(json.dumps({
                "sha256": sha, "source": "synth",
                "node_count": 500 + (i % 5), "edge_count": 1000 + (i % 5),
                "schema_version": "v9",
            }))
        results = compute_feature_drift(
            baseline_labels, baseline_rep,
            new_labels, new_rep,
            features=["node_count"], min_sample=30,
        )
        assert len(results) == 1
        r = results[0]
        assert r.feature == "node_count"
        assert r.warning, f"Expected drift WARNING, got p={r.pvalue}, stat={r.statistic}, n={r.n_baseline}/{r.n_new}"
        assert r.pvalue < 0.01

    def test_label_drift_flags_intentional_drift(self, tmp_path: Path):
        from sentinel_data.analysis.drift_monitor import compute_label_drift
        base = tmp_path / "base" / "labels" / "merged"
        base.mkdir(parents=True)
        # Baseline: 50 contracts, 10% Reentrancy (5 positive)
        for i in range(50):
            sha = f"b{i:062d}"
            entry = {"sha256": sha, "sources": ["synth"], "classes": {
                "Reentrancy": {"value": 1 if i < 5 else 0, "tier": "T2", "source": "synth"}
            }}
            (base / f"{sha}.labels.json").write_text(json.dumps(entry))
        # New: 50 contracts, 50% Reentrancy
        new = tmp_path / "new" / "labels" / "merged"
        new.mkdir(parents=True)
        for i in range(50):
            sha = f"n{i:062d}"
            entry = {"sha256": sha, "sources": ["synth"], "classes": {
                "Reentrancy": {"value": 1 if i < 25 else 0, "tier": "T2", "source": "synth"}
            }}
            (new / f"{sha}.labels.json").write_text(json.dumps(entry))
        results = compute_label_drift(base, new)
        r = next(x for x in results if x.class_name == "Reentrancy")
        assert r.warning, f"Expected Reentrancy WARNING, got p={r.pvalue}"
        assert abs(r.rate_baseline - 0.10) < 0.01
        assert abs(r.rate_new - 0.50) < 0.01

    def test_insufficient_sample_marked(self, tmp_path: Path):
        from sentinel_data.analysis.drift_monitor import compute_feature_drift
        base = tmp_path / "base" / "labels" / "merged"
        base_rep = tmp_path / "base" / "rep"
        base.mkdir(parents=True)
        base_rep.mkdir(parents=True)
        # Only 2 contracts (below min_sample=30)
        for i in range(2):
            sha = f"b{i:062d}"
            (base / f"{sha}.labels.json").write_text(json.dumps(
                {"sha256": sha, "sources": ["synth"], "classes": {}}
            ))
            (base_rep / "synth" / f"{sha}.rep.json").parent.mkdir(parents=True, exist_ok=True)
            (base_rep / "synth" / f"{sha}.rep.json").write_text(json.dumps({
                "sha256": sha, "source": "synth",
                "node_count": 50, "edge_count": 100, "schema_version": "v9",
            }))
        new = tmp_path / "new" / "labels" / "merged"
        new_rep = tmp_path / "new" / "rep"
        new.mkdir(parents=True)
        new_rep.mkdir(parents=True)
        for i in range(2):
            sha = f"n{i:062d}"
            (new / f"{sha}.labels.json").write_text(json.dumps(
                {"sha256": sha, "sources": ["synth"], "classes": {}}
            ))
            (new_rep / "synth" / f"{sha}.rep.json").parent.mkdir(parents=True, exist_ok=True)
            (new_rep / "synth" / f"{sha}.rep.json").write_text(json.dumps({
                "sha256": sha, "source": "synth",
                "node_count": 500, "edge_count": 1000, "schema_version": "v9",
            }))
        results = compute_feature_drift(base, base_rep, new, new_rep, features=["node_count"])
        assert results[0].insufficient_sample is True
        assert results[0].warning is False

    def test_report(self, tmp_path: Path):
        from sentinel_data.analysis.drift_monitor import (
            DriftReport, FeatureKSResult, LabelKSResult, write_drift_report,
        )
        report = DriftReport(
            feature_results=[FeatureKSResult(
                feature="node_count", n_baseline=50, n_new=50,
                statistic=0.5, pvalue=0.001, warning=True,
            )],
            label_results=[LabelKSResult(
                class_name="Reentrancy", n_baseline_pos=5, n_new_pos=25,
                n_baseline_total=50, n_new_total=50,
                rate_baseline=0.1, rate_new=0.5,
                statistic=0.4, pvalue=0.001, warning=True,
            )],
            overall_warning=True,
        )
        out = write_drift_report(report, tmp_path / "drift.md")
        text = out.read_text()
        assert "DRIFT DETECTED" in text
        assert "node_count" in text
        assert "Reentrancy" in text


# ─────────────────────────────────────────────────────────────────────
# probe_dataset re-export
# ─────────────────────────────────────────────────────────────────────

class TestProbeDatasetReexport:

    def test_reexport_works(self):
        from sentinel_data.analysis.probe_dataset import (
            ProbeDataset, ProbeEntry, ClassProbeBucket, build_probe_dataset,
        )
        # Should be the same identity as the verification module's
        from sentinel_data.verification.probe_dataset import (
            ProbeDataset as OrigPD, ProbeEntry as OrigPE,
            ClassProbeBucket as OrigCPB, build_probe_dataset as orig_bpd,
        )
        assert ProbeDataset is OrigPD
        assert ProbeEntry is OrigPE
        assert ClassProbeBucket is OrigCPB
        assert build_probe_dataset is orig_bpd
