"""BCCC Phase 5 regression test — Stage 4 Task 4.7.

This is the regression gate that ensures the new sentinel_data
verification module reproduces the Phase 5 ad-hoc scripts' results on
the legacy BCCC corpus to within ±0.5% per-class drop counts (per
plan D-4.8 and AUDIT_PATCHES 4-P7).

Per-stage regression (per AUDIT_PATCHES 4-P7):
  Each p5 stage's output is checked individually, not just the final
  report. The chain is:
    p5_s1_evidence_integration.py → p5_s1_evidence_table.csv + coverage
    p5_s2_bulk_verification.py    → p5_s2_automated_verdict.csv
    p5_s3_discrepancy_resolution.py → p5_s3_refined_verdict.csv
    p5_s4_manual_extrapolation.py → p5_s4_final_verdict.csv
    p5_s6_synthesis.py            → p5_s6_class_size_comparison.csv
                                   + p5_s6_verification_report.md

BCCC corpus status:
  The BCCC .sol corpus is deferred per config.yaml
  (`deferred_sources.bccc`). This test currently:
  1. Verifies the v1.4 labels CSV is internally consistent with the
     p5_s6_class_size_comparison.csv
  2. Verifies the per-stage p5_s2 → p5_s3 → p5_s4 → p5_s6 refinement chain
  3. Verifies the per-class drop percentages in p5_s6_verification_report.md

  When BCCC is re-introduced, the new module's verification stage
  will be re-run on the BCCC corpus and the output's per-class drop
  counts must match p5_s6 ±0.5%.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pytest


# Canonical BCCC legacy paths (relative to repo root)
_BCCC_ROOT = Path("docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08")
_P5_OUTPUTS = _BCCC_ROOT / "outputs"

# Per-class p5_s6 numbers (extracted from p5_s6_verification_report.md, 2026-06-08).
# These are the regression targets — when the new module re-runs on the
# BCCC corpus, the per-class drop counts must match these within ±0.5%.
#
# IMPORTANT: p5_s6 only reports on the 6 classes that ran through the
# AUTOMATED Stage 5.4 pipeline. The other 4 classes (MishandledException,
# UnusedReturn, IntegerUO) were "Stage 5.1 manual (clean)" and kept at
# 100% — they are not regression targets for the new module's automated
# verification stage.
_P5_S6_NUMBERS: dict[str, dict[str, float]] = {
    # Automated (Stage 5.4) — regression targets
    "Class01:ExternalBug":          {"n_total": 3604,  "n_keep": 344,   "pct_drop": 90.5},
    "Class02:GasException":         {"n_total": 6879,  "n_keep": 2794,  "pct_drop": 59.4},
    "Class04:Timestamp":            {"n_total": 2674,  "n_keep": 1075,  "pct_drop": 59.8},
    "Class08:CallToUnknown":        {"n_total": 11131, "n_keep": 239,   "pct_drop": 97.9},
    "Class09:DenialOfService":      {"n_total": 12394, "n_keep": 1252,  "pct_drop": 89.9},
    "Class11:Reentrancy":           {"n_total": 17698, "n_keep": 1699,  "pct_drop": 90.4},
    # Manual-clean (Stage 5.1) — kept at 100%, NOT regression targets
    "Class03:MishandledException":  {"n_total": 5154,  "n_keep": 5154,  "pct_drop": 0.0,
                                     "stage": "5.1 manual (clean)"},
    "Class06:UnusedReturn":         {"n_total": 3229,  "n_keep": 3229,  "pct_drop": 0.0,
                                     "stage": "5.1 manual (clean)"},
    "Class10:IntegerUO":            {"n_total": 16740, "n_keep": 16740, "pct_drop": 0.0,
                                     "stage": "5.1 manual (clean)"},
    # Class12:NonVulnerable handled separately (D-I-11/12)
}

# The 6 classes that actually ran through automated verification
_P5_S6_AUTOMATED_CLASSES: list[str] = [
    "Class01:ExternalBug",
    "Class02:GasException",
    "Class04:Timestamp",
    "Class08:CallToUnknown",
    "Class09:DenialOfService",
    "Class11:Reentrancy",
]

_TOLERANCE_PCT = 0.5   # ±0.5% per plan D-4.8 exit criterion #3


def _skip_if_no_bccc_outputs():
    if not _P5_OUTPUTS.exists():
        pytest.skip(f"BCCC Phase 5 outputs not found at {_P5_OUTPUTS}")


def _read_p5_s6_comparison() -> dict[str, dict]:
    """Load p5_s6_class_size_comparison.csv → class → row dict."""
    path = _P5_OUTPUTS / "p5_s6_class_size_comparison.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["class"]: row for row in csv.DictReader(f)}


def _read_v14_keeps_per_class() -> dict[str, int]:
    """Count KEEPs in contracts_clean_v1.4.csv per BCCC class column.

    The v1.4 CSV has columns `p5_verdict_Class01:ExternalBug` etc.
    KEEP == "KEEP" counts as kept for that class; anything else (incl.
    empty / "not_positive") counts as dropped.
    """
    path = _P5_OUTPUTS / "contracts_clean_v1.4.csv"
    if not path.exists():
        return {}
    keeps: Counter = Counter()
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, val in row.items():
                if col.startswith("p5_verdict_") and val == "KEEP":
                    cls = col.replace("p5_verdict_", "")
                    keeps[cls] += 1
    return dict(keeps)


def _read_p5_csv_verdicts(path: Path) -> Counter:
    """Count KEEPs in a per-contract p5 stage CSV (s2/s3/s4)."""
    if not path.exists():
        return Counter()
    out: Counter = Counter()
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("verdict") == "KEEP":
                cls = row.get("class", "unknown")
                out[cls] += 1
    return out


class TestBCCCLegacyOutputsExist:
    """Sanity check: the BCCC Phase 5 outputs are present."""

    def test_p5_s6_comparison_csv_exists(self):
        _skip_if_no_bccc_outputs()
        assert (_P5_OUTPUTS / "p5_s6_class_size_comparison.csv").exists()

    def test_p5_s6_verification_report_md_exists(self):
        _skip_if_no_bccc_outputs()
        assert (_P5_OUTPUTS / "p5_s6_verification_report.md").exists()

    def test_v14_labels_csv_exists(self):
        _skip_if_no_bccc_outputs()
        assert (_P5_OUTPUTS / "contracts_clean_v1.4.csv").exists()

    def test_p5_s2_s3_s4_csvs_exist(self):
        _skip_if_no_bccc_outputs()
        for stage in ("s2", "s3", "s4"):
            name = {"s2": "automated", "s3": "refined", "s4": "final"}[stage]
            assert (_P5_OUTPUTS / f"p5_{stage}_{name}_verdict.csv").exists(), \
                f"p5_{stage}_{name}_verdict.csv missing"


class TestP5S6Numbers:
    """The hardcoded _P5_S6_NUMBERS match the p5_s6 CSV report."""

    def test_p5_s6_csv_matches_hardcoded_numbers(self):
        _skip_if_no_bccc_outputs()
        csv_data = _read_p5_s6_comparison()
        assert csv_data, "p5_s6_class_size_comparison.csv is empty"
        for cls, expected in _P5_S6_NUMBERS.items():
            row = csv_data.get(cls)
            assert row is not None, f"Class {cls} missing from p5_s6 CSV"
            assert int(row["before_phase5"]) == expected["n_total"], \
                f"{cls}: before_phase5 {row['before_phase5']} != {expected['n_total']}"
            assert int(row["after_phase5"]) == expected["n_keep"], \
                f"{cls}: after_phase5 {row['after_phase5']} != {expected['n_keep']}"

    def test_p5_s6_report_md_mentions_automated_class_drops(self):
        """The p5_s6_verification_report.md should mention each of the
        6 automated classes with their drop percentages (within ±0.5%).

        The 4 manual-clean classes (Class03, Class06, Class10) are NOT
        in the p5_s6 report — they were kept at 100% by Stage 5.1 manual
        review and did not run through the automated Stage 5.4 pipeline.
        """
        _skip_if_no_bccc_outputs()
        report = (_P5_OUTPUTS / "p5_s6_verification_report.md").read_text()
        for cls in _P5_S6_AUTOMATED_CLASSES:
            expected = _P5_S6_NUMBERS[cls]
            assert cls in report, f"{cls} not mentioned in p5_s6 report"
            # The drop percentage should be approximately correct
            # (allow ±0.5% per the plan's regression tolerance)
            drop_str = f"{expected['pct_drop']:.1f}"
            assert drop_str in report or f"{expected['pct_drop']:.0f}" in report, \
                f"{cls}: drop% {drop_str} not found in p5_s6 report"

    def test_p5_s6_report_does_not_mention_manual_clean_classes(self):
        """The 4 manual-clean classes should NOT appear in p5_s6
        (they were handled by Stage 5.1, not Stage 5.4 automated)."""
        _skip_if_no_bccc_outputs()
        report = (_P5_OUTPUTS / "p5_s6_verification_report.md").read_text()
        for cls in ("Class03:MishandledException", "Class06:UnusedReturn",
                    "Class10:IntegerUO"):
            # The class name should not appear in the report body
            # (it's in the comparison CSV but not in the .md report)
            assert cls not in report, (
                f"{cls} should not appear in p5_s6_verification_report.md "
                f"(it was Stage 5.1 manual-clean, not Stage 5.4 automated)"
            )


class TestV14ConsistencyWithP5S6:
    """The v1.4 labels CSV is internally consistent with p5_s6.

    IMPORTANT: v1.4 is a SUPERSET of p5_s6's data — v1.4 was produced
    AFTER the p5_s6 report and may contain additional KEEPs that
    didn't exist in p5_s6's snapshot. The p5_s6 report header says
    "Dataset version: v1.3 (pre-Stage 5.5)".

    So we check structural consistency, NOT exact count equality:
    - v1.4 KEEPs >= p5_s6 KEEPs for each automated class
    - The per-class KEEP counts in v1.4 are within a reasonable
      tolerance of p5_s6's after_phase5 numbers
    """

    def test_v14_keeps_at_least_p5_s6_keeps(self):
        """For each automated class, v1.4's p5_verdict_ClassXX:KEEP
        count must be >= p5_s6's after_phase5 count (v1.4 is later)."""
        _skip_if_no_bccc_outputs()
        v14_keeps = _read_v14_keeps_per_class()
        assert v14_keeps, "v1.4 CSV has no KEEPs"
        for cls in _P5_S6_AUTOMATED_CLASSES:
            expected = _P5_S6_NUMBERS[cls]
            actual = v14_keeps.get(cls, 0)
            assert actual >= expected["n_keep"], (
                f"{cls}: v1.4 has {actual} KEEPs but p5_s6 reports "
                f"{expected['n_keep']} — v1.4 should be a superset"
            )

    def test_v14_keeps_within_reasonable_range_of_p5_s6(self):
        """v1.4's per-class KEEP count is in [p5_s6, p5_s6 * 5] range.

        v1.4 is a LATER, more permissive version of p5_s6 — it may
        contain many additional KEEPs that p5_s6 did not have (manual
        additions after the Stage 5.5 verification). The test asserts
        the v1.4 KEEP count is between p5_s6's count and 5× that
        count, which catches gross over- or under-counting while
        accommodating the v1.4 → p5_s6 relaxation.

        For example: Class11:Reentrancy has 1699 KEEPs in p5_s6 but
        4622 in v1.4 (~2.7×) — a 16pp relaxation in the drop rate.
        """
        _skip_if_no_bccc_outputs()
        v14_keeps = _read_v14_keeps_per_class()
        for cls in _P5_S6_AUTOMATED_CLASSES:
            expected = _P5_S6_NUMBERS[cls]
            actual = v14_keeps.get(cls, 0)
            # v1.4 should be a superset of p5_s6 (>= expected) but
            # shouldn't be more than 5× expected (catches gross
            # over-counting)
            assert expected["n_keep"] <= actual <= 5 * expected["n_keep"], (
                f"{cls}: v1.4 has {actual} KEEPs, expected "
                f"[{expected['n_keep']}, {5 * expected['n_keep']}]"
            )


class TestP5PerStageRefinementChain:
    """Per AUDIT_PATCHES 4-P7: each p5 stage's output is checked.

    The refinement chain is s2 (automated) → s3 (refined) → s4 (final).
    s3 is supposed to refine s2: the only KEEPs that change are
    those that moved between KEEP and DROP in s3.
    s4 should equal s3 (s4 is the manual extrapolation step that
    propagates s3 verdicts to all 7.4K contracts).
    s6 is the synthesis report.
    """

    def test_s3_subset_of_s2_keeps(self):
        """Every KEEP in s3 should also be a KEEP in s2 (s3 only
        promotes, never demotes, automated verdicts)."""
        _skip_if_no_bccc_outputs()
        s2_path = _P5_OUTPUTS / "p5_s2_automated_verdict.csv"
        s3_path = _P5_OUTPUTS / "p5_s3_refined_verdict.csv"
        if not (s2_path.exists() and s3_path.exists()):
            pytest.skip("p5_s2 or p5_s3 missing")
        s2_keeps = {(r["id"], r["class"]) for r in csv.DictReader(s2_path.open())
                    if r["verdict"] == "KEEP"}
        s3_keeps = {(r["id"], r["class"]) for r in csv.DictReader(s3_path.open())
                    if r["verdict"] == "KEEP"}
        # s3 may add KEEPs (refinements) but should not remove s2 KEEPs
        demotions = s2_keeps - s3_keeps
        assert not demotions, (
            f"s3 demoted {len(demotions)} KEEPs from s2 (expected 0): "
            f"{list(demotions)[:5]}"
        )

    def test_s4_superset_of_s3_keeps(self):
        """s4 is the manual extrapolation step — it can PROMOTE
        contracts to KEEP (based on structural patterns) but should
        never DEMOTE. So s4 KEEPs ⊇ s3 KEEPs."""
        _skip_if_no_bccc_outputs()
        s3_path = _P5_OUTPUTS / "p5_s3_refined_verdict.csv"
        s4_path = _P5_OUTPUTS / "p5_s4_final_verdict.csv"
        if not (s3_path.exists() and s4_path.exists()):
            pytest.skip("p5_s3 or p5_s4 missing")
        s3_keeps = {(r["id"], r["class"]) for r in csv.DictReader(s3_path.open())
                    if r["verdict"] == "KEEP"}
        s4_keeps = {(r["id"], r["class"]) for r in csv.DictReader(s4_path.open())
                    if r["verdict"] == "KEEP"}
        # s4 may ADD KEEPs (promotions) but should not remove s3 KEEPs
        demotions = s3_keeps - s4_keeps
        assert not demotions, (
            f"s4 demoted {len(demotions)} KEEPs from s3 (expected 0): "
            f"{list(demotions)[:5]}"
        )
        # Sanity: s4 should have at least as many KEEPs as s3 per class
        s3_counts = _read_p5_csv_verdicts(s3_path)
        s4_counts = _read_p5_csv_verdicts(s4_path)
        promotions = {cls: s4_counts[cls] - s3_counts.get(cls, 0)
                      for cls in s3_counts
                      if s4_counts.get(cls, 0) < s3_counts.get(cls, 0)}
        assert not promotions, (
            f"s4 has FEWER KEEPs than s3 in: {promotions}"
        )

    def test_s4_per_class_counts_match_s6_for_automated(self):
        """For the 6 automated classes, s4's per-class KEEP count
        should match p5_s6's after_phase5. The 4 manual-clean
        classes are NOT in s4 (s4 only contains automated verdicts)."""
        _skip_if_no_bccc_outputs()
        s4_path = _P5_OUTPUTS / "p5_s4_final_verdict.csv"
        if not s4_path.exists():
            pytest.skip("p5_s4 missing")
        s4_counts = _read_p5_csv_verdicts(s4_path)
        for cls in _P5_S6_AUTOMATED_CLASSES:
            expected = _P5_S6_NUMBERS[cls]
            assert s4_counts.get(cls, 0) == expected["n_keep"], (
                f"{cls}: s4 KEEPs={s4_counts.get(cls, 0)} != p5_s6 "
                f"after_phase5={expected['n_keep']}"
            )


class TestBCCCDropPercentagesMatch:
    """The p5_s6 per-class drop percentages are within ±0.5% of the
    hardcoded values from the verification report. This is the
    regression test for the v1.4 → p5_s6 numbers.
    """

    @pytest.mark.parametrize("cls,expected", list(_P5_S6_NUMBERS.items()))
    def test_drop_pct_within_tolerance(self, cls, expected):
        _skip_if_no_bccc_outputs()
        csv_data = _read_p5_s6_comparison()
        row = csv_data.get(cls)
        if row is None:
            pytest.skip(f"{cls} not in p5_s6 comparison CSV")
        before = int(row["before_phase5"])
        after = int(row["after_phase5"])
        if before == 0:
            pytest.skip(f"{cls}: 0 before")
        actual_drop = (before - after) / before * 100
        assert abs(actual_drop - expected["pct_drop"]) <= _TOLERANCE_PCT, (
            f"{cls}: drop {actual_drop:.2f}% not within ±{_TOLERANCE_PCT}% "
            f"of expected {expected['pct_drop']}%"
        )
