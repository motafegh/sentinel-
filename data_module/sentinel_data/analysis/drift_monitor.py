"""Stage 6 — drift_monitor: KS test for feature + label distribution drift.

D-6.5 — the version-update gate.

Per AUDIT_PATCHES 6-P3, the drift monitor reports BOTH:
  1. Per-feature KS test (numerical features like node_count, loc)
  2. Per-class label distribution KS test (binary: count_positive / total)

This catches two distinct drift patterns:
  - Feature drift: the contracts themselves got bigger/smaller
  - Label drift: the class balance shifted even if the contracts look the same

Inputs:
  - new_labels_dir: Path to the new version's merged labels
  - new_rep_root: Path to the new version's representations
  - baseline_labels_dir: Path to the baseline version's merged labels
  - baseline_rep_root: Path to the baseline version's representations

The CLI passes these via the `--corpus` and `--baseline-version` flags.

Outputs:
  - data/analysis/<run_id>/drift_report.md
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sentinel_data.labeling.schema import class_names


@dataclass
class FeatureKSResult:
    feature: str
    n_baseline: int
    n_new: int
    statistic: float       # KS statistic
    pvalue: float
    warning: bool
    insufficient_sample: bool = False


@dataclass
class LabelKSResult:
    class_name: str
    n_baseline_pos: int
    n_new_pos: int
    n_baseline_total: int
    n_new_total: int
    rate_baseline: float
    rate_new: float
    statistic: float
    pvalue: float
    warning: bool
    insufficient_sample: bool = False


@dataclass
class DriftReport:
    feature_results: list[FeatureKSResult] = field(default_factory=list)
    label_results: list[LabelKSResult] = field(default_factory=list)
    overall_warning: bool = False


def _ks_test_2samp(values_a: list[float], values_b: list[float]) -> tuple[float, float]:
    """Two-sample KS test (independent). Returns (statistic, pvalue).

    Uses scipy.stats.ks_2samp if available; otherwise falls back to a manual
    implementation. scipy is the standard library for this; we don't want
    to reinvent it.
    """
    if not values_a or not values_b:
        return 0.0, 1.0
    try:
        from scipy import stats
        result = stats.ks_2samp(values_a, values_b)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        # Fallback: crude approximation using absolute difference of empirical CDFs
        all_vals = sorted(set(values_a + values_b))
        a_set = set(values_a)
        b_set = set(values_b)
        max_diff = 0.0
        for v in all_vals:
            cdf_a = sum(1 for x in values_a if x <= v) / len(values_a)
            cdf_b = sum(1 for x in values_b if x <= v) / len(values_b)
            max_diff = max(max_diff, abs(cdf_a - cdf_b))
        return max_diff, float("nan")


def _iter_label_features(
    labels_dir: Path, rep_root: Path, preproc_root: Path,
) -> Iterable[tuple[dict, dict]]:
    """Yield (label_dict, features_dict) for each contract with both available.

    Single pass: reads each label file once and computes all features per
    contract (the 2 node/edge counts from .rep.json + the 4 .sol proxies).
    """
    from sentinel_data.analysis.feature_dist import _features_for_contract

    if not labels_dir.exists():
        return
    for p in sorted(Path(labels_dir).glob("*.labels.json")):
        try:
            lj = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sha = lj.get("sha256", "")
        sources = lj.get("sources") or ["unknown"]
        source = sources[0] if sources else "unknown"
        feats = _features_for_contract(sha, source, rep_root, preproc_root)
        if feats:
            yield lj, feats


def compute_feature_drift(
    baseline_labels: Path, baseline_rep: Path, baseline_preproc: Path,
    new_labels: Path, new_rep: Path, new_preproc: Path,
    features: list[str] | None = None,
    pvalue_warn: float = 0.01,
    min_sample: int = 30,
) -> list[FeatureKSResult]:
    """Per-feature KS test (2-sample, independent) between baseline and new.

    Single pass: for each (label_dir, rep_root, preproc_root) we read every
    label file once, compute ALL 6 features per contract, and bucket by
    feature. Avoids the O(F * N) trap of re-scanning labels for each feature.
    """
    feats = features or ["node_count", "edge_count", "loc", "function_count",
                          "cyclomatic_complexity", "call_depth"]

    def _bucket_by_feature(
        labels_dir: Path, rep_root: Path, preproc_root: Path,
    ) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {f: [] for f in feats}
        for _, feature_dict in _iter_label_features(labels_dir, rep_root, preproc_root):
            for f in feats:
                if f in feature_dict:
                    out[f].append(feature_dict[f])
        return out

    base_buckets = _bucket_by_feature(baseline_labels, baseline_rep, baseline_preproc)
    new_buckets = _bucket_by_feature(new_labels, new_rep, new_preproc)

    results = []
    for feat in feats:
        v_base = base_buckets[feat]
        v_new = new_buckets[feat]
        if len(v_base) < min_sample or len(v_new) < min_sample:
            results.append(FeatureKSResult(
                feature=feat, n_baseline=len(v_base), n_new=len(v_new),
                statistic=0.0, pvalue=1.0, warning=False, insufficient_sample=True,
            ))
            continue
        stat, p = _ks_test_2samp(v_base, v_new)
        results.append(FeatureKSResult(
            feature=feat, n_baseline=len(v_base), n_new=len(v_new),
            statistic=round(stat, 4), pvalue=round(p, 4) if not math.isnan(p) else p,
            warning=(p < pvalue_warn),
        ))
    return results


def compute_label_drift(
    baseline_labels: Path, new_labels: Path,
    pvalue_warn: float = 0.01,
    min_sample: int = 30,
) -> list[LabelKSResult]:
    classes = class_names()

    def _rates(labels_dir: Path) -> dict[str, tuple[int, int]]:
        """Return {class: (n_positive, n_total)} per class."""
        if not labels_dir.exists():
            return {c: (0, 0) for c in classes}
        out = {c: (0, 0) for c in classes}
        for p in sorted(Path(labels_dir).glob("*.labels.json")):
            try:
                lj = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for c in classes:
                entry = lj.get("classes", {}).get(c, {})
                if entry.get("value") == 1:
                    out[c] = (out[c][0] + 1, out[c][1] + 1)
                else:
                    out[c] = (out[c][0], out[c][1] + 1)
        return out

    base_rates = _rates(baseline_labels)
    new_rates = _rates(new_labels)
    results = []
    for c in classes:
        bp, bt = base_rates[c]
        np_, nt = new_rates[c]
        if bt < min_sample or nt < min_sample:
            results.append(LabelKSResult(
                class_name=c,
                n_baseline_pos=bp, n_new_pos=np_,
                n_baseline_total=bt, n_new_total=nt,
                rate_baseline=bp / bt if bt else 0.0,
                rate_new=np_ / nt if nt else 0.0,
                statistic=0.0, pvalue=1.0, warning=False, insufficient_sample=True,
            ))
            continue
        # Two-sample KS on binary labels: 0/1 arrays
        base_arr = [1.0] * bp + [0.0] * (bt - bp)
        new_arr = [1.0] * np_ + [0.0] * (nt - np_)
        stat, p = _ks_test_2samp(base_arr, new_arr)
        results.append(LabelKSResult(
            class_name=c,
            n_baseline_pos=bp, n_new_pos=np_,
            n_baseline_total=bt, n_new_total=nt,
            rate_baseline=round(bp / bt, 4) if bt else 0.0,
            rate_new=round(np_ / nt, 4) if nt else 0.0,
            statistic=round(stat, 4), pvalue=round(p, 4) if not math.isnan(p) else p,
            warning=(p < pvalue_warn),
        ))
    return results


def write_drift_report(report: DriftReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Drift Report",
        "",
        "**Generated by:** `sentinel_data.analysis.drift_monitor` (Stage 6 — D-6.5).",
        "**Per AUDIT_PATCHES 6-P3:** both feature drift AND label distribution drift are checked.",
        "",
        f"**Overall:** {'⚠ DRIFT DETECTED' if report.overall_warning else '✓ no significant drift'}",
        "",
        "## Feature Drift (per-feature KS test)",
        "",
        "| Feature | n_baseline | n_new | statistic | pvalue | WARNING | Note |",
        "|---------|-----------:|------:|----------:|-------:|---------|------|",
    ]
    for r in report.feature_results:
        note = "insufficient sample" if r.insufficient_sample else ""
        lines.append(
            f"| {r.feature} | {r.n_baseline} | {r.n_new} | "
            f"{r.statistic:.4f} | {r.pvalue:.4f} | "
            f"{'⚠' if r.warning else '✓'} | {note} |"
        )
    lines += [
        "",
        "## Label Distribution Drift (per-class KS test)",
        "",
        "| Class | rate_baseline | rate_new | n_baseline_pos / total | n_new_pos / total | statistic | pvalue | WARNING |",
        "|-------|--------------:|---------:|------------------------|-------------------|----------:|-------:|---------|",
    ]
    for r in report.label_results:
        lines.append(
            f"| {r.class_name} | {r.rate_baseline:.4f} | {r.rate_new:.4f} | "
            f"{r.n_baseline_pos} / {r.n_baseline_total} | "
            f"{r.n_new_pos} / {r.n_new_total} | "
            f"{r.statistic:.4f} | {r.pvalue:.4f} | "
            f"{'⚠' if r.warning else '✓'} |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        "A WARNING in either table indicates significant drift between the baseline",
        "and the new version. The ML training pipeline can opt to require explicit",
        "acknowledgement of any WARNING before training.",
        "",
    ]
    output_path.write_text("\n".join(lines))
    return output_path


def run_drift_monitor(
    baseline_labels: Path, baseline_rep: Path, baseline_preproc: Path,
    new_labels: Path, new_rep: Path, new_preproc: Path,
    output_dir: Path,
    pvalue_warn: float = 0.01,
    min_sample: int = 30,
) -> dict:
    feat = compute_feature_drift(
        baseline_labels, baseline_rep, baseline_preproc,
        new_labels, new_rep, new_preproc,
        pvalue_warn=pvalue_warn, min_sample=min_sample,
    )
    lbl = compute_label_drift(baseline_labels, new_labels,
                              pvalue_warn=pvalue_warn, min_sample=min_sample)
    report = DriftReport(
        feature_results=feat, label_results=lbl,
        overall_warning=any(r.warning for r in feat) or any(r.warning for r in lbl),
    )
    report_path = write_drift_report(report, output_dir / "drift_report.md")
    return {
        "report": str(report_path),
        "overall_warning": report.overall_warning,
        "feature_warnings": [r.feature for r in feat if r.warning],
        "label_warnings": [r.class_name for r in lbl if r.warning],
    }
