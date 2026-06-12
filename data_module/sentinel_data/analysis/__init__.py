"""Stage 6 — Analysis (the Run-9-failure catcher).

This module runs 6 read-only exploratory tools that surface dataset
properties before and after pipeline runs. The headline output is
`complexity_proxy_risk.md`, which catches the model learning complexity
as a proxy for vulnerability (L4 finding from Run 7/9 interpretability).

Tools:
  - balance_viz: per-class / per-source / per-tier counts + bar plot
  - feature_dist: per-class feature distributions + complexity_proxy_risk.md
  - cooccurrence: directed + conditional co-occurrence matrices + heatmap
  - overlap_detector: pairwise Jaccard similarity between source datasets
  - drift_monitor: KS test for feature + label distribution drift
  - probe_dataset: re-export from verification
"""
from sentinel_data.analysis import balance_viz, cooccurrence, drift_monitor, feature_dist, overlap_detector, probe_dataset  # noqa: F401


__all__ = [
    "balance_viz",
    "cooccurrence",
    "drift_monitor",
    "feature_dist",
    "overlap_detector",
    "probe_dataset",
]
