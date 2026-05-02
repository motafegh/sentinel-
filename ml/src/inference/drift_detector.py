"""
drift_detector.py — Feature-distribution drift detection (T2-B)

WHY THIS EXISTS
───────────────
SENTINEL's model was trained on BCCC-SCsVul-2024 (a 2024 historical snapshot).
Production contracts from 2026+ may have different structural distributions —
different average function counts, state-variable densities, CFG complexity.
If the input distribution drifts far enough from training, model accuracy
degrades silently without any error signal.

A Kolmogorov–Smirnov (KS) test compares the current rolling window of requests
against a pre-computed baseline. p < 0.05 fires a Prometheus counter.

BASELINE STRATEGY — read this before calling compute_drift_baseline.py
────────────────────────────────────────────────────────────────────────
DO NOT use ml/data/graphs/ (training data) as the baseline.  The BCCC-2024
corpus is a historical snapshot; using it will cause the KS test to fire on
almost every modern 2026 production contract.

Correct approach:
  Phase 1 (warm-up): collect stats from the first N_WARMUP real requests;
                     suppress all alerts.
  Phase 2 (active): write drift_baseline.json from warm-up data;
                    enable KS alerts.

INTEGRATION
───────────
  detector = DriftDetector(baseline_path="ml/data/drift_baseline.json")
  ...
  # per request:
  detector.update_stats({"num_nodes": result["num_nodes"], "num_edges": ...})
  # every 50 requests:
  detector.check()
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

from prometheus_client import Counter

logger = logging.getLogger(__name__)

# Prometheus counter: labels by stat name so dashboards can pivot per feature.
_drift_counter = Counter(
    "sentinel_drift_alerts_total",
    "Number of KS drift alerts fired (p < 0.05)",
    ["stat"],
)

# p-value threshold below which an alert fires.
KS_ALPHA = 0.05

# Minimum number of current-window samples required before running KS.
# Fewer samples produce unreliable p-values.
MIN_SAMPLES_FOR_KS = 30


class DriftDetector:
    """
    Rolling KS-based drift detector for SENTINEL inference requests.

    Args:
        baseline_path:  Path to drift_baseline.json produced by
                        compute_drift_baseline.py.  If None or the file does
                        not exist, the detector runs in warm-up mode only.
        n_warmup:       Number of requests to collect before enabling alerts
                        (ignored if baseline_path is provided and exists).
        buffer_size:    Size of the rolling request buffer for KS comparison.
    """

    def __init__(
        self,
        baseline_path: str | Path | None = None,
        n_warmup:      int = 500,
        buffer_size:   int = 200,
    ) -> None:
        from scipy.stats import ks_2samp  # lazy import — scipy is optional dep
        self._ks_2samp = ks_2samp

        self._n_warmup    = n_warmup
        self._buffer: deque[dict[str, float]] = deque(maxlen=buffer_size)
        self._n_seen      = 0
        self._baseline:   dict[str, list[float]] | None = None
        self._warmup_done = False

        if baseline_path is not None:
            bp = Path(baseline_path)
            if bp.exists():
                with open(bp) as f:
                    self._baseline = json.load(f)
                self._warmup_done = True
                logger.info(
                    f"DriftDetector: baseline loaded from {bp} "
                    f"({len(self._baseline)} stats)"
                )
            else:
                logger.warning(
                    f"DriftDetector: baseline path {bp} not found — "
                    f"warm-up mode active (alerts suppressed for first {n_warmup} requests)"
                )
        else:
            logger.info(
                f"DriftDetector: no baseline_path — "
                f"warm-up mode active ({n_warmup} requests before alerts)"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def update_stats(self, stats: dict[str, float]) -> None:
        """
        Record per-request feature statistics in the rolling buffer.

        Args:
            stats: Dict of {stat_name: float_value}.  Keys must match those
                   in the baseline JSON.  Extra keys are stored and ignored
                   at check() time if they are not in the baseline.
        """
        self._buffer.append(stats)
        self._n_seen += 1

        if not self._warmup_done and self._n_seen >= self._n_warmup:
            self._warmup_done = True
            logger.info(
                f"DriftDetector warm-up complete after {self._n_seen} requests. "
                "Alerts will now fire on detected drift."
            )

    def check(self) -> dict[str, float]:
        """
        Run KS tests for all stats that appear in both the baseline and the
        current rolling buffer.

        Returns:
            {stat_name: p_value} for every stat tested.
            Empty dict if warm-up is not complete, no baseline is loaded,
            or the buffer has fewer than MIN_SAMPLES_FOR_KS entries.

        Side effects:
            Increments sentinel_drift_alerts_total{stat=<name>} for each
            stat whose p-value falls below KS_ALPHA (0.05).
        """
        if not self._warmup_done or self._baseline is None:
            return {}
        if len(self._buffer) < MIN_SAMPLES_FOR_KS:
            return {}

        results: dict[str, float] = {}
        for stat_name, baseline_values in self._baseline.items():
            current_values = [
                s[stat_name] for s in self._buffer if stat_name in s
            ]
            if len(current_values) < MIN_SAMPLES_FOR_KS:
                continue

            _, p_value = self._ks_2samp(baseline_values, current_values)
            results[stat_name] = float(p_value)

            if p_value < KS_ALPHA:
                _drift_counter.labels(stat=stat_name).inc()
                logger.warning(
                    f"Drift detected: {stat_name} p={p_value:.4f} "
                    f"(baseline n={len(baseline_values)}, "
                    f"current n={len(current_values)})"
                )

        return results

    def dump_warmup_stats(self) -> list[dict[str, float]]:
        """Return current buffer contents as a list for baseline construction."""
        return list(self._buffer)

    # ─────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def warmup_done(self) -> bool:
        return self._warmup_done

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)
