"""
training_logger.py — SENTINEL Structured Training Logger (Phase 4.6)

Implements the three-stream logging infrastructure specified in:
  docs/pre-run-fixes/SENTINEL-Run5-Training-Log-Specification.md

Three output streams:
  step_metrics.jsonl   — per-step granular data (loss, lr, grad_norm, vram, …)
  epoch_summary.jsonl  — one JSON line per epoch (37-field schema from Spec §8)
  alerts.jsonl         — warnings and kill events with timestamps

Alert tiers (Spec §9):
  KILL     — raise TrainingAbortError immediately (NaN loss/params/Adam state)
  WARN_SKIP — log alert, return skip=True to caller (poisoned batch, bad input)
  WARN     — log alert, continue training (VRAM, grad explosion, JK collapse, …)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM

if TYPE_CHECKING:
    from ml.src.models.sentinel_model import SentinelModel

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class TrainingAbortError(Exception):
    """Raised when a KILL-level alert fires. Caller must NOT save checkpoint."""


# ---------------------------------------------------------------------------
# Alert level constants
# ---------------------------------------------------------------------------

KILL      = "KILL"
WARN_SKIP = "WARN_SKIP"
WARN      = "WARN"

# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------

class StructuredLogger:
    """
    Three-stream structured logger for SENTINEL Run 5 training.

    Usage::

        sl = StructuredLogger("ml/logs/run5")
        sl.log_startup(dataset_paths=[...], archive_dir=Path("ml/data/archive"))

        # inside training step:
        skip = sl.check_batch(labels, graphs, step, epoch)
        if skip:
            continue
        sl.check_loss(loss_val, step, epoch)           # raises on KILL
        sl.log_step(step_metrics_dict)

        # inside epoch end:
        summary = sl.build_epoch_summary(...)
        sl.log_epoch(summary)

        sl.close()
    """

    # Spec §9.3 thresholds
    VRAM_WARN_MB       = 7500.0
    GRAD_SPIKE_FACTOR  = 100.0
    AUX_HEAD_NORM_MIN  = 1e-6
    JK_ENTROPY_MIN     = 0.5
    AUC_PR_MIN         = 0.1
    BRIER_MAX          = 0.4
    GRAD_NORM_HISTORY  = 100   # rolling window length

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._step_file  = open(self.log_dir / "step_metrics.jsonl",  "a", encoding="utf-8")
        self._epoch_file = open(self.log_dir / "epoch_summary.jsonl", "a", encoding="utf-8")
        self._alert_file = open(self.log_dir / "alerts.jsonl",        "a", encoding="utf-8")

        self._global_step: int = 0
        self._grad_norm_history: deque[float] = deque(maxlen=self.GRAD_NORM_HISTORY)
        self._loss_history: deque[float] = deque(maxlen=100)
        self._loss_spike_count: int = 0

        # AUC deltas — stored across epochs for Spec §3B.12/3B.13
        self._prev_auc_roc: dict[str, float] = {}
        self._prev_auc_pr:  dict[str, float] = {}

        # Data integrity hash — set once at startup
        self._dataset_hash: str | None = None

        # Log that all three streams are open (Gate 4.6: files exist)
        self.alert(WARN, "StructuredLogger initialised — all three log streams open", {
            "step_metrics":   str(self.log_dir / "step_metrics.jsonl"),
            "epoch_summary":  str(self.log_dir / "epoch_summary.jsonl"),
            "alerts":         str(self.log_dir / "alerts.jsonl"),
        })

    # ------------------------------------------------------------------
    # Startup checks (Spec §1.8, §1.9) — call once before first step
    # ------------------------------------------------------------------

    def log_startup(
        self,
        dataset_paths: list[Path],
        archive_dir:   Path | None = None,
        expected_node_dim: int = 11,
    ) -> None:
        """Log data integrity hash (§1.8) and archive verification (§1.9)."""
        # §1.8 — dataset files hash
        h = hashlib.sha256()
        for p in sorted(str(x) for x in dataset_paths):
            h.update(p.encode())
            if Path(p).is_file():
                stat = Path(p).stat()
                h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        self._dataset_hash = h.hexdigest()[:16]
        self._write_alert(WARN, f"[1.8] Dataset integrity hash: {self._dataset_hash}", {
            "dataset_hash":  self._dataset_hash,
            "paths_checked": [str(p) for p in dataset_paths],
        })

        # §1.9 — archive verification
        if archive_dir is not None:
            archive_ok = archive_dir.exists()
            archive_count = len(list(archive_dir.rglob("*.pt"))) if archive_ok else 0
            self._write_alert(
                WARN,
                f"[1.9] Archive verification: {'OK' if archive_ok else 'MISSING'} "
                f"({archive_count} .pt files in {archive_dir})",
                {
                    "archive_dir":   str(archive_dir),
                    "archive_exists": archive_ok,
                    "archive_pt_count": archive_count,
                },
            )
        else:
            self._write_alert(WARN, "[1.9] Archive dir not provided — skipped.", {})

    # ------------------------------------------------------------------
    # Per-step checks
    # ------------------------------------------------------------------

    def check_batch(
        self,
        labels:     torch.Tensor,
        graphs_x:   torch.Tensor | None,
        step:       int,
        epoch:      int,
    ) -> bool:
        """
        Check §1.1 (label distribution), §1.3 (feature sanity), §1.5 (NaN/Inf).
        Returns True if batch should be SKIPPED.
        """
        # §1.1 — poisoned-label check (9.2.1)
        if labels.numel() > 0:
            label_sums = labels.sum(dim=0)
            total = labels.shape[0]
            if (label_sums == 0).all():
                self._write_alert(WARN_SKIP,
                    f"[9.2.1] All-zero label batch at step={step} epoch={epoch} — skipping.",
                    {"step": step, "epoch": epoch, "label_row_count": int(total)},
                )
                return True

        # §1.5 — NaN/Inf in inputs
        if graphs_x is not None and not torch.isfinite(graphs_x).all():
            self._write_alert(WARN_SKIP,
                f"[9.2.2] NaN/Inf in graphs.x at step={step} epoch={epoch} — skipping.",
                {"step": step, "epoch": epoch},
            )
            return True

        return False

    def check_inputs(
        self,
        graphs_x:   "torch.Tensor | None",
        edge_index: "torch.Tensor | None",
        step:       int,
        epoch:      int,
    ) -> bool:
        """§1.3/1.7 — feature dim and edge_index sanity checks. Returns True if batch should be skipped."""
        if graphs_x is not None:
            if graphs_x.shape[-1] != NODE_FEATURE_DIM:
                self._write_alert(WARN,
                    f"[1.3] graphs.x feature dim={graphs_x.shape[-1]} != {NODE_FEATURE_DIM} at step={step} epoch={epoch}.",
                    {"step": step, "epoch": epoch, "feature_dim": int(graphs_x.shape[-1])},
                )
        if edge_index is not None and edge_index.numel() > 0:
            if edge_index.min() < 0:
                self._write_alert(WARN_SKIP,
                    f"[1.7] Negative edge_index at step={step} epoch={epoch} — skipping.",
                    {"step": step, "epoch": epoch},
                )
                return True
        return False

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch accumulators. Call at the start of each epoch."""
        self._loss_spike_count = 0

    def check_loss(self, loss_val: float, step: int, epoch: int) -> None:
        """§2.1 + §9.1.1 — raise TrainingAbortError if loss is NaN/Inf."""
        if not math.isfinite(loss_val):
            self._write_alert(KILL,
                f"[9.1.1] Loss is {loss_val} at step={step} epoch={epoch}. "
                "KILL: do NOT save checkpoint.",
                {"step": step, "epoch": epoch, "loss": loss_val},
            )
            raise TrainingAbortError(f"[9.1.1] Loss={loss_val} — Adam state may be corrupted. Restart from last clean checkpoint.")

        # §2.7 — loss spike detection (> 5× rolling mean)
        if self._loss_history:
            rolling_mean = sum(self._loss_history) / len(self._loss_history)
            if rolling_mean > 1e-8 and loss_val > rolling_mean * 5.0:
                self._loss_spike_count += 1
                self._write_alert(WARN,
                    f"[2.7] Loss spike: {loss_val:.4f} > 5×rolling_mean={rolling_mean:.4f} "
                    f"at step={step} epoch={epoch} (spike #{self._loss_spike_count}).",
                    {"step": step, "epoch": epoch, "loss": loss_val, "rolling_mean": rolling_mean},
                )
        self._loss_history.append(loss_val)

    def check_parameters(self, model: "SentinelModel", step: int, epoch: int) -> None:
        """§2.2 + §9.1.2 — scan all param.data for NaN/Inf; KILL if found."""
        for name, param in model.named_parameters():
            if not torch.isfinite(param.data).all():
                self._write_alert(KILL,
                    f"[9.1.2] NaN/Inf in param '{name}' at step={step} epoch={epoch}. "
                    "KILL: Adam state likely corrupted.",
                    {"step": step, "epoch": epoch, "param": name},
                )
                raise TrainingAbortError(f"[9.1.2] Parameter '{name}' contains NaN/Inf.")

    def check_adam_state(self, optimizer: torch.optim.Optimizer, step: int, epoch: int) -> None:
        """§2.6 + §9.1.3 — check exp_avg / exp_avg_sq for NaN/Inf."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state is None:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    buf = state.get(key)
                    if buf is not None and not torch.isfinite(buf).all():
                        self._write_alert(KILL,
                            f"[9.1.3] Adam state '{key}' is NaN/Inf at step={step} epoch={epoch}. "
                            "KILL: optimizer state is permanently corrupted.",
                            {"step": step, "epoch": epoch, "adam_key": key},
                        )
                        raise TrainingAbortError(f"[9.1.3] Adam '{key}' is NaN/Inf — must restart from clean checkpoint.")

    def check_vram(self, step: int, epoch: int) -> None:
        """§6.1 + §9.3.1 — warn if VRAM exceeds 7500 MB."""
        if not torch.cuda.is_available():
            return
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        if peak_mb > self.VRAM_WARN_MB:
            self._write_alert(WARN,
                f"[9.3.1] VRAM peak {peak_mb:.0f} MB > {self.VRAM_WARN_MB:.0f} MB at step={step}.",
                {"step": step, "epoch": epoch, "vram_peak_mb": round(peak_mb, 1)},
            )

    def check_grad_norm(self, grad_norm: float, step: int, epoch: int) -> None:
        """§2.4 + §2.8 + §9.3.2 — warn on gradient explosion."""
        self._grad_norm_history.append(grad_norm)
        if len(self._grad_norm_history) >= 10:
            rolling_mean = sum(self._grad_norm_history) / len(self._grad_norm_history)
            if rolling_mean > 1e-8 and grad_norm > rolling_mean * self.GRAD_SPIKE_FACTOR:
                self._write_alert(WARN,
                    f"[9.3.2] Grad norm spike: {grad_norm:.2f} > {self.GRAD_SPIKE_FACTOR}×rolling_mean={rolling_mean:.3f}.",
                    {"step": step, "epoch": epoch, "grad_norm": grad_norm, "rolling_mean": rolling_mean},
                )

    # ------------------------------------------------------------------
    # Per-epoch model health checks (Spec §4.3, §4.2, §9.3.3/4)
    # ------------------------------------------------------------------

    def check_aux_head(self, model: "SentinelModel", epoch: int) -> dict[str, float]:
        """§4.3/4.4 + §9.3.3 — aux_phase2 final-Linear weight/bias norms."""
        result: dict[str, float] = {}
        head = getattr(model, "aux_phase2", None)
        if head is None:
            return result
        head = getattr(head, "_orig_mod", head)  # unwrap torch.compile OptimizedModule
        final_linear = head[-1]
        w_norm = final_linear.weight.data.norm().item()
        b_norm = final_linear.bias.data.norm().item() if final_linear.bias is not None else 0.0
        result["aux_weight_norm"] = w_norm
        result["aux_bias_norm"]   = b_norm
        if w_norm < self.AUX_HEAD_NORM_MIN:
            self._write_alert(WARN,
                f"[9.3.3] aux_phase2 final-Linear weight norm={w_norm:.2e} < {self.AUX_HEAD_NORM_MIN:.0e} at epoch={epoch}. "
                "Aux head may be disconnected from gradient flow.",
                {"epoch": epoch, "aux_weight_norm": w_norm},
            )
        return result

    def check_jk_entropy(self, jk_weights: list[float], epoch: int) -> float:
        """§4.2 + §9.3.4 — Shannon entropy of JK attention weights."""
        if not jk_weights:
            return 0.0
        p = np.array(jk_weights, dtype=float)
        p = p / (p.sum() + 1e-12)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        if ent < self.JK_ENTROPY_MIN:
            self._write_alert(WARN,
                f"[9.3.4] JK weight entropy={ent:.3f} < {self.JK_ENTROPY_MIN} at epoch={epoch}. "
                "JK fusion collapsed to a single phase.",
                {"epoch": epoch, "jk_entropy": ent, "jk_weights": jk_weights},
            )
        return ent

    # ------------------------------------------------------------------
    # AUC / Brier / ECE computations (Spec §3B)
    # ------------------------------------------------------------------

    def compute_auc_metrics(
        self,
        y_true:      np.ndarray,  # [N, C]
        y_probs:     np.ndarray,  # [N, C]
        class_names: list[str],
    ) -> dict:
        """§3B.1–3B.6, 3B.12/13 — per-label and macro/micro AUC-ROC + AUC-PR."""
        from sklearn.metrics import roc_auc_score, average_precision_score

        auc_roc: dict[str, float] = {}
        auc_pr:  dict[str, float] = {}

        for c, name in enumerate(class_names):
            y_c = y_true[:, c]
            p_c = y_probs[:, c]
            if y_c.sum() == 0 or y_c.sum() == len(y_c):
                # Cannot compute AUC on all-same-class column
                auc_roc[name] = float("nan")
                auc_pr[name]  = float("nan")
                continue
            try:
                auc_roc[name] = float(roc_auc_score(y_c, p_c))
            except Exception:
                auc_roc[name] = float("nan")
            try:
                auc_pr[name] = float(average_precision_score(y_c, p_c))
            except Exception:
                auc_pr[name] = float("nan")

        valid_roc = [v for v in auc_roc.values() if math.isfinite(v)]
        valid_pr  = [v for v in auc_pr.values()  if math.isfinite(v)]
        auc_roc_macro = float(np.mean(valid_roc)) if valid_roc else float("nan")
        auc_pr_macro  = float(np.mean(valid_pr))  if valid_pr  else float("nan")

        # §3B.5/6 — micro (pool all labels)
        try:
            auc_roc_micro = float(roc_auc_score(y_true, y_probs, average="micro"))
        except Exception:
            auc_roc_micro = float("nan")
        try:
            auc_pr_micro = float(average_precision_score(y_true, y_probs, average="micro"))
        except Exception:
            auc_pr_micro = float("nan")

        # §3B.12/13 — epoch-over-epoch deltas
        auc_roc_delta: dict[str, float] = {}
        auc_pr_delta:  dict[str, float] = {}
        for name in class_names:
            prev_roc = self._prev_auc_roc.get(name)
            prev_pr  = self._prev_auc_pr.get(name)
            auc_roc_delta[name] = auc_roc[name] - prev_roc if prev_roc is not None and math.isfinite(auc_roc[name]) else float("nan")
            auc_pr_delta[name]  = auc_pr[name]  - prev_pr  if prev_pr  is not None and math.isfinite(auc_pr[name])  else float("nan")

        self._prev_auc_roc = {k: v for k, v in auc_roc.items() if math.isfinite(v)}
        self._prev_auc_pr  = {k: v for k, v in auc_pr.items()  if math.isfinite(v)}

        # §9.3.6b — warn if AUC-PR < threshold for any label
        for name, ap in auc_pr.items():
            if math.isfinite(ap) and ap < self.AUC_PR_MIN:
                self._write_alert(WARN,
                    f"[9.3.6b] AUC-PR={ap:.3f} < {self.AUC_PR_MIN} for label '{name}'. "
                    "Agent will receive near-random probability signal for this vulnerability.",
                    {"label": name, "auc_pr": ap},
                )

        return {
            "auc_roc_per_label":  auc_roc,
            "auc_pr_per_label":   auc_pr,
            "auc_roc_macro":      auc_roc_macro,
            "auc_pr_macro":       auc_pr_macro,
            "auc_roc_micro":      auc_roc_micro,
            "auc_pr_micro":       auc_pr_micro,
            "auc_roc_delta":      auc_roc_delta,
            "auc_pr_delta":       auc_pr_delta,
        }

    def compute_brier(
        self,
        y_true:      np.ndarray,  # [N, C]
        y_probs:     np.ndarray,  # [N, C]
        class_names: list[str],
    ) -> dict:
        """§3B.7/8 — per-label and overall Brier Score."""
        brier_per: dict[str, float] = {}
        for c, name in enumerate(class_names):
            brier_per[name] = float(np.mean((y_probs[:, c] - y_true[:, c]) ** 2))
        brier_overall = float(np.mean((y_probs - y_true) ** 2))

        # §9.3.6d — warn on severe miscalibration
        for name, bs in brier_per.items():
            if bs > self.BRIER_MAX:
                self._write_alert(WARN,
                    f"[9.3.6d] Brier Score={bs:.3f} > {self.BRIER_MAX} for label '{name}'. "
                    "Severe miscalibration — agent cannot trust probabilities for this label.",
                    {"label": name, "brier_score": bs},
                )

        return {
            "brier_score_per_label": brier_per,
            "brier_score_overall":   brier_overall,
        }

    def compute_ece(
        self,
        y_true:  np.ndarray,  # [N, C]
        y_probs: np.ndarray,  # [N, C]
        n_bins:  int = 10,
    ) -> float:
        """§3.9 — Expected Calibration Error (pooled across all labels)."""
        bins   = np.linspace(0.0, 1.0, n_bins + 1)
        total  = y_true.size
        ece    = 0.0
        flat_p = y_probs.ravel()
        flat_y = y_true.ravel()
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            is_last = (i == n_bins - 1)
            mask = (flat_p >= lo) & (flat_p <= hi if is_last else flat_p < hi)
            if mask.sum() == 0:
                continue
            acc  = flat_y[mask].mean()
            conf = flat_p[mask].mean()
            ece += (mask.sum() / total) * abs(acc - conf)
        return float(ece)

    def compute_prob_stats(
        self,
        y_probs:     np.ndarray,  # [N, C]
        class_names: list[str],
    ) -> dict[str, dict]:
        """§3B.10 — probability distribution stats per label."""
        stats: dict[str, dict] = {}
        for c, name in enumerate(class_names):
            p = y_probs[:, c]
            stats[name] = {
                "min":  float(p.min()),
                "max":  float(p.max()),
                "mean": float(p.mean()),
                "std":  float(p.std()),
                "p5":   float(np.percentile(p, 5)),
                "p50":  float(np.percentile(p, 50)),
                "p95":  float(np.percentile(p, 95)),
            }
        return stats

    def check_f1_auc_divergence(
        self,
        f1_per_class:    dict[str, float],
        auc_roc_per:     dict[str, float],
        auc_roc_delta:   dict[str, float],
        epoch: int,
    ) -> bool:
        """§3B.15 + §9.3.6c — flag F1 improving while AUC degrades."""
        diverged = False
        for name in f1_per_class:
            f1  = f1_per_class.get(name, 0.0)
            roc = auc_roc_per.get(name, float("nan"))
            d   = auc_roc_delta.get(name, float("nan"))
            if not math.isfinite(roc) or not math.isfinite(d):
                continue
            if d < -0.02 and f1 > 0.0:
                self._write_alert(WARN,
                    f"[9.3.6c] F1-AUC divergence for '{name}' at epoch={epoch}: "
                    f"F1={f1:.3f} but AUC-ROC delta={d:+.4f} (degrading). "
                    "Model is getting worse at ranking while threshold-level metric improves.",
                    {"epoch": epoch, "label": name, "f1": f1, "auc_roc_delta": d},
                )
                diverged = True
        return diverged

    # ------------------------------------------------------------------
    # Epoch summary (Spec §8 — 37 fields)
    # ------------------------------------------------------------------

    def build_epoch_summary(
        self,
        epoch:              int,
        train_loss:         float,
        val_loss:           float | None,
        main_loss:          float,
        aux_loss:           float,
        total_loss:         float,
        lr:                 float,
        grad_norm_total:    float,
        grad_norm_max_layer: tuple[str, float],
        param_nan_count:    int,
        grad_nan_count:     int,
        vram_peak_mb:       float,
        vram_current_mb:    float,
        label_dist_train:   dict[str, int],
        label_dist_val:     dict[str, int],
        aux_weight_norm:    float,
        aux_bias_norm:      float,
        jk_weight_entropy:  float,
        prediction_entropy: float,
        per_class_f1:       dict[str, float],
        auc_metrics:        dict,
        brier_metrics:      dict,
        ece:                float,
        temperature:        float,
        prob_dist_stats:    dict,
        f1_auc_divergence:  bool,
        epoch_duration_sec: float,
        steps_per_epoch:    int,
        gpu_util_mean_pct:  float,
        loss_spike_count:   int,
        grad_zero_count:    int,
    ) -> dict:
        """Assemble the 37-field epoch summary dict (Spec §8)."""
        summary = {
            # §8.1–8.7
            "epoch":                epoch,
            "train_loss":           round(train_loss, 6),
            "val_loss":             round(val_loss, 6) if val_loss is not None else None,
            "main_loss":            round(main_loss, 6),
            "aux_loss":             round(aux_loss, 6),
            "total_loss":           round(total_loss, 6),
            "lr":                   lr,
            # §8.8–8.11
            "grad_norm_total":      round(grad_norm_total, 6),
            "grad_norm_max_layer":  {"name": grad_norm_max_layer[0], "norm": round(grad_norm_max_layer[1], 6)},
            "param_nan_count":      param_nan_count,
            "grad_nan_count":       grad_nan_count,
            # §8.12–8.13
            "vram_peak_mb":         round(vram_peak_mb, 1),
            "vram_current_mb":      round(vram_current_mb, 1),
            # §8.14–8.15
            "label_dist_train":     label_dist_train,
            "label_dist_val":       label_dist_val,
            # §8.16–8.18
            "aux_weight_norm":      round(aux_weight_norm, 6),
            "aux_bias_norm":        round(aux_bias_norm, 6),
            "jk_weight_entropy":    round(jk_weight_entropy, 6),
            # §8.19–8.20
            "prediction_entropy_mean": round(prediction_entropy, 6),
            "per_class_f1":         {k: round(v, 6) for k, v in per_class_f1.items()},
            # §8.21–8.24
            "auc_roc_per_label":    auc_metrics.get("auc_roc_per_label", {}),
            "auc_pr_per_label":     auc_metrics.get("auc_pr_per_label", {}),
            "auc_roc_macro":        auc_metrics.get("auc_roc_macro"),
            "auc_pr_macro":         auc_metrics.get("auc_pr_macro"),
            # §8.25–8.26
            "brier_score_per_label": brier_metrics.get("brier_score_per_label", {}),
            "brier_score_overall":   brier_metrics.get("brier_score_overall"),
            # §8.27–8.29
            "ece":                  round(ece, 6),
            "temperature":          temperature,
            "prob_dist_stats":      prob_dist_stats,
            # §8.30–8.32
            "auc_roc_delta":        auc_metrics.get("auc_roc_delta", {}),
            "auc_pr_delta":         auc_metrics.get("auc_pr_delta", {}),
            "f1_auc_divergence":    f1_auc_divergence,
            # §8.33–8.37
            "epoch_duration_sec":   round(epoch_duration_sec, 2),
            "steps_per_epoch":      steps_per_epoch,
            "gpu_util_mean_pct":    round(gpu_util_mean_pct, 1),
            "loss_spike_count":     loss_spike_count,
            "grad_zero_count":      grad_zero_count,
        }
        return summary

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def log_calibration(
        self,
        temperatures: dict,
        ece_pre:      float,
        ece_post:     float,
        epoch:        int,
    ) -> None:
        """§5 — log temperature scaling calibration result."""
        entry = {
            "type":         "calibration",
            "epoch":        epoch,
            "temperatures": temperatures,
            "ece_pre":      round(ece_pre, 6),
            "ece_post":     round(ece_post, 6),
            "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
        }
        self._epoch_file.write(json.dumps(entry, default=_json_default) + "\n")
        self._epoch_file.flush()

    def log_graph_stats(
        self,
        edge_type_counts: dict,
        cei_label_dist:   dict,
        epoch:            int,
    ) -> None:
        """§7 — log graph-level dataset statistics."""
        entry = {
            "type":             "graph_stats",
            "epoch":            epoch,
            "edge_type_counts": edge_type_counts,
            "cei_label_dist":   cei_label_dist,
            "timestamp":        datetime.now(tz=timezone.utc).isoformat(),
        }
        self._epoch_file.write(json.dumps(entry, default=_json_default) + "\n")
        self._epoch_file.flush()

    def log_step(self, metrics: dict) -> None:
        """Write one step record to step_metrics.jsonl (Spec §10.1)."""
        self._global_step += 1
        self._step_file.write(json.dumps(metrics, default=_json_default) + "\n")
        self._step_file.flush()

    def log_epoch(self, summary: dict) -> None:
        """Write one epoch summary record to epoch_summary.jsonl (Spec §8)."""
        self._epoch_file.write(json.dumps(summary, default=_json_default) + "\n")
        self._epoch_file.flush()

    def alert(self, level: str, message: str, data: dict | None = None) -> None:
        """
        Log an alert and, for KILL level, raise TrainingAbortError.
        WARN_SKIP alerts do NOT raise — caller is responsible for skip logic.
        """
        self._write_alert(level, message, data or {})
        if level == KILL:
            raise TrainingAbortError(f"KILL: {message}")

    def _write_alert(self, level: str, message: str, data: dict) -> None:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level":     level,
            "message":   message,
        }
        entry.update(data)
        self._alert_file.write(json.dumps(entry, default=_json_default) + "\n")
        self._alert_file.flush()

    def close(self) -> None:
        """Flush and close all three log streams."""
        for f in (self._step_file, self._epoch_file, self._alert_file):
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utility: compute label distribution dict from a batch/epoch label tensor
# ---------------------------------------------------------------------------

def label_dist_from_tensor(labels: torch.Tensor, class_names: list[str]) -> dict[str, int]:
    """Return per-class positive counts for a [N, C] label tensor."""
    counts = labels.sum(dim=0).long().tolist()
    return {name: int(c) for name, c in zip(class_names, counts)}


def compute_grad_stats(model: nn.Module) -> tuple[float, tuple[str, float], int]:
    """
    Compute total grad norm, (name, norm) of layer with highest grad norm,
    and count of parameters with zero gradient.
    Returns (total_norm, (max_name, max_norm), zero_count).
    """
    total_sq   = 0.0
    max_norm   = 0.0
    max_name   = ""
    zero_count = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            zero_count += 1
            continue
        gn = p.grad.data.norm(2).item()
        total_sq += gn ** 2
        if gn > max_norm:
            max_norm = gn
            max_name = name
        if gn == 0.0:
            zero_count += 1
    return math.sqrt(total_sq), (max_name, max_norm), zero_count


def _json_default(obj):
    """Fallback JSON serialiser for numpy/torch scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return str(obj)
