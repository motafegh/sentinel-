"""
exp_a3_jk_entropy_logging.py — Training Monitor: JK Entropy Analysis (Log Parser + MLflow Snippet)

PURPOSE
───────
Two-function script:

1. LOG PARSER — Parse an existing training log to extract per-epoch JK attention
   weight statistics (Phase1/Phase2/Phase3 mean ± std).  Compute an approximate
   Shannon entropy H from the normalized phase weights and plot the trajectory
   over training epochs.

2. MLFLOW SNIPPET — Print a ready-to-paste code block for adding per-step JK
   entropy logging to trainer.py in future runs.

LAYER / PRIORITY
─────────────────
Training Monitor — no layer classification; supports post-hoc analysis of
Run 4 JK routing behaviour.

LOG FORMAT PARSED
─────────────────
Lines matching:
  "JK attention weights — Phase1=<mean>±<std> Phase2=<mean>±<std> Phase3=<mean>±<std>"

Epoch number inferred from interleaved "Epoch N/60" plain-text markers or
from a running counter (one JK line per epoch).

PASS CRITERIA
─────────────
- Mean JK entropy (from phase weights) at LAST epoch < 0.3:
  WARN "JK has effectively collapsed to single phase"
- Mean JK entropy > 0.5: PASS

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_a3_jk_entropy_logging.py \\
        --log-file ml/logs/graphcodebert-p1-run4-20260525.log \\
        --out ml/logs/interpretability/a3_jk_entropy.json

OUTPUT
──────
    - Epoch table printed to stdout
    - Entropy plot PNG: <out_dir>/a3_jk_entropy.png
    - MLflow snippet printed to stdout
    - JSON report

EXIT CODES
──────────
    0  log parsed successfully
    1  log file not found or no JK entries found
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Matches:  JK attention weights — Phase1=0.332±0.014 Phase2=0.328±0.015 Phase3=0.340±0.025
_JK_PATTERN = re.compile(
    r"JK attention weights.*?"
    r"Phase1=(?P<p1_mean>[0-9.]+)±(?P<p1_std>[0-9.]+)\s+"
    r"Phase2=(?P<p2_mean>[0-9.]+)±(?P<p2_std>[0-9.]+)\s+"
    r"Phase3=(?P<p3_mean>[0-9.]+)±(?P<p3_std>[0-9.]+)"
)

# Matches plain-text epoch boundary headers: "Epoch 13/60"
_EPOCH_PATTERN = re.compile(r"^Epoch\s+(\d+)/\d+\s*$")

# Also matches structured log epoch summary:
# "Epoch  13/60 | Loss=..."
_EPOCH_SUMMARY_PATTERN = re.compile(r"Epoch\s+(\d+)/\d+\s*\|")


# ── Entropy computation ───────────────────────────────────────────────────────

def phase_weights_to_entropy(p1: float, p2: float, p3: float) -> float:
    """
    Compute Shannon entropy of the 3-way phase weight distribution.

    H = -sum(w_i * log(w_i + 1e-8))

    The weights are first normalized to sum to 1 (in case they don't exactly).
    """
    w = np.array([p1, p2, p3], dtype=np.float64)
    total = w.sum()
    if total > 1e-12:
        w = w / total
    return float(-np.sum(w * np.log(w + 1e-8)))


# ── Log parser ────────────────────────────────────────────────────────────────

def parse_jk_log(log_file: Path) -> list[dict]:
    """
    Parse training log and extract per-epoch JK weight statistics.

    Strategy:
    1. Scan for plain-text "Epoch N/60" lines to track current epoch.
    2. The next JK attention weights line after each epoch header belongs to
       that epoch (one JK summary line per epoch).
    3. When multiple training runs are concatenated in the same file (log
       restart), epoch numbering resets — we track restart boundaries and
       continue a global epoch counter.

    Returns:
        List of dicts, one per epoch:
            epoch, p1_mean, p1_std, p2_mean, p2_std, p3_mean, p3_std, entropy
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    entries: list[dict] = []
    current_epoch: int | None = None
    prev_epoch: int = 0
    global_epoch_offset: int = 0

    with open(str(log_file), "r", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\n")

        # Check for plain "Epoch N/60" line
        m_plain = _EPOCH_PATTERN.match(line.strip())
        if m_plain:
            ep = int(m_plain.group(1))
            # Detect restart: if ep < prev_epoch, the log was restarted
            if ep < prev_epoch:
                global_epoch_offset = prev_epoch
            current_epoch = global_epoch_offset + ep
            prev_epoch = ep
            continue

        # Check for structured log epoch summary
        m_summary = _EPOCH_SUMMARY_PATTERN.search(line)
        if m_summary and "trainer" in line:
            ep = int(m_summary.group(1))
            if ep < prev_epoch:
                global_epoch_offset = prev_epoch
            current_epoch = global_epoch_offset + ep
            prev_epoch = ep
            continue

        # Check for JK weights line
        m_jk = _JK_PATTERN.search(line)
        if m_jk:
            p1 = float(m_jk.group("p1_mean"))
            p2 = float(m_jk.group("p2_mean"))
            p3 = float(m_jk.group("p3_mean"))
            entropy = phase_weights_to_entropy(p1, p2, p3)

            ep_label = current_epoch if current_epoch is not None else len(entries) + 1

            entries.append({
                "epoch": ep_label,
                "p1_mean": round(p1, 4),
                "p1_std": round(float(m_jk.group("p1_std")), 4),
                "p2_mean": round(p2, 4),
                "p2_std": round(float(m_jk.group("p2_std")), 4),
                "p3_mean": round(p3, 4),
                "p3_std": round(float(m_jk.group("p3_std")), 4),
                "entropy": round(entropy, 4),
            })

    return entries


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_jk_entropy(entries: list[dict], out_path: Path) -> None:
    """
    Plot phase weights and entropy over epochs.

    Panel 1: Phase1 / Phase2 / Phase3 mean weights stacked area chart.
    Panel 2: Shannon entropy over epochs with pass/warn threshold lines.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs  = [e["epoch"] for e in entries]
    p1_vals = np.array([e["p1_mean"] for e in entries])
    p2_vals = np.array([e["p2_mean"] for e in entries])
    p3_vals = np.array([e["p3_mean"] for e in entries])
    h_vals  = np.array([e["entropy"] for e in entries])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Phase weights
    ax1.plot(epochs, p1_vals, "o-", label="Phase1 (struct)", color="#1f77b4", linewidth=1.5, ms=3)
    ax1.plot(epochs, p2_vals, "s-", label="Phase2 (CF/ICFG)", color="#ff7f0e", linewidth=1.5, ms=3)
    ax1.plot(epochs, p3_vals, "^-", label="Phase3 (rev-CONTAINS)", color="#2ca02c", linewidth=1.5, ms=3)
    ax1.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Uniform (1/3)")
    ax1.set_ylabel("Mean JK Phase Weight")
    ax1.set_title("JK Attention Weights per Phase over Training (Run 4)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, alpha=0.3)

    # Fill ±std bands for Phase1
    p1_std = np.array([e["p1_std"] for e in entries])
    p2_std = np.array([e["p2_std"] for e in entries])
    p3_std = np.array([e["p3_std"] for e in entries])
    ax1.fill_between(epochs, p1_vals - p1_std, p1_vals + p1_std, alpha=0.15, color="#1f77b4")
    ax1.fill_between(epochs, p2_vals - p2_std, p2_vals + p2_std, alpha=0.15, color="#ff7f0e")
    ax1.fill_between(epochs, p3_vals - p3_std, p3_vals + p3_std, alpha=0.15, color="#2ca02c")

    # Panel 2: Entropy
    ax2.plot(epochs, h_vals, "D-", color="#d62728", linewidth=2, ms=4, label="H (phase weights)")
    ax2.axhline(0.30, color="orange", linestyle="--", alpha=0.8, linewidth=1.5,
                label="WARN threshold (H<0.3 = collapsed)")
    ax2.axhline(0.50, color="green", linestyle="--", alpha=0.8, linewidth=1.5,
                label="PASS threshold (H>0.5)")
    ax2.set_ylabel("Shannon Entropy H")
    ax2.set_xlabel("Epoch")
    ax2.set_title("JK Phase-Weight Shannon Entropy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Entropy plot saved: {out_path}")


# ── MLflow snippet ────────────────────────────────────────────────────────────

_MLFLOW_SNIPPET = '''
# ────────────────────────────────────────────────────────────────────────────
# ADD TO trainer.py — after loss.backward() in the training step
# Logs per-step JK phase weights to MLflow for real-time monitoring.
# ────────────────────────────────────────────────────────────────────────────

# After: loss.backward() / optimizer.step()
# In: Trainer.train_one_epoch() loop

if self.use_mlflow and global_step % self.log_every_n_steps == 0:
    # Retrieve JK attention weights from the JK aggregator
    gnn = getattr(model, "gnn", None) or getattr(model, "gnn_encoder", None)
    if gnn is not None and hasattr(gnn, "jk") and gnn.jk is not None:
        jk = gnn.jk
        # AttentiveAggregation stores last-forward weights in .last_weights
        # (a [3] tensor on the same device as the model)
        last_w = getattr(jk, "last_weights", None)
        if last_w is not None:
            w = last_w.detach().cpu().float()
            if w.numel() == 3:
                import mlflow
                mlflow.log_metrics(
                    {
                        "train/jk_weight_phase1": float(w[0]),
                        "train/jk_weight_phase2": float(w[1]),
                        "train/jk_weight_phase3": float(w[2]),
                        "train/jk_entropy": float(
                            -(w * (w + 1e-8).log()).sum()
                        ),
                    },
                    step=global_step,
                )

# ────────────────────────────────────────────────────────────────────────────
# NOTE: If jk.last_weights is not yet stored by your JK implementation,
#       add this line inside AttentiveAggregation.forward() in gnn_encoder.py:
#
#   self.last_weights = weights.detach()   # [3] normalized phase weights
#
# Or retrieve from the JK entropy scalar already returned by GNNEncoder.forward():
#
#   _, _, jk_entropy_scalar = model.gnn(x, ei, batch, ea)
#   mlflow.log_metric("train/jk_entropy", float(jk_entropy_scalar), step=step)
# ────────────────────────────────────────────────────────────────────────────
'''


def print_mlflow_snippet() -> None:
    print("\n" + "=" * 72)
    print("MLflow JK Entropy Logging Snippet (copy into trainer.py)")
    print("=" * 72)
    print(_MLFLOW_SNIPPET)
    print("=" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse JK entropy from training log and print MLflow snippet"
    )
    parser.add_argument(
        "--log-file",
        default="ml/logs/graphcodebert-p1-run4-20260525.log",
        dest="log_file",
        help="Path to training log (default: ml/logs/graphcodebert-p1-run4-20260525.log)",
    )
    parser.add_argument(
        "--out",
        default="ml/logs/interpretability/a3_jk_entropy.json",
        help="Output JSON path (default: ml/logs/interpretability/a3_jk_entropy.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_file = Path(args.log_file)
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Parse log ──────────────────────────────────────────────────────
    try:
        entries = parse_jk_log(log_file)
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    if not entries:
        log.error(
            f"No JK attention weight lines found in {log_file}.\n"
            "Expected format: 'JK attention weights — Phase1=X.XXX±Y.YYY ...'"
        )
        return 1

    log.info(f"Parsed {len(entries)} JK epoch entries from {log_file}")

    # ── Print epoch table ──────────────────────────────────────────────────────
    header = f"{'Epoch':6s} | {'P1 mean':8s} | {'P2 mean':8s} | {'P3 mean':8s} | {'Entropy H':10s}"
    log.info("\n" + "=" * len(header))
    log.info("JK PHASE WEIGHTS PER EPOCH")
    log.info("=" * len(header))
    log.info(header)
    log.info("-" * len(header))
    for e in entries:
        log.info(
            f"  {e['epoch']:4d}   | {e['p1_mean']:8.4f} | {e['p2_mean']:8.4f} | "
            f"{e['p3_mean']:8.4f} | {e['entropy']:10.4f}"
        )
    log.info("=" * len(header))

    # ── Evaluate pass criteria ─────────────────────────────────────────────────
    last_entropy = entries[-1]["entropy"]
    log.info(f"\nLast epoch entropy: H = {last_entropy:.4f}")

    if last_entropy < 0.30:
        log.warning(
            f"WARN: JK entropy at final epoch = {last_entropy:.4f} < 0.30 — "
            "JK has effectively collapsed to a single phase. "
            "Per-node routing has become degenerate."
        )
        criterion = "WARN_COLLAPSED"
    elif last_entropy > 0.50:
        log.info("PASS: JK entropy > 0.50 — healthy phase diversity maintained.")
        criterion = "PASS"
    else:
        log.info(f"INFO: JK entropy = {last_entropy:.4f} (between 0.30 and 0.50 — moderate diversity).")
        criterion = "MODERATE"

    # Summary statistics
    h_vals = np.array([e["entropy"] for e in entries])
    log.info(f"Entropy stats: mean={h_vals.mean():.4f}, min={h_vals.min():.4f}, max={h_vals.max():.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_path = out_dir / "a3_jk_entropy.png"
    try:
        plot_jk_entropy(entries, plot_path)
    except Exception as exc:
        log.warning(f"Plot failed: {exc}")

    # ── Part 2: Print MLflow snippet ──────────────────────────────────────────
    print_mlflow_snippet()

    # ── JSON output ───────────────────────────────────────────────────────────
    report = {
        "experiment": "exp_a3_jk_entropy_logging",
        "log_file": str(log_file),
        "n_epochs_parsed": len(entries),
        "last_entropy": round(float(last_entropy), 4),
        "entropy_mean": round(float(h_vals.mean()), 4),
        "entropy_min": round(float(h_vals.min()), 4),
        "entropy_max": round(float(h_vals.max()), 4),
        "pass_criterion": criterion,
        "epochs": entries,
        "plot": str(plot_path),
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
