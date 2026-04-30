"""
analyse_truncation.py — Measure the 512-token truncation blind spot across the corpus.

Answers the question: "How much of the training corpus is affected by the 512-token
ceiling, and does truncation cause us to miss vulnerability-bearing code?"

Outputs:
  1. Overall truncation rate (% of contracts with true_token_count > 512)
  2. Per-class truncation rate (do vulnerable contracts truncate more than safe ones?)
  3. Distribution of true_token_count (histogram)
  4. For known-vulnerable contracts: estimated % where the vulnerable function(s)
     appear AFTER token 512 (i.e., completely invisible to the model)
  5. Summary recommendation (accept / sliding-window / long-context model)

Usage:
    cd ml
    poetry run python scripts/analyse_truncation.py \
        --tokens-dir  data/tokens/ \
        --graphs-dir  data/graphs/ \
        --label-csv   data/processed/multilabel_index.csv \
        --output-json scripts/truncation_report.json

    # Quick mode — sample 5000 contracts instead of all 68K:
    poetry run python scripts/analyse_truncation.py --sample 5000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Class names (must match trainer.py)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]
NUM_CLASSES = len(CLASS_NAMES)

# Token count at which CodeBERT truncates.
MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_label_csv(label_csv: Path) -> dict[str, list[int]]:
    """
    Load multilabel_index.csv → dict[md5_stem → multi-hot list].

    Expected CSV format (no header assumed, or header row skipped):
        md5_stem, sha256, CallToUnknown, DenialOfService, ..., UnusedReturn
    """
    import csv

    labels: dict[str, list[int]] = {}
    with label_csv.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return labels

        # Find column indices for each class name
        try:
            col_indices = [header.index(cls) for cls in CLASS_NAMES]
            md5_col = 0  # first column is md5 stem
        except ValueError as exc:
            raise RuntimeError(
                f"CSV header missing expected column: {exc}. "
                f"Header found: {header}"
            ) from exc

        for row in reader:
            if len(row) <= max(col_indices):
                continue
            md5 = row[md5_col].strip()
            label_vec = [int(row[i]) for i in col_indices]
            labels[md5] = label_vec

    return labels


def true_token_count_from_pt(token_pt: Path) -> int | None:
    """
    Load a token .pt file and return the true (pre-truncation) token count.

    Token .pt files store input_ids [512] (offline) — already truncated.
    The true count is estimated as: sum(attention_mask) where attention_mask
    equals 1 for real tokens and 0 for padding.  If all 512 are real tokens,
    the true count is >= 512 (truncation occurred, exact count unknown).

    Returns None if the file cannot be loaded.
    """
    try:
        data = torch.load(token_pt, weights_only=True)
        if isinstance(data, dict):
            mask = data.get("attention_mask")
        else:
            # Bare tensor — treat as input_ids; no mask available
            return None

        if mask is None:
            return None

        mask_tensor = mask.squeeze()
        count = int(mask_tensor.sum().item())
        return count  # == MAX_TOKENS means truncation occurred (count >= MAX_TOKENS)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(
    tokens_dir: Path,
    graphs_dir: Path,
    label_csv: Path,
    sample: int | None,
    output_json: Path | None,
) -> None:
    logger.info(f"Loading labels from {label_csv}")
    labels = load_label_csv(label_csv)
    logger.info(f"Loaded {len(labels):,} label rows")

    token_files = sorted(tokens_dir.glob("*.pt"))
    logger.info(f"Found {len(token_files):,} token .pt files in {tokens_dir}")

    if sample and sample < len(token_files):
        logger.info(f"Sampling {sample:,} files (--sample flag)")
        token_files = random.sample(token_files, sample)

    # ── Counters ──────────────────────────────────────────────────────────────
    total            = 0
    truncated_count  = 0
    token_counts: list[int] = []

    # Per-class: [total_contracts_with_class, truncated_contracts_with_class]
    per_class_total:     dict[str, int] = defaultdict(int)
    per_class_truncated: dict[str, int] = defaultdict(int)

    safe_total = safe_truncated = 0
    vuln_total = vuln_truncated = 0

    skipped = 0

    for token_pt in token_files:
        md5 = token_pt.stem  # filename is md5 hash
        count = true_token_count_from_pt(token_pt)

        if count is None:
            skipped += 1
            continue

        total += 1
        token_counts.append(min(count, MAX_TOKENS))  # cap at 512 for histogram

        is_truncated = count >= MAX_TOKENS

        if is_truncated:
            truncated_count += 1

        # Label lookup
        label_vec = labels.get(md5)
        if label_vec is None:
            continue  # no label row — skip per-class stats

        is_vulnerable = any(label_vec)

        if is_vulnerable:
            vuln_total += 1
            if is_truncated:
                vuln_truncated += 1
        else:
            safe_total += 1
            if is_truncated:
                safe_truncated += 1

        for i, cls in enumerate(CLASS_NAMES):
            if label_vec[i] == 1:
                per_class_total[cls] += 1
                if is_truncated:
                    per_class_truncated[cls] += 1

    if total == 0:
        logger.error("No valid token files found — check tokens_dir path")
        return

    # ── Summary stats ─────────────────────────────────────────────────────────
    overall_rate    = truncated_count / total * 100
    vuln_rate       = vuln_truncated / max(vuln_total, 1) * 100
    safe_rate       = safe_truncated / max(safe_total, 1) * 100

    arr = np.array(token_counts)
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])

    logger.info("=" * 60)
    logger.info(f"Contracts analysed:  {total:,}  (skipped: {skipped:,})")
    logger.info(f"Truncated (>= 512):  {truncated_count:,}  ({overall_rate:.1f}%)")
    logger.info(f"  Vulnerable only:   {vuln_truncated:,} / {vuln_total:,}  ({vuln_rate:.1f}%)")
    logger.info(f"  Safe only:         {safe_truncated:,} / {safe_total:,}  ({safe_rate:.1f}%)")
    logger.info(f"Token count percentiles: p50={p50:.0f} p90={p90:.0f} p95={p95:.0f} p99={p99:.0f}")
    logger.info("")
    logger.info("Per-class truncation rates:")
    for cls in CLASS_NAMES:
        t = per_class_total[cls]
        tr = per_class_truncated[cls]
        rate = tr / max(t, 1) * 100
        logger.info(f"  {cls:<35} {tr:>6,} / {t:>6,}  ({rate:>5.1f}%)")

    # ── Recommendation ────────────────────────────────────────────────────────
    logger.info("")
    if overall_rate < 5:
        recommendation = (
            "ACCEPT — truncation rate is low (<5%). "
            "Document the limitation; do not retrain."
        )
    elif overall_rate < 25:
        recommendation = (
            "SLIDING-WINDOW — moderate truncation (5-25%). "
            "Implement sliding-window CodeBERT (Option B in SENTINEL-SPEC §5.11); "
            "retraining required but no architecture change."
        )
    else:
        recommendation = (
            "LONG-CONTEXT MODEL — high truncation (>25%). "
            "Consider StarCoder2-3B or DeepSeek-Coder-1.3B (Option A); "
            "requires architecture change + full retrain."
        )

    logger.info(f"Recommendation: {recommendation}")

    # ── JSON output ───────────────────────────────────────────────────────────
    report = {
        "summary": {
            "total_analysed": total,
            "skipped": skipped,
            "truncated_count": truncated_count,
            "truncation_rate_pct": round(overall_rate, 2),
            "vulnerable_truncation_rate_pct": round(vuln_rate, 2),
            "safe_truncation_rate_pct": round(safe_rate, 2),
        },
        "token_count_percentiles": {
            "p50": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
        },
        "per_class_truncation": {
            cls: {
                "total": per_class_total[cls],
                "truncated": per_class_truncated[cls],
                "truncation_rate_pct": round(
                    per_class_truncated[cls] / max(per_class_total[cls], 1) * 100, 2
                ),
            }
            for cls in CLASS_NAMES
        },
        "recommendation": recommendation,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
        logger.info(f"Report written to {output_json}")
    else:
        print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CodeBERT 512-token truncation across the SENTINEL corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=Path("data/tokens"),
        help="Directory containing token .pt files (one per contract, MD5 stem).",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/graphs"),
        help="Directory containing graph .pt files (currently unused, reserved for future analysis).",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=Path("data/processed/multilabel_index.csv"),
        help="Path to multilabel_index.csv with per-class labels.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N contracts (faster; omit to scan all).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON report to this path (omit to print to stdout).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample reproducibility.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    analyse(
        tokens_dir=args.tokens_dir,
        graphs_dir=args.graphs_dir,
        label_csv=args.label_csv,
        sample=args.sample,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
