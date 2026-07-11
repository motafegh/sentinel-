#!/usr/bin/env python3
# agents/scripts/build_reliability_matrix.py
"""
build_reliability_matrix.py — Build the per-(source, class) confusion matrix
from a directory of audit reports against ground-truth labels.

P3 (B-3 / D-C) data-collection step. Output is consumed by the shrinkage
fitter `agents/src/eval/reliability_fit.py` to produce a fitted
`configs/reliability_v1.yaml` (Bayesian-shrinkage per-(source, class) weights).

Sources detected from `consensus_verdict[cls]` per report:
    ml_signal (0/1), slither_match (int), aderyn_match (int).

Usage:
    cd agents && poetry run python scripts/build_reliability_matrix.py \
        --reports test_audit_reports_p0 \
        --corpus ../manual_hand_written_contracts \
        --out eval/reliability/confusion_matrix_v1.json

Classes scored: the 10 classes in `configs/verdicts_default.yaml`
(`consensus.accuracy_weights`), with any extraneous labels surfaced as a
warning (data-quality signal, not silently dropped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make agents/ importable when run as `python scripts/build_reliability_matrix.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.eval.reliability_matrix import KNOWN_SOURCES, build_matrix
from src.eval.run_benchmark import load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="build_reliability_matrix",
        description="Build per-(source, class) confusion matrix from audit reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reports", type=Path, required=True,
                        help="Dir containing <stem>_report.json files")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Corpus root with .sol + .json sidecars")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON path for the confusion matrix")
    args = parser.parse_args()

    if not args.reports.is_dir():
        print(f"ERROR: --reports dir not found: {args.reports}", file=sys.stderr)
        sys.exit(2)
    if not args.corpus.is_dir():
        print(f"ERROR: --corpus dir not found: {args.corpus}", file=sys.stderr)
        sys.exit(2)

    cfg = get_config()
    canonical_classes = list(cfg.consensus.accuracy_weights.keys())

    rows = load_corpus(args.reports, args.corpus)

    # Surface label-quality signal: any label outside the scored class set.
    label_universe = set()
    for r in rows:
        label_universe.update(r.labels)
    extra = label_universe - set(canonical_classes)
    if extra:
        print(f"WARNING: {len(extra)} label(s) outside the {len(canonical_classes)} scored "
              f"classes — ignored in the matrix: {sorted(extra)}", file=sys.stderr)

    matrix = build_matrix(
        rows=rows,
        classes=canonical_classes,
        sources=KNOWN_SOURCES,
        reports_dir=str(args.reports),
        corpus_dir=str(args.corpus),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix.save(args.out)

    # Console summary — cells with TP=0 AND FP=0 mean "source never emitted
    # for this class": zero-sample → fitter falls back to the L1 prior.
    n_cells = len(matrix.cells)
    zero_sample = sum(1 for c in matrix.cells.values() if c.measured_precision == 0.0 and c.n == matrix.n_contracts and c.tp == 0)
    # Note: zero_pred = tp==0 and fp==0; that's the actual zero-sample signal.
    zero_pred = sum(1 for c in matrix.cells.values() if c.tp == 0 and c.fp == 0)

    print("=" * 72)
    print(f"Contracts: {matrix.n_contracts}  |  Classes: {len(canonical_classes)}  "
          f"|  Sources: {len(KNOWN_SOURCES)}  |  Cells: {n_cells}")
    print(f"Zero-sample cells (source never flagged this class): {zero_pred}")
    print(f"Matrix written → {args.out}")
    print("=" * 72)




if __name__ == "__main__":
    main()