# agents/src/eval/reliability_fit.py
"""
Bayesian shrinkage fitter for per-(source, class) reliability — P3 (B-3 / D-C).

Reads a confusion matrix produced by `reliability_matrix.build_matrix()` and
fits a shrinkage estimator:

    fitted = (n * measured + alpha * prior) / (n + alpha)

where:
    n             = total cell count (tp + fp + fn + tn)
    measured      = tp / (tp + fp), 0.0 if undefined
    alpha         = 5 (the P3 shrinkage strength, from the proposal)
    prior         = the L1 hand-set value from verdicts_default.yaml's
                    `consensus.accuracy_weights[cls][source]`
                    (the P2 placeholder P3 replaces)

Zero-sample cells (n == 0) return the prior verbatim — no fiction, no
fabrication. The prior itself is the L1 starting point, externally versioned
in `configs/verdicts_default.yaml`.

Gate (Rule B maturity ladder — Level 2):
    A drop of |fitted − prior| ≥ 0.05 against the prior fails the build
    unless an entry is present in `--justify <yaml>` mapping the (source,
    cls) cell to a reason. The justification file is recorded in
    `fit_metadata.justifications` on the produced YAML so a future reader
    knows the drop was deliberate.

Output: `configs/reliability_v1.yaml` (schema_version="1") consumed by
`verdict/reliability.py` (T3.3).

Usage:
    cd agents && poetry run python -m src.eval.reliability_fit \\
        --matrix eval/reliability/confusion_matrix_v2.json \\
        --out configs/reliability_v1.yaml \\
        [--alpha 5] [--config configs/verdicts_default.yaml] \\
        [--justify configs/reliability_justifications.yaml]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Decision-numbers — these are POLICY, not constants. Changes here are
# Rule B Level-2: must be measured on a held-out set before adopting.
# (T3.4 will produce the measurement; the cell fitter only uses them.)
DEFAULT_ALPHA: float = 5.0
DROP_GATE_PCT: float = 0.05  # |fitted - prior| >= this → fail (without --justify)
SCHEMA_VERSION: str = "1"


@dataclass
class FittedCell:
    source: str
    cls: str
    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    measured: float
    prior: float
    fitted: float
    n_zero_sample: bool  # n == 0 → fitted == prior verbatim


@dataclass
class FitMetadata:
    matrix_path: str
    matrix_n_contracts: int
    n_contracts_used: int  # contracts that contributed at least one signal
    alpha: float
    prior_schema_version: str
    fit_at: str
    drop_gate_pct: float
    justifications: dict[str, str] = field(default_factory=dict)
    n_cells: int = 0
    n_zero_sample: int = 0
    n_cells_over_gate: int = 0


def _load_prior(accuracy_weights: dict[str, dict[str, float]]) -> dict[tuple[str, str], float]:
    """Flatten `consensus.accuracy_weights[cls][source]` into a (source, cls) → prior dict."""
    priors: dict[tuple[str, str], float] = {}
    for cls, by_source in accuracy_weights.items():
        for source, val in by_source.items():
            priors[(source, cls)] = float(val)
    return priors


def _fit_cell(
    cell: dict[str, Any],
    prior: float,
    alpha: float,
) -> FittedCell:
    n = int(cell["n"])
    tp = int(cell["tp"])
    fp = int(cell["fp"])
    fn = int(cell["fn"])
    tn = int(cell["tn"])
    denom = tp + fp
    measured = (tp / denom) if denom > 0 else 0.0

    if n == 0:
        # Zero-sample — fall back to the prior verbatim. No fiction.
        fitted = prior
        n_zero = True
    else:
        # Bayesian shrinkage: pull measured toward prior with strength alpha.
        # alpha=5 is the proposal's chosen shrinkage; tune via the Rule B L2
        # gate (T3.4) only if a measurement justifies a different value.
        fitted = (n * measured + alpha * prior) / (n + alpha)
        n_zero = False

    return FittedCell(
        source=cell["source"],
        cls=cell["cls"],
        n=n,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        measured=round(measured, 6),
        prior=round(prior, 6),
        fitted=round(fitted, 6),
        n_zero_sample=n_zero,
    )


def fit(
    matrix: dict[str, Any],
    accuracy_weights: dict[str, dict[str, float]],
    alpha: float = DEFAULT_ALPHA,
    drop_gate_pct: float = DROP_GATE_PCT,
    justifications: dict[str, str] | None = None,
) -> tuple[list[FittedCell], FitMetadata, list[str]]:
    """
    Fit Bayesian-shrunk reliability for every (source, cls) cell in the matrix.

    Returns:
        fitted_cells: per-cell shrinkage results
        metadata: provenance (matrix path, n_contracts, alpha, fit_at, etc.)
        gate_failures: list of (source, cls, fitted, prior) cells exceeding
                       drop_gate_pct that lack a justification entry.

    The caller decides what to do with `gate_failures` — the function
    itself is a pure transformer; it does NOT raise. This makes it
    composable in the gate and CI flows.
    """
    priors = _load_prior(accuracy_weights)
    fitted_cells: list[FittedCell] = []
    gate_failures: list[str] = []
    justifications = justifications or {}

    for cell_data in matrix["cells"]:
        src, cls = cell_data["source"], cell_data["cls"]
        prior = priors.get((src, cls))
        if prior is None:
            # Matrix cell with no matching prior — surface as gate failure
            # (no L1 prior means we have no principled starting point).
            gate_failures.append(
                f"{src}|{cls}: no L1 prior in consensus.accuracy_weights "
                f"(matrix has {src}/{cls} but config has no entry for it)"
            )
            continue
        fc = _fit_cell(cell_data, prior=prior, alpha=alpha)
        fitted_cells.append(fc)

        if not fc.n_zero_sample:
            delta = abs(fc.fitted - fc.prior)
            if delta >= drop_gate_pct:
                key = f"{src}|{cls}"
                if key not in justifications:
                    gate_failures.append(
                        f"{key}: fitted={fc.fitted:.4f} prior={fc.prior:.4f} "
                        f"delta={delta:.4f} >= gate={drop_gate_pct:.4f} (no justification)"
                    )

    n_contracts_used = sum(1 for cell in matrix["cells"] if int(cell["n"]) > 0)
    n_zero_sample = sum(1 for fc in fitted_cells if fc.n_zero_sample)
    n_over_gate = sum(1 for f in gate_failures if "no L1 prior" not in f)

    metadata = FitMetadata(
        matrix_path=matrix.get("matrix_path", "?"),
        matrix_n_contracts=int(matrix.get("n_contracts", 0)),
        n_contracts_used=n_contracts_used,
        alpha=alpha,
        prior_schema_version=SCHEMA_VERSION,
        fit_at=datetime.now(timezone.utc).isoformat(),
        drop_gate_pct=drop_gate_pct,
        justifications=justifications,
        n_cells=len(fitted_cells),
        n_zero_sample=n_zero_sample,
        n_cells_over_gate=n_over_gate,
    )
    return fitted_cells, metadata, gate_failures


def _table_to_dict(cells: list[FittedCell]) -> dict[str, dict[str, float]]:
    """Restructure for the YAML output: {source: {cls: fitted}}."""
    table: dict[str, dict[str, float]] = {}
    for fc in cells:
        table.setdefault(fc.source, {})[fc.cls] = fc.fitted
    return table


def _metadata_to_dict(m: FitMetadata) -> dict[str, Any]:
    return {
        "matrix_path": m.matrix_path,
        "matrix_n_contracts": m.matrix_n_contracts,
        "n_contracts_used": m.n_contracts_used,
        "alpha": m.alpha,
        "prior_schema_version": m.prior_schema_version,
        "fit_at": m.fit_at,
        "drop_gate_pct": m.drop_gate_pct,
        "justifications": m.justifications,
        "n_cells": m.n_cells,
        "n_zero_sample": m.n_zero_sample,
        "n_cells_over_gate": m.n_cells_over_gate,
    }


def _cells_to_detail(cells: list[FittedCell]) -> list[dict[str, Any]]:
    return [
        {
            "source": c.source,
            "cls": c.cls,
            "n": c.n,
            "tp": c.tp,
            "fp": c.fp,
            "fn": c.fn,
            "tn": c.tn,
            "measured_precision": c.measured,
            "prior": c.prior,
            "fitted": c.fitted,
            "n_zero_sample": c.n_zero_sample,
        }
        for c in cells
    ]


def write_yaml(
    cells: list[FittedCell],
    metadata: FitMetadata,
    out_path: Path,
) -> None:
    """Write the versioned reliability YAML consumed by `verdict/reliability.py`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "fit_metadata": _metadata_to_dict(metadata),
        "table": _table_to_dict(cells),
        "cells_detail": _cells_to_detail(cells),
    }
    out_path.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=120)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reliability_fit",
        description="Bayesian-shrinkage fitter for per-(source, class) reliability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--matrix", type=Path, required=True,
                        help="Path to confusion matrix JSON (build_reliability_matrix output).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output YAML path (e.g. configs/reliability_v1.yaml).")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/verdicts_default.yaml"),
                        help="YAML config providing the L1 priors (consensus.accuracy_weights).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Bayesian shrinkage strength (prior weight).")
    parser.add_argument("--drop-gate-pct", type=float, default=DROP_GATE_PCT,
                        help="|fitted-prior| >= this → fail without --justify.")
    parser.add_argument("--justify", type=Path, default=None,
                        help="Optional YAML mapping {source|cls: reason} for cells that "
                             "exceed the drop gate. Required to pass the gate for "
                             "deltas >= --drop-gate-pct.")
    parser.add_argument("--allow-failures", action="store_true",
                        help="Write the YAML even if the gate fires (default: fail-fast).")
    args = parser.parse_args()

    if not args.matrix.is_file():
        print(f"ERROR: matrix not found: {args.matrix}", file=sys.stderr)
        sys.exit(2)
    if not args.config.is_file():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(2)

    matrix = json.loads(args.matrix.read_text())
    config = yaml.safe_load(args.config.read_text())
    try:
        accuracy_weights = config["consensus"]["accuracy_weights"]
    except (KeyError, TypeError):
        print(f"ERROR: config missing consensus.accuracy_weights", file=sys.stderr)
        sys.exit(2)

    justifications: dict[str, str] = {}
    if args.justify is not None:
        if not args.justify.is_file():
            print(f"ERROR: --justify file not found: {args.justify}", file=sys.stderr)
            sys.exit(2)
        j = yaml.safe_load(args.justify.read_text()) or {}
        justifications = {str(k): str(v) for k, v in j.items()}

    cells, metadata, gate_failures = fit(
        matrix=matrix,
        accuracy_weights=accuracy_weights,
        alpha=args.alpha,
        drop_gate_pct=args.drop_gate_pct,
        justifications=justifications,
    )

    # Print a summary to stdout (one row per cell, easy to grep/diff)
    print(f"Fitted {len(cells)} cells | alpha={args.alpha} | drop_gate={args.drop_gate_pct}")
    print(f"matrix: {args.matrix}  n_contracts={metadata.matrix_n_contracts}")
    print()
    header = f'{"source":<10}{"cls":<32}{"tp":>4}{"fp":>4}{"prior":>7}{"fit":>7}  delta  zero?'
    print(header); print("-" * 84)
    for fc in sorted(cells, key=lambda c: (c.source, c.cls)):
        delta = fc.fitted - fc.prior
        print(
            f"{fc.source:<10}{fc.cls:<32}{fc.tp:>4}{fc.fp:>4}"
            f"{fc.prior:>7.3f}{fc.fitted:>7.3f}  {delta:+.3f}  {fc.n_zero_sample}"
        )
    print()
    print(f"zero-sample cells (n==0 → prior verbatim): {metadata.n_zero_sample}")
    print(f"cells over drop gate (unjustified):        {metadata.n_cells_over_gate}")

    if gate_failures:
        print()
        print("=== GATE FAILURES ===")
        for f in gate_failures:
            print(f"  ✗ {f}")
        if not args.allow_failures:
            print()
            print(f"FAIL: {len(gate_failures)} gate failure(s) — refusing to write {args.out}.")
            print("Provide a --justify file or pass --allow-failures to inspect output anyway.")
            sys.exit(1)

    write_yaml(cells, metadata, args.out)
    print()
    print(f"OK: wrote {args.out}")


if __name__ == "__main__":
    main()
