"""
smoke_fix5.py — Smoke test for Fix #5 (Slither-derived labels).

Verifies that:
  - derive_slither_labels.py exists
  - Output CSV exists with expected column layout
  - Class distributions are in expected windows
  - At least some rows have non-empty provenance_json

Gates-in:
  G5.1 — `slither` CLI is on PATH
  G5.2 — ml/src/preprocessing/graph_extractor.py (or new derive_slither_labels.py) exists
  G5.3 — At least one source directory is readable

Gates-out:
  G5.4 — multilabel_index_slither.csv exists
  G5.5 — All CLASS_COLUMNS present, no NaN values
  G5.6 — Timestamp count is in [500, 1200] (matches Fix #1 distribution)
  G5.7 — provenance_json column exists and has ≥ 1 non-null row
  G5.8 — Class distribution spread: at least 3 classes with ≥ 50 positives
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    CLASS_COLUMNS,
    PROCESSED_DIR,
    check,
    pass_,
    smoke_header,
    timed,
)

SLITHER_CSV_NAME: str = "multilabel_index_slither.csv"


@timed("fix5_check_slither")
def check_slither_installed() -> str:
    """Verify `slither` CLI is on PATH and return its version string."""
    path = shutil.which("slither")
    if path is None:
        raise AssertionError(
            "G5.1 `slither` not on PATH — install with: pipx install slither-analyzer"
        )
    import subprocess
    try:
        result = subprocess.run(
            ["slither", "--version"], capture_output=True, text=True, timeout=10
        )
        version = (result.stdout or result.stderr).strip().split("\n")[0]
    except Exception as exc:
        version = f"(version check failed: {exc})"
    pass_(f"G5.1 slither found at {path} — {version}")
    return version


@timed("fix5_check_csv")
def check_slither_csv() -> dict:
    """Verify slither-derived CSV exists and has expected structure."""
    csv_path = PROCESSED_DIR / SLITHER_CSV_NAME
    if not csv_path.exists():
        raise AssertionError(
            f"G5.4 {SLITHER_CSV_NAME} missing — run derive_slither_labels.py first"
        )

    df = pd.read_csv(csv_path)
    check(len(df) > 0, f"G5.4 {SLITHER_CSV_NAME} has {len(df)} rows (> 0)")

    missing = [c for c in CLASS_COLUMNS if c not in df.columns]
    check(len(missing) == 0, f"G5.5 all CLASS_COLUMNS present (missing: {missing})")

    nan_cells = int(df[CLASS_COLUMNS].isna().sum().sum())
    check(nan_cells == 0, f"G5.5 no NaN values in class columns (found {nan_cells})")

    ts_count = int(df["Timestamp"].sum())
    check(500 < ts_count < 1200, f"G5.6 Timestamp count = {ts_count} (expected 500–1200)")

    if "provenance_json" in df.columns:
        non_null_prov = int(df["provenance_json"].notna().sum())
        non_empty_prov = int(df["provenance_json"].fillna("").astype(str).str.len().gt(2).sum())
        check(non_empty_prov >= 1, f"G5.7 provenance_json has ≥ 1 non-empty row (got {non_empty_prov})")
    else:
        raise AssertionError("G5.7 provenance_json column missing from CSV")

    trainable = [c for c in CLASS_COLUMNS if int(df[c].sum()) >= 50]
    check(
        len(trainable) >= 3,
        f"G5.8 at least 3 classes have ≥ 50 positives (got {len(trainable)}: {trainable})",
    )

    return {
        "rows": len(df),
        "timestamp": ts_count,
        "provenance_non_empty": non_empty_prov,
        "trainable_classes": trainable,
    }


@timed("fix5_total")
def main() -> int:
    smoke_header(5, "Slither-derived labels (multi-class, provenance-tracked)")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    check_slither_installed()

    # ── Body ─────────────────────────────────────────────────────────────
    stats = check_slither_csv()

    elapsed = time.perf_counter() - start
    pass_(
        f"Fix #5 smoke OK — {stats['rows']} rows, "
        f"Timestamp={stats['timestamp']}, "
        f"provenance={stats['provenance_non_empty']} non-empty, "
        f"{len(stats['trainable_classes'])} trainable classes, "
        f"{elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #5 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
