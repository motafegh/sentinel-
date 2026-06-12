"""Probe dataset — Stage 4 Task 4.6.

Builds a hand-curated set of contracts for the model interpretability
suite. Per class: N (default 40) real contracts from the highest-quality
source available, plus 1 trivial positive (simplest example) and 1
trivial negative (clean OZ-style contract). Total per class: N+2.

Design decisions (per AUDIT_PATCHES 4-P5, 4-P6, F28):
  - 40 per class is the v2 baseline; 50 is the v2.1 expansion.
  - The trivial positive is the simplest possible Solidity contract
    that exhibits the pattern — used to verify the model has learned
    the pattern, not the surface features.
  - The trivial negative is a clean OZ-style contract that exhibits
    NONE of the 10 patterns — used to verify the model does not
    over-predict on safe code.
  - Source priority for the real N contracts:
      1. BCCC review_batches/ (KEEPs only) — 6 of 10 classes available
         (Reentrancy, CallToUnknown, DoS, Timestamp, ExternalBug, GasException)
      2. DIVE positives (if a class is not in BCCC)
      3. None available → class entry has 0 real contracts (trivial
         pos/neg still produced)

Output structure:
  data/probe_dataset/
  ├── manifest.json
  ├── reentrancy/
  │   ├── <sha>.sol
  │   ├── ...
  │   ├── trivial_positive.sol
  │   └── trivial_negative.sol
  ├── calltounknown/
  │   └── ...
  └── ... (10 class directories)

The BCCC review_batches CSVs have the contract source code embedded in
the `source_snippet` column, so we can build the probe dataset without
needing the BCCC source corpus on disk (BCCC is deferred per
config.yaml — `deferred_sources.bccc`).
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.probe_trivials import (
    TRIVIAL_POSITIVES, TRIVIAL_NEGATIVE, bccc_class_to_sentinel,
)

log = logging.getLogger("sentinel_data.verification.probe_dataset")

DEFAULT_N_PER_CLASS = 40  # per plan D-4.7 / 4-P5 (v2 baseline; 50 is v2.1)


@dataclass
class ProbeEntry:
    """One contract in the probe dataset."""
    sha256: str
    class_name: str
    kind: str                     # "real" | "trivial_positive" | "trivial_negative"
    source: str                   # "bccc" | "dive" | "trivial_positive" | "trivial_negative"
    tier: Optional[str] = None    # from the source (None for trivial)
    verdict: Optional[str] = None # KEEP/DROP from BCCC review (real only)
    confidence: Optional[float] = None
    contract_path: Path = field(default_factory=lambda: Path(""))  # absolute path


@dataclass
class ClassProbeBucket:
    """One class's worth of probe contracts (real + trivial pos/neg)."""

    class_name: str
    real_entries: list[ProbeEntry] = field(default_factory=list)
    trivial_positive: Optional[ProbeEntry] = None
    trivial_negative: Optional[ProbeEntry] = None

    @property
    def total(self) -> int:
        n = len(self.real_entries)
        if self.trivial_positive:
            n += 1
        if self.trivial_negative:
            n += 1
        return n


@dataclass
class ProbeDataset:
    """Built probe dataset (10 class buckets)."""
    by_class: dict[str, ClassProbeBucket] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("data/probe_dataset"))
    n_per_class: int = DEFAULT_N_PER_CLASS
    duration_s: float = 0.0

    def to_manifest(self) -> dict:
        """Return a JSON-serializable manifest of the dataset."""
        out: dict = {
            "schema_version": "1",
            "n_per_class_target": self.n_per_class,
            "classes": {},
        }
        for cls, bucket in self.by_class.items():
            entries: list[dict] = []
            for e in bucket.real_entries:
                entries.append({
                    "sha256": e.sha256,
                    "kind": "real",
                    "source": e.source,
                    "tier": e.tier,
                    "verdict": e.verdict,
                    "confidence": e.confidence,
                    "contract_path": str(e.contract_path.relative_to(self.output_dir)),
                })
            out["classes"][cls] = {
                "real_count": len(bucket.real_entries),
                "trivial_positive": (
                    str(bucket.trivial_positive.contract_path.relative_to(self.output_dir))
                    if bucket.trivial_positive else None
                ),
                "trivial_negative": (
                    str(bucket.trivial_negative.contract_path.relative_to(self.output_dir))
                    if bucket.trivial_negative else None
                ),
                "total": bucket.total,
                "entries": entries,
            }
        return out


def _read_bccc_review(review_csv: Path) -> list[dict]:
    """Read a BCCC review_batch CSV and return only rows with verdict_s4 == KEEP."""
    if not review_csv.exists():
        return []
    with review_csv.open() as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("verdict_s4") == "KEEP"]


def _read_dive_positives_for_class(
    data_dir: Path,
    cls: str,
    *,
    limit: int,
) -> list[dict]:
    """Read DIVE-positive contracts for `cls` from preprocessed/ + labels/merged/.

    Returns list of {sha256, source, tier, sol_path}.
    """
    pre = data_dir / "preprocessed" / "dive"
    merged = data_dir / "labels" / "merged"
    if not pre.exists() or not merged.exists():
        return []

    out: list[dict] = []
    for lf in sorted(merged.glob("*.labels.json")):
        try:
            lj = json.loads(lf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "dive" not in (lj.get("sources") or []):
            continue
        entry = lj.get("classes", {}).get(cls, {})
        if entry.get("value") != 1:
            continue
        sha = lj["sha256"]
        sol = pre / f"{sha}.sol"
        if not sol.exists():
            continue
        out.append({
            "sha256": sha,
            "source": "dive",
            "tier": entry.get("tier") or "T2",
            "sol_path": sol,
        })
        if len(out) >= limit:
            break
    return out


def build_probe_dataset(
    *,
    data_dir: Optional[Path] = None,
    n_per_class: int = DEFAULT_N_PER_CLASS,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    bccc_review_dir: Optional[Path] = None,
    add_trivial: bool = True,
) -> ProbeDataset:
    """Build the probe dataset and write it to `output_dir`.

    Args:
        data_dir: Path to the data/ directory (for DIVE fallback).
                  Optional — only used for classes not in BCCC.
        n_per_class: Per-class real-contract target (default 40 per plan).
        seed: RNG seed for deterministic sampling.
        output_dir: Where to write the probe dataset. Default:
                    data_dir / "probe_dataset" if data_dir is set,
                    else ./data/probe_dataset.
        bccc_review_dir: Path to the BCCC review_batches/ directory.
                        Default: Data/docs/legacy/bccc_deep_dive/.../review_batches/
        add_trivial: If True (default), add trivial_positive.sol and
                     trivial_negative.sol for every class.

    Returns:
        ProbeDataset with the built buckets.
    """
    if output_dir is None:
        if data_dir is not None:
            # Resolve data_dir first so the child path is always absolute,
            # regardless of CWD at call time.
            output_dir = Path(data_dir).resolve() / "probe_dataset"
        else:
            output_dir = Path("data/probe_dataset")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if bccc_review_dir is None:
        # Look for the canonical BCCC review_batches location. Walk up from
        # this module's location to find the repo root, so the path resolves
        # whether the caller is in the repo root, Data/, or any subdir.
        here = Path(__file__).resolve()
        for ancestor in [here.parent, *here.parents]:
            candidate = ancestor / "docs" / "legacy" / "bccc_deep_dive" \
                / "Phase5_LabelVerification_2026-06-08" \
                / "outputs" / "review_batches"
            if candidate.exists():
                bccc_review_dir = candidate
                break
        if bccc_review_dir is None:
            # Fallback: relative path (works when CWD is the repo root)
            bccc_review_dir = (
                Path("docs") / "legacy" / "bccc_deep_dive"
                / "Phase5_LabelVerification_2026-06-08"
                / "outputs" / "review_batches"
            )

    rng = random.Random(seed)
    dataset = ProbeDataset(
        by_class={c: ClassProbeBucket(class_name=c) for c in class_names()},
        output_dir=output_dir,
        n_per_class=n_per_class,
    )
    t0 = time.monotonic()

    # Discover BCCC review CSVs (one per class)
    bccc_by_class: dict[str, list[dict]] = {}
    if bccc_review_dir.exists():
        for csv_path in sorted(bccc_review_dir.glob("review_class*.csv")):
            stem = csv_path.stem            # e.g. "review_class11_reentrancy"
            try:
                bccc_class = stem.split("_", 2)[2]  # "reentrancy"
            except IndexError:
                continue
            sentinel_cls = bccc_class_to_sentinel(bccc_class)
            if sentinel_cls is None:
                continue
            bccc_by_class.setdefault(sentinel_cls, []).extend(
                _read_bccc_review(csv_path)
            )

    log.info(
        f"  BCCC review_batches found for {len(bccc_by_class)} classes: "
        f"{sorted(bccc_by_class.keys())}"
    )

    for cls in class_names():
        bucket = dataset.by_class[cls]
        class_dir = output_dir / cls.lower()
        class_dir.mkdir(parents=True, exist_ok=True)

        # 1) Try BCCC first
        bccc_rows = bccc_by_class.get(cls, [])
        if bccc_rows:
            rng.shuffle(bccc_rows)
            for row in bccc_rows[:n_per_class]:
                sha = row["id"]
                snippet = row.get("source_snippet", "").strip()
                if not snippet:
                    continue
                # Some snippets start with a leading newline (from CSV embedding)
                if not snippet.startswith("//") and not snippet.startswith("pragma") \
                        and not snippet.startswith("contract") and not snippet.startswith("library") \
                        and not snippet.startswith("abstract") and not snippet.startswith("interface"):
                    snippet = "// " + snippet
                out_path = class_dir / f"{sha}.sol"
                out_path.write_text(snippet)
                bucket.real_entries.append(ProbeEntry(
                    sha256=sha, class_name=cls, kind="real",
                    source="bccc", tier="T3", verdict="KEEP",
                    confidence=_safe_float(row.get("confidence")),
                    contract_path=out_path,
                ))

        # 2) Fall back to DIVE for classes with insufficient BCCC KEEPs
        if len(bucket.real_entries) < n_per_class and data_dir is not None:
            need = n_per_class - len(bucket.real_entries)
            dive_pos = _read_dive_positives_for_class(data_dir, cls, limit=need)
            for d in dive_pos:
                out_path = class_dir / f"{d['sha256']}.sol"
                out_path.write_text(d["sol_path"].read_text())
                bucket.real_entries.append(ProbeEntry(
                    sha256=d["sha256"], class_name=cls, kind="real",
                    source="dive", tier=d["tier"], verdict=None,
                    confidence=None, contract_path=out_path,
                ))

        if bucket.real_entries:
            log.info(f"  {cls}: {len(bucket.real_entries)} real contracts "
                     f"({sum(1 for e in bucket.real_entries if e.source == 'bccc')} BCCC + "
                     f"{sum(1 for e in bucket.real_entries if e.source == 'dive')} DIVE)")
        else:
            log.warning(f"  {cls}: 0 real contracts (no BCCC/DIVE data)")

        # 3) Trivial positive
        if add_trivial and cls in TRIVIAL_POSITIVES:
            tp_path = class_dir / "trivial_positive.sol"
            tp_path.write_text(TRIVIAL_POSITIVES[cls])
            bucket.trivial_positive = ProbeEntry(
                sha256="trivial_positive", class_name=cls,
                kind="trivial_positive", source="trivial_positive",
                contract_path=tp_path,
            )

        # 4) Trivial negative (same for all classes)
        if add_trivial:
            tn_path = class_dir / "trivial_negative.sol"
            tn_path.write_text(TRIVIAL_NEGATIVE)
            bucket.trivial_negative = ProbeEntry(
                sha256="trivial_negative", class_name=cls,
                kind="trivial_negative", source="trivial_negative",
                contract_path=tn_path,
            )

    # 5) Write manifest
    manifest = dataset.to_manifest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"  Wrote manifest: {output_dir / 'manifest.json'}")

    dataset.duration_s = time.monotonic() - t0
    total = sum(b.total for b in dataset.by_class.values())
    log.info(
        f"  Probe dataset: {total} contracts across {len(class_names())} classes  "
        f"({dataset.duration_s:.1f}s)"
    )
    return dataset


def _safe_float(s) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None
