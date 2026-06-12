"""Multi-source label merger — Task 3.10.

Reads per-source .labels.json files, merges contracts that appear in
multiple sources, applies the 99% DoS↔Reentrancy co-occurrence rule,
and writes one canonical .labels.json to data/labels/merged/.

Design decisions:
  D-3.3: Conflict resolution precedence T0 > T1 > T2 > T3 > T4.
         Within a tier, positive wins over negative.
  D-3.3: DoS+Reentrancy co-occurrence from a SINGLE low-confidence source
         (T3/T4) is flagged as suspect noise when the co-occurrence rate
         for that source exceeds CO_OCCUR_NOISE_THRESHOLD (default 0.50).
         DIVE (T2) at 12% co-occurrence is NOT flagged — that is legitimate
         multi-label signal. BCCC's 99% rate was the target of this rule.
  D-3.5: Merged .labels.json is the canonical record for Stage 4+ stages.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from sentinel_data.labeling.schema import class_names

log = logging.getLogger("sentinel_data.labeling.merger")

# Tier precedence: lower index = higher confidence
_TIER_ORDER = ["T0", "T1", "T2", "T3", "T4", None]

# Co-occurrence noise threshold: if a single T3/T4 source labels both
# DoS AND Reentrancy for more than this fraction of its contracts,
# those contracts are flagged (not dropped — Stage 4 verifies).
CO_OCCUR_NOISE_THRESHOLD = 0.50

# Sources whose tier warrants co-occurrence scrutiny
_LOW_CONFIDENCE_TIERS = {"T3", "T4"}

# Canonical sources in precedence order
_SOURCE_PRECEDENCE = ["solidifi", "defihacklabs", "smartbugs_curated",
                      "web3bugs", "dive", "disl"]

# Maps source name → expected confidence tier (from crosswalk)
_SOURCE_TIER = {
    "solidifi":        "T0",
    "defihacklabs":    "T0",
    "smartbugs_curated": "T1",
    "web3bugs":        "T1",
    "dive":            "T2",
    "disl":            "T4",
}


@dataclass
class MergeResult:
    """Aggregated statistics from a merger run."""

    contracts_merged: int = 0       # unique sha256s written
    single_source: int = 0          # contracts from exactly one source
    multi_source: int = 0           # contracts from 2+ sources
    co_occurrence_flagged: int = 0  # DoS+Reentrancy flags applied
    cached: int = 0
    failed: int = 0
    duration_s: float = 0.0


def _tier_rank(tier: str | None) -> int:
    """Lower = higher confidence."""
    try:
        return _TIER_ORDER.index(tier)
    except ValueError:
        return len(_TIER_ORDER)


def _merge_class_entries(
    entries: list[tuple[str, dict]]  # list of (source, class_entry)
) -> dict:
    """Merge class entries from multiple sources into one.

    Strategy: keep the highest-confidence (lowest tier rank) positive.
    If no source is positive, keep negative with lowest tier rank.
    """
    positives = [(src, e) for src, e in entries if e["value"] == 1]
    if positives:
        best_src, best_entry = min(positives, key=lambda x: _tier_rank(x[1]["tier"]))
        return {
            "value": 1,
            "tier": best_entry["tier"],
            "source": best_src,
        }
    best_src, best_entry = min(entries, key=lambda x: _tier_rank(x[1]["tier"]))
    return {
        "value": 0,
        "tier": None,
        "source": best_src,
    }


def _check_co_occurrence_flag(
    classes: dict,
    sources: list[str],
    source_cooccur_rates: dict[str, float],
) -> bool:
    """Return True if DoS+Reentrancy co-occurrence looks like noise.

    Conditions (ALL must hold):
    1. Both DoS and Reentrancy are positive in the merged output.
    2. Only one source contributed (no independent attesting source).
    3. That source has a tier in _LOW_CONFIDENCE_TIERS.
    4. That source's DoS+Reentrancy co-occurrence rate > CO_OCCUR_NOISE_THRESHOLD.
    """
    if classes["DenialOfService"]["value"] != 1:
        return False
    if classes["Reentrancy"]["value"] != 1:
        return False
    if len(sources) > 1:
        return False  # independently attested — not noise
    sole_source = sources[0]
    tier = _SOURCE_TIER.get(sole_source, "T4")
    if tier not in _LOW_CONFIDENCE_TIERS:
        return False
    rate = source_cooccur_rates.get(sole_source, 0.0)
    return rate > CO_OCCUR_NOISE_THRESHOLD


def _compute_cooccur_rates(labels_dir: Path, sources: list[str]) -> dict[str, float]:
    """Compute per-source DoS+Reentrancy co-occurrence rates."""
    rates: dict[str, float] = {}
    for source in sources:
        src_dir = labels_dir / source
        if not src_dir.exists():
            continue
        files = list(src_dir.glob("*.labels.json"))
        if not files:
            continue
        cooccur = sum(
            1 for f in files
            if (lj := json.loads(f.read_text()))
            and lj["classes"]["DenialOfService"]["value"] == 1
            and lj["classes"]["Reentrancy"]["value"] == 1
        )
        rates[source] = cooccur / len(files)
    return rates


def run_merger(
    data_dir: Path,
    sources: list[str],
    *,
    force: bool = False,
    output_dir: Path | None = None,
) -> MergeResult:
    """Merge per-source label files into canonical merged labels.

    Args:
        data_dir: Path to data/ directory.
        sources: List of source names to merge (e.g. ["solidifi", "dive"]).
        force: Overwrite existing merged .labels.json files.
        output_dir: Override output dir (default: data_dir/labels/merged).
    """
    labels_dir = data_dir / "labels"
    out_dir = output_dir if output_dir is not None else labels_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_classes = class_names()
    result = MergeResult()
    t0 = time.monotonic()

    # Compute co-occurrence rates upfront for noise detection
    cooccur_rates = _compute_cooccur_rates(labels_dir, sources)
    for src, rate in cooccur_rates.items():
        log.info(f"  {src}: DoS+Reentrancy co-occurrence rate = {rate:.1%}")

    # Collect all sha256 → {source: labels_json}
    sha_to_sources: dict[str, dict[str, dict]] = {}
    for source in sources:
        src_dir = labels_dir / source
        if not src_dir.exists():
            log.warning(f"Labels dir not found for source '{source}': {src_dir}")
            continue
        for lf in src_dir.glob("*.labels.json"):
            sha = lf.name.removesuffix(".labels.json")
            try:
                sha_to_sources.setdefault(sha, {})[source] = json.loads(lf.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Cannot read {lf}: {e}")
                result.failed += 1

    # Merge
    for sha, source_labels in sha_to_sources.items():
        out_path = out_dir / f"{sha}.labels.json"

        if not force and out_path.exists():
            result.cached += 1
            continue

        contributing_sources = sorted(source_labels.keys())

        if len(contributing_sources) == 1:
            # Single source — pass through, add merger metadata
            sole = contributing_sources[0]
            lj = source_labels[sole]
            merged_classes = {
                cls: {**entry, "source": sole}
                for cls, entry in lj["classes"].items()
            }
            result.single_source += 1
        else:
            # Multi-source — merge per class
            merged_classes = {}
            for cls in all_classes:
                entries = [
                    (src, lj["classes"][cls])
                    for src, lj in source_labels.items()
                ]
                merged_classes[cls] = _merge_class_entries(entries)
            result.multi_source += 1

        # Co-occurrence noise check
        flagged = _check_co_occurrence_flag(
            merged_classes, contributing_sources, cooccur_rates
        )
        if flagged:
            result.co_occurrence_flagged += 1
            log.info(f"  Co-occurrence flag: {sha[:12]} (sources={contributing_sources})")

        n_pos = sum(v["value"] for v in merged_classes.values())
        merged = {
            "sha256": sha,
            "sources": contributing_sources,
            "classes": merged_classes,
            "n_pos": n_pos,
            "flags": ["dos_reentrancy_cooccur_suspect"] if flagged else [],
        }
        out_path.write_text(json.dumps(merged, indent=2))
        result.contracts_merged += 1

    result.duration_s = time.monotonic() - t0
    return result
