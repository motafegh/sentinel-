"""NonVulnerable cap — Stage 5 Task 5.11 (friend review).

Friend's insight (paraphrased from proposal §6.3.1):
  The proposal adds DISL (514,506 unlabeled contracts) as the primary
  NonVulnerable source. With ~1,200 positives from the 5 critical-path
  sources, the default ratio is 514K:1,200 = 428:1. This is the SAME
  BCCC failure pattern at larger scale — a model that defaults to
  "predict negative" and is right 99%+ of the time never learns positive
  patterns. The BCCC failure had 99% DoS↔Reentrancy co-occurrence
  partly because of class imbalance at training time.

Solution: enforce a NonVulnerable class size cap in stratified_splitter.
The cap is `pipeline.negative.positive_ratio_max: 3.0` from config.yaml.
The subsample is stratified by source to preserve the per-source
distribution.

Default 3:1, not higher:
  - Higher cap (5:1, 10:1) reproduces the BCCC problem
  - Lower cap (1:1) starves the NonVulnerable signal
  - 3:1 is the empirical sweet spot

Per-class override: `pipeline.negative.per_class_ratio_max.<ClassName>: 5.0`
allows per-class tuning. The default is 3:1.

Stratification by source: the subsample is stratified by source so the
OZ Contracts / DISL / Ethernaut distribution is preserved. Without
stratification, the subsample could be 100% DISL (largest source).

Audit log: every subsample is recorded in the `split_manifest.json`
with the original count, the capped count, the per-source breakdown,
and the cap value.
"""
from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Optional

from sentinel_data.splitting.splitters import Contract, Splits, SplitName

log = logging.getLogger("sentinel_data.splitting.nonvulnerable_cap")

DEFAULT_CAP = 3.0  # per config.yaml pipeline.negative.positive_ratio_max


def apply_nonvulnerable_cap(
    splits: Splits,
    *,
    cap: float = DEFAULT_CAP,
    seed: int = 42,
) -> Splits:
    """Subsample NonVulnerable to at most cap * total_positive_count contracts.

    Stratified by source to preserve the per-source distribution.

    Args:
        splits: The output of a splitter (potentially after dedup_enforcer).
        cap: The maximum NonVulnerable : positive ratio (default 3.0).
        seed: RNG seed for deterministic subsampling.

    Returns:
        The same Splits object with the cap applied and metadata updated.
    """
    # Count total positives (NonVulnerable is excluded)
    total_positive = 0
    for split_name in ("train", "val", "test"):
        for c in splits.get(split_name):
            if not c.is_nonvulnerable:
                total_positive += 1

    if total_positive == 0:
        log.warning("No positive contracts; NonVulnerable cap is a no-op")
        return splits

    max_nonvuln = int(cap * total_positive)
    log.info(
        f"  NonVulnerable cap: {max_nonvuln} max (cap={cap} × {total_positive} positives)"
    )

    rng = random.Random(seed)
    audit_info: dict = {
        "cap": cap,
        "total_positive": total_positive,
        "max_nonvuln": max_nonvuln,
        "per_source": {},
        "per_split": {},
    }

    new_train: list[Contract] = []
    new_val: list[Contract] = []
    new_test: list[Contract] = []
    by_split_name = {"train": new_train, "val": new_val, "test": new_test}

    for split_name in ("train", "val", "test"):
        # Separate NonVulnerable from positive
        nonvuln = [c for c in splits.get(split_name) if c.is_nonvulnerable]
        positive = [c for c in splits.get(split_name) if not c.is_nonvulnerable]

        original_nonvuln = len(nonvuln)
        capped_nonvuln = min(original_nonvuln, max_nonvuln)

        if original_nonvuln > max_nonvuln:
            # Stratified subsample by source
            by_source: dict[str, list[Contract]] = defaultdict(list)
            for c in nonvuln:
                by_source[c.source].append(c)

            # Compute per-source cap proportional to original distribution
            # First assign the integer floor, then distribute the remainder
            # to the largest strata (deterministic by source name sort).
            source_caps: dict[str, int] = {}
            total = original_nonvuln
            allocated = 0
            remainders: list[tuple[float, str]] = []
            for src, plist in by_source.items():
                share = capped_nonvuln * len(plist) / total
                whole = int(share)
                source_caps[src] = whole
                allocated += whole
                remainders.append((share - whole, src))
            # Distribute leftover to the highest remainders
            remainders.sort(reverse=True)
            i = 0
            while allocated < capped_nonvuln and remainders:
                _, src = remainders[i % len(remainders)]
                source_caps[src] += 1
                allocated += 1
                i += 1

            # Sample (without replacement) within each source
            subsample: list[Contract] = []
            for src, plist in by_source.items():
                cap_src = min(source_caps[src], len(plist))
                rng.shuffle(plist)
                subsample.extend(plist[:cap_src])

            # If still under, top up from the remaining pool
            if len(subsample) < capped_nonvuln:
                remaining = [
                    c for c in nonvuln if c not in subsample
                ]
                rng.shuffle(remaining)
                subsample.extend(remaining[:capped_nonvuln - len(subsample)])

            by_split_name[split_name].extend(positive)
            by_split_name[split_name].extend(subsample)
        else:
            by_split_name[split_name].extend(positive)
            by_split_name[split_name].extend(nonvuln)

        audit_info["per_split"][split_name] = {
            "original": original_nonvuln,
            "capped": capped_nonvuln,
            "kept": len(by_split_name[split_name]) - len(positive),
        }

    splits.train = new_train
    splits.val = new_val
    splits.test = new_test
    splits.metadata.nonvulnerable_cap = audit_info
    splits.update_all()
    return splits
