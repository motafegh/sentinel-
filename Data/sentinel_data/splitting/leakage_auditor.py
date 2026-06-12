"""Leakage auditor — Stage 5 Task 5.3.

Independent post-split similarity check (D-5.3). This is the **safety
net**: if `dedup_enforcer` uses AST similarity and the auditor uses
text similarity, the two methods can disagree, and the auditor is
the safety net for cases the enforcer misses.

The auditor does its own near-dup check (using text-shingle similarity
or any method the auditor is configured to use) and reports any leak
it finds. The report is informational — the auditor does NOT block
the split (that's `dedup_enforcer`'s job). The leak count is recorded
in `split_manifest.json` for the data team to review.

Differences from `dedup_enforcer`:
  - dedup_enforcer uses Stage 1's pre-computed `dedup_group` field
    (AST similarity at threshold 0.85)
  - leakage_auditor does its own ad-hoc check (default: text shingle
    similarity at threshold 0.5, per AUDIT_PATCHES 5-P3)
  - dedup_enforcer REASSIGNS contracts; leakage_auditor REPORTS only

The auditor is invoked AFTER the dedup_enforcer. If it finds leaks,
that's a bug to fix in dedup_enforcer (or a config tweak).
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from sentinel_data.splitting.splitters import Contract, Splits

log = logging.getLogger("sentinel_data.splitting.leakage_auditor")

DEFAULT_TEXT_SIMILARITY_THRESHOLD = 0.5  # per AUDIT_PATCHES 5-P3
SHINGLE_SIZE = 3


def _shingles(s: str, n: int = SHINGLE_SIZE) -> set[str]:
    """Return the set of n-gram shingles in `s`."""
    s = re.sub(r"\s+", " ", s.lower())
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _text_similarity(a: str, b: str) -> float:
    """Jaccard similarity of 3-shingles."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class LeakPair:
    """One near-dup pair found across split boundaries."""
    sha_a: str
    split_a: str
    sha_b: str
    split_b: str
    similarity: float


@dataclass
class LeakageReport:
    """Output of the auditor."""
    threshold: float
    pairs: list[LeakPair] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def find_leaks(
    splits: Splits,
    *,
    texts: dict[str, str],        # sha256 -> source code (for text sim)
    threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,
    sources_for_text: Optional[set[str]] = None,  # only check these sources
) -> LeakageReport:
    """Scan split boundaries for near-dup contracts (text shingle sim >= threshold).

    For each contract in train, compute its shingle set. For each contract
    in val/test, compute shingle set. If Jaccard similarity >= threshold,
    it's a leak. We do O(N²) comparisons — for the v2 baseline (22K
    contracts) this is ~500M comparisons, which takes ~10-30 min.

    A faster implementation would use LSH (Locality-Sensitive Hashing)
    to find candidate pairs in O(N). For the v2 baseline, O(N²) is
    acceptable; for larger corpora, swap in an LSH implementation.
    """
    # Build (sha, split_name) → text map
    text_by_split: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split_name in ("train", "val", "test"):
        for c in splits.get(split_name):
            if c.sha256 in texts:
                if sources_for_text is None or c.source in sources_for_text:
                    text_by_split[split_name].append((c.sha256, texts[c.sha256]))

    # Pre-compute shingle sets
    shingles_by_sha: dict[str, set[str]] = {}
    for split_name, items in text_by_split.items():
        for sha, text in items:
            shingles_by_sha[sha] = _shingles(text)

    pairs: list[LeakPair] = []
    seen_pairs: set[tuple[str, str]] = set()

    for split_a, split_b in (("train", "val"), ("train", "test"), ("val", "test")):
        items_a = text_by_split[split_a]
        items_b = text_by_split[split_b]
        for sha_a, _ in items_a:
            sh_a = shingles_by_sha.get(sha_a)
            if not sh_a:
                continue
            for sha_b, _ in items_b:
                if sha_a == sha_b:
                    continue  # same contract in two splits (shouldn't happen)
                pair_key = tuple(sorted((sha_a, sha_b)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                sh_b = shingles_by_sha.get(sha_b)
                if not sh_b:
                    continue
                sim = len(sh_a & sh_b) / len(sh_a | sh_b) if (sh_a | sh_b) else 0.0
                if sim >= threshold:
                    pairs.append(LeakPair(
                        sha_a=sha_a, split_a=split_a,
                        sha_b=sha_b, split_b=split_b,
                        similarity=sim,
                    ))

    return LeakageReport(threshold=threshold, pairs=pairs)


def run_audit(
    splits: Splits,
    *,
    data_dir: Path,
    threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,
    sources: Optional[list[str]] = None,
) -> LeakageReport:
    """High-level: load the preprocessed .sol texts and run find_leaks.

    `data_dir` is the data/ root (e.g. `Data/data`). The auditor reads
    `<data_dir>/preprocessed/<source>/<sha>.sol` for each contract.
    """
    texts: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for c in splits.get(split_name):
            if c.sha256 in texts:
                continue
            sol_path = data_dir / "preprocessed" / c.source / f"{c.sha256}.sol"
            if sol_path.exists():
                texts[c.sha256] = sol_path.read_text(errors="ignore")

    sources_set = set(sources) if sources else None
    return find_leaks(splits, texts=texts, threshold=threshold,
                     sources_for_text=sources_set)
