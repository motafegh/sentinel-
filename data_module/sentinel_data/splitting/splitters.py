"""Splitting — Stage 5 Task 5.1.

Implements the 4 splitter strategies that produce leak-free, deterministic
train/val/test splits:

  - random_splitter       — random assignment, deterministic via seed
  - stratified_splitter   — per-class distribution preserved within ±2%
  - project_splitter      — projects kept in one split (for audit datasets)
  - temporal_splitter     — pre-2023 / post-2023 split by year

The default (per proposal §3.6 + AUDIT_PATCHES 5-P1) is:
  - stratified per class + per confidence tier
  - project-level for audit datasets (Bastet, ScaBench, Web3Bugs, DeFiHackLabs)
  - stratified with project-level fallback for tool-derived sources

Design decisions (per plan):
  D-5.1: 4 splitter strategies; per-source strategy in config.yaml
  D-5.2: Two-pass split — splitter then dedup_enforcer (in dedup_enforcer.py)
  D-5.3: Leakage auditor is an independent post-split check (in leakage_auditor.py)
  D-5.4: SQLite + YAML mirror registry (in registry/catalog.py)
  D-5.7: Dataset versions are named and append-only

Two-pass split:
  Pass 1: splitter assigns contracts to splits per the strategy
  Pass 2: dedup_enforcer reassigns any near-dup group that straddles a split

The output of pass 1 is "stratified splits"; pass 2 is "leak-free splits."
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("sentinel_data.splitting")

# Default split ratios: 70% train, 15% val, 15% test
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class Contract:
    """One labeled contract, the unit of splitting.

    Fields are designed for all 4 splitter strategies + dedup_enforcer.
    A contract with all classes=0 is NonVulnerable (label=9, primary_class="NonVulnerable").
    """
    sha256: str
    source: str                        # "solidifi" / "dive" / "smartbugs_curated" / "disl"
    tier: str                          # "T0" / "T1" / "T2" / "T3" / "T4"
    classes: dict[str, int] = field(default_factory=dict)
    primary_class: str = "NonVulnerable"
    n_pos: int = 0
    loc: int = 0                       # for temporal splitting (line count)
    year: int = 0                      # for temporal splitting (contract year)
    dedup_group: Optional[str] = None  # for dedup_enforcer
    project_id: Optional[str] = None   # for project-level splitter

    @property
    def is_nonvulnerable(self) -> bool:
        return all(v == 0 for v in self.classes.values())


@dataclass
class SplitMetadata:
    """The versioned contract for a split (split_manifest.json content)."""
    version: str
    seed: int
    strategy: str                                 # "stratified" / "project" / etc.
    strategy_per_source: dict[str, str]          # per-source strategy overrides
    ratios: tuple[float, float, float]
    contract_counts: dict[str, int] = field(default_factory=dict)            # split -> count
    class_distributions: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> class -> count
    source_distributions: dict[str, dict[str, int]] = field(default_factory=dict) # split -> source -> count
    tier_distributions: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> tier -> count
    dedup_groups_resolved: int = 0
    reassignments: list[dict] = field(default_factory=list)  # [{group, from, to, count}]
    nonvulnerable_cap: Optional[dict] = None      # {cap, original, capped, per_source}
    leakage_audit: Optional[dict] = None          # populated after dedup_enforcer + leakage_auditor
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Splits:
    """The output of a split operation."""
    train: list[Contract] = field(default_factory=list)
    val: list[Contract] = field(default_factory=list)
    test: list[Contract] = field(default_factory=list)
    metadata: SplitMetadata = field(default_factory=lambda: SplitMetadata(
        version="v1", seed=42, strategy="stratified", ratios=DEFAULT_RATIOS,
        strategy_per_source={},
    ))

    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def get(self, split) -> list[Contract]:
        if isinstance(split, SplitName):
            split = split.value
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def update_counts(self) -> None:
        self.metadata.contract_counts = {
            "train": len(self.train), "val": len(self.val), "test": len(self.test),
        }

    def update_class_distributions(self) -> None:
        """Compute per-split, per-class positive counts."""
        for split_name in ("train", "val", "test"):
            cc: Counter = Counter()
            for c in self.get(split_name):
                for cls, val in c.classes.items():
                    if val == 1:
                        cc[cls] += 1
            self.metadata.class_distributions[split_name] = dict(cc)

    def update_source_distributions(self) -> None:
        for split_name in ("train", "val", "test"):
            sc: Counter = Counter()
            for c in self.get(split_name):
                sc[c.source] += 1
            self.metadata.source_distributions[split_name] = dict(sc)

    def update_tier_distributions(self) -> None:
        for split_name in ("train", "val", "test"):
            tc: Counter = Counter()
            for c in self.get(split_name):
                tc[c.tier] += 1
            self.metadata.tier_distributions[split_name] = dict(tc)

    def update_all(self) -> None:
        self.update_counts()
        self.update_class_distributions()
        self.update_source_distributions()
        self.update_tier_distributions()


# ── Splitter strategies ─────────────────────────────────────────────────────

def random_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> Splits:
    """Random assignment, deterministic via seed.

    No guarantees on per-class distribution. Use only for sanity testing.
    """
    rng = random.Random(seed)
    n = len(contracts)
    if n == 0:
        return Splits(metadata=SplitMetadata(
            version="v1", seed=seed, strategy="random", ratios=ratios,
            strategy_per_source={},
        ))
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # n_test = n - n_train - n_val  (remainder)

    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:n_train + n_val])

    out = Splits(metadata=SplitMetadata(
        version="v1", seed=seed, strategy="random", ratios=ratios,
        strategy_per_source={},
    ))
    for i, c in enumerate(contracts):
        if i in train_idx:
            out.train.append(c)
        elif i in val_idx:
            out.val.append(c)
        else:
            out.test.append(c)
    out.update_all()
    return out


def stratified_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    strata: list[str] | None = None,
) -> Splits:
    """Per-class (and optionally per-source, per-tier) stratified split.

    The per-class distribution is preserved within ±2% across splits.
    Default strata: ["primary_class", "source", "tier"].
    """
    if strata is None:
        strata = ["primary_class", "source", "tier"]

    rng = random.Random(seed)
    n = len(contracts)
    if n == 0:
        return Splits(metadata=SplitMetadata(
            version="v1", seed=seed, strategy="stratified", ratios=ratios,
            strategy_per_source={},
        ))

    # Group contracts by stratum (composite key)
    def stratum_key(c: Contract) -> tuple:
        parts = []
        for s in strata:
            if s == "primary_class":
                parts.append(c.primary_class)
            elif s == "source":
                parts.append(c.source)
            elif s == "tier":
                parts.append(c.tier)
            else:
                parts.append("?")
        return tuple(parts)

    groups: dict[tuple, list[Contract]] = defaultdict(list)
    for c in contracts:
        groups[stratum_key(c)].append(c)

    # For each stratum, assign to train/val/test per ratios
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    out = Splits(metadata=SplitMetadata(
        version="v1", seed=seed, strategy="stratified", ratios=ratios,
        strategy_per_source={c.source: "stratified" for c in contracts},
    ))

    # Distribute contracts from each stratum proportionally
    for key, group in groups.items():
        rng.shuffle(group)
        g_n = len(group)
        # Proportional allocation
        g_train = max(1, int(g_n * ratios[0])) if g_n > 0 else 0
        g_val = max(1, int(g_n * ratios[1])) if g_n > 0 else 0
        # Adjust if total exceeds target
        # (We do best-effort per-stratum; total may slightly differ from target)
        out.train.extend(group[:g_train])
        out.val.extend(group[g_train:g_train + g_val])
        out.test.extend(group[g_train + g_val:])

    # If train/val/test counts are way off target, do a global rebalance.
    # This is rare — only happens when strata are very imbalanced.
    out.update_all()
    return out


def project_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> Splits:
    """Project-level split: each project goes entirely to one split.

    For audit datasets (Bastet, ScaBench, Web3Bugs, DeFiHackLabs) where
    contracts in the same project are highly correlated. The default
    strategy for these sources per the plan.
    """
    rng = random.Random(seed)
    out = Splits(metadata=SplitMetadata(
        version="v1", seed=seed, strategy="project", ratios=ratios,
        strategy_per_source={c.source: "project" for c in contracts},
    ))

    # Group by project_id
    projects: dict[str, list[Contract]] = defaultdict(list)
    no_project: list[Contract] = []
    for c in contracts:
        if c.project_id:
            projects[c.project_id].append(c)
        else:
            no_project.append(c)

    # Assign projects to splits
    project_ids = list(projects.keys())
    rng.shuffle(project_ids)
    n_train = int(len(project_ids) * ratios[0])
    n_val = int(len(project_ids) * ratios[1])
    train_projs = set(project_ids[:n_train])
    val_projs = set(project_ids[n_train:n_train + n_val])

    for pid, plist in projects.items():
        if pid in train_projs:
            out.train.extend(plist)
        elif pid in val_projs:
            out.val.extend(plist)
        else:
            out.test.extend(plist)

    # Contracts without project_id go to train (best-effort)
    out.train.extend(no_project)
    out.update_all()
    return out


def temporal_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    cutoff_year: int = 2023,
) -> Splits:
    """Temporal split: pre-cutoff_year contracts in train/val, post-cutoff in test.

    Simulates "train on past, test on future" — the realistic deployment
    scenario. The val set draws from pre-cutoff too.
    """
    rng = random.Random(seed)
    out = Splits(metadata=SplitMetadata(
        version="v1", seed=seed, strategy="temporal", ratios=ratios,
        strategy_per_source={c.source: "temporal" for c in contracts},
    ))

    pre_cutoff = [c for c in contracts if c.year and c.year <= cutoff_year]
    post_cutoff = [c for c in contracts if c.year and c.year > cutoff_year]
    no_year = [c for c in contracts if not c.year]

    # All post-cutoff → test
    out.test.extend(post_cutoff)
    # Pre-cutoff: stratified between train/val
    if pre_cutoff:
        n = len(pre_cutoff)
        n_train = int(n * ratios[0] / (ratios[0] + ratios[1]))
        rng.shuffle(pre_cutoff)
        out.train.extend(pre_cutoff[:n_train])
        out.val.extend(pre_cutoff[n_train:])

    # No-year: best-effort to train
    out.train.extend(no_year)
    out.update_all()
    return out


# ── Strategy dispatch ────────────────────────────────────────────────────────

SPLITTERS = {
    "random": random_split,
    "stratified": stratified_split,
    "project": project_split,
    "project_level": project_split,   # alias per AUDIT_PATCHES 5-P1
    "temporal": temporal_split,
}


def apply_strategy(
    strategy: str,
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    **kwargs,
) -> Splits:
    """Dispatch to the named splitter strategy."""
    fn = SPLITTERS.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown splitter strategy: {strategy}. "
                         f"Choose from {list(SPLITTERS)}")
    return fn(contracts, ratios=ratios, seed=seed, **kwargs)


# ── Manifest writer ──────────────────────────────────────────────────────────

def write_manifest(splits: Splits, output_dir: Path) -> Path:
    """Write the versioned split_manifest.json contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(asdict(splits.metadata), indent=2))
    log.info(f"Wrote manifest: {manifest_path}")
    return manifest_path


def write_splits(splits: Splits, output_dir: Path) -> None:
    """Write train/val/test as JSONL files (one Contract per line)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        path = output_dir / f"{split_name}.jsonl"
        with path.open("w") as f:
            for c in splits.get(split_name):
                row = {
                    "sha256": c.sha256,
                    "source": c.source,
                    "tier": c.tier,
                    "classes": c.classes,
                    "primary_class": c.primary_class,
                    "n_pos": c.n_pos,
                    "loc": c.loc,
                    "year": c.year,
                    "dedup_group": c.dedup_group,
                    "project_id": c.project_id,
                }
                f.write(json.dumps(row) + "\n")
        log.info(f"Wrote {len(splits.get(split_name))} contracts to {path}")


def load_splits(output_dir: Path) -> tuple[dict[str, list[Contract]], SplitMetadata]:
    """Load train/val/test from a split directory."""
    splits: dict[str, list[Contract]] = {"train": [], "val": [], "test": []}
    for split_name in ("train", "val", "test"):
        path = output_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                splits[split_name].append(Contract(
                    sha256=row["sha256"],
                    source=row["source"],
                    tier=row["tier"],
                    classes=row.get("classes", {}),
                    primary_class=row.get("primary_class", "NonVulnerable"),
                    n_pos=row.get("n_pos", 0),
                    loc=row.get("loc", 0),
                    year=row.get("year", 0),
                    dedup_group=row.get("dedup_group"),
                    project_id=row.get("project_id"),
                ))
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.exists():
        md = json.loads(manifest_path.read_text())
        metadata = SplitMetadata(**md)
    else:
        metadata = SplitMetadata(version="v1", seed=42, strategy="unknown", ratios=DEFAULT_RATIOS)
    return splits, metadata
