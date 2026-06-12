"""Deduplication enforcer — Stage 5 Task 5.2.

Takes the output of any splitter and reassigns any near-dup group that
straddles a split boundary. This is the BCCC-failure pattern fix:

  BCCC had 38.8% duplication. Many contracts appeared in BOTH train
  and test, inflating Run 9's F1 by ~0.05 (estimated). The dedup
  enforcer eliminates this.

Design decision (per plan D-5.2):
  Two-pass split: stratified_splitter → dedup_enforcer.
  Reassignment rule: the group goes to the split where the majority
  of its members are; ties go to train. The enforcer records all
  reassignments in the split manifest.

The enforcer looks up dedup_groups from the per-contract `dedup_group`
field, set by Stage 1's deduplicator. The groups are pre-computed
during Stage 1 preprocessing, so this is a fast lookup.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Iterable

from sentinel_data.splitting.splitters import Contract, Splits, SplitName

log = logging.getLogger("sentinel_data.splitting.dedup_enforcer")


def apply_dedup_enforcer(splits: Splits) -> Splits:
    """Reassign straddling dedup groups to the split with majority members.

    Returns the same Splits object with the dedup enforcement applied
    and the metadata updated. Does NOT modify the input lists in place
    (it builds new lists internally).
    """
    # Build a map: group_id -> {split_name -> [contracts]}
    group_to_splits: dict[str, dict[str, list[Contract]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )
    group_to_all: dict[str, list[Contract]] = defaultdict(list)

    for split_name in ("train", "val", "test"):
        for c in splits.get(split_name):
            if c.dedup_group is not None:
                group_to_splits[c.dedup_group][split_name].append(c)
                group_to_all[c.dedup_group].append(c)

    # For each group, find the majority split
    reassigned_groups: list[dict] = []
    for group_id, by_split in group_to_splits.items():
        # Count members per split
        counts = {s: len(by_split[s]) for s in ("train", "val", "test")}
        total = sum(counts.values())
        if total <= 1:
            continue  # group of size 1 — nothing to enforce

        # Check if the group straddles a split boundary
        straddles = sum(1 for s in ("train", "val", "test") if counts[s] > 0)
        if straddles <= 1:
            continue  # already in one split

        # Determine target split: majority (ties → train)
        target = max(("train", "val", "test"), key=lambda s: (counts[s], s == "train"))
        target_count = counts[target]

        # Reassign all members of this group to the target split
        for s in ("train", "val", "test"):
            if s != target and counts[s] > 0:
                for c in by_split[s]:
                    reassigned_groups.append({
                        "group": group_id,
                        "from_split": s,
                        "to_split": target,
                        "contract_count": 1,
                    })
        by_split[target].extend(
            c for s in ("train", "val", "test") if s != target for c in by_split[s]
        )
        # Clear non-target splits for this group
        for s in ("train", "val", "test"):
            if s != target:
                by_split[s].clear()

    # Now rebuild the Splits lists: for each contract, check if it was
    # in a dedup group that was reassigned. If so, place it in the target.
    # Otherwise, keep its original split.
    reassigned_set: dict[str, str] = {}  # sha256 -> new_split
    for r in reassigned_groups:
        # Find the contracts in the group that were moved
        group_id = r["group"]
        target = r["to_split"]
        for c in group_to_all[group_id]:
            reassigned_set[c.sha256] = target

    # Build new train/val/test
    new_train: list[Contract] = []
    new_val: list[Contract] = []
    new_test: list[Contract] = []
    by_split_name = {"train": new_train, "val": new_val, "test": new_test}

    for split_name in ("train", "val", "test"):
        for c in splits.get(split_name):
            target = reassigned_set.get(c.sha256, split_name)
            by_split_name[target].append(c)

    splits.train = new_train
    splits.val = new_val
    splits.test = new_test

    # Update metadata
    splits.metadata.dedup_groups_resolved = len(reassigned_groups)
    splits.metadata.reassignments = reassigned_groups
    splits.update_all()
    return splits
