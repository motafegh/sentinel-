"""Group-balanced sampler for R4 Phase 8."""
from __future__ import annotations

import hashlib
import random
from typing import Mapping

from torch.utils.data import Sampler


class DeterministicGroupSampler(Sampler[int]):
    """Yield one rotating contract from each frozen group per epoch."""

    def __init__(self, group_to_indices: Mapping[str, tuple[int, ...]], *, seed: int) -> None:
        self.groups = {
            str(group): tuple(int(i) for i in indices)
            for group, indices in group_to_indices.items()
        }
        if not self.groups:
            raise ValueError("at least one group is required")
        if any(not values for values in self.groups.values()):
            raise ValueError("empty group is not allowed")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _pick(self, group: str, indices: tuple[int, ...]) -> int:
        raw = f"{self.seed}:{self.epoch}:{group}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        return indices[value % len(indices)]

    def __iter__(self):
        group_names = sorted(self.groups)
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(group_names)
        for group in group_names:
            yield self._pick(group, self.groups[group])

    def __len__(self) -> int:
        return len(self.groups)


__all__ = ["DeterministicGroupSampler"]
