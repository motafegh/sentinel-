"""
verdict.py — ClassVerdict: the fused output for one vulnerability class.

Contains both the provable (deterministic-tier) and full (all-evidence)
verdicts, plus the confidence and the driving evidence for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassVerdict:
    cls: str
    verdict_provable: str   # CONFIRMED | LIKELY | DISPUTED | SAFE
    verdict_full: str       # CONFIRMED | LIKELY | DISPUTED | SAFE
    confidence: float       # [0, 1] aggregated post-asymmetry confidence
    driving_evidence: list[Any] = field(default_factory=list)  # list[Evidence]
