"""
verdict — Uniform evidence model + deterministic fusion (P2, 2026-06-24).

Proposal §5.1–§5.2: every evidence channel emits zero or more Evidence items;
one fuse() function groups, de-correlates, aggregates, applies FN/FP asymmetry,
and produces a dual verdict (provable + full, per the ZK boundary in D-B).
"""

from src.orchestration.verdict.evidence import Evidence, Polarity, Kind
from src.orchestration.verdict.verdict import ClassVerdict
from src.orchestration.verdict.fuse import fuse

__all__ = [
    "Evidence",
    "Polarity",
    "Kind",
    "ClassVerdict",
    "fuse",
]
