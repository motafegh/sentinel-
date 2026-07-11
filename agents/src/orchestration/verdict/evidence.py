"""
evidence.py — The uniform Evidence record (proposal §5.1).

Every channel — ML, Slither, Aderyn, RAG, debate, and future Halmos/Z3,
Gigahorse, taint, access-control, economic — emits zero or more Evidence items.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Polarity(str, enum.Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NEUTRAL = "NEUTRAL"


class Kind(str, enum.Enum):
    STATISTICAL = "STATISTICAL"
    SYNTACTIC = "SYNTACTIC"
    SEMANTIC = "SEMANTIC"
    FORMAL = "FORMAL"
    ECONOMIC = "ECONOMIC"


@dataclass(frozen=True)
class Evidence:
    source: str
    vuln_class: str
    polarity: Polarity
    strength: float           # [0, 1]
    reliability: float         # [0, 1]
    kind: Kind
    deterministic: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength must be in [0,1], got {self.strength}")
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(f"reliability must be in [0,1], got {self.reliability}")

    # ── Helper constructors ────────────────────────────────────────────────

    @staticmethod
    def ml(vuln_class: str, probability: float, reliability: float,
           tier: str = "", detail: dict | None = None) -> Evidence:
        return Evidence(
            source="ml",
            vuln_class=vuln_class,
            polarity=Polarity.SUPPORTS,
            strength=round(float(probability), 4),
            reliability=round(float(reliability), 4),
            kind=Kind.STATISTICAL,
            deterministic=True,
            detail={"tier": tier, **(detail or {})},
        )

    @staticmethod
    def slither(vuln_class: str, impact: str, description: str,
                reliability: float, detector: str = "",
                lines: list | None = None) -> Evidence:
        _impact_map = {"High": 1.0, "Medium": 0.6, "Low": 0.3,
                       "Informational": 0.15, "Optimization": 0.1}
        strength = _impact_map.get(impact, 0.3)
        return Evidence(
            source="slither",
            vuln_class=vuln_class,
            polarity=Polarity.SUPPORTS,
            strength=strength,
            reliability=round(float(reliability), 4),
            kind=Kind.SYNTACTIC,
            deterministic=True,
            detail={
                "detector": detector, "impact": impact,
                "description": description[:200],
                "lines": lines or [],
            },
        )

    @staticmethod
    def aderyn(vuln_class: str, impact: str, description: str,
               reliability: float, detector: str = "",
               lines: list | None = None) -> Evidence:
        _impact_map = {"High": 1.0, "Medium": 0.6, "Low": 0.3,
                       "Informational": 0.15, "NC": 0.1}
        strength = _impact_map.get(impact, 0.3)
        return Evidence(
            source="aderyn",
            vuln_class=vuln_class,
            polarity=Polarity.SUPPORTS if impact in ("High", "Medium") else Polarity.NEUTRAL,
            strength=strength,
            reliability=round(float(reliability), 4),
            kind=Kind.SYNTACTIC,
            deterministic=True,
            detail={
                "detector": detector, "impact": impact,
                "description": description[:200],
                "lines": lines or [],
            },
        )

    @staticmethod
    def rag(vuln_class: str, similarity: float, reliability: float,
            chunk_id: str = "", title: str = "") -> Evidence:
        return Evidence(
            source="rag",
            vuln_class=vuln_class,
            polarity=Polarity.SUPPORTS,
            strength=min(float(similarity), 1.0),
            reliability=round(float(reliability), 4),
            kind=Kind.SEMANTIC,
            deterministic=True,
            detail={"chunk_id": chunk_id, "title": title, "similarity": similarity},
        )

    @staticmethod
    def debate(vuln_class: str, verdict: str, confidence: float,
               judge_rationale: str = "") -> Evidence:
        _polarity_map = {
            "CONFIRMED": Polarity.SUPPORTS,
            "LIKELY": Polarity.SUPPORTS,
            "DISPUTED": Polarity.NEUTRAL,
            "WATCH": Polarity.NEUTRAL,
            "INCONCLUSIVE": Polarity.NEUTRAL,
            "SAFE": Polarity.REFUTES,
        }
        polarity = _polarity_map.get(verdict, Polarity.NEUTRAL)
        return Evidence(
            source="debate",
            vuln_class=vuln_class,
            polarity=polarity,
            strength=round(float(confidence), 4),
            reliability=0.55,  # LLM debate — L3 after measurement
            kind=Kind.SEMANTIC,
            deterministic=False,
            detail={"debate_verdict": verdict, "rationale": judge_rationale[:200]},
        )

    @staticmethod
    def quick_screen(vuln_class: str, detector: str, impact: str,
                     reliability: float = 0.40) -> Evidence:
        _impact_map = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
        strength = _impact_map.get(impact, 0.3)
        return Evidence(
            source="quick_screen",
            vuln_class=vuln_class,
            polarity=Polarity.SUPPORTS,
            strength=strength,
            reliability=round(float(reliability), 4),
            kind=Kind.SYNTACTIC,
            deterministic=True,
            detail={"detector": detector, "impact": impact},
        )

    @staticmethod
    def formal(source: str, vuln_class: str, polarity: "Polarity",
               invariant: str, proven: bool, counterexample: str = "",
               reliability: float = 0.95) -> Evidence:
        """
        Formal verification evidence (Halmos, Z3, Gigahorse).

        - polarity=SUPPORTS: invariant violation found (vulnerability confirmed)
        - polarity=REFUTES: invariant proven to hold (vulnerability refuted)
        - strength: 1.0 for SUPPORTS (formal proof of violation),
                   0.9 for REFUTES (formal proof of safety, but limited by tool coverage)
        """
        return Evidence(
            source=source,
            vuln_class=vuln_class,
            polarity=polarity,
            strength=1.0 if polarity == Polarity.SUPPORTS else 0.9,
            reliability=round(float(reliability), 4),
            kind=Kind.FORMAL,
            deterministic=True,
            detail={
                "invariant": invariant,
                "proven": proven,
                "counterexample": counterexample[:200] if counterexample else "",
            },
        )
