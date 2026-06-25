"""
Tests for the uniform Evidence record (P2, 2026-06-24).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.verdict.evidence import Evidence, Polarity, Kind


class TestEvidenceConstruction:
    """Evidence dataclass construction and immutability."""

    def test_basic_construction(self):
        e = Evidence(
            source="ml", vuln_class="Reentrancy",
            polarity=Polarity.SUPPORTS, strength=0.87,
            reliability=0.6, kind=Kind.STATISTICAL,
            deterministic=True, detail={},
        )
        assert e.source == "ml"
        assert e.vuln_class == "Reentrancy"
        assert e.polarity == Polarity.SUPPORTS
        assert e.strength == 0.87
        assert e.reliability == 0.6
        assert e.kind == Kind.STATISTICAL
        assert e.deterministic is True

    def test_frozen_immutable(self):
        e = Evidence(
            source="ml", vuln_class="Reentrancy",
            polarity=Polarity.SUPPORTS, strength=0.87,
            reliability=0.6, kind=Kind.STATISTICAL,
            deterministic=True,
        )
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            e.strength = 0.99  # type: ignore[misc]

    def test_strength_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Evidence(source="x", vuln_class="c", polarity=Polarity.SUPPORTS,
                     strength=1.5, reliability=0.5, kind=Kind.STATISTICAL,
                     deterministic=True)

    def test_reliability_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Evidence(source="x", vuln_class="c", polarity=Polarity.SUPPORTS,
                     strength=0.5, reliability=-0.1, kind=Kind.STATISTICAL,
                     deterministic=True)

    def test_default_detail(self):
        e = Evidence(source="x", vuln_class="c", polarity=Polarity.NEUTRAL,
                     strength=0.5, reliability=0.5, kind=Kind.SYNTACTIC,
                     deterministic=True)
        assert e.detail == {}


class TestHelperConstructors:
    """Static helper constructors create correct Evidence."""

    def test_ml_constructor(self):
        e = Evidence.ml("Reentrancy", 0.87, 0.6, tier="CONFIRMED")
        assert e.source == "ml"
        assert e.polarity == Polarity.SUPPORTS
        assert e.strength == 0.87
        assert e.reliability == 0.6
        assert e.kind == Kind.STATISTICAL
        assert e.deterministic is True
        assert e.detail["tier"] == "CONFIRMED"

    def test_slither_constructor_high_impact(self):
        e = Evidence.slither("Reentrancy", "High", "reentrancy", 0.82,
                             detector="reentrancy-eth")
        assert e.source == "slither"
        assert e.polarity == Polarity.SUPPORTS
        assert e.strength == 1.0
        assert e.kind == Kind.SYNTACTIC
        assert e.deterministic is True

    def test_slither_constructor_medium_impact(self):
        e = Evidence.slither("IntegerUO", "Medium", "overflow", 0.80)
        assert e.strength == 0.6

    def test_slither_constructor_low_impact(self):
        e = Evidence.slither("GasException", "Low", "costly", 0.65)
        assert e.strength == 0.3

    def test_aderyn_constructor(self):
        e = Evidence.aderyn("CallToUnknown", "High", "delegate", 0.60,
                            detector="delegate-call")
        assert e.source == "aderyn"
        assert e.polarity == Polarity.SUPPORTS

    def test_aderyn_low_impact_neutral(self):
        e = Evidence.aderyn("GasException", "Low", "loop", 0.55)
        assert e.polarity == Polarity.NEUTRAL

    def test_rag_constructor(self):
        e = Evidence.rag("Reentrancy", 0.72, 0.50, chunk_id="ch001", title="DAO hack")
        assert e.source == "rag"
        assert e.polarity == Polarity.SUPPORTS
        assert e.strength == pytest.approx(0.72)
        assert e.kind == Kind.SEMANTIC
        assert e.deterministic is True

    def test_debate_constructor_confirmed(self):
        e = Evidence.debate("Reentrancy", "CONFIRMED", 0.85)
        assert e.source == "debate"
        assert e.polarity == Polarity.SUPPORTS
        assert e.deterministic is False

    def test_debate_constructor_safe_refutes(self):
        e = Evidence.debate("Reentrancy", "SAFE", 0.80)
        assert e.polarity == Polarity.REFUTES

    def test_debate_constructor_disputed_neutral(self):
        e = Evidence.debate("Reentrancy", "DISPUTED", 0.40)
        assert e.polarity == Polarity.NEUTRAL

    def test_quick_screen_constructor(self):
        e = Evidence.quick_screen("Reentrancy", "reentrancy-eth", "High")
        assert e.source == "quick_screen"
        assert e.kind == Kind.SYNTACTIC
        assert e.deterministic is True
        assert e.strength == 1.0


class TestPolarityAndKindEnums:
    """Enum value checks."""

    def test_polarity_values(self):
        assert Polarity.SUPPORTS.value == "SUPPORTS"
        assert Polarity.REFUTES.value == "REFUTES"
        assert Polarity.NEUTRAL.value == "NEUTRAL"

    def test_kind_values(self):
        assert Kind.STATISTICAL.value == "STATISTICAL"
        assert Kind.SYNTACTIC.value == "SYNTACTIC"
        assert Kind.SEMANTIC.value == "SEMANTIC"
        assert Kind.FORMAL.value == "FORMAL"
        assert Kind.ECONOMIC.value == "ECONOMIC"
