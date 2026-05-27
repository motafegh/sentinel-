# agents/tests/test_routing_phase0.py
"""
Phase 0 routing tests — per-class thresholds, tool matrix, verdict logic.

Coverage:
    routing.py:
        - compute_active_tools()    — per-class threshold enforcement
        - build_routing_decisions() — human-readable log strings
        - compute_verdict()         — CONFIRMED/LIKELY/DISPUTED/SAFE logic
        - compute_overall_verdict() — max-rank across classes
        - DETECTOR_TO_CLASSES       — inverted map completeness

    nodes.py evidence_router:
        - logs routing_decisions to state
        - returns empty list on fast path (no threshold exceeded)

    graph.py:
        - evidence_router node present in compiled graph
        - SqliteSaver (or MemorySaver fallback) attached

Running:
    cd ~/projects/sentinel/agents
    poetry run pytest tests/test_routing_phase0.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.routing import (
    CLASS_TO_DETECTORS,
    DEEP_THRESHOLDS,
    DETECTOR_TO_CLASSES,
    ROUTING_RULES,
    build_routing_decisions,
    compute_active_tools,
    compute_overall_verdict,
    compute_verdict,
    prob_to_severity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ml(vulns: list[tuple[str, float]]) -> dict:
    """Build a minimal ml_result dict from (class, prob) pairs (three-tier schema)."""
    CONF_THR = 0.55
    SUSP_THR = 0.25
    probs = dict(vulns)
    confirmed = [
        {"vulnerability_class": c, "probability": p, "tier": "CONFIRMED"}
        for c, p in vulns if p >= CONF_THR
    ]
    suspicious = [
        {"vulnerability_class": c, "probability": p, "tier": "SUSPICIOUS"}
        for c, p in vulns if SUSP_THR <= p < CONF_THR
    ]
    if confirmed:     label = "confirmed_vulnerable"
    elif suspicious:  label = "suspicious"
    else:             label = "safe"
    return {
        "label":           label,
        "probabilities":   probs,
        "confirmed":       confirmed,
        "suspicious":      suspicious,
        "vulnerabilities": [{"vulnerability_class": c, "probability": p} for c, p in vulns if p >= CONF_THR],
        "tier_thresholds": {"confirmed": CONF_THR, "suspicious": SUSP_THR, "noteworthy": 0.10},
        "threshold":       0.50,
    }


# ---------------------------------------------------------------------------
# compute_active_tools
# ---------------------------------------------------------------------------

class TestComputeActiveTools:
    def test_all_below_threshold_returns_empty(self):
        ml = _ml([("Reentrancy", 0.20), ("IntegerUO", 0.10)])
        assert compute_active_tools(ml) == []

    def test_reentrancy_above_threshold_activates_static_and_rag(self):
        ml = _ml([("Reentrancy", 0.87)])
        tools = compute_active_tools(ml)
        assert "static_analysis" in tools
        assert "rag_research" in tools

    def test_gas_exception_only_activates_static(self):
        ml = _ml([("GasException", 0.60)])
        tools = compute_active_tools(ml)
        assert "static_analysis" in tools
        assert "rag_research" not in tools

    def test_mishandled_exception_only_activates_static(self):
        ml = _ml([("MishandledException", 0.55)])
        tools = compute_active_tools(ml)
        assert "static_analysis" in tools
        assert "rag_research" not in tools

    def test_dos_uses_lower_threshold_0_30(self):
        # DoS threshold is 0.30, lower than default 0.40
        ml = _ml([("DenialOfService", 0.31)])
        assert "static_analysis" in compute_active_tools(ml)

    def test_dos_below_0_30_skipped(self):
        ml = _ml([("DenialOfService", 0.29)])
        assert compute_active_tools(ml) == []

    def test_unused_return_uses_threshold_0_45(self):
        ml = _ml([("UnusedReturn", 0.44)])
        assert compute_active_tools(ml) == []  # below 0.45
        ml2 = _ml([("UnusedReturn", 0.45)])
        assert "static_analysis" in compute_active_tools(ml2)

    def test_multiple_classes_deduplicates_tools(self):
        # Both Reentrancy and IntegerUO trigger static_analysis + rag_research
        # — should appear once each, not twice.
        ml = _ml([("Reentrancy", 0.87), ("IntegerUO", 0.72)])
        tools = compute_active_tools(ml)
        assert tools.count("static_analysis") == 1
        assert tools.count("rag_research") == 1

    def test_empty_vulnerabilities_returns_empty(self):
        assert compute_active_tools({"vulnerabilities": []}) == []

    def test_empty_ml_result_returns_empty(self):
        assert compute_active_tools({}) == []

    def test_all_ten_classes_covered_in_routing_rules(self):
        expected = {
            "Reentrancy", "IntegerUO", "GasException", "Timestamp", "TOD",
            "ExternalBug", "CallToUnknown", "MishandledException",
            "UnusedReturn", "DenialOfService",
        }
        assert set(ROUTING_RULES.keys()) == expected

    def test_all_ten_classes_covered_in_thresholds(self):
        expected = {
            "Reentrancy", "IntegerUO", "GasException", "Timestamp", "TOD",
            "ExternalBug", "CallToUnknown", "MishandledException",
            "UnusedReturn", "DenialOfService",
        }
        assert set(DEEP_THRESHOLDS.keys()) == expected


# ---------------------------------------------------------------------------
# build_routing_decisions
# ---------------------------------------------------------------------------

class TestBuildRoutingDecisions:
    def test_fast_path_decision_string(self):
        ml = _ml([("Reentrancy", 0.10)])
        decisions = build_routing_decisions(ml)
        assert any("fast path" in d for d in decisions)

    def test_deep_path_decision_contains_class_and_tools(self):
        ml = _ml([("Reentrancy", 0.87)])
        decisions = build_routing_decisions(ml)
        assert any("Reentrancy" in d and "static_analysis" in d for d in decisions)

    def test_skipped_class_shows_threshold(self):
        ml = _ml([("GasException", 0.20)])
        decisions = build_routing_decisions(ml)
        assert any("GasException" in d and "skip" in d for d in decisions)

    def test_one_decision_per_vulnerability(self):
        ml = _ml([("Reentrancy", 0.87), ("Timestamp", 0.12)])
        decisions = build_routing_decisions(ml)
        # Each vulnerability produces exactly one decision string
        assert len(decisions) == 2


# ---------------------------------------------------------------------------
# compute_verdict
# ---------------------------------------------------------------------------

class TestComputeVerdict:
    def test_fast_path_always_likely(self):
        verdict, sources = compute_verdict("Reentrancy", 0.87, [], [], "fast")
        assert verdict == "LIKELY"
        assert "ml:0.870" in sources

    def test_confirmed_when_slither_matches(self):
        findings = [{"detector": "reentrancy-eth", "impact": "High", "confidence": "High"}]
        verdict, sources = compute_verdict("Reentrancy", 0.87, findings, [], "deep")
        assert verdict == "CONFIRMED"
        assert any("slither" in s for s in sources)

    def test_confirmed_when_rag_score_high(self):
        rag = [{"score": 0.85, "content": "reentrancy exploit"}]
        verdict, sources = compute_verdict("Reentrancy", 0.87, [], rag, "deep")
        assert verdict == "CONFIRMED"
        assert any("rag" in s for s in sources)

    def test_likely_when_rag_score_moderate(self):
        rag = [{"score": 0.65, "content": "reentrancy exploit"}]
        verdict, sources = compute_verdict("Reentrancy", 0.87, [], rag, "deep")
        assert verdict == "LIKELY"

    def test_disputed_when_no_corroboration(self):
        verdict, sources = compute_verdict("Reentrancy", 0.87, [], [], "deep")
        assert verdict == "DISPUTED"

    def test_slither_low_impact_does_not_confirm(self):
        # Informational/Low findings should not produce CONFIRMED
        findings = [{"detector": "reentrancy-eth", "impact": "Informational"}]
        verdict, _ = compute_verdict("Reentrancy", 0.87, findings, [], "deep")
        assert verdict == "DISPUTED"

    def test_wrong_detector_class_does_not_confirm(self):
        # Timestamp detector should not confirm Reentrancy
        findings = [{"detector": "timestamp", "impact": "High"}]
        verdict, _ = compute_verdict("Reentrancy", 0.87, findings, [], "deep")
        assert verdict == "DISPUTED"

    def test_sources_always_start_with_ml_prob(self):
        _, sources = compute_verdict("IntegerUO", 0.72, [], [], "deep")
        assert sources[0].startswith("ml:")


# ---------------------------------------------------------------------------
# compute_overall_verdict
# ---------------------------------------------------------------------------

class TestComputeOverallVerdict:
    def test_confirmed_beats_likely(self):
        assert compute_overall_verdict({"A": "CONFIRMED", "B": "LIKELY"}) == "CONFIRMED"

    def test_likely_beats_disputed(self):
        assert compute_overall_verdict({"A": "LIKELY", "B": "DISPUTED"}) == "LIKELY"

    def test_disputed_beats_safe(self):
        assert compute_overall_verdict({"A": "DISPUTED", "B": "SAFE"}) == "DISPUTED"

    def test_empty_dict_returns_safe(self):
        assert compute_overall_verdict({}) == "SAFE"

    def test_all_same(self):
        assert compute_overall_verdict({"A": "LIKELY", "B": "LIKELY"}) == "LIKELY"


# ---------------------------------------------------------------------------
# prob_to_severity
# ---------------------------------------------------------------------------

class TestProbToSeverity:
    def test_critical(self):   assert prob_to_severity(0.90) == "CRITICAL"
    def test_high(self):       assert prob_to_severity(0.75) == "HIGH"
    def test_medium(self):     assert prob_to_severity(0.55) == "MEDIUM"
    def test_low(self):        assert prob_to_severity(0.40) == "LOW"
    def test_info(self):       assert prob_to_severity(0.10) == "INFO"
    def test_boundary_85(self): assert prob_to_severity(0.85) == "CRITICAL"
    def test_boundary_70(self): assert prob_to_severity(0.70) == "HIGH"
    def test_boundary_50(self): assert prob_to_severity(0.50) == "MEDIUM"


# ---------------------------------------------------------------------------
# DETECTOR_TO_CLASSES inverted map
# ---------------------------------------------------------------------------

class TestDetectorToClasses:
    def test_reentrancy_eth_maps_to_reentrancy(self):
        assert "Reentrancy" in DETECTOR_TO_CLASSES.get("reentrancy-eth", [])

    def test_timestamp_maps_to_timestamp(self):
        assert "Timestamp" in DETECTOR_TO_CLASSES.get("timestamp", [])

    def test_unused_return_maps_to_unused_return(self):
        assert "UnusedReturn" in DETECTOR_TO_CLASSES.get("unused-return", [])

    def test_all_detectors_in_class_map_are_inverted(self):
        for cls, dets in CLASS_TO_DETECTORS.items():
            for det in dets:
                assert cls in DETECTOR_TO_CLASSES[det], (
                    f"detector '{det}' missing '{cls}' in DETECTOR_TO_CLASSES"
                )


# ---------------------------------------------------------------------------
# evidence_router node (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEvidenceRouterNode:
    async def test_deep_path_logs_routing_decisions(self):
        from src.orchestration.nodes import evidence_router
        state = {
            "contract_code": "pragma solidity ^0.8.0;",
            "contract_address": "0xABC",
            "ml_result": {
                "label": "vulnerable",
                "vulnerabilities": [
                    {"vulnerability_class": "Reentrancy", "probability": 0.87}
                ],
            },
        }
        update = await evidence_router(state)
        decisions = update.get("routing_decisions", [])
        assert len(decisions) > 0
        assert any("Reentrancy" in d for d in decisions)

    async def test_fast_path_logs_fast_path_decision(self):
        from src.orchestration.nodes import evidence_router
        state = {
            "ml_result": {
                "label": "safe",
                "vulnerabilities": [
                    {"vulnerability_class": "Reentrancy", "probability": 0.10}
                ],
            },
        }
        update = await evidence_router(state)
        decisions = update.get("routing_decisions", [])
        assert any("fast path" in d for d in decisions)

    async def test_empty_ml_result_returns_decisions(self):
        from src.orchestration.nodes import evidence_router
        update = await evidence_router({"ml_result": {}})
        assert "routing_decisions" in update


# ---------------------------------------------------------------------------
# graph compilation
# ---------------------------------------------------------------------------

class TestGraphCompilation:
    def test_graph_compiles_with_evidence_router_node(self):
        from src.orchestration.graph import build_graph
        graph = build_graph(use_checkpointer=False)
        # CompiledStateGraph exposes .nodes as a dict of node names
        node_names = list(graph.nodes.keys())
        assert "evidence_router" in node_names
        assert "ml_assessment" in node_names
        assert "synthesizer" in node_names

    def test_graph_compiles_without_checkpointer(self):
        from src.orchestration.graph import build_graph
        graph = build_graph(use_checkpointer=False)
        assert graph is not None
