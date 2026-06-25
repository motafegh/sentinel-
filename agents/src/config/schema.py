"""
schema.py — Pydantic v2 models for all decision-number groups.

Each model mirrors the constants in the corresponding orchestration module.
Current code values are the field defaults (behaviour preservation).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConsensusConfig(BaseModel):
    accuracy_weights: dict[str, dict[str, float]] = {
        "Reentrancy":                 {"ml": 0.78, "slither": 0.82, "aderyn": 0.60},
        "IntegerUO":                  {"ml": 0.62, "slither": 0.80, "aderyn": 0.70},
        "GasException":               {"ml": 0.40, "slither": 0.65, "aderyn": 0.55},
        "Timestamp":                  {"ml": 0.80, "slither": 0.45, "aderyn": 0.40},
        "TransactionOrderDependence": {"ml": 0.70, "slither": 0.60, "aderyn": 0.45},
        "ExternalBug":                {"ml": 0.45, "slither": 0.50, "aderyn": 0.45},
        "CallToUnknown":              {"ml": 0.60, "slither": 0.70, "aderyn": 0.60},
        "MishandledException":        {"ml": 0.55, "slither": 0.72, "aderyn": 0.62},
        "UnusedReturn":               {"ml": 0.55, "slither": 0.75, "aderyn": 0.65},
        "DenialOfService":            {"ml": 0.65, "slither": 0.55, "aderyn": 0.50},
    }
    default_weights: dict[str, float] = {"ml": 0.60, "slither": 0.65, "aderyn": 0.55}
    ml_weight_scale: float = 0.5
    ml_positive_threshold: float = 0.50
    confirmed_band: float = 0.70
    likely_band: float = 0.50
    disputed_band: float = 0.30


class ConfidenceConfig(BaseModel):
    slither_agree: float = 1.10
    slither_disagree: float = 0.90
    aderyn_agree: float = 1.05
    aderyn_disagree: float = 0.97
    rag_agree: float = 1.05
    rag_relevance: float = 0.70


class RoutingConfig(BaseModel):
    deep_thresholds: dict[str, float] = {
        "Reentrancy":          0.35,
        "IntegerUO":           0.35,
        "GasException":        0.40,
        "Timestamp":           0.35,
        "TransactionOrderDependence": 0.35,
        "ExternalBug":         0.40,
        "CallToUnknown":       0.40,
        "MishandledException": 0.40,
        "UnusedReturn":        0.45,
        "DenialOfService":     0.30,
    }
    routing_rules: dict[str, list[str]] = {
        "Reentrancy":          ["static_analysis", "rag_research"],
        "IntegerUO":           ["static_analysis", "rag_research"],
        "GasException":        ["static_analysis"],
        "Timestamp":           ["static_analysis", "rag_research"],
        "TransactionOrderDependence": ["static_analysis", "rag_research"],
        "ExternalBug":         ["static_analysis", "rag_research"],
        "CallToUnknown":       ["static_analysis", "rag_research"],
        "MishandledException": ["static_analysis"],
        "UnusedReturn":        ["static_analysis"],
        "DenialOfService":     ["static_analysis", "rag_research"],
    }
    prob_to_severity: dict[str, float] = {
        "CRITICAL": 0.85,
        "HIGH": 0.70,
        "MEDIUM": 0.50,
        "LOW": 0.35,
    }
    overall_verdict_rank: dict[str, int] = {
        "CONFIRMED": 5, "LIKELY": 4, "DISPUTED": 3, "WATCH": 2,
        "INCONCLUSIVE": 1, "SAFE": 0,
    }
    compute_verdict_prob_cutoff: float = 0.50
    compute_verdict_rag_confirmed_cutoff: float = 0.80
    compute_verdict_rag_likely_cutoff: float = 0.50


class AttributionConfig(BaseModel):
    rag_relevance_floor: float = 0.30


class EvalConfig(BaseModel):
    positive_verdicts: list[str] = ["CONFIRMED", "LIKELY"]
    borderline_band_low: float = 0.35
    borderline_band_high: float = 0.50
    fbeta_beta: float = 2.0


class SentinelConfig(BaseModel):
    schema_version: str = "1"
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    attribution: AttributionConfig = Field(default_factory=AttributionConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
