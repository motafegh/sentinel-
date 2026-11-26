"""Synthetic report-policy regressions."""
from unittest.mock import patch
import pytest
from src.contracts.execution import bind_status
from src.orchestration.nodes.synthesizer import synthesizer
from src.orchestration.routing import CLASS_TO_DETECTORS
from src.orchestration.verdict.evidence import Evidence
from src.eval.provenance import EvaluationInputError

SOURCE = "contract Example {}"

def live_ml():
    return bind_status({
        "label": "safe", "probabilities": dict.fromkeys(CLASS_TO_DETECTORS, 0.01),
        "confirmed": [], "suspicious": [], "truncated": False, "model_hash": "a" * 64,
    }, dependency="module1-inference", input_payload={"source_code": SOURCE}, duration_ms=1)

def clean_state():
    return {"contract_code": SOURCE, "analysis_path": "fast", "no_llm": True,
            "ml_result": live_ml(), "quick_screen_hits": {"slither": [], "aderyn": []},
            "tool_status": {"slither": {"ran": True}, "aderyn": {"ran": True}}}

async def report(state):
    with patch("src.orchestration.nodes.synthesizer.persist_report", return_value={}):
        return (await synthesizer({"contract_code": SOURCE, "no_llm": True, **state}))["final_report"]

@pytest.mark.asyncio
async def test_no_evidence_abstains():
    r = await report({})
    assert r["overall_verdict"] == "INCONCLUSIVE"
    assert r["assessment"]["coverage"] == "INCOMPLETE"
    assert r["assessment"]["abstained"] is True

@pytest.mark.asyncio
async def test_completed_screen_not_safety():
    r = await report(clean_state())
    assert r["overall_verdict"] == "NO_FINDINGS_IN_SCREEN_SCOPE"
    assert r["assessment"]["coverage"] == "SCREEN_SCOPE_COMPLETED"
    assert "not a safety assurance" in r["recommendation"]

@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"analysis_path": "deep"}, {"analysis_path": None},
    {"tool_status": {"slither": {"ran": True}, "aderyn": {"ran": False}}},
    {"quick_screen_hits": {"slither": []}}, {"ml_result": {}},
])
async def test_incomplete_screen_cannot_clear(change):
    r = await report({**clean_state(), **change})
    assert r["overall_verdict"] == "INCONCLUSIVE"
    assert r["assessment"]["abstained"] is True

@pytest.mark.asyncio
async def test_direct_positive_preserved_without_ml():
    e = Evidence.slither("Reentrancy", "High", "synthetic finding", 1.0,
                         detector="reentrancy-eth", lines=[1])
    r = await report({"analysis_path": "deep", "tool_status": {"slither": {"ran": True}},
                      "evidence_list": [e]})
    assert r["overall_verdict"] == "CONFIRMED"
    assert r["assessment"]["coverage"] == "INCOMPLETE"
    assert "manual review" in r["recommendation"].lower()
    assert r["verdicts"]["Reentrancy"] == "CONFIRMED"
    assert r["vulnerability_verdicts"][0]["ml_probability_available"] is False
    assert r["vulnerability_verdicts"][0]["severity"] == "UNKNOWN"

@pytest.mark.asyncio
async def test_rag_only_not_target_evidence():
    e = Evidence.rag("Reentrancy", 1.0, 1.0, chunk_id="historical")
    r = await report({"analysis_path": "deep", "evidence_list": [e]})
    assert r["overall_verdict"] == "INCONCLUSIVE"
    assert r["verdicts"]["Reentrancy"] == "INCONCLUSIVE"
    assert r["assessment"]["raw_fusion_verdicts"]["Reentrancy"] == "CONFIRMED"

def test_legacy_metrics_reject_versioned_assessment():
    from src.eval.provenance import require_legacy_report_policy
    require_legacy_report_policy({"final_report": {"overall_verdict": "SAFE"}}, report_path="old.json")
    with pytest.raises(EvaluationInputError, match="coverage-aware"):
        require_legacy_report_policy({"final_report": {"assessment": {"policy": "evidence-adequacy-v1"}}},
                                     report_path="new.json")


@pytest.mark.asyncio
async def test_context_cannot_inflate_direct_signal_score():
    e = Evidence.slither("Reentrancy", "Low", "weak signal", 0.2, detector="reentrancy-eth")
    context = Evidence.rag("Reentrancy", 1.0, 1.0, chunk_id="historical")
    r = await report({"analysis_path": "deep", "tool_status": {"slither": {"ran": True}},
                      "evidence_list": [e, context]})
    assert r["verdicts"]["Reentrancy"] == "WATCH"
    assert r["assessment"]["raw_fusion_verdicts"]["Reentrancy"] == "CONFIRMED"
    assert r["vulnerability_verdicts"][0]["confidence"] == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_truncated_and_partial_vectors_abstain():
    for payload_delta in ({"truncated": True}, {"probabilities": {"Reentrancy": 0.01}}):
        payload = {k: v for k, v in live_ml().items() if k != "execution_status"}
        payload.update(payload_delta)
        ml = bind_status(payload, dependency="module1-inference",
                         input_payload={"source_code": SOURCE}, duration_ms=1)
        r = await report({**clean_state(), "ml_result": ml})
        assert r["overall_verdict"] == "INCONCLUSIVE"
