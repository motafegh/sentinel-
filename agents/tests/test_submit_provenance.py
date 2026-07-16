from __future__ import annotations

from typing import Any

import pytest
from src.contracts.execution import ExecutionState, bind_status, failure_status, mock_status
from src.mcp.servers.audit._submit import _run_submit

SOURCE = "contract C {}"
ADDRESS = "0x" + "1" * 40
MODEL_HASH = "a" * 64


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _submit(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> dict[str, Any]:
    import requests

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Response(payload))
    return _run_submit(SOURCE, ADDRESS, MODEL_HASH)


@pytest.mark.parametrize("case", ["missing", "mock", "degraded", "mutated"])
def test_submission_stops_before_proof_for_ineligible_fusion(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    material = {"fusion_embedding": [0.0] * 128, "model_hash": MODEL_HASH}
    if case == "missing":
        payload = material
    elif case == "mock":
        payload = mock_status(
            material,
            dependency="module1-inference",
            input_payload={"source_code": SOURCE},
        )
    elif case == "degraded":
        payload = {
            **material,
            "execution_status": failure_status(
                ExecutionState.DEGRADED,
                dependency="module1-inference",
                reason_code="partial_model",
                detail="model output was degraded",
                attempted=True,
                input_payload={"source_code": SOURCE},
            ),
        }
    else:
        payload = bind_status(
            material,
            dependency="module1-inference",
            input_payload={"source_code": SOURCE},
            duration_ms=1,
        )
        payload["fusion_embedding"][0] = 1.0

    result = _submit(monkeypatch, payload)

    assert result["status"] == "failed"
    assert result["failed_step"] == "ml_provenance"
    assert result["tx_hash"] is None
    assert result["class_scores"] is None
    assert result["proof_hash"] is None


def test_live_eligible_fusion_passes_provenance_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    payload = bind_status(
        {"fusion_embedding": [0.0] * 128, "model_hash": MODEL_HASH},
        dependency="module1-fusion",
        input_payload={"source_code": SOURCE},
        duration_ms=1,
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after provenance")),
    )

    result = _submit(monkeypatch, payload)

    assert result["failed_step"] == "proxy_inference"
    assert "stop after provenance" in result["reason"]
    assert result["tx_hash"] is None


def test_submission_rejects_bound_but_mismatched_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bind_status(
        {"fusion_embedding": [0.0] * 128, "model_hash": "b" * 64},
        dependency="module1-fusion",
        input_payload={"source_code": SOURCE},
        duration_ms=1,
    )

    result = _submit(monkeypatch, payload)

    assert result["failed_step"] == "ml_provenance"
    assert "model hash" in result["reason"]
    assert result["tx_hash"] is None
