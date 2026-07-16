from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ml.src.inference.api import PredictRequest, fusion_embedding
from ml.src.inference.execution_status import bind_live_result, digest_payload


def test_live_wire_status_binds_exact_input_and_output() -> None:
    payload = {"fusion_embedding": [0.0] * 128, "model_hash": "a" * 64}
    input_payload = {"source_code": "contract C {}"}

    result = bind_live_result(
        payload,
        dependency="module1-fusion",
        input_payload=input_payload,
        duration_ms=2,
    )

    status = result["execution_status"]
    assert status["schema_version"] == "1"
    assert status["status"] == "SUCCEEDED"
    assert status["input_digest"] == digest_payload(input_payload)
    assert status["output_digest"] == digest_payload(payload)


def test_fusion_endpoint_emits_bound_live_status() -> None:
    class _Predictor:
        def predict_fusion_embedding(self, source_code: str) -> dict:
            return {
                "fusion_embedding": [0.0] * 128,
                "num_nodes": 1,
                "num_edges": 0,
                "model_hash": "a" * 64,
                "windows_used": 1,
            }

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(predictor=_Predictor())))
    source_code = "contract C {}"
    response = asyncio.run(fusion_embedding(request, PredictRequest(source_code=source_code)))
    payload = response.model_dump()
    status = payload["execution_status"]

    assert status["status"] == "SUCCEEDED"
    assert status["input_digest"] == digest_payload({"source_code": source_code})
    material = {key: value for key, value in payload.items() if key != "execution_status"}
    assert status["output_digest"] == digest_payload(material)
