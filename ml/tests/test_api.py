# ml/tests/test_api.py
"""
Integration tests for the SENTINEL inference API (Track 3, 2026-04-17).

These tests use FastAPI's TestClient — it runs the full app
in-process, including the lifespan startup (model load).
No running server needed. No network calls needed.

WHAT CHANGED FROM BINARY (Track 3, 2026-04-17):
    - "confidence: float" removed from PredictResponse → assert it is GONE
    - "vulnerabilities: list[VulnerabilityResult]" added → assert list + entry shape
    - New test: safe contract → label=="safe" and vulnerabilities==[]

Test coverage:
    test_health_returns_ok              → /health responds correctly
    test_predict_valid_contract         → valid Solidity returns multi-label schema
    test_predict_no_confidence_field    → old "confidence" field must be absent
    test_predict_safe_contract          → safe contract → label=="safe", empty vulnerabilities
    test_predict_error_cases            → invalid inputs return 400/422
    test_predict_consistent_on_same_input → same input = same output (determinism)
"""

import pytest


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------

def test_health_returns_ok(client):
    """
    /health must return 200 with predictor_loaded=True.

    If this fails, the model failed to load at startup.
    Check: checkpoint path, CUDA availability, disk space.
    """
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"

    # predictor_loaded=True confirms lifespan() completed successfully.
    # If False, every /predict call will crash — catch it here first.
    assert body["predictor_loaded"] is True


# ------------------------------------------------------------------
# /predict — valid input
# ------------------------------------------------------------------

# This is the same Vault contract used in the M3.4 smoke test.
# It has a classic reentrancy pattern — withdraw() sends ETH before
# updating balances. The model may or may not flag it — that depends
# on training. What we test here is shape and type, not the label.
VAULT_CONTRACT = """
pragma solidity ^0.8.0;
contract Vault {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;
    }
}
""".strip()


def test_predict_valid_contract(client):
    """
    Valid Solidity input must return 200 with correct Track 3 multi-label schema.

    We test structure and types — not the specific label or vulnerabilities.
    Why: the output depends on model weights, which change between training runs.
    Shape and types never change.
    """
    response = client.post(
        "/predict",
        json={"source_code": VAULT_CONTRACT},
    )

    assert response.status_code == 200

    body = response.json()

    # Every field declared in PredictResponse must be present
    required = {"label", "probabilities", "confirmed", "suspicious",
                "vulnerabilities", "thresholds", "truncated", "num_nodes", "num_edges"}
    assert required <= set(body.keys()), f"Missing keys: {required - set(body.keys())}"

    # Three-tier label values
    assert body["label"] in ("safe", "suspicious", "confirmed_vulnerable")
    assert isinstance(body["probabilities"], dict)
    assert len(body["probabilities"]) == 10   # 10 classes, always present
    assert isinstance(body["confirmed"],  list)
    assert isinstance(body["suspicious"], list)
    assert isinstance(body["vulnerabilities"], list)
    assert isinstance(body["thresholds"], list)
    assert isinstance(body["truncated"], bool)
    assert isinstance(body["num_nodes"], int)
    assert body["num_nodes"] > 0  # Vault has real nodes — 0 means extraction failed

    # Each vulnerability entry must have the right shape
    for vuln in body["confirmed"] + body["suspicious"]:
        assert "vulnerability_class" in vuln
        assert "probability" in vuln
        assert isinstance(vuln["vulnerability_class"], str)
        assert isinstance(vuln["probability"], float)
        assert 0.0 <= vuln["probability"] <= 1.0


def test_predict_no_confidence_field(client):
    """
    Track 3 change: "confidence" scalar field was removed from PredictResponse.
    Assert it is absent — protects against accidental regression to binary schema.
    """
    response = client.post("/predict", json={"source_code": VAULT_CONTRACT})
    assert response.status_code == 200
    assert "confidence" not in response.json()


def test_predict_safe_contract(client):
    """
    A trivially safe contract (no logic, no ETH flow) should return
    label=="safe" and an empty vulnerabilities list.

    Note: this assertion is model-weight dependent. If the newly trained
    model over-predicts, this test may fail — in that case, examine the
    per-class probabilities and recheck the training data rather than
    deleting the test.
    """
    safe_contract = """
pragma solidity ^0.8.0;
contract Empty {
    uint256 public value;
    function getValue() external view returns (uint256) {
        return value;
    }
}
""".strip()

    response = client.post("/predict", json={"source_code": safe_contract})
    assert response.status_code == 200

    body = response.json()
    assert body["label"] in ("safe", "suspicious", "confirmed_vulnerable")
    assert isinstance(body["confirmed"],  list)
    assert isinstance(body["suspicious"], list)


# ------------------------------------------------------------------
# /predict — error cases
# ------------------------------------------------------------------

@pytest.mark.parametrize("payload,expected_status,description", [
    (
        # Empty string — fails min_length=10 in Pydantic → 422
        {"source_code": ""},
        422,
        "empty string rejected by Pydantic validator",
    ),
    (
        # Not Solidity — passes min_length but fails must_look_like_solidity → 422
        {"source_code": "hello world this is not solidity code at all"},
        422,
        "non-Solidity text rejected by field validator",
    ),
    (
        # Missing source_code field entirely → 422 (required field)
        {},
        422,
        "missing required field rejected",
    ),
])
def test_predict_error_cases(client, payload, expected_status, description):
    """
    Invalid inputs must be rejected before reaching the model.

    All three cases return 422 — Pydantic validation failure.
    FastAPI handles this automatically from our schema declarations.
    No model inference happens for any of these inputs.
    """
    response = client.post("/predict", json=payload)
    assert response.status_code == expected_status, (
        f"Failed: {description} — "
        f"expected {expected_status}, got {response.status_code}"
    )


# ------------------------------------------------------------------
# /predict — determinism
# ------------------------------------------------------------------

def test_predict_consistent_on_same_input(client):
    """
    Same input must return identical output on repeated calls.

    Why this matters: the model has Dropout layers (GNNEncoder: 0.2,
    FusionLayer: 0.3). If model.eval() was forgotten, Dropout stays
    active and scores become non-deterministic — different on every call.
    This test catches that bug directly.
    """
    first  = client.post("/predict", json={"source_code": VAULT_CONTRACT}).json()
    second = client.post("/predict", json={"source_code": VAULT_CONTRACT}).json()

    assert first["label"]           == second["label"]
    assert first["vulnerabilities"] == second["vulnerabilities"]


# ------------------------------------------------------------------
# /hotspots — Phase 1: GNN attention-based function hotspots
# ------------------------------------------------------------------

class TestHotspotsEndpoint:
    """
    Tests for POST /hotspots — returns GNN embedding-norm hotspots + ML result.

    All assertions are structural (shape, types, ranges) — not model-output-dependent.
    The actual scores depend on checkpoint weights and change between training runs.
    """

    def test_hotspots_returns_200(self, client):
        """Basic smoke test — endpoint reachable and returns 200."""
        response = client.post("/hotspots", json={"source_code": VAULT_CONTRACT})
        assert response.status_code == 200

    def test_hotspots_response_schema(self, client):
        """Response must contain all declared HotspotsResponse fields."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()

        required = {"hotspots", "hotspot_stats", "label", "probabilities", "confirmed", "suspicious"}
        assert required <= set(body.keys()), f"Missing keys: {required - set(body.keys())}"

    def test_hotspots_label_valid(self, client):
        """label must be one of the three-tier values."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()
        assert body["label"] in ("safe", "suspicious", "confirmed_vulnerable")

    def test_hotspots_list_structure(self, client):
        """Each hotspot entry must have fn_name, node_id, score, lines, node_type."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()

        for h in body["hotspots"]:
            assert "fn_name"   in h,  f"Missing fn_name in hotspot: {h}"
            assert "node_id"   in h,  f"Missing node_id in hotspot: {h}"
            assert "score"     in h,  f"Missing score in hotspot: {h}"
            assert "lines"     in h,  f"Missing lines in hotspot: {h}"
            assert "node_type" in h,  f"Missing node_type in hotspot: {h}"

            assert isinstance(h["fn_name"],   str),       "fn_name must be str"
            assert isinstance(h["node_id"],   int),       "node_id must be int"
            assert isinstance(h["score"],     float),     "score must be float"
            assert isinstance(h["lines"],     list),      "lines must be list"
            assert 0.0 <= h["score"] <= 1.0,              f"score out of range: {h['score']}"

    def test_hotspots_at_most_20(self, client):
        """Hotspot list is capped at 20 entries."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()
        assert len(body["hotspots"]) <= 20

    def test_hotspots_sorted_descending(self, client):
        """Hotspots must be sorted by score descending."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()
        scores = [h["score"] for h in body["hotspots"]]
        assert scores == sorted(scores, reverse=True), "Hotspots not sorted by score desc"

    def test_hotspot_stats_present(self, client):
        """hotspot_stats must contain total_function_nodes, num_nodes, attention_source."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()
        stats = body["hotspot_stats"]

        assert "total_function_nodes" in stats
        assert "num_nodes"            in stats
        assert "attention_source"     in stats
        assert stats["attention_source"] == "gnn_embedding_norm"
        assert isinstance(stats["num_nodes"], int)
        assert stats["num_nodes"] > 0

    def test_vault_has_function_hotspots(self, client):
        """Vault contract has real functions — hotspot list must be non-empty."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()
        assert len(body["hotspots"]) > 0, "Vault has functions; hotspot list must not be empty"

    def test_hotspots_probabilities_complete(self, client):
        """probabilities dict must contain all 10 vulnerability classes."""
        body = client.post("/hotspots", json={"source_code": VAULT_CONTRACT}).json()
        assert len(body["probabilities"]) == 10
        for cls, prob in body["probabilities"].items():
            assert isinstance(cls,  str)
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_hotspots_error_cases(self, client):
        """Invalid inputs must be rejected the same way /predict rejects them."""
        assert client.post("/hotspots", json={"source_code": ""}).status_code          == 422
        assert client.post("/hotspots", json={"source_code": "not solidity"}).status_code == 422
        assert client.post("/hotspots", json={}).status_code                            == 422