from __future__ import annotations

import pytest

from src.contracts.submission_v3 import normalize_v3_submission

ADDR = "0x" + "11" * 20
AGENT = "0x" + "22" * 20
SIGNER = "0x" + "33" * 20
VERIFIER = "0x" + "44" * 20
H = "0x" + "aa" * 32


def test_default_v3_truth_never_claims_finality() -> None:
    value = normalize_v3_submission()
    assert value["schema_version"] == "3"
    assert value["submission_protocol"] == "context_attested_v3"
    assert value["proof_scope"] == "legacy_proxy_only_unbound"
    assert value["verified_audit_finality"] is False
    assert value["proof_state"] == "not_requested"
    assert value["context_state"] == "not_requested"
    assert value["signer_state"] == "not_requested"
    assert value["transaction_state"] == "not_requested"
    assert value["finality_state"] == "not_requested"


def test_context_protocol_does_not_upgrade_neural_proof_scope() -> None:
    with pytest.raises(ValueError, match="does not upgrade the neural proof scope"):
        normalize_v3_submission({"proof_scope": "typed_identity_bound_v3"})


def test_positive_finality_requires_all_independent_states() -> None:
    with pytest.raises(ValueError, match="independent states"):
        normalize_v3_submission(
            {
                "verified_audit_finality": True,
                "proof_state": "verified",
                "context_state": "attested",
                "signer_state": "signed",
                "transaction_state": "pending",
                "finality_state": "confirmed",
            }
        )


def test_confirmed_v3_truth_requires_exact_identity() -> None:
    value = normalize_v3_submission(
        {
            "verified_audit_finality": True,
            "proof_state": "verified",
            "context_state": "attested",
            "signer_state": "signed",
            "transaction_state": "confirmed",
            "finality_state": "confirmed",
            "request_digest": H,
            "chain_id": 11155111,
            "registry_address": ADDR,
            "contract_address": ADDR,
            "agent": AGENT,
            "policy_signer": SIGNER,
            "verifier": VERIFIER,
            "round_id": 8,
            "teacher_model_hash": H,
            "proxy_bundle_hash": H,
            "data_version_hash": H,
            "class_schema_hash": H,
            "proof_hash": H,
            "public_signals_hash": H,
        }
    )
    assert value["verified_audit_finality"] is True
    assert value["chain_id"] == 11155111
    assert value["request_digest"] == H
    assert value["transaction_state"] == "confirmed"


def test_confirmed_finality_without_request_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires request_digest"):
        normalize_v3_submission(
            {
                "verified_audit_finality": True,
                "proof_state": "verified",
                "context_state": "attested",
                "signer_state": "signed",
                "transaction_state": "confirmed",
                "finality_state": "confirmed",
            }
        )


def test_invalid_state_hash_address_and_chain_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid proof_state"):
        normalize_v3_submission({"proof_state": "probably_verified"})
    with pytest.raises(ValueError, match="32-byte hex"):
        normalize_v3_submission({"request_digest": "0x1234"})
    with pytest.raises(ValueError, match="20-byte"):
        normalize_v3_submission({"registry_address": "0x1234"})
    with pytest.raises(ValueError, match="chain_id must be positive"):
        normalize_v3_submission({"chain_id": 0})
