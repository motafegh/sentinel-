"""V3 policy-signer boundary tests.

These tests do not sign transactions or touch a private key. They verify the
unsigned EIP-712 request that may be handed to the isolated signer service.
"""

from dataclasses import replace

import pytest

from agents.src.security.policy_signer import (
    LEGACY_PROOF_SCOPE,
    PolicyDecision,
    REJECT_REASON_INVALID_V3,
    REJECT_REASON_UNBOUND,
    V3_SUBMISSION_PROTOCOL,
    build_v3_request,
    compute_v3_digest,
    evaluate_submission,
    evaluate_v3_request,
)

AGENT = "0x1111111111111111111111111111111111111111"
TARGET = "0x2222222222222222222222222222222222222222"
REGISTRY = "0x3333333333333333333333333333333333333333"
CHAIN_ID = 31337
ROUND_ID = 77
DEADLINE = 2_000_000_000

CODE_HASH = "0x" + "44" * 32
TEACHER_HASH = "0x" + "55" * 32
BUNDLE_HASH = "0x" + "66" * 32
DATA_HASH = "0x" + "77" * 32
SCHEMA_HASH = "0x" + "88" * 32
PROOF = bytes.fromhex("deadbeef")
SCORES = [1200 + i * 41 for i in range(10)]
SIGNALS = [i + 17 for i in range(128)] + SCORES


def _request():
    return build_v3_request(
        agent=AGENT,
        contract_address=TARGET,
        contract_code_hash=CODE_HASH,
        chain_id=CHAIN_ID,
        registry_address=REGISTRY,
        round_id=ROUND_ID,
        teacher_model_hash=TEACHER_HASH,
        proxy_bundle_hash=BUNDLE_HASH,
        data_version_hash=DATA_HASH,
        class_schema_hash=SCHEMA_HASH,
        proof_bytes=PROOF,
        public_signals=SIGNALS,
        class_score_felts=SCORES,
        deadline=DEADLINE,
    )


def test_valid_v3_request_is_eligible_for_isolated_signer():
    request = _request()
    result = evaluate_v3_request(request, now_timestamp=DEADLINE - 1)
    assert result.decision is PolicyDecision.ACCEPTED
    assert result.reason is None
    assert request.proof_scope == LEGACY_PROOF_SCOPE
    assert request.submission_protocol == V3_SUBMISSION_PROTOCOL
    assert request.digest.startswith("0x") and len(request.digest) == 66


def test_digest_is_deterministic_and_matches_recomputation():
    request = _request()
    recomputed = compute_v3_digest(
        agent=request.agent,
        contract_address=request.contract_address,
        contract_code_hash=request.contract_code_hash,
        chain_id=request.chain_id,
        registry_address=request.registry_address,
        round_id=request.round_id,
        teacher_model_hash=request.teacher_model_hash,
        proxy_bundle_hash=request.proxy_bundle_hash,
        data_version_hash=request.data_version_hash,
        class_schema_hash=request.class_schema_hash,
        proof_hash=request.proof_hash,
        public_signals_hash=request.public_signals_hash,
        class_score_felts_hash=request.class_score_felts_hash,
        deadline=request.deadline,
    )
    assert recomputed == request.digest
    assert _request().digest == request.digest


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("contract_address", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("registry_address", "0xcccccccccccccccccccccccccccccccccccccccc"),
        ("chain_id", 1),
        ("round_id", ROUND_ID + 1),
        ("teacher_model_hash", "0x" + "99" * 32),
        ("proxy_bundle_hash", "0x" + "aa" * 32),
        ("data_version_hash", "0x" + "bb" * 32),
        ("class_schema_hash", "0x" + "cc" * 32),
        ("proof_hash", "0x" + "dd" * 32),
        ("public_signals_hash", "0x" + "ee" * 32),
        ("class_score_felts_hash", "0x" + "ff" * 32),
        ("deadline", DEADLINE + 1),
    ],
)
def test_every_bound_field_changes_digest(field, value):
    request = _request()
    kwargs = request.to_dict()
    kwargs[field] = value
    kwargs.pop("digest")
    kwargs.pop("proof_scope")
    kwargs.pop("submission_protocol")
    changed = compute_v3_digest(**kwargs)
    assert changed != request.digest


def test_public_signal_output_mismatch_fails_closed():
    bad_scores = list(SCORES)
    bad_scores[3] += 1
    with pytest.raises(ValueError, match="do not match public proof outputs"):
        build_v3_request(
            agent=AGENT,
            contract_address=TARGET,
            contract_code_hash=CODE_HASH,
            chain_id=CHAIN_ID,
            registry_address=REGISTRY,
            round_id=ROUND_ID,
            teacher_model_hash=TEACHER_HASH,
            proxy_bundle_hash=BUNDLE_HASH,
            data_version_hash=DATA_HASH,
            class_schema_hash=SCHEMA_HASH,
            proof_bytes=PROOF,
            public_signals=SIGNALS,
            class_score_felts=bad_scores,
            deadline=DEADLINE,
        )


def test_wrong_signal_count_fails_closed():
    with pytest.raises(ValueError, match="exactly 138"):
        build_v3_request(
            agent=AGENT,
            contract_address=TARGET,
            contract_code_hash=CODE_HASH,
            chain_id=CHAIN_ID,
            registry_address=REGISTRY,
            round_id=ROUND_ID,
            teacher_model_hash=TEACHER_HASH,
            proxy_bundle_hash=BUNDLE_HASH,
            data_version_hash=DATA_HASH,
            class_schema_hash=SCHEMA_HASH,
            proof_bytes=PROOF,
            public_signals=SIGNALS[:-1],
            class_score_felts=SCORES,
            deadline=DEADLINE,
        )


def test_tampered_digest_is_rejected_before_signing():
    request = replace(_request(), digest="0x" + "00" * 32)
    result = evaluate_v3_request(request, now_timestamp=DEADLINE - 1)
    assert result.decision is PolicyDecision.REJECTED
    assert result.reason == REJECT_REASON_INVALID_V3
    assert result.details["error"] == "request_digest_mismatch"


def test_expired_request_is_rejected_before_signing():
    request = _request()
    result = evaluate_v3_request(request, now_timestamp=DEADLINE + 1)
    assert result.decision is PolicyDecision.REJECTED
    assert result.reason == REJECT_REASON_INVALID_V3
    assert result.details["error"] == "request_expired"


def test_legacy_proof_only_path_remains_rejected():
    result = evaluate_submission(
        proof_scope=LEGACY_PROOF_SCOPE,
        contract_address=TARGET,
        chain_id=CHAIN_ID,
        round_id=ROUND_ID,
        model_hash=TEACHER_HASH,
    )
    assert result.decision is PolicyDecision.REJECTED
    assert result.reason == REJECT_REASON_UNBOUND
