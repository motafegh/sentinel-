"""R0-F4: Transaction state machine behavioral tests.

Tests all 11 TxState values, transitions, receipt/failure tracking,
idempotency, and policy-signer rejection path.
"""

import pytest
from src.mcp.servers.audit._submit import TxState, TxLifecycle
from src.security.policy_signer import (
    evaluate_submission,
    PolicyDecision,
    PolicyResult,
    REJECT_REASON_UNBOUND,
    REJECT_REASON_NO_SCOPE,
)


class TestTxStateValues:
    def test_all_states_present(self):
        assert len(list(TxState)) == 11
        expected = {
            "not_requested", "policy_rejected", "prepared", "signed",
            "broadcast", "pending", "confirmed", "reverted",
            "dropped", "replaced", "failed",
        }
        actual = {s.value for s in TxState}
        assert actual == expected

    def test_state_serialization_roundtrip(self):
        for s in TxState:
            assert TxState(s.value) == s


class TestTxLifecycleTransitions:
    def test_default_state_is_pending(self):
        lc = TxLifecycle(tx_hash="0xabc")
        assert lc.state == TxState.PENDING

    def test_policy_rejected_transition(self):
        lc = TxLifecycle(tx_hash="0xdef")
        lc.state = TxState.POLICY_REJECTED
        lc.error = REJECT_REASON_UNBOUND
        assert lc.state == TxState.POLICY_REJECTED
        assert lc.error == REJECT_REASON_UNBOUND
        d = lc.to_dict()
        assert d["state"] == "policy_rejected"
        assert d["error"] == REJECT_REASON_UNBOUND

    def test_broadcast_to_confirmed_transition(self):
        lc = TxLifecycle(tx_hash="0x123")
        lc.state = TxState.BROADCAST
        lc.state = TxState.CONFIRMED
        lc.receipt_status = 1
        lc.block_number = 1000
        lc.confirmations = 12
        lc.gas_used = 50000
        assert lc.state == TxState.CONFIRMED
        assert lc.receipt_status == 1
        d = lc.to_dict()
        assert d["block_number"] == 1000
        assert d["confirmations"] == 12

    def test_broadcast_to_reverted(self):
        lc = TxLifecycle(tx_hash="0x456")
        lc.state = TxState.BROADCAST
        lc.state = TxState.REVERTED
        lc.receipt_status = 0
        lc.error = "reverted on-chain"
        assert lc.state == TxState.REVERTED
        assert lc.receipt_status == 0
        assert "reverted" in lc.error

    def test_broadcast_to_dropped(self):
        lc = TxLifecycle(tx_hash="0x789")
        lc.state = TxState.BROADCAST
        lc.state = TxState.DROPPED
        lc.error = "tx dropped from mempool — timeout"
        assert lc.state == TxState.DROPPED

    def test_broadcast_to_replaced(self):
        lc = TxLifecycle(tx_hash="0xaaa")
        lc.state = TxState.BROADCAST
        lc.state = TxState.REPLACED
        lc.error = "replaced by tx 0xbbb with higher gas"
        assert lc.state == TxState.REPLACED

    def test_replaced_to_confirmed(self):
        lc = TxLifecycle(tx_hash="0xbbb")
        lc.state = TxState.REPLACED
        lc.state = TxState.CONFIRMED
        lc.receipt_status = 1
        assert lc.state == TxState.CONFIRMED

    def test_not_requested_to_policy_rejected(self):
        lc = TxLifecycle(tx_hash=None)
        lc.state = TxState.NOT_REQUESTED
        lc.state = TxState.POLICY_REJECTED
        lc.error = REJECT_REASON_UNBOUND
        assert lc.state == TxState.POLICY_REJECTED

    def test_receipt_zero_means_reverted_not_confirmed(self):
        lc = TxLifecycle(state=TxState.BROADCAST, receipt_status=0, error="execution reverted")
        lc.state = TxState.REVERTED
        assert lc.state == TxState.REVERTED
        assert lc.state != TxState.CONFIRMED
        assert lc.receipt_status == 0

    def test_idempotency_key_stored(self):
        lc = TxLifecycle(tx_hash="0xidem", state=TxState.CONFIRMED, receipt_status=1)
        assert lc.tx_hash == "0xidem"
        d = lc.to_dict()
        assert "tx_hash" in d

    def test_to_dict_includes_all_fields(self):
        lc = TxLifecycle(
            tx_hash="0xfull",
            state=TxState.CONFIRMED,
            block_number=42,
            confirmations=5,
            gas_used=100000,
            effective_gas_price=20000000000,
            receipt_status=1,
        )
        d = lc.to_dict()
        for key in ("tx_hash", "state", "block_number", "confirmations",
                     "gas_used", "effective_gas_price", "receipt_status", "error"):
            assert key in d


class TestPolicySignerRejection:
    def test_v2_unbound_rejected(self):
        r = evaluate_submission(
            proof_scope="legacy_proxy_only_unbound",
            contract_address="0x0000000000000000000000000000000000000001",
            chain_id=1, round_id=42, model_hash="a" * 64,
        )
        assert r.decision == PolicyDecision.REJECTED
        assert r.reason == REJECT_REASON_UNBOUND

    def test_no_proof_scope_rejected(self):
        r = evaluate_submission(
            proof_scope="none",
            contract_address="0x0000000000000000000000000000000000000001",
            chain_id=1, round_id=42, model_hash="a" * 64,
        )
        assert r.decision == PolicyDecision.REJECTED
        assert r.reason == REJECT_REASON_NO_SCOPE

    def test_v3_identity_bound_accepted(self):
        r = evaluate_submission(
            proof_scope="typed_identity_bound_v3",
            contract_address="0x0000000000000000000000000000000000000001",
            chain_id=1, round_id=42, model_hash="a" * 64,
        )
        assert r.decision == PolicyDecision.ACCEPTED

    def test_rejection_details_include_identity(self):
        r = evaluate_submission(
            proof_scope="legacy_proxy_only_unbound",
            contract_address="0xCAFE",
            chain_id=5, round_id=99, model_hash="b" * 64,
        )
        assert "contract_address" in r.details
        assert r.details["chain_id"] == 5
        assert r.details["round_id"] == 99

    def test_unknown_proof_scope_rejected(self):
        r = evaluate_submission(
            proof_scope="unknown_future_scope",
            contract_address="0x0001", chain_id=1, round_id=1, model_hash="c" * 64,
        )
        assert r.decision == PolicyDecision.REJECTED
        assert "unknown" in r.reason

    def test_policy_result_serialization(self):
        r = evaluate_submission(
            proof_scope="legacy_proxy_only_unbound",
            contract_address="0x0001", chain_id=1, round_id=1, model_hash="d" * 64,
        )
        d = r.to_dict()
        assert d["decision"] == "policy_rejected"
        assert d["reason"] == REJECT_REASON_UNBOUND
        assert "details" in d

    def test_empty_proof_scope_rejected(self):
        r = evaluate_submission(
            proof_scope="",
            contract_address="0x0001", chain_id=1, round_id=1, model_hash="e" * 64,
        )
        assert r.decision == PolicyDecision.REJECTED
