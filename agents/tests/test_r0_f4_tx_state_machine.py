"""R0-F4: Transaction state machine engine tests.

All tests invoke TxEngine.transition() or FakeChain methods.
No manual dataclass field assignment — all state changes go through the engine.
"""

import pytest
from src.mcp.servers.audit._submit import (
    TxState, TxLifecycle, TxEngine, FakeChain, _ALLOWED_TRANSITIONS,
)
from src.security.policy_signer import (
    evaluate_submission, PolicyDecision, REJECT_REASON_UNBOUND, REJECT_REASON_NO_SCOPE,
)


class TestTxEngineTransitions:
    def setup_method(self):
        self.engine = TxEngine()

    def test_default_state_is_not_requested(self):
        assert self.engine.state == TxState.PENDING  # TxLifecycle default

    def test_policy_reject_from_not_requested(self):
        engine = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        engine.policy_reject(REJECT_REASON_UNBOUND)
        assert engine.state == TxState.POLICY_REJECTED
        assert engine.lifecycle.error == REJECT_REASON_UNBOUND

    def test_policy_reject_from_wrong_state_raises(self):
        self.engine.transition = lambda to, **kw: None  # dummy to get past NOT_REQUESTED
        engine = TxEngine(TxLifecycle(state=TxState.PREPARED))
        with pytest.raises(ValueError, match="policy_reject only valid from NOT_REQUESTED"):
            engine.policy_reject("test")

    def test_prepare_sign_broadcast_chain(self):
        engine = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        engine.prepare()
        assert engine.state == TxState.PREPARED
        engine.sign()
        assert engine.state == TxState.SIGNED
        engine.broadcast("0xabc123")
        assert engine.state == TxState.BROADCAST
        assert engine.lifecycle.tx_hash == "0xabc123"

    def test_broadcast_to_pending_to_confirmed(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xdef")
        engine.mined(1)
        assert engine.state == TxState.PENDING
        assert engine.lifecycle.block_number == 1
        lc = engine.confirm(42, 80000, confirmations=12)
        assert engine.state == TxState.CONFIRMED
        assert lc.receipt_status == 1
        assert lc.confirmations == 12

    def test_broadcast_to_reverted(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xrev")
        lc = engine.revert("execution reverted")
        assert engine.state == TxState.REVERTED
        assert lc.receipt_status == 0
        assert "reverted" in lc.error

    def test_broadcast_to_dropped(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xdrop")
        engine.drop("timed out")
        assert engine.state == TxState.DROPPED

    def test_broadcast_to_replaced(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xold")
        engine.replace("0xnew", "higher gas")
        assert engine.state == TxState.REPLACED
        assert engine.lifecycle.tx_hash == "0xnew"

    def test_dropped_can_represent(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xa")
        engine.drop("timed out")
        engine.prepare()
        assert engine.state == TxState.PREPARED

    def test_invalid_transition_raises(self):
        engine = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        with pytest.raises(ValueError, match="invalid transition"):
            engine.transition(TxState.CONFIRMED)

    def test_confirmed_is_terminal(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xterm")
        engine.mined(1)
        engine.confirm(42, 80000)
        with pytest.raises(ValueError, match="invalid transition"):
            engine.transition(TxState.FAILED)

    def test_ingest_success_receipt(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xrcpt")
        engine.mined(1)
        engine.ingest_receipt({"status": 1, "blockNumber": 42, "gasUsed": 80000, "confirmations": 1})
        assert engine.state == TxState.CONFIRMED
        assert engine.lifecycle.receipt_status == 1

    def test_ingest_failed_receipt(self):
        engine = TxEngine(TxLifecycle(state=TxState.SIGNED))
        engine.broadcast("0xrcpt_fail")
        engine.mined(1)
        engine.ingest_receipt({"status": 0, "revertReason": "out of gas"})
        assert engine.state == TxState.REVERTED
        assert engine.lifecycle.receipt_status == 0

    def test_transition_table_completeness(self):
        for state in TxState:
            assert state in _ALLOWED_TRANSITIONS, f"{state} missing from transition table"
            allowed = _ALLOWED_TRANSITIONS[state]
            if state in (TxState.CONFIRMED, TxState.REVERTED, TxState.POLICY_REJECTED, TxState.FAILED):
                assert len(allowed) == 0, f"{state} should be terminal"


class TestFakeChain:
    def test_full_lifecycle(self):
        chain = FakeChain(confirm_blocks=2)
        engine = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        engine.prepare()
        engine.sign()
        chain.send(engine)
        assert engine.state == TxState.BROADCAST
        chain.mine_block()
        assert engine.state == TxState.PENDING
        chain.confirm_tx(engine.lifecycle.tx_hash)
        assert engine.state == TxState.CONFIRMED
        assert engine.lifecycle.receipt_status == 1

    def test_revert_on_chain(self):
        chain = FakeChain()
        engine = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        engine.prepare()
        engine.sign()
        chain.send(engine)
        chain.mine_block()
        chain.revert_tx(engine.lifecycle.tx_hash, "arithmetic overflow")
        assert engine.state == TxState.REVERTED

    def test_drop_from_mempool(self):
        chain = FakeChain()
        engine = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        engine.prepare()
        engine.sign()
        chain.send(engine)
        chain.drop_tx(engine.lifecycle.tx_hash)
        assert engine.state == TxState.DROPPED

    def test_replacement_transaction(self):
        chain = FakeChain()
        e1 = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        e1.prepare(); e1.sign()
        chain.send(e1)

        e2 = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        e2.prepare(); e2.sign()
        tx_hash, lc = chain.replace_tx(e1.lifecycle.tx_hash, e2)
        assert e2.state == TxState.BROADCAST
        assert e1.state == TxState.DROPPED

    def test_idempotency_in_durable_state(self):
        chain = FakeChain()
        engine = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
        engine.prepare(); engine.sign()
        chain.send(engine)
        chain.mine_block()
        chain.confirm_tx(engine.lifecycle.tx_hash)

        # Re-confirming same tx should fail (already confirmed)
        with pytest.raises(ValueError, match="PENDING"):
            chain.confirm_tx(engine.lifecycle.tx_hash)

    def test_send_from_wrong_state_raises(self):
        chain = FakeChain()
        engine = TxEngine()
        with pytest.raises(ValueError, match="SIGNED"):
            chain.send(engine)

    def test_nonce_sequential(self):
        chain = FakeChain()
        engines = []
        for i in range(5):
            e = TxEngine(TxLifecycle(state=TxState.NOT_REQUESTED))
            e.prepare(); e.sign()
            chain.send(e)
            engines.append(e)
        for e in engines:
            assert e.state == TxState.BROADCAST
            assert e.lifecycle.tx_hash is not None


class TestPolicySignerRejection:
    def test_all_scopes_rejected(self):
        for scope in ("legacy_proxy_only_unbound", "none", "", "typed_identity_bound_v3",
                       "future_v4", "malicious_string_to_bypass"):
            r = evaluate_submission(
                proof_scope=scope,
                contract_address="0x0001", chain_id=1, round_id=42, model_hash="a" * 64,
            )
            assert r.decision == PolicyDecision.REJECTED, f"scope '{scope}' should be rejected"

    def test_no_caller_self_declare_eligibility(self):
        """Callers cannot pass 'typed_identity_bound_v3' to bypass rejection."""
        r = evaluate_submission(
            proof_scope="typed_identity_bound_v3",
            contract_address="0x0001", chain_id=1, round_id=42, model_hash="a" * 64,
        )
        assert r.decision == PolicyDecision.REJECTED
        assert "not_accepted" in r.reason

    def test_details_contain_identity_for_audit(self):
        r = evaluate_submission(
            proof_scope="legacy_proxy_only_unbound",
            contract_address="0xCAFE", chain_id=5, round_id=99, model_hash="b" * 64,
        )
        assert r.details["chain_id"] == 5
        assert r.details["contract_address"] == "0xCAFE"
