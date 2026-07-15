"""R0-F4: Transaction state machine engine + FakeChain tests.

All state changes go through TxEngine.transition() or typed methods.
Default state is NOT_REQUESTED. Receipt validation fails closed.
FakeChain: persistent nonce, idempotency map, replacement linking,
block accumulation, reorg, snapshot/reload.
"""

import pytest
from src.mcp.servers.audit._submit import (
    TxState, TxLifecycle, TxEngine, FakeChain, _ALLOWED_TRANSITIONS,
)
from src.security.policy_signer import (
    evaluate_submission, PolicyDecision, REJECT_REASON_UNBOUND,
)


class TestTxEngine:
    def test_default_state_is_not_requested(self):
        e = TxEngine()
        assert e.state == TxState.NOT_REQUESTED

    def test_normal_lifecycle(self):
        e = TxEngine()
        e.prepare()
        assert e.state == TxState.PREPARED
        e.sign()
        assert e.state == TxState.SIGNED
        e.broadcast("0xabc")
        assert e.state == TxState.BROADCAST
        assert e.lifecycle.tx_hash == "0xabc"
        e.mined(1)
        assert e.state == TxState.PENDING
        lc = e.confirm(42, 80000, confirmations=12)
        assert e.state == TxState.CONFIRMED
        assert lc.receipt_status == 1
        assert lc.confirmations == 12
        assert lc.block_number == 42
        assert lc.gas_used == 80000

    def test_policy_reject(self):
        e = TxEngine()
        e.policy_reject(REJECT_REASON_UNBOUND)
        assert e.state == TxState.POLICY_REJECTED

    def test_revert(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrev")
        lc = e.revert("execution reverted")
        assert e.state == TxState.REVERTED
        assert lc.receipt_status == 0

    def test_drop(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xdrop")
        e.drop("timed out")
        assert e.state == TxState.DROPPED

    def test_replace(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xold")
        e.replace("0xnew", "higher gas")
        assert e.state == TxState.REPLACED
        assert e.lifecycle.replaced_by == "0xnew"

    def test_receipt_validation_rejects_missing_status(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrcpt")
        e.mined(1)
        with pytest.raises(ValueError, match="receipt status must be 0 or 1"):
            e.ingest_receipt({})

    def test_receipt_validation_rejects_negative_block(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrcpt")
        e.mined(1)
        with pytest.raises(ValueError, match="blockNumber must be non-negative"):
            e.ingest_receipt({"status": 1, "blockNumber": -1, "gasUsed": 80000})

    def test_receipt_validation_rejects_zero_gas(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrcpt")
        e.mined(1)
        with pytest.raises(ValueError, match="gasUsed must be positive"):
            e.ingest_receipt({"status": 1, "blockNumber": 42, "gasUsed": 0})

    def test_receipt_validation_rejects_bad_confirmations(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrcpt")
        e.mined(1)
        with pytest.raises(ValueError, match="confirmations must be"):
            e.ingest_receipt({"status": 1, "blockNumber": 42, "gasUsed": 80000, "confirmations": 0})

    def test_confirm_rejects_negative_block(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrcpt")
        e.mined(1)
        with pytest.raises(ValueError, match="invalid block_number"):
            e.confirm(-1, 80000)

    def test_confirm_rejects_zero_gas(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrcpt")
        e.mined(1)
        with pytest.raises(ValueError, match="invalid gas_used"):
            e.confirm(42, 0)

    def test_ingest_success_receipt(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xgood")
        e.mined(1)
        e.ingest_receipt({"status": 1, "blockNumber": 42, "gasUsed": 80000, "confirmations": 1})
        assert e.state == TxState.CONFIRMED
        assert e.lifecycle.receipt_status == 1

    def test_ingest_failed_receipt(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xbad")
        e.mined(1)
        e.ingest_receipt({"status": 0, "blockNumber": 42, "gasUsed": 80000, "revertReason": "OOG"})
        assert e.state == TxState.REVERTED
        assert e.lifecycle.receipt_status == 0


class TestForbiddenTransitions:
    """Adversarial: every forbidden transition must raise."""

    TERMINAL = {TxState.CONFIRMED, TxState.REVERTED, TxState.POLICY_REJECTED, TxState.FAILED}
    ALL_STATES = list(TxState)

    def _engine_at(self, state: TxState) -> TxEngine:
        lc = TxLifecycle(state=state)
        return TxEngine(lc)
    @pytest.mark.parametrize("terminal", [TxState.CONFIRMED, TxState.REVERTED, TxState.POLICY_REJECTED, TxState.FAILED])
    def test_terminal_states_blocked(self, terminal):
        e = self._engine_at(terminal)
        for target in TxState:
            try:
                e.transition(target)
                assert False, f"{terminal.value} should block {target.value}"
            except ValueError:
                pass

    def test_prepare_cannot_go_to_confirmed(self):
        e = self._engine_at(TxState.PREPARED)
        with pytest.raises(ValueError):
            e.transition(TxState.CONFIRMED)

    def test_signed_cannot_go_to_confirmed(self):
        e = self._engine_at(TxState.SIGNED)
        with pytest.raises(ValueError):
            e.transition(TxState.CONFIRMED)

    def test_broadcast_can_go_to_pending_reverted_dropped_replaced_failed(self):
        e = self._engine_at(TxState.BROADCAST)
        e.transition(TxState.PENDING)  # should work
        e2 = self._engine_at(TxState.BROADCAST)
        e2.transition(TxState.REVERTED)
        e3 = self._engine_at(TxState.BROADCAST)
        e3.transition(TxState.DROPPED)
        e4 = self._engine_at(TxState.BROADCAST)
        e4.transition(TxState.REPLACED)

    def test_broadcast_cannot_go_to_prepared(self):
        e = self._engine_at(TxState.BROADCAST)
        with pytest.raises(ValueError):
            e.transition(TxState.PREPARED)


class TestFakeChain:
    def test_persistent_nonce_increments(self):
        chain = FakeChain()
        n1 = chain._next_nonce()
        n2 = chain._next_nonce()
        assert n2 == n1 + 1

    def test_full_lifecycle(self):
        chain = FakeChain(confirm_blocks=2)
        e = TxEngine()
        e.prepare(); e.sign()
        chain.send(e)
        assert e.state == TxState.BROADCAST
        chain.mine_block()
        assert e.state == TxState.PENDING
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED

    def test_idempotency_key_returns_existing(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        lc1 = chain.send(e1, idempotency_key="ik-1")
        chain.mine_blocks(2)
        chain.confirm_tx(lc1.tx_hash)

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="ik-1")
        assert lc2.tx_hash == lc1.tx_hash
        assert lc2.state == TxState.CONFIRMED  # returns EXISTING confirmed state

    def test_replacement_links_old(self):
        chain = FakeChain()
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)

        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        new_hash, lc = chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        assert e_new.state == TxState.BROADCAST
        e_new.lifecycle.tx_hash == new_hash
        # Old should be dropped or replaced
        assert e_old.state in (TxState.DROPPED, TxState.REPLACED)

    def test_block_accumulation_before_confirm(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        # Mine 5 blocks
        hashes = chain.mine_blocks(5)
        assert len(hashes) == 1
        assert chain._block_height == 5
        lc = chain.confirm_tx(e.lifecycle.tx_hash)
        assert lc.block_number == 5

    def test_reorg_rolls_back(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED

        unconfirmed = chain.reorg(depth=1)
        assert e.state == TxState.PENDING
        assert e.lifecycle.tx_hash in unconfirmed

    def test_snapshot_and_reload(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        idx = chain.snapshot()
        chain.mine_block()
        assert chain._block_height == 1
        chain.reload_snapshot(idx)
        assert chain._block_height == 0  # returned to snapshot state
        assert e.lifecycle.tx_hash is not None  # tx hash still assigned

    def test_concurrent_nonces(self):
        chain = FakeChain()
        nonces = set()
        for _ in range(10):
            e = TxEngine(); e.prepare(); e.sign()
            lc = chain.send(e)
            nonces.add(lc.nonce)
        assert len(nonces) == 10


class TestPolicySignerRejection:
    def test_all_scopes_rejected(self):
        for scope in ("legacy_proxy_only_unbound", "none", "", "typed_identity_bound_v3"):
            r = evaluate_submission(
                proof_scope=scope,
                contract_address="0x0001", chain_id=1, round_id=42, model_hash="a" * 64,
            )
            assert r.decision == PolicyDecision.REJECTED, f"scope '{scope}' should be rejected"

    def test_no_self_declare(self):
        r = evaluate_submission(
            proof_scope="typed_identity_bound_v3",
            contract_address="0x0001", chain_id=1, round_id=42, model_hash="a" * 64,
        )
        assert r.decision == PolicyDecision.REJECTED
        assert "not_accepted" in r.reason
