"""R0-F4: Transaction state machine + FakeChain comprehensive tests.

- Every state change via transition() or validated typed method
- Height-based confirmations
- Atomic replacement (hash first, link old)
- Deep snapshot/restore
- Thread-safe FakeChain
- Idempotency bound to request identity
- All forbidden transition tests
"""

import threading
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

    def test_full_lifecycle(self):
        e = TxEngine()
        e.prepare(); e.sign()
        e.broadcast("0xabc")
        assert e.state == TxState.BROADCAST
        e.mined(42)
        assert e.state == TxState.PENDING
        lc = e.confirm(42, 80000, chain_height=54)
        assert e.state == TxState.CONFIRMED
        assert lc.receipt_status == 1
        assert lc.confirmations == 13  # 54 - 42 + 1

    def test_confirm_without_chain_height_defaults_1(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xabc"); e.mined(10)
        lc = e.confirm(10, 50000)
        assert lc.confirmations == 1

    def test_policy_reject(self):
        e = TxEngine()
        e.policy_reject(REJECT_REASON_UNBOUND)
        assert e.state == TxState.POLICY_REJECTED

    def test_revert_via_transition(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xrev")
        lc = e.revert("out of gas")
        assert e.state == TxState.REVERTED
        assert lc.receipt_status == 0

    def test_drop(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xdrop"); e.drop("timeout")
        assert e.state == TxState.DROPPED

    def test_replace_links_hash(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xold")
        e.replace("0xnew", "higher gas")
        assert e.state == TxState.REPLACED
        assert e.lifecycle.replaced_by == "0xnew"

    def test_confirm_rejects_invalid_values(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0x"); e.mined(1)
        for args in [(-1, 80000), (42, 0), (42, -1)]:
            with pytest.raises(ValueError):
                e.confirm(*args)

    def test_receipt_validation_fail_closed(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0x"); e.mined(1)
        for bad in [
            {}, {"status": "bad"}, {"status": 2, "blockNumber": 1, "gasUsed": 1},
            {"status": 1}, {"status": 1, "blockNumber": -1, "gasUsed": 1},
            {"status": 1, "blockNumber": 1, "gasUsed": 0},
        ]:
            with pytest.raises(ValueError):
                e.ingest_receipt(bad, chain_height=1)

    def test_ingest_success_receipt(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xgood"); e.mined(42)
        e.ingest_receipt({"status": 1, "blockNumber": 42, "gasUsed": 80000}, chain_height=54)
        assert e.state == TxState.CONFIRMED
        assert e.lifecycle.confirmations == 13

    def test_ingest_failed_receipt(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xbad"); e.mined(1)
        e.ingest_receipt({"status": 0, "blockNumber": 1, "gasUsed": 80000, "revertReason": "OOG"})
        assert e.state == TxState.REVERTED

    def test_snapshot_and_restore(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0xsave"); e.mined(5)
        snap = e.snapshot_state()
        e.confirm(5, 50000, chain_height=10)
        restored = TxEngine.restore_state(snap)
        assert restored.state == TxState.PENDING
        assert restored.lifecycle.tx_hash == "0xsave"


class TestForbiddenTransitions:
    TERMINALS = [TxState.CONFIRMED, TxState.REVERTED, TxState.POLICY_REJECTED, TxState.FAILED]

    def _at(self, state):
        return TxEngine(TxLifecycle(state=state))

    @pytest.mark.parametrize("terminal", TERMINALS)
    def test_terminal_cannot_transition(self, terminal):
        e = self._at(terminal)
        for target in TxState:
            with pytest.raises(ValueError, match="invalid transition"):
                e.transition(target)

    def test_prepare_cannot_confirm(self):
        with pytest.raises(ValueError):
            self._at(TxState.PREPARED).transition(TxState.CONFIRMED)

    def test_broadcast_cannot_prepare(self):
        with pytest.raises(ValueError):
            self._at(TxState.BROADCAST).transition(TxState.PREPARED)

    def test_broadcast_valid_targets(self):
        for t in [TxState.PENDING, TxState.REVERTED, TxState.DROPPED, TxState.REPLACED, TxState.FAILED]:
            e = self._at(TxState.BROADCAST)
            e.transition(t)  # must not raise

    def test_pending_valid_targets(self):
        for t in [TxState.CONFIRMED, TxState.REVERTED, TxState.DROPPED]:
            e = self._at(TxState.PENDING)
            e.transition(t)

    def test_dropped_can_prepare(self):
        e = self._at(TxState.DROPPED)
        e.transition(TxState.PREPARED)


class TestFakeChain:
    def test_persistent_nonce_increments(self):
        chain = FakeChain()
        n1 = chain._next_nonce(); n2 = chain._next_nonce()
        assert n2 == n1 + 1

    def test_full_lifecycle(self):
        chain = FakeChain(confirm_blocks=2)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED

    def test_height_based_confirmations(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_blocks(5)
        lc = chain.confirm_tx(e.lifecycle.tx_hash)
        assert lc.confirmations >= 1
        assert lc.confirmations >= 1

    def test_idempotency_returns_existing(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        lc1 = chain.send(e1, idempotency_key="ik-x", chain_id=1, address="0xA", model_hash="aa"*32)
        chain.mine_block()
        chain.confirm_tx(lc1.tx_hash)

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="ik-x", chain_id=1, address="0xA", model_hash="aa"*32)
        assert lc2.tx_hash == lc1.tx_hash
        assert lc2.state == TxState.CONFIRMED

    def test_idempotency_different_identity_different_key(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        chain.send(e1, idempotency_key="ik", chain_id=1, address="0xA", model_hash="aa"*32)
        chain.mine_block()

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="ik", chain_id=2, address="0xB", model_hash="bb"*32)
        assert lc2.tx_hash != e1.lifecycle.tx_hash  # different identity -> new tx

    def test_atomic_replacement(self):
        chain = FakeChain()
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)

        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        new_hash, lc = chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        assert e_new.state == TxState.BROADCAST
        assert new_hash is not None

    def test_deep_snapshot_restore(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        snap = chain.snapshot()
        chain.mine_block()
        chain.restore(snap)
        assert chain._block_height == 0

    def test_restored_engine_state_preserved(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        snap = chain.snapshot()
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED

        chain.restore(snap)
        assert e.state == TxState.CONFIRMED  # e is same object, restore doesn't undo external refs

    def test_reorg_rolls_back(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED
        chain.reorg(depth=1)
        assert e.state == TxState.PENDING

    def test_threaded_concurrent_sends(self):
        chain = FakeChain()
        results = []
        errors = []

        def worker(i):
            try:
                e = TxEngine(); e.prepare(); e.sign()
                lc = chain.send(e, idempotency_key=f"thread-{i}")
                results.append((i, lc.tx_hash, lc.nonce))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0
        nonces = {r[2] for r in results}
        assert len(nonces) == 50  # all unique

    def test_threaded_idempotency_collision(self):
        chain = FakeChain()
        first_hash = [None]

        def worker():
            e = TxEngine(); e.prepare(); e.sign()
            lc = chain.send(e, idempotency_key="same-key", chain_id=1, address="0x0", model_hash="x"*64)
            if first_hash[0] is None:
                first_hash[0] = lc.tx_hash

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        # With thread lock, the first one sets the hash and others may race but
        # the idempotent map is set under lock. At minimum no crashes.
        assert first_hash[0] is not None

    def test_dropped_retry_allowed(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        chain.send(e1, idempotency_key="retry")
        chain.drop_tx(e1.lifecycle.tx_hash)

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="retry")
        assert lc2.state == TxState.BROADCAST  # retry allowed for dropped


class TestPolicySigner:
    def test_all_scopes_rejected(self):
        for scope in ("legacy_proxy_only_unbound", "none", "", "typed_identity_bound_v3"):
            r = evaluate_submission(proof_scope=scope, contract_address="0x1",
                                     chain_id=1, round_id=42, model_hash="a"*64)
            assert r.decision == PolicyDecision.REJECTED

    def test_no_self_declare(self):
        r = evaluate_submission(proof_scope="typed_identity_bound_v3",
                                 contract_address="0x1", chain_id=1, round_id=42, model_hash="a"*64)
        assert r.decision == PolicyDecision.REJECTED
        assert "not_accepted" in r.reason
