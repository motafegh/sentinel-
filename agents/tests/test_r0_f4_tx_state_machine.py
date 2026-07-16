"""R0-F4: Comprehensive adversarial transaction tests.

Covers: threshold enforcement, future inclusion, replacement lifecycle,
reorg replayability, idempotency identity binding, real concurrency,
state-machine enforcement, snapshot/restore.
"""

import threading
import pytest
from src.mcp.servers.audit._submit import (
    TxState, TxLifecycle, TxEngine, FakeChain, _ALLOWED_TRANSITIONS,
)
from src.security.policy_signer import (
    evaluate_submission, PolicyDecision, REJECT_REASON_UNBOUND,
)


class TestConfirmationThreshold:
    def test_confirm_before_threshold_raises(self):
        chain = FakeChain(confirm_blocks=3)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_blocks(2)  # height=2, inclusion=1, depth=2 < 3
        with pytest.raises(ValueError, match="insufficient confirmations"):
            chain.confirm_tx(e.lifecycle.tx_hash)

    def test_confirm_at_threshold(self):
        chain = FakeChain(confirm_blocks=3)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_blocks(2)  # height=2, inc=1, depth=2 < 3 — not yet
        with pytest.raises(ValueError):
            chain.confirm_tx(e.lifecycle.tx_hash)
        chain.mine_block()  # height=3, inc=1, depth=3 == 3 — at threshold
        lc = chain.confirm_tx(e.lifecycle.tx_hash)
        assert lc.state == TxState.CONFIRMED
        assert lc.confirmations == 3

    def test_confirm_above_threshold(self):
        chain = FakeChain(confirm_blocks=2)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_blocks(5)
        lc = chain.confirm_tx(e.lifecycle.tx_hash)
        assert lc.confirmations >= 2  # 5-1+1=5 >= 2

    def test_n_minus_1_rejected(self):
        chain = FakeChain(confirm_blocks=5)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_blocks(4)  # depth=4 < 5
        with pytest.raises(ValueError):
            chain.confirm_tx(e.lifecycle.tx_hash)

    def test_n_plus_1_accepted(self):
        chain = FakeChain(confirm_blocks=5)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_blocks(6)  # depth=6 > 5
        lc = chain.confirm_tx(e.lifecycle.tx_hash)
        assert lc.state == TxState.CONFIRMED


class TestFutureInclusion:
    def test_reject_chain_height_below_inclusion(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0x"); e.mined(100)
        with pytest.raises(ValueError, match="chain_height"):
            e.confirm(100, 80000, chain_height=50)

    def test_reject_negative_inclusion(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0x"); e.mined(1)
        with pytest.raises(ValueError):
            e.confirm(-1, 80000, chain_height=1)

    def test_reject_zero_gas(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0x"); e.mined(1)
        with pytest.raises(ValueError):
            e.confirm(1, 0, chain_height=1)

    def test_ingest_receipt_rejects_future_block(self):
        e = TxEngine(TxLifecycle(state=TxState.SIGNED))
        e.broadcast("0x"); e.mined(1)
        with pytest.raises(ValueError, match="chain_height"):
            e.ingest_receipt({"status": 1, "blockNumber": 100, "gasUsed": 80000}, chain_height=50)


class TestReplacementLifecycle:
    def test_replacement_hash_generated_first(self):
        chain = FakeChain()
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)
        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        new_hash, _ = chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        assert new_hash is not None
        assert len(new_hash) == 66  # 0x + 64 hex

    def test_old_linked_to_exact_new_hash(self):
        chain = FakeChain()
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)
        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        new_hash, _ = chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        assert e_new.lifecycle.tx_hash == new_hash

    def test_replacement_in_mempool(self):
        chain = FakeChain()
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)
        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        new_hash, _ = chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        assert new_hash in chain._mempool

    def test_replacement_can_be_mined(self):
        chain = FakeChain()
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)
        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        mined = chain.mine_block()
        assert len(mined) >= 1
        assert e_new.state == TxState.PENDING

    def test_replacement_can_be_confirmed(self):
        chain = FakeChain(confirm_blocks=1)
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)
        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        new_hash, _ = chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        chain.mine_block()
        chain.confirm_tx(new_hash)
        assert e_new.state == TxState.CONFIRMED

    def test_replacement_idempotency_map_updated(self):
        chain = FakeChain()
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old, idempotency_key="rep-ik")
        before = len(chain._idempotent)
        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        assert len(chain._idempotent) == before  # still one mapping, updated

    def test_replacement_survives_snapshot_restore(self):
        chain = FakeChain(confirm_blocks=1)
        e_old = TxEngine(); e_old.prepare(); e_old.sign()
        chain.send(e_old)
        snap = chain.snapshot()
        e_new = TxEngine(); e_new.prepare(); e_new.sign()
        new_hash, _ = chain.replace_tx(e_old.lifecycle.tx_hash, e_new)
        chain.restore(snap)
        # after restore, old tx should be back in mempool
        assert e_old.lifecycle.tx_hash in chain._mempool


class TestReorgBehavior:
    def test_pending_receipt_orphaned_without_crash(self):
        chain = FakeChain(confirm_blocks=3)
        e = TxEngine(); e.prepare(); e.sign()
        tx_hash = chain.send(e).tx_hash
        chain.mine_block()
        assert e.state == TxState.PENDING
        assert chain.reorg(depth=1) == [tx_hash]
        assert e.state == TxState.BROADCAST
        assert e.lifecycle.block_number is None
        assert e.lifecycle.receipt_status is None

    def test_reorg_clears_orphaned_receipt_truth(self):
        chain = FakeChain(confirm_blocks=1)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e); chain.mine_block(); chain.confirm_tx(e.lifecycle.tx_hash)
        chain.reorg(depth=1)
        assert e.state == TxState.BROADCAST
        assert e.lifecycle.receipt_status is None
        assert e.lifecycle.confirmations == 0
        assert e.lifecycle.gas_used is None
        assert e.lifecycle.block_number is None

    def test_reorg_reduces_finality_for_still_included_receipt(self):
        chain = FakeChain(confirm_blocks=3)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e); chain.mine_blocks(3); chain.confirm_tx(e.lifecycle.tx_hash)
        chain.reorg(depth=1)
        assert e.state == TxState.PENDING
        assert e.lifecycle.receipt_status == 1
        assert e.lifecycle.confirmations == 2

    def test_reorg_replayable(self):
        chain = FakeChain(confirm_blocks=1)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED
        chain.reorg(depth=1)
        assert e.state == TxState.BROADCAST

    def test_reorged_tx_can_be_re_mined(self):
        chain = FakeChain(confirm_blocks=1)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)
        chain.reorg(depth=1)
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED

    def test_reorg_unaffected_tx_unchanged(self):
        chain = FakeChain(confirm_blocks=1)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()
        chain.confirm_tx(e.lifecycle.tx_hash)

        e2 = TxEngine(); e2.prepare(); e2.sign()
        chain.send(e2)
        chain.mine_block()
        chain.confirm_tx(e2.lifecycle.tx_hash)

        chain.reorg(depth=1)
        assert e.state == TxState.CONFIRMED  # unaffected
        assert e2.state == TxState.BROADCAST  # affected, rolled back to re-mine

    def test_multi_block_reorg(self):
        chain = FakeChain(confirm_blocks=1)
        engines = []
        for _ in range(3):
            e = TxEngine(); e.prepare(); e.sign()
            chain.send(e)
            chain.mine_block()
            chain.confirm_tx(e.lifecycle.tx_hash)
            engines.append(e)

        chain.reorg(depth=2)
        assert engines[0].state == TxState.CONFIRMED  # not rolled back
        assert engines[1].state == TxState.BROADCAST
        assert engines[2].state == TxState.BROADCAST

    def test_deep_reorg_replay(self):
        chain = FakeChain(confirm_blocks=1)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_blocks(5)  # height=5, inclusion=1
        chain.confirm_tx(e.lifecycle.tx_hash)  # confirmed at height 5
        chain.reorg(depth=3)  # height drops to 2, inclusion=1 <= 2 — NOT affected
        assert e.state == TxState.CONFIRMED  # below reorg depth

    def test_shallow_reorg_affects_recent_tx(self):
        chain = FakeChain(confirm_blocks=1)
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()  # inclusion=1
        chain.confirm_tx(e.lifecycle.tx_hash)  # confirmed at height 1
        chain.reorg(depth=1)  # height drops to 0, inclusion=1 > 0 — AFFECTED
        assert e.state == TxState.BROADCAST
        chain.mine_block()  # re-mine
        chain.confirm_tx(e.lifecycle.tx_hash)
        assert e.state == TxState.CONFIRMED


class TestIdempotencyBinding:
    def test_round_id_is_bound(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        first = chain.send(e1, idempotency_key="ik", round_id=1)
        e2 = TxEngine(); e2.prepare(); e2.sign()
        second = chain.send(e2, idempotency_key="ik", round_id=2)
        assert first.tx_hash != second.tx_hash

    def test_request_digest_is_bound(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        first = chain.send(e1, idempotency_key="ik", request_digest="a" * 64)
        e2 = TxEngine(); e2.prepare(); e2.sign()
        second = chain.send(e2, idempotency_key="ik", request_digest="b" * 64)
        assert first.tx_hash != second.tx_hash

    def test_full_model_hash_not_truncated(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        chain.send(e1, idempotency_key="ik", model_hash="a"*63 + "b")
        chain.mine_block()

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="ik", model_hash="a"*64)  # different last char
        assert lc2.tx_hash != e1.lifecycle.tx_hash

    def test_same_identity_same_tx(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        chain.send(e1, idempotency_key="same", chain_id=1, address="0xAbC", model_hash="m"*64)
        chain.mine_block()

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="same", chain_id=1, address="0xabc", model_hash="m"*64)
        assert lc2.tx_hash == e1.lifecycle.tx_hash  # canonical address

    def test_different_chain_id_different_tx(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        chain.send(e1, idempotency_key="ik", chain_id=1, address="0xA", model_hash="m"*64)
        chain.mine_block()

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="ik", chain_id=5, address="0xA", model_hash="m"*64)
        assert lc2.tx_hash != e1.lifecycle.tx_hash

    def test_dropped_retry_allowed(self):
        chain = FakeChain()
        e1 = TxEngine(); e1.prepare(); e1.sign()
        chain.send(e1, idempotency_key="retry-ik")
        chain.drop_tx(e1.lifecycle.tx_hash)

        e2 = TxEngine(); e2.prepare(); e2.sign()
        lc2 = chain.send(e2, idempotency_key="retry-ik")
        assert lc2.state == TxState.BROADCAST


class TestRealConcurrency:
    def test_concurrent_identical_requests_one_hash(self):
        chain = FakeChain()
        results = []
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                e = TxEngine(); e.prepare(); e.sign()
                lc = chain.send(e, idempotency_key="same", chain_id=1, address="0x0", model_hash="x"*64)
                with lock:
                    results.append(lc.tx_hash)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0
        assert len(results) == 20
        unique = set(results)
        assert len(unique) == 1, f"all must get the same hash, got {len(unique)}"

    def test_concurrent_different_unique_nonces(self):
        chain = FakeChain()
        results = []
        errors = []
        lock = threading.Lock()

        def worker(i):
            try:
                e = TxEngine(); e.prepare(); e.sign()
                lc = chain.send(e, idempotency_key=f"diff-{i}")
                with lock:
                    results.append((lc.tx_hash, lc.nonce))
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0
        assert len(results) == 50
        nonces = [r[1] for r in results]
        hashes = [r[0] for r in results]
        assert len(set(nonces)) == 50
        assert len(set(hashes)) == 50

    def test_concurrent_after_snapshot_restore(self):
        chain = FakeChain()
        e = TxEngine(); e.prepare(); e.sign()
        chain.send(e)
        chain.mine_block()
        snap = chain.snapshot()
        chain.restore(snap)

        results = []
        def worker(i):
            en = TxEngine(); en.prepare(); en.sign()
            lc = chain.send(en, idempotency_key=f"snap-{i}")
            results.append(lc.tx_hash)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(results) == 10
        assert len(set(results)) == 10


class TestStateMachineEnforcement:
    """Every typed helper routes through transition(). Test from forbidden states."""

    def _at(self, state):
        return TxEngine(TxLifecycle(state=state))

    @pytest.mark.parametrize("terminal_state,method,args", [
        (TxState.POLICY_REJECTED, "prepare", ()),
        (TxState.FAILED, "sign", ()),
        (TxState.REVERTED, "broadcast", ("0xa",)),
    ])
    def test_helper_from_forbidden_state_raises(self, terminal_state, method, args):
        e = self._at(terminal_state)
        before = e.snapshot_state()
        with pytest.raises(ValueError, match="invalid transition"):
            getattr(e, method)(*args)
        assert e.snapshot_state() == before

    def test_confirm_from_broadcast_raises(self):
        e = self._at(TxState.BROADCAST)
        before = e.snapshot_state()
        with pytest.raises(ValueError):
            e.confirm(1, 80000, chain_height=1)
        assert e.snapshot_state() == before

    @pytest.mark.parametrize("method,args", [
        ("mined", (7,)),
        ("revert", ("no receipt",)),
        ("replace", ("0xnew", "not broadcast")),
    ])
    def test_all_mutating_helpers_are_failure_atomic(self, method, args):
        e = self._at(TxState.NOT_REQUESTED)
        before = e.snapshot_state()
        with pytest.raises(ValueError):
            getattr(e, method)(*args)
        assert e.snapshot_state() == before

    def test_confirm_from_pending_ok(self):
        e = self._at(TxState.PENDING)
        e.lifecycle.block_number = 1
        e.confirm(1, 80000, chain_height=1)

    def test_reorg_rollback_from_confirmed_ok(self):
        e = self._at(TxState.CONFIRMED)
        e.reorg_rollback()  # must not raise

    def test_reorg_rollback_from_pending_clears_inclusion(self):
        e = self._at(TxState.PENDING)
        e.lifecycle.block_number = 9
        e.lifecycle.receipt_status = 1
        e.reorg_rollback()
        assert e.state == TxState.SIGNED
        assert e.lifecycle.block_number is None
        assert e.lifecycle.receipt_status is None

    def test_transition_table_consistent(self):
        for state in TxState:
            assert state in _ALLOWED_TRANSITIONS
            for target in _ALLOWED_TRANSITIONS[state]:
                assert isinstance(target, TxState)


class TestPolicySigner:
    def test_all_scopes_rejected(self):
        for scope in ("legacy_proxy_only_unbound", "none", "", "typed_identity_bound_v3"):
            r = evaluate_submission(proof_scope=scope, contract_address="0x1",
                                     chain_id=1, round_id=42, model_hash="a"*64)
            assert r.decision == PolicyDecision.REJECTED
