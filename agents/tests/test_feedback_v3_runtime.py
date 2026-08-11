from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ingestion.audit_observation import format_v1_event, format_v3_event
from src.ingestion.feedback_observer import OnChainFeedbackObserver
from src.ingestion.feedback_policy import (
    LEGACY_V1_SCORE_THRESHOLD,
    evaluate_feedback_eligibility,
)
from src.ingestion.feedback_runtime import FeedbackRuntime, PendingObservationJournal

ADDR = "0x" + "11" * 20
AGENT = "0x" + "22" * 20
TX = bytes.fromhex("33" * 32)
PROOF = bytes.fromhex("44" * 32)
DIGEST = bytes.fromhex("55" * 32)


def v1_event(*, block: int = 10, score: int = LEGACY_V1_SCORE_THRESHOLD, log_index: int = 0):
    return {
        "args": {
            "contractAddress": ADDR,
            "proofHash": PROOF,
            "agent": AGENT,
            "scoreFieldElement": score,
        },
        "blockNumber": block,
        "transactionIndex": 0,
        "logIndex": log_index,
        "transactionHash": TX,
    }


def v3_event(*, block: int = 11, round_id: int = 7, log_index: int = 0):
    return {
        "args": {
            "contractAddress": ADDR,
            "requestDigest": DIGEST,
            "proofHash": PROOF,
            "agent": AGENT,
            "roundId": round_id,
        },
        "blockNumber": block,
        "transactionIndex": 0,
        "logIndex": log_index,
        "transactionHash": TX,
    }


def test_event_decoders_preserve_protocol_and_exact_identity() -> None:
    old = format_v1_event(v1_event())
    new = format_v3_event(v3_event())
    assert old["protocol_version"] == "v1"
    assert old["score_field_element"] == LEGACY_V1_SCORE_THRESHOLD
    assert new["protocol_version"] == "v3"
    assert new["request_digest"] == "0x" + DIGEST.hex()
    assert new["round_id"] == 7
    assert new["event_identity"].endswith(":0")


def test_malformed_v3_event_fails_closed() -> None:
    event = v3_event()
    del event["args"]["requestDigest"]
    with pytest.raises(ValueError, match="requestDigest"):
        format_v3_event(event)


def test_v1_threshold_is_preserved_only_as_legacy_policy() -> None:
    low = format_v1_event(v1_event(score=LEGACY_V1_SCORE_THRESHOLD - 1))
    high = format_v1_event(v1_event(score=LEGACY_V1_SCORE_THRESHOLD))
    assert evaluate_feedback_eligibility(low)["eligible"] is False
    decision = evaluate_feedback_eligibility(high)
    assert decision["eligible"] is True
    assert decision["policy_version"] == "legacy-v1-scalar-5734"


def test_v3_never_uses_v1_threshold_and_is_explicitly_not_evaluated() -> None:
    observation = format_v3_event(v3_event())
    decision = evaluate_feedback_eligibility(observation)
    assert decision == {
        "eligible": False,
        "status": "not_evaluated",
        "reason_code": "v3_feedback_policy_unavailable",
        "reason": (
            "V3 feedback promotion requires a measured/versioned policy from "
            "the promoted R4 model/data/calibration lineage; legacy V1 scalar "
            "thresholds are not applicable"
        ),
        "policy_version": None,
        "protocol_version": "v3",
        "attempted": False,
    }


class FakeLegacyIngester:
    def __init__(self, result: bool = True):
        self.result = result
        self.calls = []

    def process_event(self, event):
        self.calls.append(event)
        return self.result


def test_v3_is_journaled_and_never_calls_legacy_ingester(tmp_path: Path) -> None:
    ingester = FakeLegacyIngester()
    journal = PendingObservationJournal(tmp_path / "pending.jsonl")
    runtime = FeedbackRuntime(journal=journal, legacy_ingester=ingester)
    observation = format_v3_event(v3_event())

    first = runtime.process_observation(observation)
    second = runtime.process_observation(observation)

    assert first["status"] == "recorded_policy_pending"
    assert first["ingested_to_rag"] is False
    assert first["feedback_decision"]["reason_code"] == "v3_feedback_policy_unavailable"
    assert second["status"] == "already_recorded"
    assert ingester.calls == []

    rows = [json.loads(line) for line in (tmp_path / "pending.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["observation"]["request_digest"] == "0x" + DIGEST.hex()


def test_v1_eligible_observation_adapts_to_legacy_ingester(tmp_path: Path) -> None:
    ingester = FakeLegacyIngester(result=True)
    runtime = FeedbackRuntime(
        journal=PendingObservationJournal(tmp_path / "pending.jsonl"),
        legacy_ingester=ingester,
    )
    observation = format_v1_event(v1_event())
    result = runtime.process_observation(observation)
    assert result["status"] == "ingested"
    assert result["ingested_to_rag"] is True
    assert ingester.calls[0]["score"] == LEGACY_V1_SCORE_THRESHOLD
    assert "score_field_element" not in ingester.calls[0]


def test_v1_ineligible_observation_never_calls_ingester(tmp_path: Path) -> None:
    ingester = FakeLegacyIngester()
    runtime = FeedbackRuntime(
        journal=PendingObservationJournal(tmp_path / "pending.jsonl"),
        legacy_ingester=ingester,
    )
    result = runtime.process_observation(
        format_v1_event(v1_event(score=LEGACY_V1_SCORE_THRESHOLD - 1))
    )
    assert result["status"] == "ineligible"
    assert result["ingested_to_rag"] is False
    assert ingester.calls == []


class FakeEventReader:
    def __init__(self, batches=None, error=None):
        self.batches = list(batches or [])
        self.error = error
        self.calls = []

    def get_logs(self, *, from_block, to_block):
        self.calls.append((from_block, to_block))
        if self.error is not None:
            raise self.error
        if self.batches:
            return self.batches.pop(0)
        return []


class FakeContract:
    def __init__(self, v1_reader, v3_reader):
        self.events = SimpleNamespace(
            AuditSubmitted=v1_reader,
            AuditSubmittedV3=v3_reader,
        )


class FakeEth:
    def __init__(self, block_number, contract):
        self.block_number = block_number
        self._contract = contract

    def contract(self, *, address, abi):
        return self._contract


class FakeWeb3:
    def __init__(self, block_number, contract):
        self.eth = FakeEth(block_number, contract)

    @staticmethod
    def to_checksum_address(address):
        return address


def observer(tmp_path: Path, *, head: int, v1_reader, v3_reader, last: int = 0, max_range: int = 1999):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"last_block": last}), encoding="utf-8")
    return OnChainFeedbackObserver(
        registry_address=ADDR,
        state_path=state,
        max_block_range=max_range,
        web3=FakeWeb3(head, FakeContract(v1_reader, v3_reader)),
    )


def test_observer_fetches_both_versions_and_checkpoints_only_complete_chunk(tmp_path: Path) -> None:
    obs = observer(
        tmp_path,
        head=2,
        v1_reader=FakeEventReader([[v1_event(block=1)]]),
        v3_reader=FakeEventReader([[v3_event(block=2)]]),
        last=0,
    )
    batch = obs.get_new_observations()
    assert batch["status"] == "success"
    assert [row["protocol_version"] for row in batch["observations"]] == ["v1", "v3"]
    assert batch["last_completed_block"] == 2
    assert json.loads((tmp_path / "state.json").read_text())["last_block"] == 2


def test_v3_query_failure_cannot_checkpoint_v1_only_range(tmp_path: Path) -> None:
    obs = observer(
        tmp_path,
        head=2,
        v1_reader=FakeEventReader([[v1_event(block=1)]]),
        v3_reader=FakeEventReader(error=ConnectionError("v3 query failed")),
        last=0,
    )
    batch = obs.get_new_observations()
    assert batch["status"] == "unavailable"
    assert batch["observations"] == []
    assert batch["last_completed_block"] == 0
    assert json.loads((tmp_path / "state.json").read_text())["last_block"] == 0


def test_partial_failure_preserves_only_fully_completed_chunks(tmp_path: Path) -> None:
    class FailSecond(FakeEventReader):
        def get_logs(self, *, from_block, to_block):
            self.calls.append((from_block, to_block))
            if len(self.calls) == 2:
                raise ConnectionError("second chunk failed")
            return [v3_event(block=1)]

    obs = observer(
        tmp_path,
        head=2,
        v1_reader=FakeEventReader([[], []]),
        v3_reader=FailSecond(),
        last=0,
        max_range=1,
    )
    batch = obs.get_new_observations()
    assert batch["status"] == "partial"
    assert len(batch["observations"]) == 1
    assert batch["last_completed_block"] == 1
    assert batch["failed_range"] == {"from_block": 2, "to_block": 2}
    assert json.loads((tmp_path / "state.json").read_text())["last_block"] == 1
