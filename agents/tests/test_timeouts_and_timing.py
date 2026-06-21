"""Tests for src/orchestration/{timeouts,timing}.py (2026-06-21)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration import timeouts
from src.orchestration.timing import step_timer, timed_node


class TestTimeoutsConfig:
    def test_get_timeout_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("SOME_TIMEOUT_NOT_SET", raising=False)
        assert timeouts.get_timeout("SOME_TIMEOUT_NOT_SET", 42.0) == 42.0

    def test_get_timeout_reads_env_override(self, monkeypatch):
        monkeypatch.setenv("SOME_TIMEOUT_NOT_SET", "99")
        assert timeouts.get_timeout("SOME_TIMEOUT_NOT_SET", 42.0) == 99.0

    def test_get_timeout_falls_back_on_bad_value(self, monkeypatch):
        monkeypatch.setenv("SOME_TIMEOUT_NOT_SET", "not-a-number")
        assert timeouts.get_timeout("SOME_TIMEOUT_NOT_SET", 42.0) == 42.0

    def test_every_default_constant_is_a_positive_float(self):
        for name in dir(timeouts):
            if name.startswith("DEFAULT_") or name == "UNBOUNDED_TIMEOUT_S":
                value = getattr(timeouts, name)
                assert isinstance(value, float) and value > 0, f"{name}={value!r}"

    def test_every_env_var_name_is_a_nonempty_string(self):
        for name in dir(timeouts):
            if name.startswith("ENV_"):
                value = getattr(timeouts, name)
                assert isinstance(value, str) and value, f"{name}={value!r}"


class TestStepTimer:
    def test_logs_start_and_done(self):
        events = []
        from loguru import logger
        sink_id = logger.add(lambda msg: events.append(msg.record["message"]), level="INFO")
        try:
            with step_timer("unit_test_step", address="0xABC"):
                pass
        finally:
            logger.remove(sink_id)

        assert any("unit_test_step | START" in e and "address=0xABC" in e for e in events)
        assert any("unit_test_step | DONE" in e and "elapsed=" in e for e in events)

    def test_logs_done_even_on_exception(self):
        events = []
        from loguru import logger
        sink_id = logger.add(lambda msg: events.append(msg.record["message"]), level="INFO")
        try:
            try:
                with step_timer("unit_test_failing_step"):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
        finally:
            logger.remove(sink_id)

        assert any("unit_test_failing_step | DONE" in e for e in events)


class TestTimedNode:
    async def _dummy_node(self, state):
        return {"out": True}

    def test_wraps_and_preserves_behavior(self):
        import asyncio
        wrapped = timed_node("dummy", self._dummy_node)
        result = asyncio.run(wrapped({"contract_address": "0xXYZ"}))
        assert result == {"out": True}

    def test_logs_node_name_and_address(self):
        import asyncio
        events = []
        from loguru import logger
        sink_id = logger.add(lambda msg: events.append(msg.record["message"]), level="INFO")
        try:
            wrapped = timed_node("dummy_node", self._dummy_node)
            asyncio.run(wrapped({"contract_address": "0xXYZ"}))
        finally:
            logger.remove(sink_id)

        assert any("dummy_node | START" in e and "address=0xXYZ" in e for e in events)
        assert any("dummy_node | DONE" in e for e in events)

    def test_handles_non_dict_state_gracefully(self):
        import asyncio
        wrapped = timed_node("dummy", self._dummy_node)
        # Should not raise even if state isn't a plain dict with .get
        result = asyncio.run(wrapped({}))
        assert result == {"out": True}
