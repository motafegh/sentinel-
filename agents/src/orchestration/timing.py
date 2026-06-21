"""
timing.py — uniform live step-timing logs for the SENTINEL audit pipeline.

WHY THIS FILE EXISTS (2026-06-21): every node logged its own start/done message
in a slightly different, ad-hoc format, and several nodes (consensus_engine,
explainer, visualizer) logged no duration at all. Tracing where an audit run
was actually spending time meant grepping for inconsistent log shapes. Every
node and every LLM sub-call (e.g. each of the debate's 3 roles individually)
now uses `step_timer()` so production logs ALWAYS show, for every run, the same
two-line shape: a START line when a step begins and a DONE line with the exact
elapsed seconds when it ends — even on failure.

Example output:
    16:35:52.919 | INFO | cross_validator.prosecutor | START | address=0xLIVE
    16:37:14.221 | INFO | cross_validator.prosecutor | DONE | elapsed=81.30s | address=0xLIVE
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator

from loguru import logger


@contextmanager
def step_timer(step_name: str, **context: object) -> Iterator[None]:
    """
    Log a START line, run the wrapped block, then log a DONE line with elapsed
    seconds — even if the block raises (the exception still propagates).

    Args:
        step_name: dotted identifier for the step, e.g. "cross_validator.judge",
            "static_analysis.slither", "graph_explain". Use dotted sub-names for
            sub-steps within a node so each LLM call/tool call is individually
            visible, not just the node's total.
        **context: extra key=value pairs appended to both log lines (e.g.
            address="0x...", classes=5) — whatever helps correlate this step
            with the rest of that run's log without re-deriving it.
    """
    suffix = "".join(f" | {k}={v}" for k, v in context.items())
    start = time.monotonic()
    logger.info("{} | START{}", step_name, suffix)
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        logger.info("{} | DONE | elapsed={:.2f}s{}", step_name, elapsed, suffix)


NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def timed_node(name: str, fn: NodeFn) -> NodeFn:
    """
    Wrap a LangGraph node coroutine so EVERY invocation logs a uniform
    START/DONE+elapsed pair via `step_timer` — added 2026-06-21 so node-level
    timing is visible in every context (production graph driven by an MCP
    client, `run_real_audit.py`, ad-hoc scripts), not only when a caller
    happens to add its own ad-hoc wrapper (the test harness used to be the
    only place this existed).

    Used at registration time in `graph.py`'s `build_graph()` — wrap once per
    node there rather than instrumenting each node function body individually,
    so internal node logic never needs to change to get timing coverage.
    """
    @functools.wraps(fn)
    async def _wrapped(state: dict[str, Any]) -> dict[str, Any]:
        address = state.get("contract_address", "unknown") if isinstance(state, dict) else "unknown"
        with step_timer(name, address=address):
            return await fn(state)
    return _wrapped
