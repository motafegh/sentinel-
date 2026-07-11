from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.nodes._helpers import AderynRunError, _run_aderyn_on_file


# High/Critical-impact Slither detectors worth escalating on even if ML is safe.
# Informational/Low detectors are intentionally excluded to limit false escalations.
_SCREEN_SLITHER_DETECTORS: frozenset[str] = frozenset({
    "reentrancy-eth",
    "reentrancy-no-eth",
    "arbitrary-send-eth",
    "controlled-delegatecall",
    "delegatecall-loop",
    # "integer-overflow" removed 2026-06-21 — does not exist as a Slither 0.11.5
    # detector ARGUMENT (removed upstream; Solidity >=0.8 has checked arithmetic).
    "msg-value-loop",
    "unchecked-send",
    "unchecked-transfer",
    "calls-loop",
    "tx-origin",
    "suicidal",
    "uninitialized-local",
    "uninitialized-state",
    "write-after-write",
})

# Removed 2026-06-21: _SCREEN_ADERYN_HIGH_IDS (a frozenset of "H-1".."C-3" labels)
# was dead code — never referenced anywhere. It also encoded a wrong assumption:
# those labels are per-report table-of-contents positions (H-1, H-2, ...), not
# stable detector identifiers — they'd renumber per contract. quick_screen now
# escalates on any Aderyn finding with impact=="High" directly (see below),
# matching real detector_name values from the verified JSON schema.


async def quick_screen(state: AuditState) -> dict[str, Any]:
    """
    Tier 0 screen — runs Slither + Aderyn on every contract before routing.

    PURPOSE: Closes the ML blind spot where all class probabilities fall below
    DEEP_THRESHOLDS. A contract scoring "safe" on ML is still escalated to deep
    path if either static tool fires a High/Critical finding.

    Two independent signals are required to agree before fast-path is allowed:
        Signal 1: ML — all class probabilities below DEEP_THRESHOLDS
        Signal 2: quick_screen — zero High/Critical findings from Slither+Aderyn

    Slither: subset of High-impact detectors (_SCREEN_SLITHER_DETECTORS).
    Aderyn:  subprocess call; parses JSON output; escalates on any finding
             with impact in (High, Medium, Critical) — same as Slither.
             Non-fatal at the NODE level (intentional — Slither-only is
             still useful) but failure MUST surface in
             `state["tool_status"]["aderyn"]` per Rule 5C (CLAUDE.md).
             An empty aderyn list alone is no longer sufficient.

    State updates:
        quick_screen_hits → {"slither": [detector_name, ...], "aderyn": [rule_id, ...]}
                           Empty lists in each key when nothing fires or tool absent.
        tool_status       → {"aderyn": {"ran": bool, "reason": str, ...}} — Rule 5C.
    """
    contract_code = state.get("contract_code", "")
    if not contract_code or not contract_code.strip():
        logger.warning("quick_screen | contract_code empty — skipping")
        return {
            "quick_screen_hits": {"slither": [], "aderyn": []},
            "tool_status": {
                "slither": {"ran": False, "reason": "empty_contract_code"},
                "aderyn":  {"ran": False, "reason": "empty_contract_code"},
            },
        }

    logger.info(
        "quick_screen | running Tier-0 screen | contract_address={}",
        state.get("contract_address", "unknown"),
    )

    slither_hits: list[str] = []
    aderyn_hits:  list[str] = []
    tmp_path: str | None    = None
    slither_status: dict = {"ran": False, "reason": "not_started"}

    try:
        import inspect

        from slither import Slither
        from slither.detectors import all_detectors
        from slither.detectors.abstract_detector import AbstractDetector

        with tempfile.NamedTemporaryFile(
            suffix=".sol",
            prefix="sentinel_screen_",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(contract_code)
            tmp_path = tmp.name

        sl = Slither(tmp_path)

        # CRITICAL (found 2026-06-21, same bug as static_analysis): Slither()
        # registers ZERO detectors on construction — must register the full set
        # before scoping/running, or quick_screen NEVER escalates on anything.
        for detector_cls in (
            d for d in (getattr(all_detectors, name) for name in dir(all_detectors))
            if inspect.isclass(d) and issubclass(d, AbstractDetector)
        ):
            sl.register_detector(detector_cls)

        # Scope to High-impact detectors only — avoids escalating on noise.
        sl._detectors = [  # type: ignore[attr-defined]
            d for d in sl._detectors  # type: ignore[attr-defined]
            if getattr(d, "ARGUMENT", "") in _SCREEN_SLITHER_DETECTORS
        ]

        for result in sl.run_detectors():
            for finding in result:
                impact = finding.get("impact", "")
                if impact in ("High", "Medium", "Critical"):
                    detector = finding.get("check", "unknown")
                    if detector not in slither_hits:
                        slither_hits.append(detector)

        logger.info(
            "quick_screen | slither done | hits={} | contract_address={}",
            slither_hits,
            state.get("contract_address", "unknown"),
        )
        slither_status = {"ran": True, "n_findings": len(slither_hits)}

    except ImportError:
        logger.warning("quick_screen | slither not installed — skipping slither screen")
        slither_status = {"ran": False, "reason": "not_installed"}

    except Exception as exc:
        logger.warning("quick_screen | slither error (non-fatal): {}", exc)
        slither_status = {"ran": False, "reason": "slither_error", "detail": str(exc)}

    # ── Aderyn ────────────────────────────────────────────────────────────────
    # Delegates to _run_aderyn_on_file (defined below) — fixed 2026-06-21 to use
    # a real directory ROOT + a real --output file path + the actual JSON schema
    # (see that function's docstring). Previously this block had its own
    # independent invocation that failed identically (file-not-directory error,
    # silently swallowed) — Aderyn never escalated anything here either.
    #
    # Rule 5C (CLAUDE.md, 2026-06-25): the helper now raises on missing
    # binary / runtime failure instead of returning []. The node stays
    # non-fatal at the pipeline level (intentional) but the failure MUST
    # surface in `state["tool_status"]["aderyn"]` — an empty aderyn_hits
    # list alone is no longer the signal for "tool absent."
    try:
        aderyn_findings = _run_aderyn_on_file(contract_code)
    except FileNotFoundError as exc:
        logger.warning("quick_screen | aderyn unavailable — slither-only: {}", exc)
        aderyn_findings = []
        aderyn_status = {
            "ran": False,
            "reason": "binary_not_found",
            "detail": str(exc),
            "fallback": "slither-only",
        }
    except AderynRunError as exc:
        logger.warning("quick_screen | aderyn run failed — slither-only: {}", exc)
        aderyn_findings = []
        aderyn_status = {
            "ran": False,
            "reason": "run_error",
            "detail": str(exc),
            "fallback": "slither-only",
        }
    else:
        aderyn_status = {
            "ran": True,
            "n_findings": len(aderyn_findings),
        }

    # WS1 (2026-06-21): align Aderyn escalation to Slither's levels.
    # Previously only "High" — Medium/Critical were silently ignored,
    # an inconsistency introduced during the Aderyn fix (Finding #11).
    for finding in aderyn_findings:
        if finding["impact"] in ("High", "Medium", "Critical") and finding["detector"] not in aderyn_hits:
            aderyn_hits.append(finding["detector"])

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            logger.warning("quick_screen | failed to delete temp file {}: {}", tmp_path, e)

    hits = {"slither": slither_hits, "aderyn": aderyn_hits}
    logger.info(
        "quick_screen complete | slither_hits={} | aderyn_hits={} | aderyn_ran={} | contract_address={}",
        len(slither_hits),
        len(aderyn_hits),
        aderyn_status.get("ran"),
        state.get("contract_address", "unknown"),
    )
    return {
        "quick_screen_hits": hits,
        "tool_status": {"slither": slither_status, "aderyn": aderyn_status},
    }
