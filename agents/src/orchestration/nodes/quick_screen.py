from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.nodes._helpers import _run_aderyn_on_file


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
    Aderyn:  subprocess call; parses JSON output; looks for H-* / C-* rule IDs.
             Non-fatal if aderyn is not installed — only slither result used.

    State updates:
        quick_screen_hits → {"slither": [detector_name, ...], "aderyn": [rule_id, ...]}
                           Empty lists in each key when nothing fires or tool absent.
    """
    contract_code = state.get("contract_code", "")
    if not contract_code or not contract_code.strip():
        logger.warning("quick_screen | contract_code empty — skipping")
        return {"quick_screen_hits": {"slither": [], "aderyn": []}}

    logger.info(
        "quick_screen | running Tier-0 screen | contract_address={}",
        state.get("contract_address", "unknown"),
    )

    slither_hits: list[str] = []
    aderyn_hits:  list[str] = []
    tmp_path: str | None    = None

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

    except ImportError:
        logger.warning("quick_screen | slither not installed — skipping slither screen")

    except Exception as exc:
        logger.warning("quick_screen | slither error (non-fatal): {}", exc)

    # ── Aderyn ────────────────────────────────────────────────────────────────
    # Delegates to _run_aderyn_on_file (defined below) — fixed 2026-06-21 to use
    # a real directory ROOT + a real --output file path + the actual JSON schema
    # (see that function's docstring). Previously this block had its own
    # independent invocation that failed identically (file-not-directory error,
    # silently swallowed) — Aderyn never escalated anything here either.
    try:
        for finding in _run_aderyn_on_file(contract_code):
            # WS1 (2026-06-21): align Aderyn escalation to Slither's levels.
            # Previously only "High" — Medium/Critical were silently ignored,
            # an inconsistency introduced during the Aderyn fix (Finding #11).
            if finding["impact"] in ("High", "Medium", "Critical") and finding["detector"] not in aderyn_hits:
                aderyn_hits.append(finding["detector"])
    except Exception as exc:
        logger.warning("quick_screen | aderyn error (non-fatal): {}", exc)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            logger.warning("quick_screen | failed to delete temp file {}: {}", tmp_path, e)

    hits = {"slither": slither_hits, "aderyn": aderyn_hits}
    logger.info(
        "quick_screen complete | slither_hits={} | aderyn_hits={} | contract_address={}",
        len(slither_hits),
        len(aderyn_hits),
        state.get("contract_address", "unknown"),
    )
    return {"quick_screen_hits": hits}
