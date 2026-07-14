"""Read-only legacy adapter for old address-keyed reports.

Existing reports at ``data/reports/{address}.json`` remain readable for
the feedback loop bridge and historical lookups. New writes never use
address as a filename — they go through the job-scoped writer instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .paths import validate_address


def find_legacy_report(root: Path, contract_address: str) -> dict[str, Any] | None:
    """Look up a legacy ``{address}.json`` report under *root*.

    Returns the parsed report dict if found, or None. The address is
    validated before any filesystem access — a malformed address can
    never become a path component.

    Raises ValueError if the address is malformed.
    """
    address = validate_address(contract_address)
    resolved_root = root.resolve()
    candidate = (resolved_root / f"{address}.json").resolve()

    if not candidate.is_relative_to(resolved_root):
        raise ValueError(
            f"legacy report path escapes root: {candidate} is not inside {resolved_root}"
        )

    if not candidate.exists():
        return None

    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("legacy_adapter | could not read {address}: {exc}")
        return None


def find_legacy_hotspot(root: Path, contract_address: str) -> Path | None:
    """Look up a legacy ``{address}_hotspot.html`` under *root*.

    Returns the Path if the file exists, or None. The address is
    validated before any filesystem access.
    """
    address = validate_address(contract_address)
    resolved_root = root.resolve()
    candidate = (resolved_root / f"{address}_hotspot.html").resolve()

    if not candidate.is_relative_to(resolved_root):
        raise ValueError(
            f"legacy hotspot path escapes root: {candidate} is not inside {resolved_root}"
        )

    if candidate.exists():
        return candidate
    return None


__all__ = ["find_legacy_hotspot", "find_legacy_report"]
