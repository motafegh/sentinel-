from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ingestion.pipeline import REPORTS_DIR
from src.orchestration.state import AuditState
from src.persistence import persist_hotspot


async def visualizer(state: AuditState) -> dict[str, Any]:
    """
    Hotspot attribution visualization (A.9) — last node before END.

    Generates a self-contained interactive HTML report (source + verdict panel
    with confidence and attribution bars) and writes it to a job-scoped
    directory. Never raises.

    State updates:
        hotspot_visualization → HTML string
    """
    from src.orchestration.visualizer import generate_hotspot_html

    try:
        html_str = generate_hotspot_html(dict(state))
    except Exception as exc:
        logger.warning("visualizer | HTML generation failed (non-fatal): {}", exc)
        return {"hotspot_visualization": None}

    persistence_status = persist_hotspot(state, html_str, REPORTS_DIR)

    logger.info("visualizer complete | html={} chars", len(html_str))
    result: dict[str, Any] = {"hotspot_visualization": html_str}
    if persistence_status:
        result["tool_status"] = persistence_status
    return result
