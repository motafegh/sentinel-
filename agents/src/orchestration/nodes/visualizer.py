from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.ingestion.pipeline import REPORTS_DIR


async def visualizer(state: AuditState) -> dict[str, Any]:
    """
    Hotspot attribution visualization (A.9) — last node before END.

    Generates a self-contained interactive HTML report (source + verdict panel
    with confidence and attribution bars) and writes it to
    data/reports/{address}_hotspot.html. Never raises.

    State updates:
        hotspot_visualization → HTML string
    """
    from src.orchestration.visualizer import generate_hotspot_html

    try:
        html_str = generate_hotspot_html(dict(state))
    except Exception as exc:
        logger.warning("visualizer | HTML generation failed (non-fatal): {}", exc)
        return {"hotspot_visualization": None}

    address = (state.get("contract_address", "") or "").strip()
    if address:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out = REPORTS_DIR / f"{address}_hotspot.html"
            out.write_text(html_str)
            logger.info("visualizer | hotspot HTML written → {}", out)
        except Exception as exc:
            logger.warning("visualizer | could not persist hotspot HTML (non-fatal): {}", exc)

    logger.info("visualizer complete | html={} chars", len(html_str))
    return {"hotspot_visualization": html_str}
