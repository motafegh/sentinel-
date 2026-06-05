"""
smoke_fix8.py — Smoke test for Fix #8 (complexity-bias documentation).

Fix #8 is documentation-only: explain why the model defaults to dropping
the complexity feature in Phase 2 (gemma-2-9b ablation insight).

This smoke test verifies:
  - The doc file exists
  - The doc references the actual flag (`drop_complexity_feature` or `drop_complexity`)
  - The doc mentions the relevant ablation result

Gates-in:
  G8.1 — docs/pre-run9-fixes/06-bonus-fixes.md exists

Gates-out:
  G8.2 — Doc contains the string "drop_complexity" (the actual flag)
  G8.3 — Doc references Run 7 or Run 8 baseline (the ablation source)
  G8.4 — Doc mentions the bias phenomenon (complexity correlates with vulnerability count)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    REPO_ROOT,
    check,
    pass_,
    smoke_header,
    timed,
)

DOC_PATH: Path = REPO_ROOT / "docs" / "pre-run9-fixes" / "06-bonus-fixes.md"


@timed("fix8_total")
def main() -> int:
    smoke_header(8, "Complexity-bias documentation (no model change)")
    start = time.perf_counter()

    # ── Gates-in ─────────────────────────────────────────────────────────
    check(DOC_PATH.exists(), f"G8.1 doc exists: {DOC_PATH}")

    # ── Body ─────────────────────────────────────────────────────────────
    text = DOC_PATH.read_text(encoding="utf-8")

    check(
        "drop_complexity" in text,
        "G8.2 doc references the drop_complexity flag",
    )

    check(
        bool(re.search(r"Run\s*[78]", text)),
        "G8.3 doc references Run 7 or Run 8 (the ablation source)",
    )

    keywords = ["complexity", "bias", "correlat", "vulnerab"]
    found = [k for k in keywords if k.lower() in text.lower()]
    check(
        len(found) >= 2,
        f"G8.4 doc mentions bias phenomenon (found {len(found)}/4 keywords: {found})",
    )

    elapsed = time.perf_counter() - start
    pass_(f"Fix #8 smoke OK — doc complete, references drop_complexity + Run 7/8 ablation, {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\nSMOKE FIX #8 FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
