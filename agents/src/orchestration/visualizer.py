"""
visualizer.py — Hotspot attribution visualization (Phase A, A.9).

Renders a self-contained HTML report that puts the audit's evidence in front of
a human: the contract source on the left with vulnerable lines highlighted, and
a per-class verdict panel on the right showing verdict, confidence, and the
metric-attribution breakdown (ML / Slither / RAG). ML hotspots (function-level
GNN attention) drive the line highlighting.

Deliberately dependency-free: pure Python string building with inlined CSS and
a tiny vanilla-JS click handler. No D3/Graphviz/Jinja — so it renders in any
browser, needs no build step, and is trivial to unit-test (string assertions,
no headless browser).

The `visualizer` node calls `generate_hotspot_html(state)` and both stores the
string in state["hotspot_visualization"] and writes it to
data/reports/{address}_hotspot.html.
"""

from __future__ import annotations

import html
from typing import Any

_VERDICT_COLORS = {
    "CONFIRMED": "#c0392b",
    "LIKELY":    "#e67e22",
    "DISPUTED":  "#f1c40f",
    "WATCH":     "#7f8c8d",
    "SAFE":      "#27ae60",
}

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#1e1e1e;color:#ddd}
header{background:#111;padding:14px 20px;border-bottom:2px solid #333}
header h1{margin:0;font-size:18px;color:#fff}
header .meta{font-size:12px;color:#888;margin-top:4px}
.wrap{display:flex;gap:0;height:calc(100vh - 60px)}
.code{flex:1;overflow:auto;background:#1e1e1e;border-right:1px solid #333}
.panel{width:380px;overflow:auto;background:#252526;padding:12px}
pre{margin:0;font-family:SFMono-Regular,Consolas,monospace;font-size:12.5px;line-height:1.5}
.ln{display:block;padding:0 12px;white-space:pre-wrap}
.ln .num{display:inline-block;width:38px;color:#555;user-select:none;text-align:right;margin-right:12px}
.hot{background:rgba(192,57,43,.22);border-left:3px solid #c0392b}
.card{background:#2d2d30;border-radius:6px;padding:10px 12px;margin-bottom:10px;border-left:4px solid #444}
.card h3{margin:0 0 6px;font-size:14px}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;color:#fff;font-weight:600}
.bar{height:8px;border-radius:4px;background:#444;overflow:hidden;display:flex;margin:6px 0}
.bar span{display:block;height:100%}
.bml{background:#3498db}.bsl{background:#9b59b6}.brag{background:#1abc9c}
.legend{font-size:11px;color:#aaa;margin-top:4px}
.conf{font-size:12px;color:#bbb;margin-top:4px}
.empty{color:#777;font-style:italic;padding:20px}
"""

_JS = """
<script>
document.querySelectorAll('.card[data-fn]').forEach(function(c){
  c.style.cursor='pointer';
  c.addEventListener('click',function(){
    var fn=c.getAttribute('data-fn');
    document.querySelectorAll('.ln').forEach(function(l){l.classList.remove('hot')});
    document.querySelectorAll('.ln[data-fn="'+fn+'"]').forEach(function(l){
      l.classList.add('hot'); l.scrollIntoView({block:'center',behavior:'smooth'});
    });
  });
});
</script>
"""


def _line_to_fn(hotspots: list[dict[str, Any]]) -> dict[int, str]:
    """Map each source line number to the function name of the hotspot on it."""
    mapping: dict[int, str] = {}
    for hs in hotspots:
        fn = hs.get("fn_name", "?")
        for ln in hs.get("lines", []) or []:
            try:
                mapping[int(ln)] = fn
            except (TypeError, ValueError):
                continue
    return mapping


def generate_hotspot_html(state: dict[str, Any]) -> str:
    """
    Build a self-contained HTML hotspot report from audit state.

    Reads: contract_code, contract_address, ml_hotspots, final_report
    (vulnerability_verdicts, confidence_by_class, metric_attribution).

    Returns an HTML document string. Never raises on missing fields — absent
    data degrades gracefully to an "(no data)" placeholder.
    """
    code = state.get("contract_code", "") or ""
    address = state.get("contract_address", "") or "unknown"
    hotspots = state.get("ml_hotspots", []) or []
    report = state.get("final_report", {}) or {}
    verdicts = report.get("vulnerability_verdicts", []) or []
    confidence = report.get("confidence_by_class", {}) or state.get("confidence_by_class", {}) or {}
    attribution = report.get("metric_attribution", {}) or state.get("metric_attribution", {}) or {}

    line_fn = _line_to_fn(hotspots)

    # ── Code column: number every line, tag hotspot lines + their function ──
    code_lines = code.splitlines() or ["(no source provided)"]
    rows = []
    for i, raw in enumerate(code_lines, start=1):
        fn = line_fn.get(i)
        cls = "ln hot" if fn else "ln"
        fn_attr = f' data-fn="{html.escape(fn)}"' if fn else ""
        rows.append(
            f'<span class="{cls}"{fn_attr}>'
            f'<span class="num">{i}</span>{html.escape(raw)}</span>'
        )
    code_html = f'<pre>{"".join(rows)}</pre>'

    # ── Panel: one card per verdict ──────────────────────────────────────────
    cards = []
    for v in verdicts:
        cls = v.get("vulnerability_class", "?")
        verdict = v.get("verdict", "?")
        color = _VERDICT_COLORS.get(verdict, "#444")
        conf = confidence.get(cls)
        conf_txt = f"{conf:.0%}" if isinstance(conf, (int, float)) else "n/a"
        attr = attribution.get(cls, {})
        ml_pct = attr.get("ml_pct", 0.0)
        sl_pct = attr.get("slither_pct", 0.0)
        rag_pct = attr.get("rag_pct", 0.0)
        # find a function name to make the card clickable → highlights its lines
        fn = next((h.get("fn_name") for h in hotspots if h.get("class") == cls), None)
        fn_attr = f' data-fn="{html.escape(fn)}"' if fn else ""
        cards.append(
            f'<div class="card"{fn_attr} style="border-left-color:{color}">'
            f'<h3>{html.escape(cls)} '
            f'<span class="badge" style="background:{color}">{html.escape(verdict)}</span></h3>'
            f'<div class="conf">Confidence: <b>{conf_txt}</b> · '
            f'probability {v.get("probability", 0.0):.1%} · {html.escape(str(v.get("severity", "")))}</div>'
            f'<div class="bar">'
            f'<span class="bml" style="width:{ml_pct}%"></span>'
            f'<span class="bsl" style="width:{sl_pct}%"></span>'
            f'<span class="brag" style="width:{rag_pct}%"></span></div>'
            f'<div class="legend">ML {ml_pct}% · Slither {sl_pct}% · RAG {rag_pct}%</div>'
            f"</div>"
        )
    panel_html = "".join(cards) if cards else '<div class="empty">No flagged vulnerabilities.</div>'

    overall = report.get("overall_verdict", "n/a")
    top = report.get("top_vulnerability", "n/a")

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>SENTINEL Hotspots — {html.escape(address)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<header><h1>SENTINEL Audit Hotspots</h1>"
        f"<div class='meta'>Contract: {html.escape(address)} · "
        f"Overall: <b>{html.escape(str(overall))}</b> · "
        f"Top: {html.escape(str(top))} · "
        f"Click a verdict card to highlight its code.</div></header>"
        f"<div class='wrap'><div class='code'>{code_html}</div>"
        f"<div class='panel'>{panel_html}</div></div>"
        f"{_JS}</body></html>"
    )
