"""Tests for A.9 hotspot visualization (src/orchestration/visualizer.py + node)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.visualizer import generate_hotspot_html
from src.orchestration.nodes import visualizer


def _state():
    return {
        "contract_code": "pragma solidity ^0.8.0;\ncontract V {\n  function withdraw() public {\n    msg.sender.call{value: 1}('');\n  }\n}",
        "contract_address": "0xABC",
        "ml_hotspots": [
            {"class": "Reentrancy", "fn_name": "withdraw", "lines": [3, 4], "score": 0.9},
        ],
        "final_report": {
            "overall_verdict": "LIKELY_VULNERABLE",
            "top_vulnerability": "Reentrancy",
            "vulnerability_verdicts": [
                {"vulnerability_class": "Reentrancy", "probability": 0.8, "verdict": "LIKELY", "severity": "High"},
            ],
            "confidence_by_class": {"Reentrancy": 0.85},
            "metric_attribution": {"Reentrancy": {"ml_pct": 40.0, "slither_pct": 50.0, "rag_pct": 10.0}},
        },
    }


class TestGenerateHotspotHtml:
    def test_returns_valid_html_document(self):
        html = generate_hotspot_html(_state())
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert html.count("<body") == 1
        assert html.count("</body>") == 1

    def test_contains_code_and_verdict_panel(self):
        html = generate_hotspot_html(_state())
        assert "withdraw" in html        # source rendered
        assert "Reentrancy" in html      # verdict card
        assert "LIKELY" in html          # verdict badge
        assert "0xABC" in html           # address in header

    def test_hotspot_lines_highlighted(self):
        html = generate_hotspot_html(_state())
        assert 'class="ln hot"' in html  # at least one highlighted line
        assert 'data-fn="withdraw"' in html

    def test_attribution_bars_present(self):
        html = generate_hotspot_html(_state())
        assert "ML 40.0%" in html
        assert "Slither 50.0%" in html

    def test_escapes_html_in_source(self):
        st = _state()
        st["contract_code"] = "contract X { /* <script>alert(1)</script> */ }"
        html = generate_hotspot_html(st)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_empty_state_degrades_gracefully(self):
        html = generate_hotspot_html({})
        assert html.startswith("<!DOCTYPE html>")
        assert "No flagged vulnerabilities" in html


class TestVisualizerNode:
    @pytest.mark.asyncio
    async def test_node_sets_html_and_writes_file(self, tmp_path, monkeypatch):
        import src.orchestration.nodes as nodes_mod
        monkeypatch.setattr(nodes_mod, "REPORTS_DIR", tmp_path)
        out = await visualizer(_state())
        assert out["hotspot_visualization"].startswith("<!DOCTYPE html>")
        written = tmp_path / "0xABC_hotspot.html"
        assert written.exists()
        assert "withdraw" in written.read_text()

    @pytest.mark.asyncio
    async def test_node_no_address_no_file(self, tmp_path, monkeypatch):
        import src.orchestration.nodes as nodes_mod
        monkeypatch.setattr(nodes_mod, "REPORTS_DIR", tmp_path)
        st = _state()
        st["contract_address"] = ""
        out = await visualizer(st)
        assert out["hotspot_visualization"] is not None
        assert list(tmp_path.glob("*.html")) == []
