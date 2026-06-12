"""Tests for report_generator (Stage 4)."""
import tempfile
from pathlib import Path

import pytest

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.class_auditor import run_audit
from sentinel_data.verification.gate import run_gate
from sentinel_data.verification.report_generator import generate_report
from sentinel_data.verification.semantic_checker import run_semantic_check

_DATA_DIR = Path("data_module/data")


def _skip_if_no_merged():
    merged = _DATA_DIR / "labels" / "merged"
    if not merged.exists() or not any(merged.glob("*.labels.json")):
        pytest.skip("Merged labels not found — run merger first")


class TestReportGenerator:
    def test_to_markdown_contains_all_classes(self):
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=5)
        gate = run_gate(audit, semantic)
        report = generate_report(audit, semantic, gate, corpus_tag="test")
        md = report.to_markdown()
        for cls in class_names():
            assert cls in md, f"Class {cls} missing from report"

    def test_report_has_gate_section(self):
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=5)
        gate = run_gate(audit, semantic)
        report = generate_report(audit, semantic, gate)
        md = report.to_markdown()
        assert "Per-Class Gate" in md

    def test_report_has_cooccurrence_section(self):
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=5)
        gate = run_gate(audit, semantic)
        report = generate_report(audit, semantic, gate)
        md = report.to_markdown()
        assert "Co-occurrence" in md

    def test_report_writes_to_file(self, tmp_path):
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=2)
        gate = run_gate(audit, semantic)
        out_path = tmp_path / "verification_report.md"
        generate_report(audit, semantic, gate, output_path=out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert len(content) > 200

    def test_corpus_tag_appears_in_report(self):
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=2)
        gate = run_gate(audit, semantic)
        report = generate_report(audit, semantic, gate, corpus_tag="SolidiFI+DIVE-test")
        md = report.to_markdown()
        assert "SolidiFI+DIVE-test" in md

    def test_no_hard_fails_shows_pass(self):
        _skip_if_no_merged()
        audit = run_audit(_DATA_DIR)
        semantic = run_semantic_check(_DATA_DIR, limit_per_class=2)
        gate = run_gate(audit, semantic)
        report = generate_report(audit, semantic, gate)
        md = report.to_markdown()
        if gate.gate_passed:
            assert "no hard failures" in md.lower() or "No hard failures" in md
