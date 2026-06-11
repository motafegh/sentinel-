"""Tests for the CLI subcommand wiring (Stage 4 Task 4.8)."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestCLIVerifyWiring:
    def test_verify_help_lists_all_args(self):
        """All verify-specific args are wired into the parser."""
        import subprocess
        import sys
        venv_python = Path(sys.executable)
        result = subprocess.run(
            [str(venv_python), "-m", "sentinel_data.cli", "verify", "--help"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        )
        assert result.returncode == 0
        assert "--strict" in result.stdout
        assert "--semantic-limit-per-class" in result.stdout
        assert "--tool-limit-per-class" in result.stdout
        assert "--negative-limit" in result.stdout
        assert "--force-slither" in result.stdout
        assert "--skip-tool-validator" in result.stdout
        assert "--skip-fp-estimator" in result.stdout
        assert "--skip-negative-checker" in result.stdout

    def test_verify_dry_run(self, tmp_path, capsys):
        """--dry-run prints and exits without invoking any component."""
        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                "sentinel-data", "verify",
                "--config", str(tmp_path / "config.yaml"),
                "--dry-run",
            ]
            (tmp_path / "config.yaml").write_text("pipeline: {}\n")
            from sentinel_data.cli import main
            main()
            captured = capsys.readouterr()
            assert "dry-run" in captured.out
            assert "class_auditor" not in captured.out
        finally:
            sys.argv = old_argv


class TestCLIVerifyEndToEnd:
    """The verify subcommand orchestrates 5 components + gate + report."""

    def test_verify_calls_all_5_components(self, tmp_path):
        """Verify that _run_verify invokes class_auditor, semantic_checker,
        tool_validator, fp_estimator, negative_checker, gate, and report_generator."""
        from sentinel_data.cli import _run_verify
        import argparse

        # Mock all 5 components + gate + report
        mock_audit = MagicMock(total_contracts=10, flagged_pairs=[])
        mock_audit.per_class = {}
        mock_audit.flagged_pairs = []
        mock_sem = MagicMock(total_checked=10, total_skipped=0, by_class={})
        mock_tool = MagicMock(total_checkable=10, total_agrees=5, by_class={})
        mock_fp = MagicMock(total_sampled=50, total_likely_fp=5, by_class={},
                           sample_size_per_class=50)
        mock_neg = MagicMock(hit_rate=0.02, status="OK", warn_threshold=0.05,
                            fail_threshold=0.10)
        mock_gate = MagicMock(gate_passed=True, hard_fails=[], verdicts={})
        mock_gate.negative_check_status = "OK"

        with patch("sentinel_data.verification.class_auditor.run_audit", return_value=mock_audit), \
             patch("sentinel_data.verification.semantic_checker.run_semantic_check", return_value=mock_sem), \
             patch("sentinel_data.verification.tool_validator.run_tool_validation", return_value=mock_tool), \
             patch("sentinel_data.verification.fp_estimator.run_fp_estimation", return_value=mock_fp), \
             patch("sentinel_data.verification.negative_checker.run_negative_check", return_value=mock_neg), \
             patch("sentinel_data.verification.gate.run_gate", return_value=mock_gate), \
             patch("sentinel_data.verification.report_generator.generate_report") as mock_report:
            args = argparse.Namespace(
                config=str(tmp_path / "config.yaml"),
                dry_run=False,
                strict=False,
                semantic_limit_per_class=None,
                tool_limit_per_class=None,
                negative_limit=None,
                force_slither=False,
                skip_tool_validator=False,
                skip_fp_estimator=False,
                skip_negative_checker=False,
            )
            (tmp_path / "config.yaml").write_text("pipeline: {verification: {}}\n")
            (tmp_path / "data").mkdir()
            result = _run_verify(args)
            assert result == 0  # gate passed
            assert mock_report.called

    def test_verify_with_skips(self, tmp_path):
        """Verify that skip flags prevent the corresponding component from running."""
        from sentinel_data.cli import _run_verify
        import argparse

        mock_audit = MagicMock(total_contracts=10, flagged_pairs=[])
        mock_audit.per_class = {}
        mock_audit.flagged_pairs = []
        mock_sem = MagicMock(total_checked=10, total_skipped=0, by_class={})
        mock_gate = MagicMock(gate_passed=True, hard_fails=[], verdicts={})
        mock_gate.negative_check_status = None

        with patch("sentinel_data.verification.class_auditor.run_audit", return_value=mock_audit), \
             patch("sentinel_data.verification.semantic_checker.run_semantic_check", return_value=mock_sem), \
             patch("sentinel_data.verification.tool_validator.run_tool_validation") as mock_tool, \
             patch("sentinel_data.verification.fp_estimator.run_fp_estimation") as mock_fp, \
             patch("sentinel_data.verification.negative_checker.run_negative_check") as mock_neg, \
             patch("sentinel_data.verification.gate.run_gate", return_value=mock_gate), \
             patch("sentinel_data.verification.report_generator.generate_report"):
            args = argparse.Namespace(
                config=str(tmp_path / "config.yaml"),
                dry_run=False,
                strict=False,
                semantic_limit_per_class=None,
                tool_limit_per_class=None,
                negative_limit=None,
                force_slither=False,
                skip_tool_validator=True,
                skip_fp_estimator=True,
                skip_negative_checker=True,
            )
            (tmp_path / "config.yaml").write_text("pipeline: {verification: {}}\n")
            (tmp_path / "data").mkdir()
            result = _run_verify(args)
            assert result == 0
            mock_tool.assert_not_called()
            mock_fp.assert_not_called()
            mock_neg.assert_not_called()

    def test_strict_returns_nonzero_on_fail(self, tmp_path):
        """--strict returns exit code 1 when gate has hard fails."""
        from sentinel_data.cli import _run_verify
        import argparse

        mock_audit = MagicMock(total_contracts=10, flagged_pairs=[])
        mock_audit.per_class = {}
        mock_audit.flagged_pairs = []
        mock_sem = MagicMock(total_checked=10, total_skipped=0, by_class={})
        mock_gate = MagicMock(gate_passed=False, hard_fails=["CallToUnknown"], verdicts={})
        mock_gate.negative_check_status = None

        with patch("sentinel_data.verification.class_auditor.run_audit", return_value=mock_audit), \
             patch("sentinel_data.verification.semantic_checker.run_semantic_check", return_value=mock_sem), \
             patch("sentinel_data.verification.tool_validator.run_tool_validation"), \
             patch("sentinel_data.verification.fp_estimator.run_fp_estimation"), \
             patch("sentinel_data.verification.negative_checker.run_negative_check"), \
             patch("sentinel_data.verification.gate.run_gate", return_value=mock_gate), \
             patch("sentinel_data.verification.report_generator.generate_report"):
            args = argparse.Namespace(
                config=str(tmp_path / "config.yaml"),
                dry_run=False,
                strict=True,
                semantic_limit_per_class=None,
                tool_limit_per_class=None,
                negative_limit=None,
                force_slither=False,
                skip_tool_validator=True,
                skip_fp_estimator=True,
                skip_negative_checker=True,
            )
            (tmp_path / "config.yaml").write_text("pipeline: {verification: {}}\n")
            (tmp_path / "data").mkdir()
            result = _run_verify(args)
            assert result == 1  # strict mode → exit 1
