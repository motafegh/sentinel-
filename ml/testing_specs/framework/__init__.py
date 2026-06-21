"""
framework/__init__.py — Project-agnostic testing framework.

The ml/testing_specs/ framework is designed to be reusable for any ML project,
not just SENTINEL. The project-specific parts (model adapters, class names,
probes) live in the project's spec files. The generic parts (gates, CLI,
config, reporters) live here.

Public API:
    from ml.testing_specs.framework import gates, cli, config
"""

from .gates import Gate, GateResult, GateStatus

__all__ = ["gates", "cli", "config", "Gate", "GateResult", "GateStatus"]
