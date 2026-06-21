"""
framework/config.py — YAML config loading for the testing framework.

The testing framework is configured via a YAML file that specifies:
  - Which gates to run
  - Which probes to use (or override defaults)
  - Which paths to check
  - Pass/fail criteria

For SENTINEL, the config is at `framework/templates/sentinel_v2.yaml`.
For new projects, copy that template and customize.

Usage:
    from ml.testing_specs.framework.config import load_config, render_template

    config = load_config("path/to/config.yaml")
    print(config["gates"]["behavioral_probes"]["checkpoint"])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# Make this work whether imported as ml.testing_specs.framework.config
# or run as a script (python ml/testing_specs/framework/config.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TEMPLATE_SENTINEL_V2 = """\
# SENTINEL v2 testing framework config
# This file is the single source of truth for what gates run, what probes
# are used, and what paths are checked.
#
# Edit the values below to customize for your run/project. See
# ml/testing_specs/MIGRATION.md for adapting to a new project.

project: sentinel
run_name: GCB-P1-Run12-v3dospatched-20260613
schema_version: v9
num_classes: 10

# Paths
paths:
  checkpoint: ml/checkpoints/${run_name}_FINAL.pt
  thresholds: ml/checkpoints/${run_name}_FINAL_thresholds.json
  behavioral_probes: ml/checkpoints/${run_name}_FINAL_behavioral_probes.json
  label_quality: ml/checkpoints/v3_label_quality.json
  export: data_module/data/exports/sentinel-v3-smartbugs-2026-06-13
  val_log: ml/logs/${run_name}/epoch_summary.jsonl

# Gates (any of these can be skipped with `enabled: false`)
gates:
  file_exists:
    enabled: true
  behavioral_probes:
    enabled: true
    all_passed_required: true
  label_quality:
    enabled: true
    no_fail_required: true
  f1_vs_prior:
    enabled: true
    min_improvement: 0.0
  contamination:
    enabled: true
    tiers: [1a, 1b, 2, 3, 4]
  calibration_files:
    enabled: true
    required: [thresholds]

# Thresholds
thresholds:
  label_quality_max_positive_rate: 0.50
  label_quality_min_positive_rate: 0.01
  label_quality_max_source_dominance: 0.80
  label_quality_max_co_occurrence: 0.60
  behavioral_probe_default_max_prob: 0.30
  behavioral_probe_default_min_prob: 0.40

# Output
output:
  report_path: ml/checkpoints/${run_name}_gate_report.json
  log_level: INFO
"""


# ---------------------------------------------------------------------------
# Config dataclass + loader
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Configuration for one gate.

    Accepts arbitrary extra fields (e.g., 'all_passed_required', 'min_improvement')
    — they're stored in the `extra` dict for later use.
    """

    name: str
    enabled: bool = True
    extra: dict = field(default_factory=dict)

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        self.name = name
        self.enabled = enabled
        self.extra = kwargs


@dataclass
class FrameworkConfig:
    """Top-level framework configuration."""

    project: str = "sentinel"
    run_name: str = ""
    schema_version: str = "v9"
    num_classes: int = 10
    paths: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)  # name -> GateConfig
    thresholds: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Dotted-key lookup, e.g., 'paths.checkpoint'."""
        keys = key.split(".")
        v: Any = self
        for k in keys:
            if isinstance(v, dict) and k in v:
                v = v[k]
            elif hasattr(v, k):
                v = getattr(v, k)
            else:
                return default
        return v

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        if not _HAS_YAML:
            return json.dumps(self.to_dict(), indent=2)
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameworkConfig":
        gates = {}
        for name, gd in d.get("gates", {}).items():
            gates[name] = GateConfig(name=name, **(gd or {}))
        return cls(
            project=d.get("project", "sentinel"),
            run_name=d.get("run_name", ""),
            schema_version=d.get("schema_version", "v9"),
            num_classes=d.get("num_classes", 10),
            paths=d.get("paths", {}),
            gates=gates,
            thresholds=d.get("thresholds", {}),
            output=d.get("output", {}),
        )


def load_config(path: Path) -> FrameworkConfig:
    """Load a YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not _HAS_YAML:
        raise ImportError(
            "PyYAML required for config loading. pip install pyyaml"
        )
    with path.open() as f:
        data = yaml.safe_load(f)
    return FrameworkConfig.from_dict(data or {})


def render_template(template_name: str = "sentinel_v2") -> str:
    """Render a built-in template to stdout (for new-project bootstrap)."""
    if template_name == "sentinel_v2":
        return TEMPLATE_SENTINEL_V2
    raise ValueError(f"Unknown template: {template_name}. Available: sentinel_v2")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="testing_config",
        description="Config tool for the testing framework.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Render a config template")
    init_p.add_argument("--template", default="sentinel_v2",
                        help="Template name (default: sentinel_v2)")
    init_p.add_argument("--output", "-o", type=Path, default=None,
                        help="Write to file instead of stdout")

    validate_p = sub.add_parser("validate", help="Validate a config file")
    validate_p.add_argument("path", type=Path)

    args = parser.parse_args()
    if args.cmd == "init":
        out = render_template(args.template)
        if args.output:
            args.output.write_text(out)
            print(f"Wrote {args.template} template to {args.output}")
        else:
            print(out)
        return 0
    if args.cmd == "validate":
        try:
            cfg = load_config(args.path)
        except Exception as e:
            print(f"FAIL: {e}")
            return 1
        print(f"OK: {args.path} is valid")
        print(f"  project: {cfg.project}")
        print(f"  run_name: {cfg.run_name}")
        print(f"  gates: {len(cfg.gates)} ({', '.join(cfg.gates.keys())})")
        print(f"  paths: {len(cfg.paths)} keys")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
