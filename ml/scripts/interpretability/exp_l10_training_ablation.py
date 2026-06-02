"""
exp_l10_training_ablation.py — Layer 3, P2: Edge Type Training Ablation
(Command Generator)

PURPOSE
───────
Generate the shell commands and tracking template for a full edge-type training
ablation study. This script does NOT run training itself (each ablation run
takes 1-2 days on this hardware) — instead it:

  1. Reads the current best checkpoint config to extract baseline hyperparams.
  2. Generates 12 training commands: 1 baseline + 11 per-edge-type ablations.
  3. Estimates expected performance impact based on exp_l2 inference ablation
     findings (faster proxy already run).
  4. Creates a CSV template to fill in as runs complete.
  5. Saves a runnable shell script ml/scripts/interpretability/run_training_ablation.sh.

WHY TRAINING ABLATION IS NEEDED
─────────────────────────────────
Inference-time ablation (exp_l2) destroys edges AFTER training — the model
has already learned to compensate (its attention patterns may partially re-route
signal). Training ablation removes the edge type from the data pipeline during
training, forcing the model to develop representations WITHOUT that edge type.
This gives a cleaner measurement of whether the edge type is architecturally
necessary.

WHEN TO USE TRAINING ABLATION VS INFERENCE ABLATION
────────────────────────────────────────────────────
Inference ablation (exp_l2) is faster and good for:
  - Initial ranking of edge type importance
  - Ruling out uninformative edge types
  - Catching distribution shift

Training ablation is needed when:
  - Inference ablation shows an unexpected result (e.g., an edge type appears
    unimportant but the domain says it should matter — training ablation
    confirms whether the model truly learned to use it)
  - Deciding which edge types to REMOVE from the schema (reducing memory/compute)

EDGE TYPES IN SCHEMA v8
────────────────────────
Type 0:  UNKNOWN
Type 1:  AST_PARENT_OF
Type 2:  CONTAINS       (CONTRACT→FUNCTION, FUNCTION→CFG node)
Type 3:  INHERITS
Type 4:  USES
Type 5:  (REVERSE_CONTAINS — built from type 2 by flipping)
Type 6:  CONTROL_FLOW   (CFG intra-function execution order)
Type 7:  DEF_USE        (definition→use data flow)
Type 8:  CALL_ENTRY     (CALL site→called FUNCTION)
Type 9:  RETURN_TO      (callee return→CALL site)
Type 10: DFG_GLOBAL     (global data flow)

REQUIRED TRAIN.PY FLAG (NOT YET IMPLEMENTED)
──────────────────────────────────────────────
The generated commands use a hypothetical --ablate-edge-type N flag. This flag
would zero out all edges of type N in the data collation step. Adding this to
train.py requires a one-line change in the batch assembly in trainer.py:

    if cfg.ablate_edge_type is not None:
        mask = graphs.edge_attr != cfg.ablate_edge_type
        graphs.edge_index = graphs.edge_index[:, mask]
        graphs.edge_attr  = graphs.edge_attr[mask]

This is left as a TODO because exp_l2 results are sufficient for Phase 1.

LAYER / PRIORITY
─────────────────
Layer 3, Priority 2 — Training-time edge ablation planning.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l10_training_ablation.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --out ml/logs/interpretability/l10_training_ablation

OUTPUT
──────
ml/logs/interpretability/l10_training_ablation/
  ablation_commands.txt                    — 12 labelled training commands
  ablation_tracking_template.csv          — fill in F1 as runs complete
  l10_results.json                        — estimated impacts + command list
ml/scripts/interpretability/run_training_ablation.sh  — executable shell script

EXIT CODES
──────────
    0  always (this script generates commands, not run training)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Edge type metadata ─────────────────────────────────────────────────────────

EDGE_TYPES = [
    {"id": 0,  "name": "UNKNOWN",         "phase": "none",   "estimated_impact": "low"},
    {"id": 1,  "name": "AST_PARENT_OF",   "phase": "none",   "estimated_impact": "low"},
    {"id": 2,  "name": "CONTAINS",        "phase": "3",      "estimated_impact": "high"},
    {"id": 3,  "name": "INHERITS",        "phase": "none",   "estimated_impact": "medium"},
    {"id": 4,  "name": "USES",            "phase": "none",   "estimated_impact": "low"},
    {"id": 5,  "name": "REVERSE_CONTAINS","phase": "3",      "estimated_impact": "high"},
    {"id": 6,  "name": "CONTROL_FLOW",    "phase": "2",      "estimated_impact": "high"},
    {"id": 7,  "name": "DEF_USE",         "phase": "none",   "estimated_impact": "medium"},
    {"id": 8,  "name": "CALL_ENTRY",      "phase": "2",      "estimated_impact": "high"},
    {"id": 9,  "name": "RETURN_TO",       "phase": "2",      "estimated_impact": "high"},
    {"id": 10, "name": "DFG_GLOBAL",      "phase": "none",   "estimated_impact": "low"},
]

# Expected F1 impact from inference ablation (exp_l2) findings.
# These are placeholder estimates — fill in real exp_l2 values when available.
EXPECTED_IMPACTS = {
    "UNKNOWN":         "Neutral — no structural role",
    "AST_PARENT_OF":   "Neutral — syntax only, no semantic role in v8",
    "CONTAINS":        "-0.03 to -0.06 F1 — loses CONTRACT→FUNCTION hierarchy",
    "INHERITS":        "-0.01 to -0.03 F1 — minor; few inheritance edges in corpus",
    "USES":            "Neutral — not used in any Phase 2/3 mask",
    "REVERSE_CONTAINS": "-0.04 to -0.08 F1 — loses CFG→FUNCTION upward signal (Phase 3 blind)",
    "CONTROL_FLOW":    "-0.05 to -0.10 F1 — PRIMARY: CEI ordering signal lost",
    "DEF_USE":         "-0.01 to -0.03 F1 — data flow signal for IntegerUO/UnusedReturn",
    "CALL_ENTRY":      "-0.03 to -0.06 F1 — cross-function call signal",
    "RETURN_TO":       "-0.02 to -0.05 F1 — return-site signal",
    "DFG_GLOBAL":      "Neutral — not in Phase 2 mask for this run config",
}


# ── Command generation ─────────────────────────────────────────────────────────

_BASE_COMMAND_TEMPLATE = """\
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. \\
  nohup ml/.venv/bin/python ml/scripts/train.py \\
  --gnn-layers 8 --gnn-prefix-k 48 --gnn-prefix-warmup-epochs 15 \\
  --epochs 60 --batch-size 8 --gradient-accumulation-steps 8 \\
  --loss-fn asl --compile --use-amp --phase2-edge-types 6 8 9 \\
  --jk-entropy-reg-lambda 0.005 \\
  --experiment-name sentinel-ablation-v1 \\
  --run-name {run_name} \\
  {ablate_flag}\\
  >> ml/logs/{log_file} 2>&1 &"""


def generate_command(run_name: str, ablate_edge_type: Optional[int], log_suffix: str) -> str:
    if ablate_edge_type is None:
        ablate_flag = ""
        log_file    = f"ablation-baseline-{log_suffix}.log"
    else:
        ablate_flag = f"--ablate-edge-type {ablate_edge_type} \\\n  "
        log_file    = f"ablation-et{ablate_edge_type}-{log_suffix}.log"

    return _BASE_COMMAND_TEMPLATE.format(
        run_name=run_name,
        ablate_flag=ablate_flag,
        log_file=log_file,
    )


def generate_all_commands(log_suffix: str = "$(date +%Y%m%d)") -> list[dict]:
    commands = []

    # Baseline
    commands.append({
        "label":              "baseline (no ablation)",
        "edge_type_id":       None,
        "edge_type_name":     "NONE",
        "estimated_impact":   "N/A — reference run",
        "run_name":           "ablation-baseline",
        "command":            generate_command("ablation-baseline", None, log_suffix),
    })

    # One ablation per edge type
    for et in EDGE_TYPES:
        run_name = f"ablation-et{et['id']}-{et['name'].lower()}"
        commands.append({
            "label":            f"ablate edge type {et['id']} ({et['name']})",
            "edge_type_id":     et["id"],
            "edge_type_name":   et["name"],
            "estimated_impact": EXPECTED_IMPACTS.get(et["name"], "unknown"),
            "run_name":         run_name,
            "command":          generate_command(run_name, et["id"], log_suffix),
        })

    return commands


# ── CSV template ───────────────────────────────────────────────────────────────

def generate_csv_template(commands: list[dict]) -> str:
    lines = [
        "run_name,edge_type_id,edge_type_name,estimated_impact,"
        "actual_best_f1,actual_best_epoch,delta_vs_baseline,notes"
    ]
    for cmd in commands:
        lines.append(
            f"{cmd['run_name']},{cmd['edge_type_id'] if cmd['edge_type_id'] is not None else ''},"
            f"{cmd['edge_type_name']},\"{cmd['estimated_impact']}\","
            f",,,"
        )
    return "\n".join(lines)


# ── Shell script ───────────────────────────────────────────────────────────────

def generate_shell_script(commands: list[dict]) -> str:
    lines = [
        "#!/bin/bash",
        "# run_training_ablation.sh — Edge-type training ablation study",
        "# Generated by exp_l10_training_ablation.py",
        "#",
        "# IMPORTANT: The --ablate-edge-type flag is NOT yet implemented in train.py.",
        "# Before running, add this to TrainConfig and trainer.py (see exp_l10 docstring).",
        "#",
        "# Run one at a time (each takes ~1-2 days on RTX 3070 8GB).",
        "# Monitor with: tail -f ml/logs/<logfile>",
        "#",
        'set -e',
        'LOG_SUFFIX=$(date +%Y%m%d)',
        'echo "Starting ablation study. LOG_SUFFIX=${LOG_SUFFIX}"',
        "",
    ]
    for cmd in commands:
        lines.append(f"# --- {cmd['label']} ---")
        lines.append(f"# Estimated F1 impact: {cmd['estimated_impact']}")
        lines.append(
            cmd["command"].replace(
                "$(date +%Y%m%d)", "${LOG_SUFFIX}"
            )
        )
        lines.append("")
        lines.append('echo "Launched: ' + cmd["run_name"] + '"')
        lines.append("sleep 5  # brief pause between launches")
        lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate edge-type training ablation commands — Layer 3, P2"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        required=False,
        help="Optional: path to checkpoint to read config from.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for commands + CSV + JSON.",
    )
    args = parser.parse_args()

    out_dir: Optional[Path] = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally read config from checkpoint
    ckpt_config = {}
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            try:
                import torch
                raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    ckpt_config = raw.get("config", {})
                log.info(f"Read config from checkpoint: {list(ckpt_config.keys())}")
            except Exception as exc:
                log.warning(f"Could not read checkpoint config: {exc}")
        else:
            log.warning(f"Checkpoint not found: {ckpt_path}")

    # Generate commands
    commands = generate_all_commands()

    # Print commands to stdout
    print()
    print("=" * 72)
    print("  EDGE-TYPE TRAINING ABLATION COMMANDS (12 runs)")
    print("=" * 72)
    print()
    print("  NOTE: --ablate-edge-type flag is NOT yet in train.py.")
    print("  See exp_l10 docstring for the one-line addition to trainer.py.")
    print()
    print("  Inference ablation (exp_l2) is a faster proxy. Training ablation")
    print("  is only needed if exp_l2 shows unexpected results for a class.")
    print()

    for i, cmd in enumerate(commands):
        print(f"  [{i:2d}] {cmd['label']}")
        print(f"       Expected impact: {cmd['estimated_impact']}")
        print()
        for line in cmd["command"].splitlines():
            print(f"       {line}")
        print()

    print("=" * 72)

    # CSV template
    csv_content = generate_csv_template(commands)

    # Shell script
    sh_content = generate_shell_script(commands)

    # Write files
    if out_dir:
        # Commands text
        cmd_path = out_dir / "ablation_commands.txt"
        with open(cmd_path, "w") as f:
            for cmd in commands:
                f.write(f"# {cmd['label']}\n")
                f.write(f"# Expected: {cmd['estimated_impact']}\n")
                f.write(cmd["command"])
                f.write("\n\n")
        log.info(f"Commands written: {cmd_path}")

        # CSV template
        csv_path = out_dir / "ablation_tracking_template.csv"
        csv_path.write_text(csv_content)
        log.info(f"CSV template written: {csv_path}")

        # JSON
        report = {
            "experiment":  "exp_l10_training_ablation",
            "layer":       3,
            "priority":    2,
            "n_runs":      len(commands),
            "note": (
                "--ablate-edge-type flag not yet in train.py. "
                "See docstring for required trainer.py change. "
                "Use exp_l2 inference ablation as faster proxy."
            ),
            "checkpoint_config": ckpt_config,
            "commands": [
                {k: v for k, v in c.items() if k != "command"}
                for c in commands
            ],
        }
        json_path = out_dir / "l10_results.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"JSON results written: {json_path}")

    # Shell script — written to interpretability dir regardless of --out
    sh_path = Path(__file__).parent / "run_training_ablation.sh"
    sh_path.write_text(sh_content)
    sh_path.chmod(0o755)
    log.info(f"Shell script written: {sh_path}")

    print(f"\nShell script: {sh_path}")
    if out_dir:
        print(f"Output dir:   {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
