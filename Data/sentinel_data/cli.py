"""sentinel-data CLI — top-level entry point for the data pipeline.

Stage 0: all stage implementations are placeholders that print what they would do.
Stages 1-7: each stage fills in its real implementation.

Usage:
  sentinel-data --help
  sentinel-data run [--from-stage STAGE] [--config CONFIG] [--dry-run]
  sentinel-data <stage> [--config CONFIG] [--dry-run] [--source SOURCE]

Stages (in pipeline order):
  ingest      Pull raw .sol contracts from all enabled sources
  preprocess  Flatten + compile + dedup + normalize + segment + version-bucket
  represent   Extract graph (.pt) and token files (Stage 2 — currently stub)
  label       Apply crosswalk YAMLs to assign class labels
  verify      AST-level semantic checks + tool corroboration
  split       Deterministic train/val/test splits with leakage audit
  register    Write to SQLite artifact catalog
  analyze     Feature distribution + complexity proxy risk report
  export      Shard export to sentinel-ml seam
"""

import argparse
import sys
import textwrap
from pathlib import Path


STAGES: list[str] = [
    "ingest",
    "preprocess",
    "represent",
    "label",
    "verify",
    "split",
    "register",
    "analyze",
    "export",
]

STAGE_DESCRIPTIONS: dict[str, str] = {
    "ingest":     "Pull raw .sol contracts from all enabled sources with SHA-256 manifests",
    "preprocess": "Flatten + two-pass compile + dedup@0.85 + normalize + segment + version-bucket",
    "represent":  "Extract v9 graph (.pt) and token files [STUB — Stage 2]",
    "label":      "Apply crosswalk YAMLs; merge labels; flag 99%% co-occurrence pairs",
    "verify":     "AST semantic checks + tool corroboration + BCCC Phase 5 regression test",
    "split":      "Deterministic train/val/test splits; leakage auditor = 0",
    "register":   "Write versioned artifact catalog to SQLite",
    "analyze":    "Feature distribution + complexity proxy risk + co-occurrence matrix",
    "export":     "Shard export to sentinel-ml; predictor tier-threshold fix; EMITS edge fix",
}


def _load_config(config_path: str) -> dict:
    import yaml  # type: ignore[import]
    with open(config_path) as f:
        return yaml.safe_load(f)


def _default_config() -> str:
    """Find config.yaml relative to the CLI entry point."""
    here = Path(__file__).resolve().parent.parent  # Data/
    candidate = here / "config.yaml"
    return str(candidate) if candidate.exists() else "config.yaml"


# ── Stage dispatch table (Stage 0: all placeholder) ──────────────────────────

def _run_ingest(args: argparse.Namespace) -> None:
    print(f"[ingest] {STAGE_DESCRIPTIONS['ingest']}")
    print(f"  config : {args.config}")
    if getattr(args, "source", None):
        print(f"  source : {args.source}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 1")


def _run_preprocess(args: argparse.Namespace) -> None:
    print(f"[preprocess] {STAGE_DESCRIPTIONS['preprocess']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 1")


def _run_represent(args: argparse.Namespace) -> None:
    print(f"[represent] {STAGE_DESCRIPTIONS['represent']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 2 (port from ml/src/preprocessing/)")


def _run_label(args: argparse.Namespace) -> None:
    print(f"[label] {STAGE_DESCRIPTIONS['label']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 3")


def _run_verify(args: argparse.Namespace) -> None:
    print(f"[verify] {STAGE_DESCRIPTIONS['verify']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 4")


def _run_split(args: argparse.Namespace) -> None:
    print(f"[split] {STAGE_DESCRIPTIONS['split']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 5")


def _run_register(args: argparse.Namespace) -> None:
    print(f"[register] {STAGE_DESCRIPTIONS['register']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 5")


def _run_analyze(args: argparse.Namespace) -> None:
    print(f"[analyze] {STAGE_DESCRIPTIONS['analyze']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 6")


def _run_export(args: argparse.Namespace) -> None:
    print(f"[export] {STAGE_DESCRIPTIONS['export']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 7")


_STAGE_FN = {
    "ingest":     _run_ingest,
    "preprocess": _run_preprocess,
    "represent":  _run_represent,
    "label":      _run_label,
    "verify":     _run_verify,
    "split":      _run_split,
    "register":   _run_register,
    "analyze":    _run_analyze,
    "export":     _run_export,
}


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel-data",
        description=textwrap.dedent("""\
            SENTINEL data pipeline — build a verified multi-source Solidity contract dataset.

            Stages (in order): ingest → preprocess → represent → label →
                                verify → split → register → analyze → export
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── sentinel-data run ────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Run the full pipeline (or resume from a stage)")
    run_p.add_argument(
        "--from-stage",
        choices=STAGES,
        default=None,
        metavar="STAGE",
        help="Resume from this stage (skips all earlier stages)",
    )
    run_p.add_argument("--config", default=_default_config(), help="Path to config.yaml")
    run_p.add_argument("--dry-run", action="store_true", help="Print planned stages without executing")

    # ── per-stage subcommands ────────────────────────────────────────────────
    for stage in STAGES:
        sp = subparsers.add_parser(stage, help=STAGE_DESCRIPTIONS[stage])
        sp.add_argument("--config", default=_default_config(), help="Path to config.yaml")
        sp.add_argument("--dry-run", action="store_true", help="Print planned action without executing")
        if stage in ("ingest", "preprocess"):
            sp.add_argument(
                "--source",
                default=None,
                metavar="NAME",
                help="Limit to a single source (default: all enabled sources)",
            )

    return parser


# ── `sentinel-data run` handler ───────────────────────────────────────────────

def _handle_run(args: argparse.Namespace) -> None:
    start_idx = STAGES.index(args.from_stage) if args.from_stage else 0
    stages_to_run = STAGES[start_idx:]

    if args.dry_run:
        print(f"[run --dry-run] Would execute {len(stages_to_run)} stage(s):")
        for s in stages_to_run:
            print(f"  {s:<12}  {STAGE_DESCRIPTIONS[s]}")
        return

    for stage in stages_to_run:
        fn = _STAGE_FN[stage]
        # inject dry_run=False and default source=None for the run dispatcher
        stage_args = argparse.Namespace(config=args.config, dry_run=False, source=None)
        fn(stage_args)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        _handle_run(args)
    else:
        fn = _STAGE_FN.get(args.command)
        if fn is None:
            parser.error(f"Unknown command: {args.command}")
        fn(args)


if __name__ == "__main__":
    main()
