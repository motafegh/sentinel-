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
  represent   Extract graph (.pt) and windowed token files
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

# Add the SENTINEL repo root and ml/ to sys.path so that thin-adapter imports
# from ml.src.preprocessing.* and ml.src.data_extraction.* resolve correctly.
# The canonical source of truth for the v9 schema lives in ml/; sentinel_data
# re-exports it via thin adapters. Without this, the CLI fails when invoked
# from outside the repo root (e.g. via installed entry-point).
_HERE = Path(__file__).resolve()
_DATA_DIR = _HERE.parent.parent          # sentinel/Data/
_REPO_ROOT = _DATA_DIR.parent            # sentinel/
_ML_ROOT   = _REPO_ROOT / "ml"
for _p in (_REPO_ROOT, _ML_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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
    "represent":  "Extract v9 graph (.pt) + windowed token files (GraphCodeBERT, [4,512])",
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
    from sentinel_data.ingestion.ingest import ingest_all, ingest_source
    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"

    print(f"[ingest] {STAGE_DESCRIPTIONS['ingest']}")
    print(f"  config : {args.config}")

    source = getattr(args, "source", None)
    if source:
        print(f"  source : {source}")
        if args.dry_run:
            print("  (dry-run — no files written)")
        ingest_source(source, cfg, data_dir, dry_run=args.dry_run)
    else:
        if args.dry_run:
            print("  (dry-run — no files written)")
        ingest_all(cfg, data_dir, dry_run=args.dry_run)


def _run_preprocess(args: argparse.Namespace) -> None:
    from sentinel_data.preprocessing.preprocess import preprocess_all, preprocess_source
    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"

    print(f"[preprocess] {STAGE_DESCRIPTIONS['preprocess']}")
    print(f"  config : {args.config}")
    n_workers = getattr(args, "workers", 1)
    sample = getattr(args, "sample", None)
    retry_failed = getattr(args, "retry_failed", False)
    if n_workers > 1:
        print(f"  workers : {n_workers}")
    if sample:
        print(f"  sample  : {sample} files (--sample)")
    if retry_failed:
        print(f"  mode    : retry-failed (re-run only files in dropped.csv)")

    source = getattr(args, "source", None)
    if source:
        print(f"  source : {source}")
        preprocess_source(source, cfg, data_dir, dry_run=args.dry_run,
                          n_workers=n_workers, sample=sample,
                          retry_failed=retry_failed)
    else:
        preprocess_all(cfg, data_dir, dry_run=args.dry_run,
                       n_workers=n_workers, sample=sample,
                       retry_failed=retry_failed)


def _run_represent(args: argparse.Namespace) -> None:
    from sentinel_data.representation.orchestrator import represent_source
    from sentinel_data.representation.versioner import check_and_evict, write_registry, current_versions

    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"
    representations_root = data_dir / "representations"

    print(f"[represent] {STAGE_DESCRIPTIONS['represent']}")
    print(f"  config  : {args.config}")

    schema_v, extractor_v = current_versions()
    print(f"  schema  : {schema_v}  extractor : {extractor_v}")

    sources = [args.source] if getattr(args, "source", None) else list(
        (cfg.get("sources") or {}).keys()
    )
    if not sources:
        print("  No sources configured in config.yaml — nothing to do.")
        return

    limit    = getattr(args, "limit",    None)
    force    = getattr(args, "force",    False)
    emit_cfg = getattr(args, "emit_cfg", False)

    if args.dry_run:
        print(f"  (dry-run) Would represent {len(sources)} source(s): {', '.join(sources)}")
        if limit:
            print(f"  limit   : {limit} contracts per source")
        if force:
            print("  force   : True (cache bypassed)")
        if emit_cfg:
            print("  emit_cfg: True (standalone CFG artifacts will be written)")
        return

    for source in sources:
        print(f"\n  → source: {source}")
        evicted = check_and_evict(representations_root, source, schema_v, extractor_v)
        if evicted:
            print(f"    evicted {evicted} stale cache entries")

        result = represent_source(
            source,
            cfg,
            data_dir,
            limit=limit,
            force=force,
            emit_cfg=emit_cfg,
        )
        print(
            f"    contracts_seen={result.contracts_seen}  "
            f"graphs_written={result.graphs_written}  graphs_cached={result.graphs_cached}  "
            f"graphs_failed={result.graphs_failed}"
        )
        print(
            f"    tokens_written={result.tokens_written}  tokens_cached={result.tokens_cached}  "
            f"tokens_failed={result.tokens_failed}  "
            f"duration={result.duration_s:.1f}s"
        )

    write_registry(representations_root, schema_v, extractor_v)


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


def _run_freshness(args: argparse.Namespace) -> None:
    from sentinel_data.ingestion.freshness import run_freshness_check
    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"
    print("[freshness] Checking source pins vs upstream HEAD + slither-analyzer version...")
    report = run_freshness_check(cfg, data_dir)
    print(report)


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
    "freshness":  _run_freshness,
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
        if stage in ("ingest", "preprocess", "represent"):
            sp.add_argument(
                "--source",
                default=None,
                metavar="NAME",
                help="Limit to a single source (default: all enabled sources)",
            )
        if stage == "represent":
            sp.add_argument(
                "--workers",
                type=int,
                default=1,
                metavar="N",
                help="Multiprocessing pool size (default: 1 = serial)",
            )
            sp.add_argument(
                "--limit",
                type=int,
                default=None,
                metavar="N",
                help="Process only the first N contracts (for fast iteration)",
            )
            sp.add_argument(
                "--force",
                action="store_true",
                help="Recompute even for cache-hit contracts",
            )
            sp.add_argument(
                "--emit-cfg",
                action="store_true",
                dest="emit_cfg",
                help="Write standalone CFG artifact (<sha256>.cfg.json) for each contract",
            )
        if stage == "preprocess":
            sp.add_argument(
                "--workers",
                type=int,
                default=1,
                metavar="N",
                help="Multiprocessing pool size (default: 1 = serial)",
            )
            sp.add_argument(
                "--sample",
                type=int,
                default=None,
                metavar="N",
                help="Process only the first N files (for fast iteration)",
            )
            sp.add_argument(
                "--retry-failed",
                action="store_true",
                help="Re-run only the files listed in the previous dropped.csv "
                     "(merge results — files that now succeed get preprocessed, "
                     "files that still fail stay in dropped.csv with updated errors). "
                     "Use after installing a missing solc version or fixing a config bug.",
            )

    # ── utility subcommands ───────────────────────────────────────────────────
    fresh_p = subparsers.add_parser(
        "freshness",
        help="Check source pin staleness + slither-analyzer version",
    )
    fresh_p.add_argument("--config", default=_default_config(), help="Path to config.yaml")

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
        stage_args = argparse.Namespace(
            config=args.config, dry_run=False, source=None,
            workers=1, sample=None, retry_failed=False,
        )
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
