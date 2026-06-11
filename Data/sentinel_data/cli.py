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
    """Stage 4 — Verification (the BCCC-failure catcher)."""
    from sentinel_data.verification.class_auditor import run_audit
    from sentinel_data.verification.semantic_checker import run_semantic_check
    from sentinel_data.verification.tool_validator import run_tool_validation
    from sentinel_data.verification.fp_estimator import run_fp_estimation
    from sentinel_data.verification.negative_checker import run_negative_check
    from sentinel_data.verification.gate import run_gate
    from sentinel_data.verification.report_generator import generate_report
    from datetime import datetime

    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"

    print(f"[verify] {STAGE_DESCRIPTIONS['verify']}")
    print(f"  config : {args.config}")
    print(f"  data   : {data_dir}")

    if args.dry_run:
        print("  (dry-run — no Slither runs, no report written)")
        return

    # Configurable thresholds (from config.yaml pipeline.verification)
    verify_cfg = (cfg or {}).get("pipeline", {}).get("verification", {})
    fp_sample = int(verify_cfg.get("fp_sample_size", 50))
    neg_warn = float(verify_cfg.get("negative_tool_hit_warn", 0.05))
    neg_fail = float(verify_cfg.get("negative_tool_hit_threshold", 0.10))
    skip_tool = bool(args.skip_tool_validator)
    skip_fp = bool(args.skip_fp_estimator)
    skip_neg = bool(args.skip_negative_checker)

    corpus_tag = " + ".join(
        (cfg.get("sources_critical_path") or {}).keys()
    ) or "manual"

    # 1) Class audit (per-class counts + co-occurrence matrix)
    print("\n  [1/5] class_auditor")
    audit = run_audit(data_dir)
    print(f"    contracts={audit.total_contracts}, flagged_pairs={len(audit.flagged_pairs)}")

    # 2) Semantic check (graph-feature-based)
    print("\n  [2/5] semantic_checker")
    sem_limit = int(args.semantic_limit_per_class) if args.semantic_limit_per_class else None
    semantic = run_semantic_check(data_dir, limit_per_class=sem_limit)
    print(f"    checked={semantic.total_checked}, skipped={semantic.total_skipped}")

    # 3) Tool validation (Slither agreement) — optional
    tool_validation = None
    if not skip_tool:
        print("\n  [3/5] tool_validator (Slither, may be slow on first run)")
        tool_limit = int(args.tool_limit_per_class) if args.tool_limit_per_class else None
        tool_validation = run_tool_validation(
            data_dir, limit_per_class=tool_limit, force=bool(args.force_slither),
        )
        print(f"    total_checked={tool_validation.total_checkable}, "
              f"agrees={tool_validation.total_agrees}")
    else:
        print("\n  [3/5] tool_validator SKIPPED (--skip-tool-validator)")

    # 4) FP estimator (stratified sampling)
    fp_estimation = None
    if not skip_fp:
        print(f"\n  [4/5] fp_estimator (stratified, N={fp_sample}/class)")
        fp_estimation = run_fp_estimation(data_dir, sample_size=fp_sample)
        print(f"    total_sampled={fp_estimation.total_sampled}, "
              f"likely_fp={fp_estimation.total_likely_fp}")
    else:
        print("\n  [4/5] fp_estimator SKIPPED (--skip-fp-estimator)")

    # 5) Negative checker (NonVulnerable contamination)
    negative_check = None
    if not skip_neg:
        print(f"\n  [5/5] negative_checker (warn>{neg_warn:.0%}, fail>{neg_fail:.0%})")
        neg_limit = int(args.negative_limit) if args.negative_limit else None
        negative_check = run_negative_check(
            data_dir, warn_threshold=neg_warn, fail_threshold=neg_fail, limit=neg_limit,
        )
        print(f"    hit_rate={negative_check.hit_rate}, status={negative_check.status}"
              if negative_check.hit_rate is not None
              else f"    hit_rate=—, status={negative_check.status}")
    else:
        print("\n  [5/5] negative_checker SKIPPED (--skip-negative-checker)")

    # Gate
    print("\n  ── Gate ──")
    gate = run_gate(
        audit, semantic,
        tool_validation=tool_validation,
        fp_estimation=fp_estimation,
        negative_check=negative_check,
    )
    print(gate)

    # Report
    out_path = data_dir / "verification" / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_report(
        audit, semantic, gate,
        tool_validation=tool_validation,
        fp_estimation=fp_estimation,
        negative_check=negative_check,
        corpus_tag=corpus_tag,
        output_path=out_path,
    )
    print(f"\n  Wrote report: {out_path}")

    # Exit code: 0 if gate passes, 1 if any FAIL (unless --strict-off)
    if gate.gate_passed:
        return 0
    if not args.strict:
        return 0   # soft fail — soft gate warns
    return 1


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
        if stage == "verify":
            sp.add_argument(
                "--strict",
                action="store_true",
                help="Exit non-zero on any FAIL (default: warn only, exit 0)",
            )
            sp.add_argument(
                "--semantic-limit-per-class",
                type=int,
                default=None,
                metavar="N",
                help="Semantic check at most N positives per class (fast smoke)",
            )
            sp.add_argument(
                "--tool-limit-per-class",
                type=int,
                default=None,
                metavar="N",
                help="Tool validation at most N checkable positives per class",
            )
            sp.add_argument(
                "--negative-limit",
                type=int,
                default=None,
                metavar="N",
                help="Negative checker at most N NonVulnerable contracts",
            )
            sp.add_argument(
                "--force-slither",
                action="store_true",
                help="Bypass Slither cache and re-run on every contract",
            )
            sp.add_argument(
                "--skip-tool-validator",
                action="store_true",
                help="Skip tool_validator (use only audit + semantic + gate)",
            )
            sp.add_argument(
                "--skip-fp-estimator",
                action="store_true",
                help="Skip fp_estimator (no Slither sampling for FP rate)",
            )
            sp.add_argument(
                "--skip-negative-checker",
                action="store_true",
                help="Skip negative_checker (no Slither run on NonVulnerable)",
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

    failed = False
    for stage in stages_to_run:
        fn = _STAGE_FN[stage]
        # inject defaults that aren't stage-specific; per-stage subcommand parsers
        # define their own --config, --dry-run, etc.
        stage_args = argparse.Namespace(
            config=args.config, dry_run=False, source=None,
            workers=1, sample=None, retry_failed=False,
            # verify-stage defaults
            strict=False, semantic_limit_per_class=None, tool_limit_per_class=None,
            negative_limit=None, force_slither=False,
            skip_tool_validator=False, skip_fp_estimator=False,
            skip_negative_checker=False,
        )
        result = fn(stage_args)
        # Stages may return an int exit code (currently only verify does this).
        if isinstance(result, int) and result != 0:
            failed = True
            print(f"  Stage '{stage}' returned exit code {result}; aborting run.")
            break
    if failed:
        sys.exit(1)


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
        result = fn(args)
        # Per-stage commands can return an int exit code (currently only verify).
        if isinstance(result, int):
            sys.exit(result)


if __name__ == "__main__":
    main()
