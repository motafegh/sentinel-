"""sentinel-data CLI — top-level entry point for the data pipeline.

This module is the **single user-facing surface** for the SENTINEL v2 data
pipeline. Every stage (ingest, preprocess, represent, label, verify, split,
register, analyze, export) is exposed as a subcommand, and a higher-level
``run`` subcommand walks multiple stages in sequence.

Why a CLI (and not just a Python library):

* **Operations**: users running the pipeline don't want to write Python;
  they want ``sentinel-data run --stage represent``.
* **Configuration loading**: the CLI centralises config loading, path
  resolution, dry-run mode, and argument validation in one place.
* **Future DVC integration**: DVC will call the CLI, not the Python API,
  for clean separation of concerns.

Usage::

    sentinel-data --help
    sentinel-data run [--from-stage STAGE] [--config CONFIG] [--dry-run]
    sentinel-data <stage> [--config CONFIG] [--dry-run] [--source SOURCE]

Stages (in pipeline order)::

    ingest      Pull raw .sol contracts from all enabled sources
    preprocess  Flatten + compile + dedup + normalize + segment + version-bucket
    represent   Extract graph (.pt) and windowed token files
    label       Apply crosswalk YAMLs to assign class labels
    verify      AST-level semantic checks + tool corroboration
    split       Deterministic train/val/test splits with leakage audit
    register    Write to SQLite artifact catalog
    analyze     Feature distribution + complexity proxy risk report
    export      Shard export to sentinel-ml seam

sys.path bootstrap
------------------
The first non-comment code in this module adds the SENTINEL repo root and
``ml/`` to ``sys.path``. This is required because the ``representation``
subpackage uses a **thin-adapter** pattern: ``sentinel_data.representation``
re-exports the graph and tokenizer code from ``ml/src/preprocessing/`` and
``ml/src/data_extraction/``. Without these paths on ``sys.path``, the
re-exports fail with ``ModuleNotFoundError`` whenever the CLI is invoked
from outside the repo root (e.g. via an installed entry-point, a Docker
container with a different CWD, or a CI runner).

This is the **only** place in the production code where ``sys.path`` is
manipulated; tests use ``conftest.py`` instead. The two strategies are
intentionally parallel: the CLI must work without pytest, the test suite
must work without the CLI.
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
    "split":      "Deterministic train/val/test splits (4 strategies, 2-pass with dedup_enforcer, NonVulnerable 3:1 cap)",
    "register":   "Register a dataset version in the SQLite catalog (4 base + 2 system tables, YAML mirror)",
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

    sources = [args.source] if getattr(args, "source", None) else [
        k for k, v in (cfg.get("sources_critical_path") or {}).items()
        if isinstance(v, dict) and v.get("enabled", False)
    ]
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
        prep_dir = data_dir / "preprocessed" / source
        if not prep_dir.exists():
            print(f"\n  → source: {source}  (skip — no preprocessed dir)")
            continue
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


def _run_split(args: argparse.Namespace) -> None:
    """Stage 5 — Splitting (the train/val/test boundary)."""
    from sentinel_data.splitting import (
        Contract, apply_dedup_enforcer, apply_nonvulnerable_cap,
        apply_strategy, stratified_split, write_manifest, write_splits,
    )
    import json

    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"

    print(f"[split] {STAGE_DESCRIPTIONS['split']}")
    print(f"  config : {args.config}")
    print(f"  data   : {data_dir}")

    if args.dry_run:
        print("  (dry-run — no splits written)")
        return

    # Read merged labels and build Contract objects
    merged_dir = data_dir / "labels" / "merged"
    if not merged_dir.exists():
        print(f"  ERROR: merged labels dir not found: {merged_dir}")
        print("  Run the labeling stage first: sentinel-data label")
        return

    # Load graph-hash dedup groups if available (fixes Level-3 stub — see deduplicator.py).
    # The file is written by `sentinel-data compute-dedup-groups` after representation.
    # Without it the dedup_enforcer has nothing to act on (all groups are singletons).
    dedup_groups_path = data_dir / "dedup_groups_graph_hash.json"
    cid_to_group: dict[str, str] = {}
    if dedup_groups_path.exists():
        dg_data = json.loads(dedup_groups_path.read_text())
        cid_to_group = dg_data.get("groups", {})
        print(f"  Loaded {len(cid_to_group)} graph-hash dedup groups "
              f"({dg_data.get('n_unique_groups',0)} unique) from {dedup_groups_path.name}")
    else:
        print(f"  WARNING: {dedup_groups_path.name} not found — "
              f"dedup_enforcer will have no groups to act on (Level-3 stub). "
              f"Run: sentinel-data compute-dedup-groups")

    print(f"\n  Loading contracts from {merged_dir}...")
    contracts = []
    for p in sorted(merged_dir.glob("*.labels.json")):
        try:
            lj = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sha = lj["sha256"]
        sources = lj.get("sources") or ["unknown"]
        source = sources[0] if sources else "unknown"
        tier = "T0"
        for cls, entry in lj.get("classes", {}).items():
            if entry.get("value") == 1:
                tier = entry.get("tier") or "T0"
                break
        classes = {cls: entry.get("value", 0) for cls, entry in lj.get("classes", {}).items()}
        primary = next((c for c, e in lj.get("classes", {}).items() if e.get("value") == 1),
                       "NonVulnerable")
        n_pos = sum(1 for e in lj.get("classes", {}).values() if e.get("value") == 1)
        contracts.append(Contract(
            sha256=sha, source=source, tier=tier,
            classes=classes, primary_class=primary, n_pos=n_pos,
            dedup_group=cid_to_group.get(sha),  # None if not in graph-hash groups
        ))
    print(f"  Loaded {len(contracts)} contracts")

    # Run splitter (default: stratified)
    print(f"\n  Splitting (strategy=stratified, seed={args.seed})...")
    splits = stratified_split(contracts, seed=args.seed)

    # Apply dedup_enforcer
    print("  Applying dedup_enforcer (BCCC-failure pattern fix)...")
    apply_dedup_enforcer(splits)

    # Apply NonVulnerable cap
    print(f"  Applying NonVulnerable {args.nonvuln_cap}:1 cap...")
    apply_nonvulnerable_cap(splits, cap=args.nonvuln_cap, seed=args.seed)

    # Write
    out_dir = data_dir / "splits" / f"v{args.version}"
    print(f"\n  Writing splits to {out_dir}...")
    write_splits(splits, out_dir)
    write_manifest(splits, out_dir)

    # Summary
    print(f"\n  ✓ Splitting complete:")
    print(f"    train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
    print(f"    dedup_groups_resolved={splits.metadata.dedup_groups_resolved}")
    nv = sum(1 for s in [splits.train, splits.val, splits.test] for c in s if c.is_nonvulnerable)
    pos = sum(1 for s in [splits.train, splits.val, splits.test] for c in s if not c.is_nonvulnerable)
    if pos > 0:
        print(f"    NonVulnerable:positive ratio = {nv/pos:.2f}:1 (cap={args.nonvuln_cap})")
    print(f"    Manifest: {out_dir}/split_manifest.json")


def _run_register(args: argparse.Namespace) -> None:
    """Stage 5 — Registry (register a dataset version in the catalog)."""
    from sentinel_data.registry import (
        Catalog, DatasetVersion, compute_hash,
    )
    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"

    print(f"[register] {STAGE_DESCRIPTIONS['register']}")
    print(f"  config : {args.config}")
    print(f"  data   : {data_dir}")

    if args.dry_run:
        print("  (dry-run — no catalog write)")
        return

    # Locate the split manifest
    split_dir = data_dir / "splits" / f"v{args.version}"
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.exists():
        print(f"  ERROR: split manifest not found: {manifest_path}")
        print("  Run the splitting stage first: sentinel-data split")
        return

    # Compute the artifact hash (over all train/val/test files in the split)
    print(f"\n  Computing artifact hash over {split_dir}/...")
    h = compute_hash(manifest_path)
    print(f"  Artifact hash: {h[:16]}...")

    # Open the catalog
    db_path = data_dir / "registry" / "catalog.db"
    yaml_path = data_dir / "registry" / "catalog.yaml"
    print(f"\n  Opening catalog at {db_path}...")
    cat = Catalog(db_path, yaml_path)

    # Register the dataset version
    version = DatasetVersion(
        name=args.name,
        source_set=args.sources or [],
        split_version=f"v{args.version}",
        preprocessing_config_hash="",  # TODO: hash of config.yaml in Stage 7
        artifact_hash=h,
        artifact_path=str(split_dir),
        verification_report_path=str(data_dir / "verification" / args.verification_report)
            if args.verification_report else "",
    )
    cat.add_dataset_version(version)
    print(f"  Registered: {args.name}")

    # If the previous version was retired, retire it
    if args.retire_previous:
        try:
            cat.retire_dataset_version(args.retire_previous, superseded_by=args.name,
                                       reason=f"Superseded by {args.name}")
            print(f"  Retired: {args.retire_previous}")
        except Exception as e:
            print(f"  Warning: could not retire {args.retire_previous}: {e}")

    # Write YAML mirror
    cat.write_yaml_mirror()
    print(f"  Wrote YAML mirror: {yaml_path}")

    # Summary
    active = [v for v in cat.list_dataset_versions() if v is not None]
    print(f"\n  ✓ Registration complete. Active dataset versions:")
    for v in active:
        print(f"    - {v.name}  (split_version={v.split_version}, generated_at={v.generated_at})")


def _run_verify(args: argparse.Namespace) -> None:
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

    print("\n  [1/5] class_auditor")
    audit = run_audit(data_dir)
    print(f"    contracts={audit.total_contracts}, flagged_pairs={len(audit.flagged_pairs)}")

    print("\n  [2/5] semantic_checker")
    sem_limit = int(args.semantic_limit_per_class) if args.semantic_limit_per_class else None
    semantic = run_semantic_check(data_dir, limit_per_class=sem_limit)
    print(f"    checked={semantic.total_checked}, skipped={semantic.total_skipped}")

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

    fp_estimation = None
    if not skip_fp:
        print(f"\n  [4/5] fp_estimator (stratified, N={fp_sample}/class)")
        fp_estimation = run_fp_estimation(data_dir, sample_size=fp_sample)
        print(f"    total_sampled={fp_estimation.total_sampled}, "
              f"likely_fp={fp_estimation.total_likely_fp}")
    else:
        print("\n  [4/5] fp_estimator SKIPPED (--skip-fp-estimator)")

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

    print("\n  \u2500\u2500 Gate \u2500\u2500")
    gate = run_gate(
        audit, semantic,
        tool_validation=tool_validation,
        fp_estimation=fp_estimation,
        negative_check=negative_check,
    )
    print(gate)

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

    if gate.gate_passed:
        return 0
    if not args.strict:
        return 0
    return 1


def _resolve_corpus_paths(args: argparse.Namespace, data_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Resolve the (labels_root, rep_root, preproc_root, merged_dir) for analysis.

    By default uses the current build's data dir. With `--corpus <name>`, looks
    up the registered version's `artifact_path` and uses the labels +
    representations + preprocessed subdirs from there. Returns None entries
    (empty Path) if the corpus lookup fails, so callers can degrade gracefully.
    """
    if not args.corpus:
        # Current build
        return (
            data_dir / "labels",
            data_dir / "representations",
            data_dir / "preprocessed",
            data_dir / "labels" / "merged",
        )
    from sentinel_data.registry import Catalog
    cat = Catalog(data_dir / "registry" / "catalog.db",
                  data_dir / "registry" / "catalog.yaml")
    v = cat.get_dataset_version(args.corpus)
    if v is None:
        print(f"  ERROR: corpus version {args.corpus} not found in catalog")
        return Path(), Path(), Path(), Path()
    artifact = Path(v.artifact_path)
    if not artifact.exists():
        print(f"  ERROR: corpus artifact path does not exist: {artifact}")
        return Path(), Path(), Path(), Path()
    return (
        artifact / "labels",
        artifact / "representations",
        artifact / "preprocessed",
        artifact / "labels" / "merged",
    )


def _run_analyze(args: argparse.Namespace) -> None:
    """Stage 6 — Analysis (the Run-9-failure catcher).

    Runs the 6 exploratory tools (5 implemented + probe_dataset re-export)
    against the current build (or a registered `--corpus` version) and
    writes outputs to `data/analysis/<run_id>/`.
    """
    from datetime import datetime

    cfg = _load_config(args.config)
    data_dir = Path(args.config).parent / "data"
    analysis_cfg = (cfg or {}).get("pipeline", {}).get("analysis", {})
    complexity_cfg = analysis_cfg.get("complexity_proxy_risk", {})
    sigma_threshold = float(complexity_cfg.get("sigma_threshold", 1.5))
    cooccur_threshold = float(analysis_cfg.get("cooccurrence", {}).get("flag_threshold", 0.5))
    drift_cfg = analysis_cfg.get("drift", {})
    pvalue_warn = float(drift_cfg.get("ks_pvalue_warn", 0.01))
    min_sample = int(drift_cfg.get("min_sample_size", 30))

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = data_dir / "analysis" / run_id
    print(f"[analyze] {STAGE_DESCRIPTIONS['analyze']}")
    print(f"  config : {args.config}")
    print(f"  data   : {data_dir}")
    print(f"  run-id : {run_id}")
    print(f"  out    : {output_dir}")
    if args.corpus:
        print(f"  corpus : {args.corpus}  (analyzing a registered version, not the current build)")
    if args.baseline_version:
        print(f"  baseline : {args.baseline_version}")

    if args.dry_run:
        print("  (dry-run — no files written)")
        return

    only = args.only
    available = ["balance_viz", "feature_dist", "cooccurrence", "overlap_detector", "drift_monitor"]
    if only and only not in available:
        print(f"  ERROR: --only must be one of {available}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Inputs — resolve from --corpus if given, else current build
    labels_root, rep_root, preproc_root, merged_dir = _resolve_corpus_paths(args, data_dir)
    if not merged_dir or not merged_dir.exists():
        if not args.corpus:
            print(f"  ERROR: merged labels not found at {merged_dir}")
            print("  Run the labeling stage first: sentinel-data label")
        return

    # ── 1. balance_viz ─────────────────────────────────────────────────────
    if not only or only == "balance_viz":
        print("\n  [1/5] balance_viz (per-class/source/tier counts)")
        from sentinel_data.analysis.balance_viz import run_balance_viz
        summary = run_balance_viz(merged_dir, output_dir)
        print(f"    total={summary['total_contracts']} multi-label={summary['multi_label_count']}")
        print(f"    csv:  {summary['csv']}")
        print(f"    plot: {summary['plot']}")

    # ── 2. feature_dist (the headline) ─────────────────────────────────────
    if not only or only == "feature_dist":
        print("\n  [2/5] feature_dist (the Run-9-failure catcher)")
        from sentinel_data.analysis.feature_dist import run_feature_dist
        summary = run_feature_dist(merged_dir, rep_root, preproc_root,
                                   output_dir, sigma_threshold=sigma_threshold)
        print(f"    high_risk_pairs={summary['high_risk_count']}")
        if summary["high_risk_pairs"]:
            for p in summary["high_risk_pairs"][:3]:
                print(f"      {p['class_a']} ↔ {p['class_b']}  "
                      f"({p['feature']} σ={p['sigma_diff']})")
        print(f"    csv:    {summary['csv']}")
        print(f"    plot:   {summary['plot']}")
        print(f"    report: {summary['report']}")

    # ── 3. cooccurrence ────────────────────────────────────────────────────
    if not only or only == "cooccurrence":
        print("\n  [3/5] cooccurrence (directed + conditional matrices)")
        from sentinel_data.analysis.cooccurrence import run_cooccurrence
        summary = run_cooccurrence(merged_dir, output_dir, flag_threshold=cooccur_threshold)
        print(f"    multi_label={summary['multi_label_count']} "
              f"flagged={summary['flagged_count']}")
        for fp in summary["flagged_pairs"][:3]:
            print(f"      {fp['class_a']} ↔ {fp['class_b']}  "
                  f"max P={fp['p_max']:.2%}")
        print(f"    csv:  {summary['csv']}")
        print(f"    plot: {summary['plot']}")

    # ── 4. overlap_detector ────────────────────────────────────────────────
    if not only or only == "overlap_detector":
        print("\n  [4/5] overlap_detector (pairwise source Jaccard)")
        from sentinel_data.analysis.overlap_detector import run_overlap_detector
        summary = run_overlap_detector(labels_root, preproc_root, output_dir)
        print(f"    sources={summary['sources']}")
        for p in summary["top_overlapping_pairs"][:3]:
            print(f"      {p['a']} ↔ {p['b']}  "
                  f"exact={p['exact_jaccard']:.3f} near={p['near_jaccard']:.3f}")
        print(f"    csv:  {summary['csv']}")
        print(f"    plot: {summary['plot']}")

    # ── 5. drift_monitor ───────────────────────────────────────────────────
    if not only or only == "drift_monitor":
        print("\n  [5/5] drift_monitor (KS test for features + labels)")
        from sentinel_data.analysis.drift_monitor import run_drift_monitor
        if args.baseline_version:
            # Look up the baseline's labels + representations + preprocessed
            # from the registry.
            from sentinel_data.registry import Catalog
            try:
                cat = Catalog(data_dir / "registry" / "catalog.db",
                              data_dir / "registry" / "catalog.yaml")
                baseline_v = cat.get_dataset_version(args.baseline_version)
                if baseline_v is None:
                    print(f"    ERROR: baseline version {args.baseline_version} not found in catalog")
                else:
                    baseline_artifact = Path(baseline_v.artifact_path)
                    if not baseline_artifact.exists():
                        print(f"    ERROR: baseline artifact path does not exist: {baseline_artifact}")
                    else:
                        baseline_labels = baseline_artifact / "labels" / "merged"
                        baseline_rep = baseline_artifact / "representations"
                        baseline_preproc = baseline_artifact / "preprocessed"
                        summary = run_drift_monitor(
                            baseline_labels, baseline_rep, baseline_preproc,
                            merged_dir, rep_root, preproc_root,
                            output_dir, pvalue_warn=pvalue_warn, min_sample=min_sample,
                        )
                        print(f"    overall_warning={summary['overall_warning']}")
                        if summary["feature_warnings"]:
                            print(f"    feature warnings: {summary['feature_warnings']}")
                        if summary["label_warnings"]:
                            print(f"    label warnings:   {summary['label_warnings']}")
                        print(f"    report: {summary['report']}")
            except Exception as e:
                print(f"    ERROR loading baseline: {e}")
        else:
            print("    skipped (no --baseline-version given; use --baseline-version <name> "
                  "to compare against a registered dataset version)")

    print(f"\n  ✓ Analysis complete. Outputs in: {output_dir}")


def _run_export(args: argparse.Namespace) -> None:
    from sentinel_data.export import chunk_export

    cfg = _load_config(args.config)
    config_path = Path(args.config)
    data_dir = config_path.parent / "data"
    split_version = getattr(args, "split_version", 1) or 1
    _shard_size_arg = getattr(args, "shard_size", None)
    shard_size = _shard_size_arg if _shard_size_arg is not None else int(
        (cfg.get("pipeline") or {}).get("export_shard_size", 5000)
    )
    dataset_version = getattr(args, "dataset_version", None) or "current-build"
    output_dir = Path(getattr(args, "output_dir", None) or
                      data_dir / "exports" / dataset_version)

    print(f"[export] {STAGE_DESCRIPTIONS['export']}")
    print(f"  config          : {args.config}")
    print(f"  dataset_version : {dataset_version}")
    print(f"  split_version   : v{split_version}")
    print(f"  shard_size      : {shard_size}")
    print(f"  output_dir      : {output_dir}")

    # Resolve enabled sources (same filter as represent/preprocess).
    all_sources = [
        k for k, v in (cfg.get("sources_critical_path") or {}).items()
        if isinstance(v, dict) and v.get("enabled", False)
    ]
    ready_sources, skipped_sources = [], []
    for s in all_sources:
        if (data_dir / "preprocessed" / s).exists():
            ready_sources.append(s)
        else:
            skipped_sources.append({"name": s, "reason": "preprocessed dir not found"})
            print(f"  WARNING: skipping source {s!r} — preprocessed dir not found")

    if not ready_sources:
        print("  ERROR: no sources ready to export")
        return

    print(f"  sources         : {ready_sources}")
    if skipped_sources:
        print(f"  skipped_sources : {[s['name'] for s in skipped_sources]}")

    if args.dry_run:
        print("  (dry-run — no files written)")
        return

    splits_dir = data_dir / "splits" / f"v{split_version}"
    rep_root = data_dir / "representations"
    preproc_root = data_dir / "preprocessed"

    manifest = chunk_export(
        rep_root=rep_root,
        preproc_root=preproc_root,
        splits_dir=splits_dir,
        output_dir=output_dir,
        config_path=config_path,
        shard_size=shard_size,
        source_set=ready_sources,
        skipped_sources=skipped_sources,
    )

    print(f"\n  ✓ Export complete.")
    print(f"    n_contracts     : {manifest.n_contracts}")
    print(f"    n_with_reps     : {manifest.n_contracts_with_reps}")
    print(f"    n_shards        : {manifest.n_shards}")
    print(f"    artifact_hash   : {manifest.artifact_hash[:16]}…")
    print(f"    output_dir      : {output_dir}")


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
        if stage == "split":
            sp.add_argument(
                "--version", type=int, default=1, metavar="N",
                help="Split version number (default 1 = v1)",
            )
            sp.add_argument(
                "--seed", type=int, default=42, metavar="N",
                help="RNG seed for reproducibility (default 42)",
            )
            sp.add_argument(
                "--nonvuln-cap", type=float, default=3.0, metavar="RATIO",
                help="NonVulnerable : positive cap (default 3.0, per friend review §6.3.1)",
            )
        if stage == "register":
            sp.add_argument(
                "--name", required=True, metavar="NAME",
                help="Dataset version name (e.g. sentinel-v2-gold-2026-08)",
            )
            sp.add_argument(
                "--version", type=int, default=1, metavar="N",
                help="Split version to register (default 1 = v1)",
            )
            sp.add_argument(
                "--sources", nargs="+", default=[], metavar="SRC",
                help="Source names that contributed to this version",
            )
            sp.add_argument(
                "--verification-report", default=None, metavar="PATH",
                help="Path to verification_report.md to link",
            )
            sp.add_argument(
                "--retire-previous", default=None, metavar="NAME",
                help="Retire this previous version (marks as superseded)",
            )
        if stage == "analyze":
            sp.add_argument(
                "--only", default=None, metavar="TOOL",
                help="Run only this tool (one of: balance_viz, feature_dist, "
                     "cooccurrence, overlap_detector, drift_monitor)",
            )
            sp.add_argument(
                "--run-id", default=None, metavar="ID",
                help="Analysis run identifier (default: timestamp YYYYMMDD_HHMMSS)",
            )
            sp.add_argument(
                "--corpus", default=None, metavar="VERSION",
                help="Analyze a specific registered dataset version (e.g. "
                     "sentinel-v2-dryrun-2026-08). Default: current build.",
            )
            sp.add_argument(
                "--baseline-version", default=None, metavar="VERSION",
                help="For drift_monitor: compare against this registered "
                     "dataset version (e.g. v1.4-bccc).",
            )
        if stage == "export":
            sp.add_argument(
                "--dataset-version", default=None, metavar="NAME",
                help="Dataset version name for the export directory "
                     "(default: 'current-build'). E.g. 'sentinel-v2-gold-2026-08'.",
            )
            sp.add_argument(
                "--split-version", type=int, default=1, metavar="N",
                help="Split version to export (default 1 = v1).",
            )
            sp.add_argument(
                "--output-dir", default=None, metavar="PATH",
                help="Override output directory (default: data/exports/<dataset-version>/).",
            )
            sp.add_argument(
                "--shard-size", type=int, default=None, metavar="N",
                help="Contracts per shard (default from config: export_shard_size, or 5000).",
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
