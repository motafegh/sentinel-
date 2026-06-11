"""v2 representation orchestrator — reads Stage 1 preprocessed output,
produces graph .pt + tokens .pt + sidecar .rep.json per contract.

Stage 2 (2026-06-10) Day 2 (Task 2.4).

This is a NEW file (not a port of ml/src/data_extraction/ast_extractor.py).
The v1 orchestrator read contracts.parquet (v1-only) and used MD5 hashing
(v1-only). The v2 orchestrator reads Stage 1's preprocessed output and
uses SHA-256 from Stage 1's meta.json.

Public surface:
  - RepresentResult         — dataclass aggregating per-source counts
  - represent_source()      — top-level entry: run for one source

What it does:
  1. Read data/preprocessed/<source>/<sha256>.meta.json for each contract
  2. For each contract, run the graph_extractor (thin adapter) and the
     tokenizer (thin adapter)
  3. Write to data/representations/<source>/<sha256>.{pt, tokens.pt, rep.json}
  4. Honor a content-addressed cache (D-2.5): if <sha256>.rep.json
     exists with matching schema_version, skip.

What's NOT here (deferred):
  - Cache invalidation logic (Task 2.8 — Day 4)
  - CFG / PDG / call-graph / opcode builders (Task 2.7 — Day 4; v3.1 for last 3)
  - Multiprocessing (Task 2.8 — Day 4; serial for now)

What's preserved from v1:
  - graph_extractor's per-file extraction (called via thin adapter, no change)
  - CodeBERT tokenizer + windowed tokens (called via thin adapter, no change)
  - Per-contract compute time tracking (in .rep.json sidecar)

What changed from v1:
  - Input: Stage 1 preprocessed .sol + .meta.json (NOT contracts.parquet)
  - Hash: SHA-256 from meta.json (NOT MD5 from src.utils.hash_utils)
  - Output: <sha256>.{pt,tokens.pt,rep.json} (NOT <md5>.pt)
  - Sidecar: .rep.json with schema_version + extractor_version (NEW)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("sentinel_data.orchestrator")


@dataclass
class RepresentResult:
    """Aggregated counts from a represent_source() run.

    Attributes:
        source:           Source name (e.g. "solidifi")
        contracts_seen:   Number of <sha256>.meta.json files read
        graphs_written:   Number of new .pt graph files written
        graphs_cached:    Number of .pt files that were cache hits
        graphs_failed:    Number of contracts that raised GraphExtractionError
        tokens_written:   Number of new .tokens.pt files written
        tokens_cached:    Number of .tokens.pt files that were cache hits
        tokens_failed:    Number of contracts where tokenize_single_contract returned None
        duration_s:       Wall-clock seconds
        schema_version:   FEATURE_SCHEMA_VERSION at the time of the run
        extractor_version: Version string for the thin-adapter bundle
    """
    source: str
    contracts_seen: int = 0
    graphs_written: int = 0
    graphs_cached: int = 0
    graphs_failed: int = 0
    tokens_written: int = 0
    tokens_cached: int = 0
    tokens_failed: int = 0
    duration_s: float = 0.0
    schema_version: str = ""
    extractor_version: str = "v2.0-thin-adapter"


# Bump this when the orchestrator's behavior changes in a way that
# invalidates existing .pt files (e.g. sidecar schema change). With
# FEATURE_SCHEMA_VERSION, this gives 2 independent cache-invalidation
# triggers: schema drift OR extractor drift.
EXTRACTOR_VERSION = "v2.0-thin-adapter"


def _load_meta(meta_path: Path) -> dict[str, Any] | None:
    """Read a Stage 1 meta.json sidecar. Returns None on parse error."""
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to read {meta_path}: {e}")
        return None


def _is_cached(rep_path: Path, schema_version: str, extractor_version: str) -> bool:
    """Return True if the sidecar exists with matching versions (= cache hit)."""
    if not rep_path.exists():
        return False
    try:
        sidecar = json.loads(rep_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        sidecar.get("schema_version") == schema_version
        and sidecar.get("extractor_version") == extractor_version
    )


def _extract_one(
    source: str,
    sol_path: Path,
    sha256: str,
    output_dir: Path,
    extractor_cfg: dict[str, Any] | None = None,
    force: bool = False,
    emit_cfg: bool = False,
) -> tuple[bool, bool, bool, bool]:
    """Extract graph + tokens for one contract.

    Returns (graph_written, graph_cached, token_written, token_cached).
    """
    from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION

    graph_path = output_dir / f"{sha256}.pt"
    tokens_path = output_dir / f"{sha256}.tokens.pt"
    rep_path = output_dir / f"{sha256}.rep.json"
    compute_t0 = time.monotonic()

    # Load meta from companion file
    meta_path = sol_path.with_suffix(".meta.json")
    meta = _load_meta(meta_path)
    if meta is None:
        log.warning(f"Cannot load meta for {sol_path.name}")
        return False, False, False, False

    # ── Cache check (D-2.5) ────────────────────────────────────────────
    graph_cached = False
    token_cached = False
    if force:
        for p in (graph_path, tokens_path, rep_path):
            if p.exists():
                p.unlink()
    else:
        if _is_cached(rep_path, FEATURE_SCHEMA_VERSION, EXTRACTOR_VERSION):
            if graph_path.exists() and tokens_path.exists():
                return False, True, False, True
            graph_cached = graph_path.exists()
            token_cached = tokens_path.exists()

    # ── Determine solc binary from pragma (D-2.5) ──────────────────
    solc_binary = _resolve_solc_binary(meta.get("solc_version", ""))

    # ── Graph extraction (thin adapter) ─────────────────────────────────
    from sentinel_data.representation.graph_extractor import (
        extract_contract_graph,
        GraphExtractionConfig,
    )

    try:
        config_kwargs: dict[str, Any] = dict(
            allow_paths=[str(sol_path.parent.parent.parent.parent)],
        )
        if solc_binary is not None:
            config_kwargs["solc_binary"] = solc_binary
            config_kwargs["solc_version"] = meta["solc_version"]
        config = GraphExtractionConfig(**config_kwargs)
        data = extract_contract_graph(str(sol_path), config=config)
    except Exception as e:
        log.warning(f"Graph extraction failed for {sol_path.name}: {e}")
        return False, graph_cached, False, token_cached

    # ── Tokenization (thin adapter — windowed, graphcodebert-base) ────────
    from sentinel_data.representation.tokenizer import tokenize_windowed_contract

    token_data = tokenize_windowed_contract(str(sol_path))
    if token_data is None:
        return True, graph_cached, False, token_cached

    # ── Write outputs ─────────────────────────────────────────────────
    import torch

    torch.save(data, graph_path)
    torch.save({
        "input_ids":      token_data["input_ids"],       # [max_windows, 512]
        "attention_mask": token_data["attention_mask"],  # [max_windows, 512]
        "sha256":         sha256,
        "source":         source,
        "num_windows":    token_data["num_windows"],
        "stride":         token_data["stride"],
        "num_tokens":     token_data["num_tokens"],
        "tokenizer_name": token_data["tokenizer_name"],
        "max_length":     token_data["max_length"],
    }, tokens_path)

    compute_time_ms = (time.monotonic() - compute_t0) * 1000

    # ── Write sidecar (D-2.6) ─────────────────────────────────────────
    sidecar = {
        "sha256": sha256,
        "source": source,
        "original_path": meta.get("original_path", ""),
        "schema_version": FEATURE_SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "node_count": int(data.num_nodes),
        "edge_count": int(data.num_edges),
        "window_count": int(token_data["num_windows"]),
        "compute_time_ms": compute_time_ms,
        "cache_hit": False,
        "pragma": meta.get("pragma", ""),
        "solc_version": meta.get("solc_version", ""),
    }
    rep_path.write_text(json.dumps(sidecar, indent=2))

    # ── Optional: standalone CFG artifact (--emit-cfg) ───────────────────
    if emit_cfg:
        try:
            from sentinel_data.representation.cfg_builder import build_cfg
            from sentinel_data.representation.graph_extractor import GraphExtractionConfig
            cfg_kwargs: dict[str, Any] = {"allow_paths": [str(sol_path.parent.parent.parent.parent)]}
            if solc_binary is not None:
                cfg_kwargs["solc_binary"] = solc_binary
                cfg_kwargs["solc_version"] = meta.get("solc_version", "")
            cfg_artifact = build_cfg(sol_path, GraphExtractionConfig(**cfg_kwargs),
                                     sha256=sha256, source=source)
            cfg_path = output_dir / f"{sha256}.cfg.json"
            cfg_path.write_text(json.dumps(cfg_artifact.to_dict(), indent=2))
        except Exception as e:
            log.warning(f"CFG build failed for {sol_path.name}: {e}")

    return True, graph_cached, True, token_cached


def _resolve_solc_binary(solc_version: str) -> Path | None:
    """Find the solc binary for a given version in solc-select's artifacts.

    Returns the absolute path to the binary, or None if the version is
    empty (pragma missing) or the binary isn't installed.
    """
    if not solc_version:
        return None
    # solc-select installs to ~/.solc-select/artifacts/solc-<v>/solc-<v>
    bin_path = Path.home() / ".solc-select" / "artifacts" / f"solc-{solc_version}" / f"solc-{solc_version}"
    return bin_path if bin_path.exists() else None


def represent_source(
    source: str,
    cfg: dict,
    data_dir: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    limit: int | None = None,
    output_dir: Path | None = None,
    emit_cfg: bool = False,
) -> RepresentResult:
    """Run the representation pipeline for one source.

    Reads data/preprocessed/<source>/<sha256>.meta.json + .sol pairs,
    runs the graph_extractor + tokenizer on each, writes outputs to
    data/representations/<source>/<sha256>.{pt, tokens.pt, rep.json}.

    Args:
        source: Source name (must match a key in cfg["sources_critical_path"]).
        cfg: The full sentinel-data config dict.
        data_dir: Path to data/ (parent of raw/, preprocessed/, representations/).
        dry_run: If True, only print the plan without executing.
        force: If True, ignore cache and recompute everything.
        limit: If set, only process the first N contracts (for smoke tests).
        output_dir: Override output directory. If None, defaults to
            data_dir/representations/<source>.

    Returns:
        RepresentResult with per-counter stats.
    """
    from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION

    prep_dir = data_dir / "preprocessed" / source
    out_dir = output_dir if output_dir is not None else data_dir / "representations" / source

    if not prep_dir.exists():
        raise FileNotFoundError(
            f"{prep_dir} not found. Run `sentinel-data preprocess --source {source}` first."
        )

    # Find all meta.json files
    meta_paths = sorted(prep_dir.glob("*.meta.json"))
    if limit:
        meta_paths = meta_paths[:limit]

    if dry_run:
        print(f"[represent] would process {len(meta_paths)} contracts from {source}")
        print(f"  output: {out_dir}")
        print(f"  schema: {FEATURE_SCHEMA_VERSION}, extractor: {EXTRACTOR_VERSION}")
        return RepresentResult(source=source, contracts_seen=len(meta_paths),
                                schema_version=FEATURE_SCHEMA_VERSION)

    out_dir.mkdir(parents=True, exist_ok=True)

    result = RepresentResult(
        source=source,
        schema_version=FEATURE_SCHEMA_VERSION,
    )
    t0 = time.monotonic()

    for i, meta_path in enumerate(meta_paths, 1):
        meta = _load_meta(meta_path)
        if meta is None:
            continue
        sol_path = prep_dir / meta_path.name.replace(".meta.json", ".sol")
        if not sol_path.exists():
            log.warning(f"Companion .sol missing for {meta_path.name}")
            continue

        result.contracts_seen += 1

        g_written, g_cached, t_written, t_cached = _extract_one(
            source, sol_path, meta["sha256"], out_dir, force=force,
            emit_cfg=emit_cfg,
        )
        if g_written:
            result.graphs_written += 1
        elif g_cached:
            result.graphs_cached += 1
        else:
            result.graphs_failed += 1
        if t_written:
            result.tokens_written += 1
        elif t_cached:
            result.tokens_cached += 1
        else:
            result.tokens_failed += 1

        if i % 20 == 0 or i == len(meta_paths):
            print(f"  [{source}] {i}/{len(meta_paths)} "
                  f"(g:+{result.graphs_written}/={result.graphs_cached} "
                  f"t:+{result.tokens_written}/={result.tokens_cached} "
                  f"f:{result.graphs_failed + result.tokens_failed})")

    result.duration_s = time.monotonic() - t0
    return result
