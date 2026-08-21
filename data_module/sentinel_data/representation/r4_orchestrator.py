"""Phase-8 repaired representation orchestrator.

Historical ``v2.1-windowed-gcb`` representations remain immutable.  This module
builds a new extractor lineage from repaired preprocessing and fails closed on
ambiguous/wrong graph targets while retaining the frozen graph schema and
``[4,512]`` token tensor contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_completeness import (
    require_complete_preprocessed_source,
)
from sentinel_data.preprocessing.r4_versions import (
    PREPROCESSING_ARTIFACT_VERSION,
    REPAIRED_REPRESENTATION_EXTRACTOR_VERSION,
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.target_selector import (
    TargetSelectionError,
    resolve_file_graph_targets,
)
from sentinel_data.representation.r4_compatibility import (
    FULL_ANALYSIS,
    extract_components_with_compatibility,
)

log = logging.getLogger("sentinel_data.r4_orchestrator")
EXTRACTOR_VERSION = REPAIRED_REPRESENTATION_EXTRACTOR_VERSION


@dataclass(frozen=True)
class RepairedRepresentResult:
    source: str
    contracts_seen: int = 0
    representations_written: int = 0
    representations_failed: int = 0
    duration_s: float = 0.0
    extractor_version: str = EXTRACTOR_VERSION


def _resolve_solc_binary(solc_version: str) -> Path | None:
    if not solc_version:
        return None
    path = (
        Path.home()
        / ".solc-select"
        / "artifacts"
        / f"solc-{solc_version}"
        / f"solc-{solc_version}"
    )
    return path if path.exists() else None


def _load_meta(path: Path) -> dict[str, Any]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read repaired preprocessing metadata {path}: {exc}") from exc
    if meta.get("preprocessing_artifact_version") != PREPROCESSING_ARTIFACT_VERSION:
        raise ValueError(
            f"{path} is not repaired preprocessing: "
            f"{meta.get('preprocessing_artifact_version')!r}"
        )
    return meta


def _explicit_target_from_provenance(meta: dict[str, Any]) -> str | None:
    """Return a unique explicit target name carried by ingestion provenance."""

    names: set[str] = set()
    for record in meta.get("source_records", []):
        entry = record.get("ingestion_entry") or {}
        for key in ("target_contract_name", "contract_name", "label_contract_name"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    if len(names) == 1:
        return next(iter(names))
    if len(names) > 1:
        raise TargetSelectionError(
            f"conflicting explicit target provenance: {sorted(names)}"
        )
    return None


def _select_targets(sol_path: Path, meta: dict[str, Any]) -> tuple[str, ...]:
    source = sol_path.read_text(encoding="utf-8", errors="replace")
    return resolve_file_graph_targets(
        source,
        explicit_target=_explicit_target_from_provenance(meta),
        provenance_contract_names=tuple(meta.get("contract_names") or ()),
    )


def _coverage_sidecar(token_data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "coverage_schema_version",
        "pre_subsampling_window_count",
        "pre_subsampling_code_tokens",
        "selected_window_indices",
        "selected_code_token_ranges",
        "retained_unique_code_tokens",
        "retained_token_ratio",
        "content_tokens_per_window",
        "coverage_interpretation",
    )
    return {key: token_data[key] for key in keys}


def _extract_one(
    source: str,
    sol_path: Path,
    meta: dict[str, Any],
    output_dir: Path,
    *,
    graph_schema_version: str = "v9",
    extractor_version: str = EXTRACTOR_VERSION,
    accepted_tokens_dir: Path | None = None,
) -> dict[str, Any]:
    """Build one strict graph/token/sidecar triple."""

    import torch

    from ml.src.data_extraction.windowed_tokenizer import (
        tokenize_windowed_contract_strict,
    )
    sha256 = meta["sha256"]
    targets = _select_targets(sol_path, meta)
    solc_binary = _resolve_solc_binary(meta.get("solc_version", ""))

    started = time.monotonic()
    extraction = extract_components_with_compatibility(
        sol_path,
        targets,
        solc_binary=solc_binary,
        solc_version=str(meta.get("solc_version", "")),
        graph_schema_version=graph_schema_version,
    )
    component_graphs = list(extraction.graphs)
    actual_targets = list(extraction.actual_targets)

    if len(component_graphs) == 1:
        graph = component_graphs[0]
    else:
        from torch_geometric.data import Data

        node_offset = 0
        edge_indexes = []
        node_metadata = []
        for target, component in zip(targets, component_graphs):
            edge_indexes.append(component.edge_index + node_offset)
            node_metadata.extend(
                {**row, "file_graph_component": target}
                for row in component.node_metadata
            )
            node_offset += int(component.num_nodes)
        graph = Data(
            x=torch.cat([item.x for item in component_graphs], dim=0),
            edge_index=torch.cat(edge_indexes, dim=1),
            edge_attr=torch.cat([item.edge_attr for item in component_graphs], dim=0),
        )
        graph.node_metadata = node_metadata
        graph.contract_name = "FILE_UNION:" + "|".join(targets)
        graph.contract_names = list(targets)
        graph.has_cei_path = int(
            any(int(getattr(item, "has_cei_path", 0)) for item in component_graphs)
        )
        graph.num_nodes = int(graph.x.shape[0])
        graph.num_edges = int(graph.edge_index.shape[1])
        if graph_schema_version == V10_GRAPH_SCHEMA_VERSION:
            graph.graph_schema_version = graph_schema_version
            graph.representation_extractor_version = extractor_version
            graph.unclassified_call_ir = [
                row
                for component in component_graphs
                for row in list(getattr(component, "unclassified_call_ir", []) or [])
            ]
            call_names = (
                "HIGH_LEVEL_CALL",
                "LOW_LEVEL_CALL",
                "ETHER_TRANSFER",
                "ETHER_SEND",
                "LIBRARY_CALL",
                "CONTRACT_CREATION",
            )
            graph.classified_call_ir_counts = {
                name: sum(
                    int((getattr(component, "classified_call_ir_counts", {}) or {}).get(name, 0))
                    for component in component_graphs
                )
                for name in call_names
            }
            graph.emitted_call_edge_counts = {
                name: sum(
                    int((getattr(component, "emitted_call_edge_counts", {}) or {}).get(name, 0))
                    for component in component_graphs
                )
                for name in call_names
            }
            graph.call_mapping_errors = [
                row
                for component in component_graphs
                for row in list(getattr(component, "call_mapping_errors", []) or [])
            ]

    # Initial v10 comparison reuses accepted token bytes exactly so graph-call
    # semantics are the only changed representation dimension.
    token_source_path: Path | None = None
    if accepted_tokens_dir is not None:
        token_source_path = accepted_tokens_dir / f"{sha256}.tokens.pt"
        if not token_source_path.is_file():
            raise FileNotFoundError(f"missing accepted token tensor {token_source_path}")
        tokens = torch.load(token_source_path, map_location="cpu", weights_only=True)
        if tokens.get("sha256") != sha256 or tokens.get("source") != source:
            raise ValueError(f"accepted token identity mismatch for {source}/{sha256}")
    else:
        # Repaired preprocessing has already performed lexical comment removal.
        # Re-stripping here would be a second mutation seam, so it is disabled.
        tokens = tokenize_windowed_contract_strict(str(sol_path), strip_comments=False)
    if tuple(tokens["input_ids"].shape) != (4, 512):
        raise ValueError(
            f"frozen token shape changed for {sha256}: {tuple(tokens['input_ids'].shape)}"
        )

    graph_path = output_dir / f"{sha256}.pt"
    token_path = output_dir / f"{sha256}.tokens.pt"
    sidecar_path = output_dir / f"{sha256}.rep.json"

    torch.save(graph, graph_path)
    if token_source_path is not None:
        shutil.copyfile(token_source_path, token_path)
    else:
        torch.save(
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "sha256": sha256,
                "source": source,
                "num_windows": tokens["num_windows"],
                "stride": tokens["stride"],
                "num_tokens": tokens["num_tokens"],
                "tokenizer_name": tokens["tokenizer_name"],
                "max_length": tokens["max_length"],
                **_coverage_sidecar(tokens),
            },
            token_path,
        )

    sidecar = {
        "sha256": sha256,
        "source": source,
        "original_path": meta.get("original_path", ""),
        "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
        "preprocessing_meta_schema_version": meta.get("meta_schema_version"),
        "normalized_text_sha256": meta.get("normalized_text_sha256"),
        "normalized_code_sha256": meta.get("normalized_code_sha256"),
        "dedup_group_id": meta.get("dedup_group_id"),
        "leakage_family_seed": meta.get("leakage_family_seed"),
        "source_record_count": meta.get("source_record_count", 0),
        "schema_version": graph_schema_version,
        "extractor_version": extractor_version,
        "graph_target_policy": "file_level_inheritance_leaf_union_v1",
        "graph_extraction_mode": extraction.mode,
        "graph_analysis_degraded": extraction.analysis_degraded,
        "graph_extraction_fallback_errors": list(extraction.fallback_errors),
        "graph_source_transform": extraction.source_transform,
        "requested_contract_names": list(targets),
        "actual_contract_names": actual_targets,
        "requested_contract_name": targets[0] if len(targets) == 1 else None,
        "actual_contract_name": actual_targets[0] if len(actual_targets) == 1 else None,
        "graph_component_count": len(component_graphs),
        "node_count": int(graph.num_nodes),
        "edge_count": int(graph.num_edges),
        "window_count": int(tokens["num_windows"]),
        "pragma": meta.get("pragma", ""),
        "solc_version": meta.get("solc_version", ""),
        "compute_time_ms": (time.monotonic() - started) * 1000.0,
        **_coverage_sidecar(tokens),
    }
    if graph_schema_version == V10_GRAPH_SCHEMA_VERSION:
        sidecar["token_lineage"] = "accepted_v9_byte_copy"
        sidecar["unclassified_call_ir"] = list(
            getattr(graph, "unclassified_call_ir", []) or []
        )
        sidecar["unclassified_call_ir_count"] = len(sidecar["unclassified_call_ir"])
        sidecar["classified_call_ir_counts"] = dict(
            getattr(graph, "classified_call_ir_counts", {}) or {}
        )
        sidecar["emitted_call_edge_counts"] = dict(
            getattr(graph, "emitted_call_edge_counts", {}) or {}
        )
        sidecar["call_mapping_errors"] = list(
            getattr(graph, "call_mapping_errors", []) or []
        )
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "graph_extraction_mode": extraction.mode,
        "graph_analysis_degraded": extraction.analysis_degraded,
        "graph_source_transform_applied": extraction.source_transform is not None,
    }


def _represent_worker(
    args: tuple[str, ...],
) -> tuple[bool, dict[str, Any] | None, dict[str, str] | None]:
    """Process-safe one-artifact wrapper with explicit failure evidence."""

    source, meta_value, preprocessed_value, output_value, *lineage_values = args
    meta_path = Path(meta_value)
    preprocessed_dir = Path(preprocessed_value)
    output_dir = Path(output_value)
    try:
        meta = _load_meta(meta_path)
        sol_path = preprocessed_dir / f"{meta['sha256']}.sol"
        if not sol_path.exists():
            raise FileNotFoundError(f"missing repaired Solidity artifact {sol_path}")
        if lineage_values:
            graph_schema_version, extractor_version, accepted_tokens_value = lineage_values
            provenance = _extract_one(
                source,
                sol_path,
                meta,
                output_dir,
                graph_schema_version=graph_schema_version,
                extractor_version=extractor_version,
                accepted_tokens_dir=Path(accepted_tokens_value),
            )
        else:
            provenance = _extract_one(source, sol_path, meta, output_dir)
        return True, provenance, None
    except Exception as exc:
        return False, None, {
            "meta_path": meta_path.name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def represent_repaired_source(
    source: str,
    preprocessed_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    verify_completeness: bool = True,
    n_workers: int = 1,
    graph_schema_version: str = "v9",
    extractor_version: str = EXTRACTOR_VERSION,
    accepted_tokens_dir: Path | None = None,
) -> RepairedRepresentResult:
    """Build repaired representations from one versioned preprocessing source.

    Output must be a fresh directory; historical representation trees are never
    overwritten.  Failures are recorded in ``representation_failures.jsonl``.
    """

    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if graph_schema_version == "v9":
        if extractor_version != EXTRACTOR_VERSION:
            raise ValueError("v9 repaired generation requires its frozen extractor")
        if accepted_tokens_dir is not None:
            raise ValueError("v9 repaired generation cannot substitute token lineage")
    elif graph_schema_version == V10_GRAPH_SCHEMA_VERSION:
        if extractor_version != V10_REPRESENTATION_EXTRACTOR_VERSION:
            raise ValueError("v10 generation requires the R4-D-010 extractor identity")
        if output_dir.parent.name != V10_REPRESENTATION_ROOT_NAME:
            raise ValueError(
                f"v10 output must be under {V10_REPRESENTATION_ROOT_NAME!r}"
            )
        if accepted_tokens_dir is None or not accepted_tokens_dir.is_dir():
            raise ValueError("v10 generation requires an accepted v9 token source")
    else:
        raise ValueError(f"unsupported graph schema {graph_schema_version!r}")
    preprocessing_manifest = (
        require_complete_preprocessed_source(source, preprocessed_dir)
        if verify_completeness
        else None
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"repaired representation output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_meta_paths = sorted(preprocessed_dir.glob("*.meta.json"))
    meta_paths = all_meta_paths
    if limit is not None:
        meta_paths = meta_paths[:limit]

    failures: list[dict[str, str]] = []
    started = time.monotonic()
    if graph_schema_version == "v9":
        worker_args = [
            (source, str(path), str(preprocessed_dir), str(output_dir))
            for path in meta_paths
        ]
    else:
        worker_args = [
            (
                source,
                str(path),
                str(preprocessed_dir),
                str(output_dir),
                graph_schema_version,
                extractor_version,
                str(accepted_tokens_dir),
            )
            for path in meta_paths
        ]
    if n_workers == 1:
        results = map(_represent_worker, worker_args)
        results = list(results)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_represent_worker, worker_args, chunksize=1))
    failures.extend(
        failure
        for passed, _, failure in results
        if not passed and failure
    )
    failures.sort(key=lambda row: row["meta_path"])
    written = sum(passed for passed, _, _ in results)
    mode_counts = Counter(
        str(provenance["graph_extraction_mode"])
        for passed, provenance, _ in results
        if passed and provenance
    )

    if failures:
        with (output_dir / "representation_failures.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in failures:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "status": "PHYSICAL_ACCEPTANCE_PENDING",
        "source": source,
        "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
        "extractor_version": extractor_version,
        "contracts_seen": len(meta_paths),
        "preprocessed_artifacts_total": len(all_meta_paths),
        "contracts_requested": len(meta_paths),
        "requested_limit": limit,
        "complete_representation_build": (
            limit is None and preprocessing_manifest is not None
        ),
        "representations_written": written,
        "representations_failed": len(failures),
        "frozen_token_shape": [4, 512],
        "coverage_policy": "telemetry_only_no_adequacy_threshold",
        "representation_workers": n_workers,
        "graph_extraction_mode_counts": dict(sorted(mode_counts.items())),
        "graph_analysis_degraded_total": sum(
            bool(provenance.get("graph_analysis_degraded"))
            for passed, provenance, _ in results
            if passed and provenance
        ),
        "graph_source_transform_total": sum(
            bool(provenance.get("graph_source_transform_applied"))
            for passed, provenance, _ in results
            if passed and provenance
        ),
        "complete_source_build_verified": preprocessing_manifest is not None,
        "preprocessing_manifest_sha256": (
            preprocessing_manifest["manifest_sha256"]
            if preprocessing_manifest is not None
            else None
        ),
    }
    if graph_schema_version == V10_GRAPH_SCHEMA_VERSION:
        manifest.update(
            {
                "graph_schema_version": graph_schema_version,
                "representation_root": V10_REPRESENTATION_ROOT_NAME,
                "token_lineage": "accepted_v9_byte_copy",
                "training_authorized": False,
                "physical_acceptance": False,
            }
        )
    (output_dir / "repaired_representation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return RepairedRepresentResult(
        source=source,
        contracts_seen=len(meta_paths),
        representations_written=written,
        representations_failed=len(failures),
        duration_s=time.monotonic() - started,
        extractor_version=extractor_version,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def recover_failed_representations(
    source: str,
    preprocessed_dir: Path,
    failed_attempt_dir: Path,
    output_dir: Path,
    *,
    n_workers: int = 1,
) -> RepairedRepresentResult:
    """Create a fresh complete root from one explicit failed full attempt.

    Successful triples are byte-reused, while only identities listed in the
    attempt's structured failure file are regenerated.  The failed attempt is
    never modified and its manifest hash is bound into the recovery manifest.
    """

    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    preprocessed_dir = Path(preprocessed_dir)
    failed_attempt_dir = Path(failed_attempt_dir)
    output_dir = Path(output_dir)
    preprocessing_manifest = require_complete_preprocessed_source(
        source, preprocessed_dir
    )
    attempt_manifest_path = (
        failed_attempt_dir / "repaired_representation_manifest.json"
    )
    failures_path = failed_attempt_dir / "representation_failures.jsonl"
    if not attempt_manifest_path.is_file() or not failures_path.is_file():
        raise FileNotFoundError(
            "failed representation recovery requires both the attempt manifest "
            "and representation_failures.jsonl"
        )
    attempt = json.loads(attempt_manifest_path.read_text(encoding="utf-8"))
    if attempt.get("source") != source:
        raise ValueError("failed representation attempt source mismatch")
    if attempt.get("complete_representation_build") is not True:
        raise ValueError("representation recovery requires a full failed attempt")
    if attempt.get("preprocessing_manifest_sha256") != preprocessing_manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("failed attempt/preprocessing manifest binding mismatch")

    all_meta_paths = sorted(preprocessed_dir.glob("*.meta.json"))
    total = len(all_meta_paths)
    attempt_written = int(attempt.get("representations_written", -1))
    attempt_failed = int(attempt.get("representations_failed", -1))
    if (
        int(attempt.get("contracts_requested", -1)) != total
        or int(attempt.get("preprocessed_artifacts_total", -1)) != total
        or attempt_written + attempt_failed != total
        or attempt_failed < 1
    ):
        raise ValueError("failed representation attempt does not reconcile")

    failure_rows = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_ids = [
        str(row.get("meta_path", "")).removesuffix(".meta.json")
        for row in failure_rows
    ]
    if len(failed_ids) != attempt_failed or len(set(failed_ids)) != len(failed_ids):
        raise ValueError("failed representation identities do not reconcile")
    meta_by_id = {path.name.removesuffix(".meta.json"): path for path in all_meta_paths}
    unknown_failures = sorted(set(failed_ids) - set(meta_by_id))
    if unknown_failures:
        raise ValueError(
            f"failed representation identities are absent from preprocessing: {unknown_failures[:5]}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"repaired representation recovery output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    failed_set = set(failed_ids)
    transfer_counts: Counter[str] = Counter()
    reused = 0
    for artifact_id in sorted(set(meta_by_id) - failed_set):
        for suffix in (".pt", ".tokens.pt", ".rep.json"):
            source_path = failed_attempt_dir / f"{artifact_id}{suffix}"
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"failed attempt is missing accepted artifact {source_path.name}"
                )
            transfer_counts[_link_or_copy(source_path, output_dir / source_path.name)] += 1
        reused += 1
    if reused != attempt_written:
        raise ValueError(
            f"failed attempt reuse count mismatch: manifest={attempt_written} physical={reused}"
        )

    worker_args = [
        (source, str(meta_by_id[artifact_id]), str(preprocessed_dir), str(output_dir))
        for artifact_id in sorted(failed_set)
    ]
    if n_workers == 1:
        results = list(map(_represent_worker, worker_args))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_represent_worker, worker_args, chunksize=1))

    recovery_failures = [
        failure
        for passed, _, failure in results
        if not passed and failure
    ]
    recovery_failures.sort(key=lambda row: row["meta_path"])
    recovered = sum(passed for passed, _, _ in results)
    if recovery_failures:
        with (output_dir / "representation_failures.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in recovery_failures:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    recovered_mode_counts = Counter(
        str(provenance["graph_extraction_mode"])
        for passed, provenance, _ in results
        if passed and provenance
    )
    mode_counts = Counter({f"{FULL_ANALYSIS}_reused": reused})
    mode_counts.update(recovered_mode_counts)
    written = reused + recovered
    manifest = {
        "status": "PHYSICAL_ACCEPTANCE_PENDING",
        "source": source,
        "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "contracts_seen": total,
        "preprocessed_artifacts_total": total,
        "contracts_requested": total,
        "requested_limit": None,
        "complete_representation_build": True,
        "representations_written": written,
        "representations_failed": len(recovery_failures),
        "frozen_token_shape": [4, 512],
        "coverage_policy": "telemetry_only_no_adequacy_threshold",
        "representation_workers": n_workers,
        "complete_source_build_verified": True,
        "preprocessing_manifest_sha256": preprocessing_manifest["manifest_sha256"],
        "graph_extraction_mode_counts": dict(sorted(mode_counts.items())),
        "graph_analysis_degraded_total": sum(
            bool(provenance.get("graph_analysis_degraded"))
            for passed, provenance, _ in results
            if passed and provenance
        ),
        "graph_source_transform_total": sum(
            bool(provenance.get("graph_source_transform_applied"))
            for passed, provenance, _ in results
            if passed and provenance
        ),
        "recovery": {
            "schema": "r4-representation-failed-tail-recovery-v1",
            "failed_attempt_manifest_sha256": _sha256_file(attempt_manifest_path),
            "failed_attempt_representations_written": attempt_written,
            "failed_attempt_representations_failed": attempt_failed,
            "reused_representations": reused,
            "retried_representations": len(failed_ids),
            "recovered_representations": recovered,
            "transfer_file_counts": dict(sorted(transfer_counts.items())),
        },
    }
    (output_dir / "repaired_representation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RepairedRepresentResult(
        source=source,
        contracts_seen=total,
        representations_written=written,
        representations_failed=len(recovery_failures),
        duration_s=time.monotonic() - started,
    )
