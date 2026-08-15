"""Phase-8 repaired representation orchestrator.

Historical ``v2.1-windowed-gcb`` representations remain immutable.  This module
builds a new extractor lineage from repaired preprocessing and fails closed on
ambiguous/wrong graph targets while retaining the frozen graph schema and
``[4,512]`` token tensor contract.
"""

from __future__ import annotations

import json
import logging
import time
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
)
from sentinel_data.representation.target_selector import (
    TargetSelectionError,
    resolve_file_graph_targets,
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
) -> None:
    """Build one strict graph/token/sidecar triple."""

    import torch

    from ml.src.data_extraction.windowed_tokenizer import (
        tokenize_windowed_contract_strict,
    )
    from sentinel_data.representation.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
    )
    from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION

    sha256 = meta["sha256"]
    targets = _select_targets(sol_path, meta)
    solc_binary = _resolve_solc_binary(meta.get("solc_version", ""))

    started = time.monotonic()
    component_graphs = []
    actual_targets: list[str] = []
    for target in targets:
        config_kwargs: dict[str, Any] = {
            "multi_contract_policy": "by_name",
            "target_contract_name": target,
            "allow_paths": str(sol_path.parent),
        }
        if solc_binary is not None:
            config_kwargs["solc_binary"] = solc_binary
            config_kwargs["solc_version"] = meta.get("solc_version", "")
        component = extract_contract_graph(
            sol_path,
            config=GraphExtractionConfig(**config_kwargs),
        )
        actual = str(getattr(component, "contract_name", ""))
        if actual != target:
            raise TargetSelectionError(
                f"graph target mismatch for {sha256}: requested={target!r}, actual={actual!r}"
            )
        component_graphs.append(component)
        actual_targets.append(actual)

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
        "schema_version": FEATURE_SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "graph_target_policy": "file_level_inheritance_leaf_union_v1",
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
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _represent_worker(
    args: tuple[str, str, str, str],
) -> tuple[bool, dict[str, str] | None]:
    """Process-safe one-artifact wrapper with explicit failure evidence."""

    source, meta_value, preprocessed_value, output_value = args
    meta_path = Path(meta_value)
    preprocessed_dir = Path(preprocessed_value)
    output_dir = Path(output_value)
    try:
        meta = _load_meta(meta_path)
        sol_path = preprocessed_dir / f"{meta['sha256']}.sol"
        if not sol_path.exists():
            raise FileNotFoundError(f"missing repaired Solidity artifact {sol_path}")
        _extract_one(source, sol_path, meta, output_dir)
        return True, None
    except Exception as exc:
        return False, {
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
) -> RepairedRepresentResult:
    """Build repaired representations from one versioned preprocessing source.

    Output must be a fresh directory; historical representation trees are never
    overwritten.  Failures are recorded in ``representation_failures.jsonl``.
    """

    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
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
    worker_args = [
        (source, str(path), str(preprocessed_dir), str(output_dir))
        for path in meta_paths
    ]
    if n_workers == 1:
        results = map(_represent_worker, worker_args)
        results = list(results)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_represent_worker, worker_args, chunksize=1))
    failures.extend(failure for passed, failure in results if not passed and failure)
    failures.sort(key=lambda row: row["meta_path"])
    written = sum(passed for passed, _ in results)

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
        "extractor_version": EXTRACTOR_VERSION,
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
        "complete_source_build_verified": preprocessing_manifest is not None,
        "preprocessing_manifest_sha256": (
            preprocessing_manifest["manifest_sha256"]
            if preprocessing_manifest is not None
            else None
        ),
    }
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
    )
