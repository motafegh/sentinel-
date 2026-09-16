"""Fresh R4-D-012 guarded-token candidate assembly.

This module never mutates or regenerates the R4-D-011 graph parent. It copies
accepted V10 graph bytes into a fresh root and replaces only the token payload
and declared token/selector metadata using the promoted
``target_aware_guarded_v1`` policy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_SELECTOR_POLICY_VERSION,
    GUARDED_TOKEN_LINEAGE_VERSION,
    PREPROCESSING_ARTIFACT_VERSION,
    TOKEN_TENSOR_SHAPE,
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.representation.r4_target_spans import target_contract_char_spans

PARENT_DECISION_ID = "R4-D-011"
PARENT_TOKEN_LINEAGE = "accepted_v9_byte_copy"
CANDIDATE_MANIFEST_SCHEMA = "sentinel-r4-guarded-token-candidate-source-v1"
TARGET_SPAN_EVIDENCE_SCHEMA = "r4-target-span-evidence-v1"

_COVERAGE_KEYS = (
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


@dataclass(frozen=True)
class GuardedCandidateResult:
    source: str
    contracts_seen: int
    representations_written: int
    representations_failed: int
    guarded_improved: int
    control_fallback: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_bound_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_parent_acceptance(
    acceptance_path: Path,
    *,
    repo_root: Path,
    parent_root: Path,
    preprocessed_root: Path,
) -> dict[str, Any]:
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read R4-D-011 acceptance {acceptance_path}: {exc}") from exc

    if acceptance.get("decision_id") != PARENT_DECISION_ID:
        raise ValueError("guarded candidate requires the R4-D-011 acceptance manifest")
    if acceptance.get("physical_acceptance") is not True:
        raise ValueError("R4-D-011 acceptance manifest does not grant physical acceptance")

    lineage = acceptance.get("accepted_lineage") or {}
    if lineage.get("graph_schema_version") != V10_GRAPH_SCHEMA_VERSION:
        raise ValueError("R4-D-011 parent graph schema mismatch")
    if lineage.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError("R4-D-011 parent extractor mismatch")
    if int(lineage.get("contracts", 0)) != 22540:
        raise ValueError("R4-D-011 accepted population is not 22,540 contracts")

    expected_parent = _resolve_bound_path(
        str(lineage.get("physical_root") or ""),
        repo_root=repo_root,
    )
    if parent_root.resolve() != expected_parent:
        raise ValueError(
            f"parent root is not the accepted R4-D-011 root: "
            f"{parent_root.resolve()} != {expected_parent}"
        )
    expected_preprocessed = _resolve_bound_path(
        str(lineage.get("preprocessed_parent") or ""),
        repo_root=repo_root,
    )
    if preprocessed_root.resolve() != expected_preprocessed:
        raise ValueError(
            "preprocessed root is not the R4-D-011 bound parent: "
            f"{preprocessed_root.resolve()} != {expected_preprocessed}"
        )
    binding_digest = str(lineage.get("binding_digest_sha256") or "")
    if not binding_digest:
        raise ValueError("R4-D-011 acceptance lacks a binding digest")
    return acceptance


def _validate_parent_sidecar(
    sidecar: dict[str, Any],
    *,
    source: str,
    contract_id: str,
) -> tuple[str, ...]:
    if str(sidecar.get("sha256") or "") != contract_id:
        raise ValueError(f"parent sidecar sha256 mismatch for {source}/{contract_id}")
    if str(sidecar.get("source") or "") != source:
        raise ValueError(f"parent sidecar source mismatch for {source}/{contract_id}")
    if sidecar.get("schema_version") != V10_GRAPH_SCHEMA_VERSION:
        raise ValueError(f"parent sidecar graph schema is not v10 for {source}/{contract_id}")
    if sidecar.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise ValueError(f"parent sidecar extractor mismatch for {source}/{contract_id}")
    if sidecar.get("token_lineage") != PARENT_TOKEN_LINEAGE:
        raise ValueError(
            f"parent token lineage is not {PARENT_TOKEN_LINEAGE!r} "
            f"for {source}/{contract_id}"
        )
    targets = tuple(
        str(value)
        for value in (sidecar.get("requested_contract_names") or ())
        if str(value)
    )
    if not targets:
        raise ValueError(f"parent sidecar has no requested_contract_names for {source}/{contract_id}")
    return targets


def _token_payload(
    token_data: dict[str, Any],
    *,
    source: str,
    contract_id: str,
    target_names: tuple[str, ...],
    target_span_evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "input_ids": token_data["input_ids"],
        "attention_mask": token_data["attention_mask"],
        "sha256": contract_id,
        "source": source,
        "num_windows": token_data["num_windows"],
        "stride": token_data["stride"],
        "num_tokens": token_data["num_tokens"],
        "tokenizer_name": token_data["tokenizer_name"],
        "max_length": token_data["max_length"],
        **{key: token_data[key] for key in _COVERAGE_KEYS},
        "token_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
        "selector_policy": GUARDED_SELECTOR_POLICY_VERSION,
        "token_selector": token_data["selector"],
        "requested_contract_names": list(target_names),
        "target_char_spans": token_data["target_char_spans"],
        "target_token_ranges": token_data["target_token_ranges"],
        "target_tokens": token_data["target_tokens"],
        "target_coverage_ratio": token_data["target_coverage_ratio"],
        "control_target_coverage_ratio": token_data["control_target_coverage_ratio"],
        "control_retained_ratio": token_data["control_retained_ratio"],
        "target_span_evidence_sha256": target_span_evidence_sha256,
        "physical_acceptance": False,
        "training_authorized": False,
    }


def build_guarded_candidate_identity(
    *,
    source: str,
    contract_id: str,
    parent_source_dir: Path,
    preprocessed_source_dir: Path,
    output_source_dir: Path,
    tokenizer: Any,
    parent_binding_digest: str,
) -> dict[str, Any]:
    """Assemble one fresh graph/token/sidecar triple from immutable parent bytes."""

    import torch

    parent_graph = parent_source_dir / f"{contract_id}.pt"
    parent_token = parent_source_dir / f"{contract_id}.tokens.pt"
    parent_sidecar = parent_source_dir / f"{contract_id}.rep.json"
    source_path = preprocessed_source_dir / f"{contract_id}.sol"
    for required in (parent_graph, parent_token, parent_sidecar, source_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_graph = output_source_dir / f"{contract_id}.pt"
    output_token = output_source_dir / f"{contract_id}.tokens.pt"
    output_sidecar = output_source_dir / f"{contract_id}.rep.json"
    if any(path.exists() for path in (output_graph, output_token, output_sidecar)):
        raise FileExistsError(f"guarded candidate identity already exists: {source}/{contract_id}")

    sidecar = json.loads(parent_sidecar.read_text(encoding="utf-8"))
    targets = _validate_parent_sidecar(
        sidecar,
        source=source,
        contract_id=contract_id,
    )
    source_text = source_path.read_text(encoding="utf-8")
    char_spans = target_contract_char_spans(source_text, targets)

    from ml.src.data_extraction.guarded_window_tokenizer import tokenize_guarded_source

    token_data = tokenize_guarded_source(
        source_text,
        target_char_spans=char_spans,
        tokenizer=tokenizer,
    )
    if tuple(token_data["input_ids"].shape) != TOKEN_TENSOR_SHAPE:
        raise ValueError(
            f"guarded token shape changed for {source}/{contract_id}: "
            f"{tuple(token_data['input_ids'].shape)}"
        )

    target_span_evidence = {
        "schema": TARGET_SPAN_EVIDENCE_SCHEMA,
        "requested_contract_names": list(targets),
        "target_char_spans": token_data["target_char_spans"],
        "target_token_ranges": token_data["target_token_ranges"],
        "target_tokens": token_data["target_tokens"],
    }
    target_span_evidence_sha256 = _sha256_json(target_span_evidence)

    parent_graph_sha256 = _sha256_file(parent_graph)
    parent_token_sha256 = _sha256_file(parent_token)
    parent_sidecar_sha256 = _sha256_file(parent_sidecar)

    output_source_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        shutil.copyfile(parent_graph, output_graph)
        written.append(output_graph)
        if _sha256_file(output_graph) != parent_graph_sha256:
            raise ValueError(f"graph byte copy mismatch for {source}/{contract_id}")

        torch.save(
            _token_payload(
                token_data,
                source=source,
                contract_id=contract_id,
                target_names=targets,
                target_span_evidence_sha256=target_span_evidence_sha256,
            ),
            output_token,
        )
        written.append(output_token)

        candidate_sidecar = dict(sidecar)
        candidate_sidecar.update(
            {
                "token_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
                "selector_policy": GUARDED_SELECTOR_POLICY_VERSION,
                "token_selector": token_data["selector"],
                "parent_physical_decision_id": PARENT_DECISION_ID,
                "parent_binding_digest_sha256": parent_binding_digest,
                "parent_graph_sha256": parent_graph_sha256,
                "parent_token_sha256": parent_token_sha256,
                "parent_sidecar_sha256": parent_sidecar_sha256,
                "parent_token_lineage": PARENT_TOKEN_LINEAGE,
                "graph_bytes_reused_from_parent": True,
                "target_span_evidence_schema": TARGET_SPAN_EVIDENCE_SCHEMA,
                "target_span_evidence_sha256": target_span_evidence_sha256,
                "target_char_spans": token_data["target_char_spans"],
                "target_token_ranges": token_data["target_token_ranges"],
                "target_tokens": token_data["target_tokens"],
                "target_coverage_ratio": token_data["target_coverage_ratio"],
                "control_target_coverage_ratio": token_data[
                    "control_target_coverage_ratio"
                ],
                "control_retained_ratio": token_data["control_retained_ratio"],
                "physical_acceptance": False,
                "training_authorized": False,
                **{key: token_data[key] for key in _COVERAGE_KEYS},
            }
        )
        output_sidecar.write_text(
            json.dumps(candidate_sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(output_sidecar)
    except Exception:
        for path in reversed(written):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    return {
        "contract_id": contract_id,
        "source": source,
        "selected_window_indices": list(token_data["selected_window_indices"]),
        "used_control_fallback": bool(
            token_data["selector"]["used_control_fallback"]
        ),
        "effective_selector_path": str(token_data["selector"]["effective_path"]),
        "target_span_evidence_sha256": target_span_evidence_sha256,
        "parent_graph_sha256": parent_graph_sha256,
        "candidate_graph_sha256": _sha256_file(output_graph),
    }


def build_guarded_candidate_source(
    source: str,
    *,
    parent_root: Path,
    preprocessed_root: Path,
    output_root: Path,
    acceptance_path: Path,
    repo_root: Path,
    tokenizer: Any | None = None,
    limit: int | None = None,
) -> GuardedCandidateResult:
    """Build one source partition of the fresh guarded-token candidate."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    parent_root = Path(parent_root).resolve()
    preprocessed_root = Path(preprocessed_root).resolve()
    output_root = Path(output_root).resolve()
    acceptance_path = Path(acceptance_path).resolve()
    repo_root = Path(repo_root).resolve()

    if output_root == parent_root:
        raise ValueError("guarded candidate output must not be the R4-D-011 parent root")
    if output_root.name != GUARDED_REPRESENTATION_ROOT_NAME:
        raise ValueError(
            f"guarded candidate output root must be named "
            f"{GUARDED_REPRESENTATION_ROOT_NAME!r}"
        )
    acceptance = _load_parent_acceptance(
        acceptance_path,
        repo_root=repo_root,
        parent_root=parent_root,
        preprocessed_root=preprocessed_root,
    )
    lineage = acceptance["accepted_lineage"]
    parent_binding_digest = str(lineage["binding_digest_sha256"])

    parent_source_dir = parent_root / source
    preprocessed_source_dir = preprocessed_root / source
    if not parent_source_dir.is_dir():
        raise FileNotFoundError(parent_source_dir)
    if not preprocessed_source_dir.is_dir():
        raise FileNotFoundError(preprocessed_source_dir)

    sidecars = sorted(parent_source_dir.glob("*.rep.json"))
    if not sidecars:
        raise ValueError(f"R4-D-011 parent has no sidecars for source {source!r}")
    selected_sidecars = sidecars[:limit] if limit is not None else sidecars
    output_source_dir = output_root / source
    if output_source_dir.exists() and any(output_source_dir.iterdir()):
        raise FileExistsError(
            f"guarded candidate source output is not empty: {output_source_dir}"
        )

    if tokenizer is None:
        from transformers import AutoTokenizer
        from ml.src.data_extraction.windowed_tokenizer import TOKENIZER_MODEL

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL,
            use_fast=True,
            local_files_only=True,
        )

    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for sidecar_path in selected_sidecars:
        contract_id = sidecar_path.name[: -len(".rep.json")]
        try:
            records.append(
                build_guarded_candidate_identity(
                    source=source,
                    contract_id=contract_id,
                    parent_source_dir=parent_source_dir,
                    preprocessed_source_dir=preprocessed_source_dir,
                    output_source_dir=output_source_dir,
                    tokenizer=tokenizer,
                    parent_binding_digest=parent_binding_digest,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "contract_id": contract_id,
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    output_source_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_source_dir / "representation_failures.jsonl"
    if failures:
        with failure_path.open("w", encoding="utf-8") as handle:
            for row in failures:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    fallback = sum(bool(row["used_control_fallback"]) for row in records)
    manifest = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "status": "PHYSICAL_ACCEPTANCE_PENDING",
        "source": source,
        "representation_root": GUARDED_REPRESENTATION_ROOT_NAME,
        "token_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
        "selector_policy": GUARDED_SELECTOR_POLICY_VERSION,
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "preprocessing_artifact_version": PREPROCESSING_ARTIFACT_VERSION,
        "parent_physical_decision_id": PARENT_DECISION_ID,
        "parent_physical_root": str(lineage["physical_root"]),
        "parent_binding_digest_sha256": parent_binding_digest,
        "parent_acceptance_manifest_sha256": _sha256_file(acceptance_path),
        "contracts_available": len(sidecars),
        "contracts_requested": len(selected_sidecars),
        "requested_limit": limit,
        "representations_written": len(records),
        "representations_failed": len(failures),
        "guarded_improved": len(records) - fallback,
        "control_fallback": fallback,
        "frozen_token_shape": list(TOKEN_TENSOR_SHAPE),
        "graph_bytes_reused_from_parent": True,
        "physical_acceptance": False,
        "training_authorized": False,
    }
    (output_source_dir / "guarded_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return GuardedCandidateResult(
        source=source,
        contracts_seen=len(selected_sidecars),
        representations_written=len(records),
        representations_failed=len(failures),
        guarded_improved=len(records) - fallback,
        control_fallback=fallback,
    )


__all__ = [
    "CANDIDATE_MANIFEST_SCHEMA",
    "GuardedCandidateResult",
    "PARENT_DECISION_ID",
    "PARENT_TOKEN_LINEAGE",
    "TARGET_SPAN_EVIDENCE_SCHEMA",
    "build_guarded_candidate_identity",
    "build_guarded_candidate_source",
]
