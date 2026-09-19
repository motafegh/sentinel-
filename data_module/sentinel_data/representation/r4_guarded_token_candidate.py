"""Fresh R4-D-012 guarded-token candidate construction.

This module constructs a new token lineage from the immutable R4-D-011 V10
graph parent. It deliberately does not modify the accepted V10 builder, does not
grant physical acceptance, and does not authorize training.

The selector implementation is reused from
ml.src.data_extraction.bounded_window_selector, whose retained source is part of
the R4-D-012 decision evidence. Invalid or ambiguous target evidence fails
closed; historical-control fallback is reserved for valid target evidence where
the guarded candidate does not strictly improve target-token coverage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sentinel_data.preprocessing.r4_versions import (
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_TOKEN_LINEAGE_VERSION,
    GUARDED_TOKEN_SELECTOR_SCHEMA_VERSION,
    GUARDED_TOKEN_SELECTOR_VERSION,
    HISTORICAL_TOKEN_SELECTOR_VERSION,
    TOKEN_TENSOR_SHAPE,
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.r4_target_spans import target_contract_char_spans

R4_D011_DECISION_ID = "R4-D-011"
R4_D011_ACCEPTANCE_SCHEMA = "sentinel-r4-v10-v26-physical-acceptance-v1"
GUARDED_CANDIDATE_MANIFEST_SCHEMA = "sentinel-r4-guarded-token-candidate-manifest-v1"
GUARDED_CANDIDATE_STATUS_BOUNDED = "BOUNDED_GUARDED_TOKEN_CANDIDATE"


class GuardedTokenCandidateError(RuntimeError):
    """Base error for guarded-token candidate construction."""


class TargetEvidenceError(GuardedTokenCandidateError):
    """Raised when target-aware evidence cannot be established safely."""


@dataclass(frozen=True)
class AcceptedV10Parent:
    """Validated identity of the immutable R4-D-011 physical parent."""

    decision_id: str
    physical_root: str
    binding_digest_sha256: str
    contracts: int
    graph_schema_version: str
    extractor_version: str


@dataclass(frozen=True)
class GuardedBuildResult:
    """Summary of one guarded-token representation triple."""

    source: str
    contract_id: str
    effective_selector: str
    used_control_fallback: bool
    selected_window_indices: tuple[int, ...]
    graph_sha256: str
    tokens_sha256: str
    sidecar_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _validate_contract_id(contract_id: str) -> None:
    if len(contract_id) != 64 or any(ch not in "0123456789abcdef" for ch in contract_id):
        raise GuardedTokenCandidateError(
            f"invalid lowercase sha256 contract identity: {contract_id!r}"
        )


def load_accepted_v10_parent(
    *,
    acceptance_path: Path,
    repo_root: Path,
    parent_root: Path,
) -> AcceptedV10Parent:
    """Validate that parent_root is exactly the accepted R4-D-011 root."""

    acceptance_path = Path(acceptance_path)
    repo_root = Path(repo_root).resolve()
    parent_root = Path(parent_root).resolve()
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardedTokenCandidateError(
            f"cannot read R4-D-011 acceptance record {acceptance_path}: {exc}"
        ) from exc

    if acceptance.get("schema") != R4_D011_ACCEPTANCE_SCHEMA:
        raise GuardedTokenCandidateError("parent acceptance schema is not R4-D-011 V10 V2.6")
    if acceptance.get("decision_id") != R4_D011_DECISION_ID:
        raise GuardedTokenCandidateError("parent acceptance decision is not R4-D-011")
    if acceptance.get("physical_acceptance") is not True:
        raise GuardedTokenCandidateError("R4-D-011 parent is not physically accepted")
    if acceptance.get("status") != "PASS":
        raise GuardedTokenCandidateError("R4-D-011 acceptance record is not PASS")

    lineage = dict(acceptance.get("accepted_lineage") or {})
    expected_root = (repo_root / str(lineage.get("physical_root") or "")).resolve()
    if parent_root != expected_root:
        raise GuardedTokenCandidateError(
            f"parent root is not the exact R4-D-011 physical root: "
            f"{parent_root} != {expected_root}"
        )
    if parent_root.name != V10_REPRESENTATION_ROOT_NAME:
        raise GuardedTokenCandidateError(
            f"R4-D-011 parent root basename changed: {parent_root.name!r}"
        )
    if lineage.get("graph_schema_version") != V10_GRAPH_SCHEMA_VERSION:
        raise GuardedTokenCandidateError("R4-D-011 graph schema identity mismatch")
    if lineage.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
        raise GuardedTokenCandidateError("R4-D-011 extractor identity mismatch")

    digest = str(lineage.get("binding_digest_sha256") or "")
    contracts = int(lineage.get("contracts", 0))
    if len(digest) != 64 or contracts < 1:
        raise GuardedTokenCandidateError("R4-D-011 acceptance lineage is incomplete")

    return AcceptedV10Parent(
        decision_id=R4_D011_DECISION_ID,
        physical_root=str(lineage["physical_root"]),
        binding_digest_sha256=digest,
        contracts=contracts,
        graph_schema_version=V10_GRAPH_SCHEMA_VERSION,
        extractor_version=V10_REPRESENTATION_EXTRACTOR_VERSION,
    )


def _load_parent_identity(
    *,
    source: str,
    contract_id: str,
    parent_root: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    _validate_contract_id(contract_id)
    parent_dir = Path(parent_root) / source
    graph_path = parent_dir / f"{contract_id}.pt"
    token_path = parent_dir / f"{contract_id}.tokens.pt"
    sidecar_path = parent_dir / f"{contract_id}.rep.json"
    missing = [
        path.name
        for path in (graph_path, token_path, sidecar_path)
        if not path.is_file()
    ]
    if missing:
        raise GuardedTokenCandidateError(
            f"R4-D-011 parent identity {source}/{contract_id} is incomplete: {missing}"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardedTokenCandidateError(
            f"invalid R4-D-011 sidecar for {source}/{contract_id}: {exc}"
        ) from exc

    checks = {
        "sha256": contract_id,
        "source": source,
        "schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "token_lineage": "accepted_v9_byte_copy",
    }
    mismatches = {
        key: {"expected": expected, "observed": sidecar.get(key)}
        for key, expected in checks.items()
        if sidecar.get(key) != expected
    }
    if mismatches:
        raise GuardedTokenCandidateError(
            f"R4-D-011 parent sidecar mismatch for {source}/{contract_id}: {mismatches}"
        )
    targets = sidecar.get("requested_contract_names")
    if not isinstance(targets, list) or not targets or any(
        not isinstance(value, str) or not value.strip() for value in targets
    ):
        raise TargetEvidenceError(
            f"R4-D-011 parent lacks valid requested_contract_names for "
            f"{source}/{contract_id}"
        )
    if sidecar.get("actual_contract_names") != targets:
        raise GuardedTokenCandidateError(
            f"R4-D-011 requested/actual graph target mismatch for {source}/{contract_id}"
        )
    return graph_path, token_path, sidecar_path, sidecar


def _selector_payload(
    *,
    source_text: str,
    target_names: list[str],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Produce guarded tensors plus the complete accepted selector decision."""

    from ml.src.data_extraction.bounded_window_selector import (
        GUARDED_STRATEGY,
        intersect_union_length,
        target_aware_greedy_indices,
        tokenize_with_selector,
        union_length,
        window_ranges,
    )
    from ml.src.data_extraction.windowed_tokenizer import (
        STRIDE,
        TOKEN_COVERAGE_SCHEMA_VERSION,
        TOKENIZER_MODEL,
        WINDOW_SIZE,
    )

    if GUARDED_STRATEGY != GUARDED_TOKEN_SELECTOR_VERSION:
        raise GuardedTokenCandidateError(
            "guarded selector constant diverged from the R4-D-012 lineage owner"
        )
    if WINDOW_SIZE != TOKEN_TENSOR_SHAPE[1]:
        raise GuardedTokenCandidateError(
            "window tokenizer size diverged from the frozen token tensor contract"
        )
    try:
        char_spans = target_contract_char_spans(source_text, target_names)
        tokenized = tokenize_with_selector(
            source_text,
            target_char_spans=char_spans,
            tokenizer=tokenizer,
            strategy=GUARDED_STRATEGY,
            max_windows=TOKEN_TENSOR_SHAPE[0],
            window_size=TOKEN_TENSOR_SHAPE[1],
            stride=STRIDE,
        )
    except (ValueError, AssertionError) as exc:
        raise TargetEvidenceError(
            f"cannot establish valid target-aware token evidence: {exc}"
        ) from exc

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    if tuple(input_ids.shape) != TOKEN_TENSOR_SHAPE:
        raise GuardedTokenCandidateError(
            f"guarded input_ids shape changed: {tuple(input_ids.shape)}"
        )
    if tuple(attention_mask.shape) != TOKEN_TENSOR_SHAPE:
        raise GuardedTokenCandidateError(
            f"guarded attention_mask shape changed: {tuple(attention_mask.shape)}"
        )

    try:
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special_tokens = 2
    content_capacity = max(1, TOKEN_TENSOR_SHAPE[1] - special_tokens)
    ranges = window_ranges(
        int(tokenized["total_code_tokens"]),
        content_capacity=content_capacity,
        stride=STRIDE,
    )
    if len(ranges) != int(tokenized["total_windows"]):
        raise GuardedTokenCandidateError(
            "selector range reconstruction diverged from tokenized window count"
        )

    target_ranges = [list(map(int, value)) for value in tokenized["target_token_ranges"]]
    greedy_indices = target_aware_greedy_indices(
        ranges,
        target_ranges,
        count=TOKEN_TENSOR_SHAPE[0],
    )
    control_indices = [
        int(value) for value in tokenized["selector"]["control_indices"]
    ]
    selected_indices = [
        int(value) for value in tokenized["selector"]["selected_indices"]
    ]
    used_fallback = bool(tokenized["selector"]["used_control_fallback"])
    effective_selector = (
        HISTORICAL_TOKEN_SELECTOR_VERSION
        if used_fallback
        else GUARDED_TOKEN_SELECTOR_VERSION
    )
    fallback_reason = (
        "candidate_target_coverage_not_strictly_greater_than_control"
        if used_fallback
        else None
    )

    greedy_ranges = [ranges[index] for index in greedy_indices]
    selected_ranges = [ranges[index] for index in selected_indices]
    control_ranges = [ranges[index] for index in control_indices]
    greedy_target = intersect_union_length(greedy_ranges, target_ranges)
    selected_target = intersect_union_length(selected_ranges, target_ranges)
    control_target = intersect_union_length(control_ranges, target_ranges)
    target_tokens = int(tokenized["target_tokens"])
    retained_tokens = union_length(selected_ranges)
    total_tokens = int(tokenized["total_code_tokens"])

    if selected_target < control_target:
        raise GuardedTokenCandidateError("guarded selector regressed target-token coverage")
    if used_fallback and selected_indices != control_indices:
        raise GuardedTokenCandidateError("control fallback did not reproduce control indices")
    if not used_fallback and greedy_target <= control_target:
        raise GuardedTokenCandidateError(
            "guarded candidate was accepted without strict target-coverage improvement"
        )
    if not used_fallback and selected_indices != greedy_indices:
        raise GuardedTokenCandidateError("effective guarded indices diverge from greedy candidate")

    def _ratio(value: int) -> float:
        return float(value) / float(target_tokens) if target_tokens else 1.0

    selector_decision = {
        "schema": GUARDED_TOKEN_SELECTOR_SCHEMA_VERSION,
        "requested_selector": GUARDED_TOKEN_SELECTOR_VERSION,
        "effective_selector": effective_selector,
        "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
        "guard": "strict_target_token_coverage_improvement",
        "used_control_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "candidate_indices": greedy_indices,
        "control_indices": control_indices,
        "selected_indices": selected_indices,
        "requested_contract_names": list(target_names),
        "target_char_spans": [list(map(int, value)) for value in char_spans],
        "target_token_ranges": target_ranges,
        "target_tokens": target_tokens,
        "candidate_target_coverage_tokens": greedy_target,
        "control_target_coverage_tokens": control_target,
        "selected_target_coverage_tokens": selected_target,
        "candidate_target_coverage_ratio": _ratio(greedy_target),
        "control_target_coverage_ratio": _ratio(control_target),
        "selected_target_coverage_ratio": _ratio(selected_target),
        "pre_subsampling_code_tokens": total_tokens,
        "pre_subsampling_window_count": len(ranges),
        "selected_code_token_ranges": selected_ranges,
        "retained_unique_code_tokens": retained_tokens,
        "retained_token_ratio": (
            float(retained_tokens) / float(total_tokens) if total_tokens else 1.0
        ),
        "content_tokens_per_window": content_capacity,
        "tokenizer_name": TOKENIZER_MODEL,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_windows": TOKEN_TENSOR_SHAPE[0],
    }
    token_fields = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_windows": len(selected_indices),
        "stride": STRIDE,
        "num_tokens": int(attention_mask.sum().item()),
        "tokenizer_name": TOKENIZER_MODEL,
        "max_length": WINDOW_SIZE,
        "coverage_schema_version": TOKEN_COVERAGE_SCHEMA_VERSION,
        "pre_subsampling_window_count": len(ranges),
        "pre_subsampling_code_tokens": total_tokens,
        "selected_window_indices": selected_indices,
        "selected_code_token_ranges": selected_ranges,
        "retained_unique_code_tokens": retained_tokens,
        "retained_token_ratio": selector_decision["retained_token_ratio"],
        "content_tokens_per_window": content_capacity,
        "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
        "token_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
        "token_selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
        "selector_decision": selector_decision,
    }
    return token_fields, selector_decision


def build_guarded_token_identity(
    *,
    source: str,
    contract_id: str,
    preprocessed_root: Path,
    parent_root: Path,
    output_root: Path,
    parent: AcceptedV10Parent,
    tokenizer: Any,
) -> GuardedBuildResult:
    """Build one fresh graph/token/sidecar triple without mutating its parent."""

    parent_root = Path(parent_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == parent_root:
        raise GuardedTokenCandidateError("guarded candidate root cannot equal R4-D-011")
    if output_root.name != GUARDED_REPRESENTATION_ROOT_NAME:
        raise GuardedTokenCandidateError(
            f"guarded candidate root must be named {GUARDED_REPRESENTATION_ROOT_NAME!r}"
        )

    graph_path, parent_token_path, parent_sidecar_path, parent_sidecar = (
        _load_parent_identity(
            source=source,
            contract_id=contract_id,
            parent_root=parent_root,
        )
    )
    source_path = Path(preprocessed_root) / source / f"{contract_id}.sol"
    if not source_path.is_file():
        raise GuardedTokenCandidateError(
            f"missing accepted preprocessed source {source_path}"
        )
    source_text = source_path.read_text(encoding="utf-8")
    target_names = [str(value) for value in parent_sidecar["requested_contract_names"]]

    token_fields, selector_decision = _selector_payload(
        source_text=source_text,
        target_names=target_names,
        tokenizer=tokenizer,
    )

    output_dir = output_root / source
    output_dir.mkdir(parents=True, exist_ok=True)
    output_graph = output_dir / f"{contract_id}.pt"
    output_tokens = output_dir / f"{contract_id}.tokens.pt"
    output_sidecar = output_dir / f"{contract_id}.rep.json"
    collisions = [
        path
        for path in (output_graph, output_tokens, output_sidecar)
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "guarded candidate refuses to overwrite existing artifacts: "
            + ", ".join(str(path) for path in collisions)
        )

    parent_graph_sha = _sha256_file(graph_path)
    parent_token_sha = _sha256_file(parent_token_path)
    parent_sidecar_sha = _sha256_file(parent_sidecar_path)
    candidate_sidecar = copy.deepcopy(parent_sidecar)
    for key in (
        "coverage_schema_version",
        "pre_subsampling_window_count",
        "pre_subsampling_code_tokens",
        "selected_window_indices",
        "selected_code_token_ranges",
        "retained_unique_code_tokens",
        "retained_token_ratio",
        "content_tokens_per_window",
        "coverage_interpretation",
    ):
        candidate_sidecar[key] = token_fields[key]
    candidate_sidecar["window_count"] = int(token_fields["num_windows"])
    candidate_sidecar["token_lineage"] = GUARDED_TOKEN_LINEAGE_VERSION
    candidate_sidecar["token_selector_policy"] = GUARDED_TOKEN_SELECTOR_VERSION
    candidate_sidecar["selector_decision"] = selector_decision
    candidate_sidecar["graph_parent"] = {
        "decision_id": parent.decision_id,
        "physical_root": parent.physical_root,
        "binding_digest_sha256": parent.binding_digest_sha256,
        "graph_sha256": parent_graph_sha,
        "tokens_sha256": parent_token_sha,
        "sidecar_sha256": parent_sidecar_sha,
    }

    token_payload = {
        **token_fields,
        "sha256": contract_id,
        "source": source,
    }

    graph_tmp = output_graph.with_suffix(output_graph.suffix + ".tmp")
    tokens_tmp = output_tokens.with_suffix(output_tokens.suffix + ".tmp")
    sidecar_tmp = output_sidecar.with_suffix(output_sidecar.suffix + ".tmp")
    temporary = (graph_tmp, tokens_tmp, sidecar_tmp)
    try:
        import torch

        shutil.copyfile(graph_path, graph_tmp)
        torch.save(token_payload, tokens_tmp)
        sidecar_tmp.write_text(
            json.dumps(candidate_sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _sha256_file(graph_tmp) != parent_graph_sha:
            raise GuardedTokenCandidateError(
                f"graph copy changed bytes for {source}/{contract_id}"
            )
        graph_tmp.replace(output_graph)
        tokens_tmp.replace(output_tokens)
        sidecar_tmp.replace(output_sidecar)
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()

    return GuardedBuildResult(
        source=source,
        contract_id=contract_id,
        effective_selector=str(selector_decision["effective_selector"]),
        used_control_fallback=bool(selector_decision["used_control_fallback"]),
        selected_window_indices=tuple(
            int(value) for value in selector_decision["selected_indices"]
        ),
        graph_sha256=_sha256_file(output_graph),
        tokens_sha256=_sha256_file(output_tokens),
        sidecar_sha256=_sha256_file(output_sidecar),
    )


def _inventory_parent(parent_root: Path) -> list[tuple[str, str]]:
    identities: list[tuple[str, str]] = []
    for sidecar in sorted(Path(parent_root).glob("*/*.rep.json")):
        source = sidecar.parent.name
        contract_id = sidecar.name.removesuffix(".rep.json")
        _validate_contract_id(contract_id)
        identities.append((source, contract_id))
    if len(set(identities)) != len(identities):
        raise GuardedTokenCandidateError("R4-D-011 parent contains duplicate identities")
    return identities


def build_guarded_token_candidate(
    *,
    acceptance_path: Path,
    repo_root: Path,
    preprocessed_root: Path,
    parent_root: Path,
    output_root: Path,
    identities: Iterable[tuple[str, str]] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Build an explicit bounded D4 candidate and write its construction manifest."""

    repo_root = Path(repo_root).resolve()
    parent_root = Path(parent_root).resolve()
    output_root = Path(output_root).resolve()
    parent = load_accepted_v10_parent(
        acceptance_path=acceptance_path,
        repo_root=repo_root,
        parent_root=parent_root,
    )
    if identities is None:
        raise GuardedTokenCandidateError(
            "full-population guarded generation is D5 and is not authorized "
            "before bounded D4 acceptance"
        )
    requested = sorted(
        set(
            (str(source), str(contract_id))
            for source, contract_id in identities
        )
    )
    if not requested:
        raise GuardedTokenCandidateError("bounded guarded candidate requires identities")
    if output_root.name != GUARDED_REPRESENTATION_ROOT_NAME:
        raise GuardedTokenCandidateError(
            f"guarded candidate root must be named {GUARDED_REPRESENTATION_ROOT_NAME!r}"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"guarded candidate output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    parent_inventory = _inventory_parent(parent_root)
    if len(parent_inventory) != parent.contracts:
        raise GuardedTokenCandidateError(
            f"R4-D-011 parent population changed: "
            f"{len(parent_inventory)} != {parent.contracts}"
        )
    parent_set = set(parent_inventory)
    missing = sorted(set(requested) - parent_set)
    if missing:
        raise GuardedTokenCandidateError(
            f"requested identities are absent from R4-D-011: {missing[:5]}"
        )
    if tokenizer is None:
        from transformers import AutoTokenizer
        from ml.src.data_extraction.windowed_tokenizer import TOKENIZER_MODEL

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL,
            use_fast=True,
            local_files_only=True,
        )

    results: list[GuardedBuildResult] = []
    for source, contract_id in requested:
        results.append(
            build_guarded_token_identity(
                source=source,
                contract_id=contract_id,
                preprocessed_root=preprocessed_root,
                parent_root=parent_root,
                output_root=output_root,
                parent=parent,
                tokenizer=tokenizer,
            )
        )

    fallback_total = sum(result.used_control_fallback for result in results)
    guarded_total = len(results) - fallback_total
    manifest = {
        "schema": GUARDED_CANDIDATE_MANIFEST_SCHEMA,
        "status": GUARDED_CANDIDATE_STATUS_BOUNDED,
        "physical_acceptance": False,
        "training_authorized": False,
        "source_commit": _source_commit(repo_root),
        "representation_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
        "candidate_root_name": GUARDED_REPRESENTATION_ROOT_NAME,
        "selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
        "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
        "graph_schema_version": parent.graph_schema_version,
        "graph_extractor_version": parent.extractor_version,
        "frozen_token_shape": list(TOKEN_TENSOR_SHAPE),
        "parent": {
            "decision_id": parent.decision_id,
            "physical_root": parent.physical_root,
            "binding_digest_sha256": parent.binding_digest_sha256,
            "contracts": parent.contracts,
        },
        "full_population": False,
        "contracts_requested": len(requested),
        "contracts_written": len(results),
        "effective_selector_counts": {
            GUARDED_TOKEN_SELECTOR_VERSION: guarded_total,
            HISTORICAL_TOKEN_SELECTOR_VERSION: fallback_total,
        },
        "control_fallback_contracts": fallback_total,
        "guarded_contracts": guarded_total,
        "records": [
            {
                "source": result.source,
                "contract_id": result.contract_id,
                "effective_selector": result.effective_selector,
                "used_control_fallback": result.used_control_fallback,
                "selected_window_indices": list(result.selected_window_indices),
                "graph_sha256": result.graph_sha256,
                "tokens_sha256": result.tokens_sha256,
                "sidecar_sha256": result.sidecar_sha256,
            }
            for result in results
        ],
    }
    manifest_path = output_root / "guarded_candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "AcceptedV10Parent",
    "GuardedBuildResult",
    "GuardedTokenCandidateError",
    "TargetEvidenceError",
    "build_guarded_token_candidate",
    "build_guarded_token_identity",
    "load_accepted_v10_parent",
]
