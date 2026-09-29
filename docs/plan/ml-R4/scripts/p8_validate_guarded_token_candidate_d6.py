#!/usr/bin/env python3
"""Independently validate the full D5 guarded-token candidate for D6 review.

This validator does not grant physical acceptance. It reconstructs the full
binding from physical artifacts, checks selector and lineage invariants for all
22,540 identities, and regenerates the required bounded probes twice. A clean
result is evidence for a later explicit accept/reject/revise decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"

from sentinel_data.preprocessing.r4_versions import (  # noqa: E402
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_TOKEN_LINEAGE_VERSION,
    GUARDED_TOKEN_SELECTOR_VERSION,
    GUARDED_TOKEN_TRANSFORMERS_VERSION,
    HISTORICAL_TOKEN_SELECTOR_VERSION,
    TOKEN_TENSOR_SHAPE,
    V10_GRAPH_SCHEMA_VERSION,
    V10_PRIMARY_SLITHER_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_SLITHER_RUNTIME_EXCEPTIONS,
)
from sentinel_data.representation.graph_schema_versions import get_graph_schema  # noqa: E402
from sentinel_data.representation.r4_guarded_token_candidate import (  # noqa: E402
    build_guarded_token_candidate,
    load_accepted_v10_parent,
)
from sentinel_data.vnext.r4_binding import _validate_graph, _validate_tokens  # noqa: E402

EXPECTED_D5_SOURCE_COMMIT = "733f0c73eb76ab107751c30345d9a169a0429fdd"
EXPECTED_D5_BINDING_DIGEST = (
    "9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881"
)
EXPECTED_D5_MANIFEST_SHA256 = (
    "5aa48ce6eb218b742af3c99823db333272d65599039b6ec550d56f713b839dca"
)
EXPECTED_ACCEPTANCE_MANIFEST_SHA256 = (
    "5fc83eff39d4a28db9a5b6b5255a95ad64ee75ca88a948ba99dadb2bc03ee165"
)
EXPECTED_CONTRACTS = 22540
EXPECTED_SELECTOR_COUNTS = {
    HISTORICAL_TOKEN_SELECTOR_VERSION: 7789,
    GUARDED_TOKEN_SELECTOR_VERSION: 14751,
}
EXPECTED_CANDIDATE_MANIFEST_SCHEMA = "sentinel-r4-guarded-token-candidate-manifest-v1"
EXPECTED_CANDIDATE_STATUS = "FULL_GUARDED_TOKEN_CANDIDATE"
REPORT_SCHEMA = "sentinel-r4-guarded-token-d6-validation-v1"

DEFAULT_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json"
)
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_PARENT = (
    DATA_ROOT
    / "v10-v26-full-candidate-attempt-2026-09-01-a"
    / "representations-r4-v3-candidate"
)
DEFAULT_CANDIDATE = (
    DATA_ROOT
    / "r4-guarded-d5-full-2026-09-29-a"
    / GUARDED_REPRESENTATION_ROOT_NAME
)

PROBE_IDENTITIES: tuple[tuple[str, str], ...] = (
    (
        "smartbugs_curated",
        "85a6581669271b86cd58b837f216e6b140f726b1dce93270dcf6291995fbfe5d",
    ),
    (
        "solidifi",
        "08378c9d432399d34e2f5a417e0b57e47b0ef63cc99a208f9efb67744d5e837f",
    ),
    (
        "solidifi",
        "397813120698b5942a0168c339310bb57dcf2d8b4041b3590ad86ce3d3accfbd",
    ),
    (
        "solidifi",
        "9b8eb361195230fb9e7d8797c3c456fce60b169564f5484ae003814ee03a6e4c",
    ),
    (
        "dive",
        "83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d",
    ),
    (
        "dive",
        "f50cd5d7df9ab644a02eb760ceab56548d327984db313015a66bca85513fa3c5",
    ),
    (
        "dive",
        "087f69b560460734f646e30aa9be314c7f9085289ba394677905d008cf3a7ae0",
    ),
    (
        "solidifi",
        "d4b90b62c2ab33ce14d403f1d995b132e8178a09ca243db3be58b610c30cd297",
    ),
    (
        "dive",
        "caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_digest(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _max_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _inventory(root: Path) -> dict[tuple[str, str], Path]:
    inventory: dict[tuple[str, str], Path] = {}
    for sidecar in sorted(root.glob("*/*.rep.json")):
        source = sidecar.parent.name
        contract_id = sidecar.name.removesuffix(".rep.json")
        if len(contract_id) != 64 or any(
            char not in "0123456789abcdef" for char in contract_id
        ):
            raise ValueError(f"invalid identity: {source}/{contract_id}")
        key = (source, contract_id)
        if key in inventory:
            raise ValueError(f"duplicate identity: {source}/{contract_id}")
        inventory[key] = sidecar
    return inventory


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_selector_decision(decision: dict[str, Any]) -> None:
    requested = decision.get("requested_selector")
    effective = decision.get("effective_selector")
    control = decision.get("control_selector")
    fallback = decision.get("used_control_fallback")
    candidate_indices = decision.get("candidate_indices")
    control_indices = decision.get("control_indices")
    selected_indices = decision.get("selected_indices")
    candidate_coverage = int(decision.get("candidate_target_coverage_tokens", -1))
    control_coverage = int(decision.get("control_target_coverage_tokens", -1))
    selected_coverage = int(decision.get("selected_target_coverage_tokens", -1))
    total_windows = int(decision.get("pre_subsampling_window_count", -1))

    if requested != GUARDED_TOKEN_SELECTOR_VERSION:
        raise ValueError("requested selector identity changed")
    if control != HISTORICAL_TOKEN_SELECTOR_VERSION:
        raise ValueError("control selector identity changed")
    if not all(
        isinstance(value, list)
        for value in (candidate_indices, control_indices, selected_indices)
    ):
        raise ValueError("selector indices are not lists")
    if selected_indices != sorted(set(selected_indices)):
        raise ValueError("selected indices are not sorted unique")
    if total_windows < 1:
        raise ValueError("pre-subsampling window count is invalid")
    for values in (candidate_indices, control_indices, selected_indices):
        if any(
            not isinstance(index, int) or index < 0 or index >= total_windows
            for index in values
        ):
            raise ValueError("selector index is outside the window population")
        if len(values) > TOKEN_TENSOR_SHAPE[0]:
            raise ValueError("selector emitted too many windows")
    if total_windows > TOKEN_TENSOR_SHAPE[0] and len(selected_indices) != TOKEN_TENSOR_SHAPE[0]:
        raise ValueError("over-cap selector did not emit the frozen window count")
    if selected_coverage < control_coverage:
        raise ValueError("selected target coverage regresses control")

    if fallback is True:
        if effective != HISTORICAL_TOKEN_SELECTOR_VERSION:
            raise ValueError("fallback effective selector is not historical control")
        if selected_indices != control_indices:
            raise ValueError("fallback indices differ from historical control")
        if candidate_coverage > control_coverage:
            raise ValueError("fallback used despite strict candidate target gain")
    elif fallback is False:
        if effective != GUARDED_TOKEN_SELECTOR_VERSION:
            raise ValueError("guarded effective selector identity changed")
        if selected_indices != candidate_indices:
            raise ValueError("guarded selected indices differ from candidate")
        if candidate_coverage <= control_coverage:
            raise ValueError("guarded selector lacks strict target gain")
    else:
        raise ValueError("used_control_fallback is not boolean")


def _validate_full_candidate(
    *,
    acceptance_path: Path,
    preprocessed_root: Path,
    parent_root: Path,
    candidate_root: Path,
    progress_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    import torch

    if _sha256_file(acceptance_path) != EXPECTED_ACCEPTANCE_MANIFEST_SHA256:
        raise ValueError("R4-D-011 acceptance manifest SHA-256 changed")

    parent = load_accepted_v10_parent(
        acceptance_path=acceptance_path,
        repo_root=REPO_ROOT,
        parent_root=parent_root,
    )
    if parent.contracts != EXPECTED_CONTRACTS:
        raise ValueError("R4-D-011 accepted contract count changed")
    expected_preprocessed = (REPO_ROOT / parent.preprocessed_parent).resolve()
    if preprocessed_root != expected_preprocessed:
        raise ValueError("preprocessed root is not the R4-D-011 accepted parent")

    manifest_path = candidate_root / "guarded_candidate_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if _sha256_file(manifest_path) != EXPECTED_D5_MANIFEST_SHA256:
        raise ValueError("D5 candidate manifest SHA-256 differs from reviewed result")
    manifest = _load_json(manifest_path)

    required_manifest = {
        "schema": EXPECTED_CANDIDATE_MANIFEST_SCHEMA,
        "status": EXPECTED_CANDIDATE_STATUS,
        "source_commit": EXPECTED_D5_SOURCE_COMMIT,
        "representation_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
        "candidate_root_name": GUARDED_REPRESENTATION_ROOT_NAME,
        "selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
        "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
        "transformers_version": GUARDED_TOKEN_TRANSFORMERS_VERSION,
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "graph_extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "full_population": True,
        "contracts_requested": EXPECTED_CONTRACTS,
        "contracts_written": EXPECTED_CONTRACTS,
        "physical_acceptance": False,
        "training_authorized": False,
        "binding_digest_sha256": EXPECTED_D5_BINDING_DIGEST,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"D5 manifest field changed: {key}={manifest.get(key)!r} "
                f"expected {expected!r}"
            )
    if manifest.get("effective_selector_counts") != EXPECTED_SELECTOR_COUNTS:
        raise ValueError("D5 selector counts differ from reviewed result")

    parent_inventory = _inventory(parent_root)
    candidate_inventory = _inventory(candidate_root)
    parent_keys = set(parent_inventory)
    candidate_keys = set(candidate_inventory)
    if len(parent_keys) != EXPECTED_CONTRACTS:
        raise ValueError("parent inventory count changed")
    if candidate_keys != parent_keys:
        missing = sorted(parent_keys - candidate_keys)
        extra = sorted(candidate_keys - parent_keys)
        raise ValueError(
            "candidate population differs from R4-D-011: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    expected_files = {manifest_path.resolve()}
    records: list[dict[str, Any]] = []
    selector_counts = {
        HISTORICAL_TOKEN_SELECTOR_VERSION: 0,
        GUARDED_TOKEN_SELECTOR_VERSION: 0,
    }
    slither_runtimes: Counter[tuple[str, str, str]] = Counter()
    schema = get_graph_schema(V10_GRAPH_SCHEMA_VERSION)
    started = time.perf_counter()

    for index, (source, contract_id) in enumerate(sorted(parent_keys), start=1):
        parent_dir = parent_root / source
        candidate_dir = candidate_root / source
        parent_graph = parent_dir / f"{contract_id}.pt"
        candidate_graph = candidate_dir / f"{contract_id}.pt"
        candidate_tokens = candidate_dir / f"{contract_id}.tokens.pt"
        candidate_sidecar = candidate_dir / f"{contract_id}.rep.json"
        expected_files.update(
            path.resolve()
            for path in (candidate_graph, candidate_tokens, candidate_sidecar)
        )
        for path in (parent_graph, candidate_graph, candidate_tokens, candidate_sidecar):
            if not path.is_file():
                raise FileNotFoundError(path)

        parent_tokens = parent_dir / f"{contract_id}.tokens.pt"
        parent_sidecar_path = parent_dir / f"{contract_id}.rep.json"
        for path in (parent_tokens, parent_sidecar_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        graph_sha = _sha256_file(candidate_graph)
        parent_graph_sha = _sha256_file(parent_graph)
        parent_tokens_sha = _sha256_file(parent_tokens)
        parent_sidecar_sha = _sha256_file(parent_sidecar_path)
        if graph_sha != parent_graph_sha:
            raise ValueError(f"graph bytes changed for {source}/{contract_id}")

        source_path = preprocessed_root / source / f"{contract_id}.sol"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if _sha256_file(source_path) != contract_id:
            raise ValueError(
                f"repaired source identity drift: {source}/{contract_id}"
            )

        sidecar = _load_json(candidate_sidecar)
        if sidecar.get("sha256") != contract_id or sidecar.get("source") != source:
            raise ValueError(f"candidate sidecar identity mismatch: {source}/{contract_id}")
        if sidecar.get("schema_version") != V10_GRAPH_SCHEMA_VERSION:
            raise ValueError(f"graph schema mismatch: {source}/{contract_id}")
        if sidecar.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
            raise ValueError(f"extractor mismatch: {source}/{contract_id}")
        if sidecar.get("token_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
            raise ValueError(f"token lineage mismatch: {source}/{contract_id}")
        if sidecar.get("token_selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
            raise ValueError(f"selector policy mismatch: {source}/{contract_id}")
        if sidecar.get("transformers_version") != GUARDED_TOKEN_TRANSFORMERS_VERSION:
            raise ValueError(f"transformers version mismatch: {source}/{contract_id}")

        runtime = dict(sidecar.get("slither_runtime") or {})
        slither_version = str(runtime.get("slither_analyzer") or "")
        crytic_compile_version = str(runtime.get("crytic_compile") or "")
        if not slither_version or not crytic_compile_version:
            raise ValueError(f"Slither runtime binding missing: {source}/{contract_id}")
        required_slither = V10_SLITHER_RUNTIME_EXCEPTIONS.get(
            contract_id, V10_PRIMARY_SLITHER_VERSION
        )
        required_role = (
            "identity_bound_exception"
            if contract_id in V10_SLITHER_RUNTIME_EXCEPTIONS
            else "primary"
        )
        if slither_version != required_slither:
            raise ValueError(f"Slither identity binding mismatch: {source}/{contract_id}")
        if runtime.get("required_for_physical_acceptance") != required_slither:
            raise ValueError(f"required Slither binding mismatch: {source}/{contract_id}")
        if runtime.get("runtime_role") != required_role:
            raise ValueError(f"Slither runtime role mismatch: {source}/{contract_id}")
        slither_runtimes[
            (slither_version, crytic_compile_version, required_role)
        ] += 1
        if str(sidecar.get("graph_extraction_mode") or "").startswith(
            "slither_parse_only"
        ) or bool(sidecar.get("graph_analysis_degraded")):
            raise ValueError(f"degraded graph analysis: {source}/{contract_id}")
        if sidecar.get("unclassified_call_ir") not in (None, []):
            raise ValueError(f"unclassified call IR: {source}/{contract_id}")
        if int(sidecar.get("unclassified_call_ir_count", 0)) != 0:
            raise ValueError(f"unclassified call IR count: {source}/{contract_id}")
        if list(sidecar.get("call_mapping_errors") or []):
            raise ValueError(f"call mapping errors: {source}/{contract_id}")
        if sidecar.get("classified_call_ir_counts") != sidecar.get(
            "emitted_call_edge_counts"
        ):
            raise ValueError(f"classified/emitted call mismatch: {source}/{contract_id}")

        graph_parent = sidecar.get("graph_parent")
        if not isinstance(graph_parent, dict):
            raise ValueError(f"graph_parent missing: {source}/{contract_id}")
        if graph_parent.get("decision_id") != "R4-D-011":
            raise ValueError(f"graph parent decision mismatch: {source}/{contract_id}")
        if graph_parent.get("physical_root") != parent.physical_root:
            raise ValueError(f"graph parent root mismatch: {source}/{contract_id}")
        if graph_parent.get("binding_digest_sha256") != parent.binding_digest_sha256:
            raise ValueError(f"graph parent digest mismatch: {source}/{contract_id}")
        if graph_parent.get("graph_sha256") != parent_graph_sha:
            raise ValueError(f"graph parent graph hash mismatch: {source}/{contract_id}")
        if graph_parent.get("tokens_sha256") != parent_tokens_sha:
            raise ValueError(f"graph parent token hash mismatch: {source}/{contract_id}")
        if graph_parent.get("sidecar_sha256") != parent_sidecar_sha:
            raise ValueError(f"graph parent sidecar hash mismatch: {source}/{contract_id}")

        targets = sidecar.get("requested_contract_names")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"requested contract targets missing: {source}/{contract_id}")
        if sidecar.get("actual_contract_names") != targets:
            raise ValueError(f"requested/actual target mismatch: {source}/{contract_id}")

        graph = torch.load(candidate_graph, map_location="cpu", weights_only=False)
        if getattr(graph, "graph_schema_version", None) != V10_GRAPH_SCHEMA_VERSION:
            raise ValueError(f"graph payload schema mismatch: {source}/{contract_id}")
        if (
            getattr(graph, "representation_extractor_version", None)
            != V10_REPRESENTATION_EXTRACTOR_VERSION
        ):
            raise ValueError(f"graph payload extractor mismatch: {source}/{contract_id}")
        if list(getattr(graph, "unclassified_call_ir", []) or []):
            raise ValueError(f"graph payload unclassified call IR: {source}/{contract_id}")
        if list(getattr(graph, "call_mapping_errors", []) or []):
            raise ValueError(f"graph payload call mapping errors: {source}/{contract_id}")
        if getattr(graph, "classified_call_ir_counts", None) != getattr(
            graph, "emitted_call_edge_counts", None
        ):
            raise ValueError(
                f"graph payload classified/emitted call mismatch: {source}/{contract_id}"
            )
        _validate_graph(torch, graph, sidecar, num_edge_types=schema.num_edge_types)

        payload = torch.load(candidate_tokens, map_location="cpu", weights_only=True)
        _validate_tokens(torch, payload, sidecar)
        if payload.get("sha256") != contract_id or payload.get("source") != source:
            raise ValueError(f"candidate token identity mismatch: {source}/{contract_id}")
        if payload.get("token_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
            raise ValueError(f"token payload lineage mismatch: {source}/{contract_id}")
        if payload.get("token_selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
            raise ValueError(f"token payload selector mismatch: {source}/{contract_id}")
        if payload.get("transformers_version") != GUARDED_TOKEN_TRANSFORMERS_VERSION:
            raise ValueError(f"token runtime mismatch: {source}/{contract_id}")

        input_ids = payload.get("input_ids")
        attention_mask = payload.get("attention_mask")
        if tuple(input_ids.shape) != TOKEN_TENSOR_SHAPE:
            raise ValueError(f"input_ids shape mismatch: {source}/{contract_id}")
        if tuple(attention_mask.shape) != TOKEN_TENSOR_SHAPE:
            raise ValueError(f"attention_mask shape mismatch: {source}/{contract_id}")
        if input_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
            raise ValueError(f"token dtype mismatch: {source}/{contract_id}")

        token_decision = payload.get("selector_decision")
        sidecar_decision = sidecar.get("selector_decision")
        if not isinstance(token_decision, dict) or token_decision != sidecar_decision:
            raise ValueError(f"selector decision persistence mismatch: {source}/{contract_id}")
        _validate_selector_decision(token_decision)
        if token_decision.get("requested_contract_names") != targets:
            raise ValueError(
                f"selector target binding mismatch: {source}/{contract_id}"
            )

        selected_indices = token_decision["selected_indices"]
        if payload.get("selected_window_indices") != selected_indices:
            raise ValueError(f"token selected indices mismatch: {source}/{contract_id}")
        if sidecar.get("selected_window_indices") != selected_indices:
            raise ValueError(f"sidecar selected indices mismatch: {source}/{contract_id}")

        effective = str(token_decision["effective_selector"])
        selector_counts[effective] += 1
        records.append(
            {
                "source": source,
                "contract_id": contract_id,
                "effective_selector": effective,
                "used_control_fallback": bool(
                    token_decision["used_control_fallback"]
                ),
                "selected_window_indices": [int(value) for value in selected_indices],
                "graph_sha256": graph_sha,
                "tokens_sha256": _sha256_file(candidate_tokens),
                "sidecar_sha256": _sha256_file(candidate_sidecar),
            }
        )

        if index == 1 or index == EXPECTED_CONTRACTS or index % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = float(index) / elapsed if elapsed > 0 else 0.0
            print(
                f"[D6 full] {index}/{EXPECTED_CONTRACTS} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
            )

    actual_files = {path.resolve() for path in candidate_root.rglob("*") if path.is_file()}
    unexpected = sorted(str(path) for path in actual_files - expected_files)
    missing_files = sorted(str(path) for path in expected_files - actual_files)
    if unexpected or missing_files:
        raise ValueError(
            "candidate file inventory mismatch: "
            f"unexpected={unexpected[:5]} missing={missing_files[:5]}"
        )

    records.sort(key=lambda row: (row["source"], row["contract_id"]))
    if _binding_digest(records) != EXPECTED_D5_BINDING_DIGEST:
        raise ValueError("independently reconstructed D5 binding digest differs")
    if manifest.get("records") != records:
        raise ValueError("D5 manifest records differ from independently reconstructed records")
    if selector_counts != EXPECTED_SELECTOR_COUNTS:
        raise ValueError(
            f"independent selector counts differ: {selector_counts}"
        )
    expected_primary = EXPECTED_CONTRACTS - len(V10_SLITHER_RUNTIME_EXCEPTIONS)
    observed_primary = sum(
        count
        for (_, _, role), count in slither_runtimes.items()
        if role == "primary"
    )
    observed_exceptions = sum(
        count
        for (_, _, role), count in slither_runtimes.items()
        if role == "identity_bound_exception"
    )
    if observed_primary != expected_primary:
        raise ValueError(
            f"primary Slither population changed: {observed_primary} != {expected_primary}"
        )
    if observed_exceptions != len(V10_SLITHER_RUNTIME_EXCEPTIONS):
        raise ValueError(
            "identity-bound Slither exception population changed: "
            f"{observed_exceptions} != {len(V10_SLITHER_RUNTIME_EXCEPTIONS)}"
        )
    manifest["_d6_slither_runtime_distribution"] = [
        {
            "slither_analyzer": slither_version,
            "crytic_compile": crytic_version,
            "runtime_role": role,
            "contracts": count,
        }
        for (slither_version, crytic_version, role), count in sorted(
            slither_runtimes.items()
        )
    ]
    return manifest, records, selector_counts


def _validate_probe_regeneration(
    *,
    acceptance_path: Path,
    preprocessed_root: Path,
    parent_root: Path,
    candidate_root: Path,
    work_root: Path,
) -> list[dict[str, Any]]:
    probe_results: list[dict[str, Any]] = []
    roots: list[Path] = []
    for label in ("repeat-a", "repeat-b"):
        output_root = work_root / label / GUARDED_REPRESENTATION_ROOT_NAME
        build_guarded_token_candidate(
            acceptance_path=acceptance_path,
            repo_root=REPO_ROOT,
            preprocessed_root=preprocessed_root,
            parent_root=parent_root,
            output_root=output_root,
            identities=PROBE_IDENTITIES,
        )
        roots.append(output_root)

    for source, contract_id in PROBE_IDENTITIES:
        candidate_dir = candidate_root / source
        first_dir = roots[0] / source
        second_dir = roots[1] / source
        row: dict[str, Any] = {
            "source": source,
            "contract_id": contract_id,
        }
        for suffix, key in (
            (".pt", "graph"),
            (".tokens.pt", "tokens"),
            (".rep.json", "sidecar"),
        ):
            candidate_path = candidate_dir / f"{contract_id}{suffix}"
            first_path = first_dir / f"{contract_id}{suffix}"
            second_path = second_dir / f"{contract_id}{suffix}"
            hashes = {
                "candidate": _sha256_file(candidate_path),
                "repeat_a": _sha256_file(first_path),
                "repeat_b": _sha256_file(second_path),
            }
            if len(set(hashes.values())) != 1:
                raise ValueError(
                    f"probe {key} bytes are not deterministic for "
                    f"{source}/{contract_id}: {hashes}"
                )
            row[f"{key}_sha256"] = hashes["candidate"]
        probe_results.append(row)
    return probe_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="Fresh D6 validation work directory used for deterministic probes.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")

    acceptance_path = args.acceptance.resolve()
    preprocessed_root = args.preprocessed_root.resolve()
    parent_root = args.parent_root.resolve()
    candidate_root = args.candidate_root.resolve()
    work_root = args.work_root.resolve()

    if candidate_root.name != GUARDED_REPRESENTATION_ROOT_NAME:
        raise ValueError("candidate root basename does not match guarded lineage")
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"D6 work root is not empty; use a fresh path: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rss_before = _max_rss_mb()
    manifest, records, selector_counts = _validate_full_candidate(
        acceptance_path=acceptance_path,
        preprocessed_root=preprocessed_root,
        parent_root=parent_root,
        candidate_root=candidate_root,
        progress_every=args.progress_every,
    )
    probes = _validate_probe_regeneration(
        acceptance_path=acceptance_path,
        preprocessed_root=preprocessed_root,
        parent_root=parent_root,
        candidate_root=candidate_root,
        work_root=work_root,
    )

    elapsed = time.perf_counter() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_D6_REVIEW_REQUIRED",
        "candidate_root": _portable_path(candidate_root),
        "candidate_source_commit": manifest["source_commit"],
        "candidate_manifest_sha256": _sha256_file(
            candidate_root / "guarded_candidate_manifest.json"
        ),
        "binding_digest_sha256": _binding_digest(records),
        "contracts_checked": len(records),
        "effective_selector_counts": selector_counts,
        "probe_identities_checked": len(probes),
        "probe_regeneration": probes,
        "graph_parent_decision": "R4-D-011",
        "graph_parent_binding_digest_sha256": manifest["parent"][
            "binding_digest_sha256"
        ],
        "representation_lineage": manifest["representation_lineage"],
        "selector_policy": manifest["selector_policy"],
        "control_selector": manifest["control_selector"],
        "transformers_version": manifest["transformers_version"],
        "graph_schema_version": manifest["graph_schema_version"],
        "graph_extractor_version": manifest["graph_extractor_version"],
        "frozen_token_shape": manifest["frozen_token_shape"],
        "slither_runtime_distribution": manifest["_d6_slither_runtime_distribution"],
        "runtime": {
            "elapsed_seconds": elapsed,
            "max_rss_mb_before": rss_before,
            "max_rss_mb_after": _max_rss_mb(),
        },
        "physical_acceptance": False,
        "acceptance_decision": "PENDING_EXPLICIT_REVIEW",
        "training_authorized": False,
        "review_required": True,
        "decision_boundary": (
            "This validator supplies independent D6 evidence only. Physical "
            "authority changes only through the explicit R4 decision/ADR and "
            "machine-readable acceptance record after human review."
        ),
    }
    report_path = work_root / "d6_guarded_token_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
