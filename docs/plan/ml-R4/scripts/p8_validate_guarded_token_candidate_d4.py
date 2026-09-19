#!/usr/bin/env python3
"""Run the protected-local D4 guarded-token bounded validation tranche.

This validator is intentionally bounded. It consumes an explicit evidence-derived
identity set, constructs the fresh R4-D-012 guarded-token candidate twice, and
compares both builds with the immutable R4-D-011 physical parent.

A PASS is D4 evidence only. It does not accept a new physical lineage, authorize
the D5 full-population build, or authorize model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

import torch
import transformers

from sentinel_data.preprocessing.r4_versions import (
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_TOKEN_LINEAGE_VERSION,
    GUARDED_TOKEN_SELECTOR_VERSION,
    GUARDED_TOKEN_TRANSFORMERS_VERSION,
    HISTORICAL_TOKEN_SELECTOR_VERSION,
    TOKEN_TENSOR_SHAPE,
)
from sentinel_data.representation.r4_guarded_token_candidate import (
    GuardedTokenCandidateError,
    build_guarded_token_candidate,
)

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json"
)
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_PARENT = (
    DATA_ROOT
    / "v10-v26-full-candidate-attempt-2026-09-01-a/representations-r4-v3-candidate"
)
DEFAULT_WORK_ROOT = DATA_ROOT / "r4-guarded-d4-bounded-2026-09-19"
REPORT_SCHEMA = "sentinel-r4-guarded-token-d4-bounded-validation-v1"
RUNTIME_EXCEPTION_CONTRACT_ID = (
    "caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9"
)
OPTIONAL_EXTRA_STRESS = (
    "dive",
    "c74bbb7fbe8eda3e6d9404b08678e9eca476aa85831e7c23b578cfa089f77b8f",
)

# These cases come from retained selector/CUDA evidence rather than ad-hoc
# sampling. Expectations express only the already-accepted R4-D-012 semantics.
EVIDENCE_CASES: tuple[dict[str, Any], ...] = (
    {
        "purpose": "under_cap_control_fallback",
        "source": "smartbugs_curated",
        "contract_id": "85a6581669271b86cd58b837f216e6b140f726b1dce93270dcf6291995fbfe5d",
        "expect_under_cap": True,
        "expect_fallback": True,
        "expected_total_windows": 1,
        "expected_control_indices": [0],
        "expected_selected_indices": [0],
        "expected_control_target_coverage_ratio": 1.0,
        "expected_selected_target_coverage_ratio": 1.0,
        "expected_control_retained_ratio": 1.0,
        "expected_retained_ratio": 1.0,
    },
    {
        "purpose": "strong_selector_improvement",
        "source": "solidifi",
        "contract_id": "08378c9d432399d34e2f5a417e0b57e47b0ef63cc99a208f9efb67744d5e837f",
        "expect_under_cap": False,
        "expect_fallback": False,
        "expected_total_windows": 11,
        "expected_control_indices": [0, 3, 7, 10],
        "expected_selected_indices": [5, 6, 8, 10],
        "expected_control_target_coverage_ratio": 0.5205566097406704,
        "expected_selected_target_coverage_ratio": 0.9854522454142948,
        "expected_control_retained_ratio": 0.6429833863556027,
        "expected_retained_ratio": 0.5510781194768469,
    },
    {
        "purpose": "over_cap_equal_control_fallback",
        "source": "solidifi",
        "contract_id": "397813120698b5942a0168c339310bb57dcf2d8b4041b3590ad86ce3d3accfbd",
        "expect_under_cap": False,
        "expect_fallback": True,
        "expected_total_windows": 8,
        "expected_control_indices": [0, 2, 5, 7],
        "expected_selected_indices": [0, 2, 5, 7],
        "expected_control_target_coverage_ratio": 0.8838709677419355,
        "expected_selected_target_coverage_ratio": 0.8838709677419355,
        "expected_control_retained_ratio": 0.8873994638069705,
        "expected_retained_ratio": 0.8873994638069705,
    },
    {
        "purpose": "reentrancy_target_shape_control_fallback",
        "source": "solidifi",
        "contract_id": "9b8eb361195230fb9e7d8797c3c456fce60b169564f5484ae003814ee03a6e4c",
        "expect_under_cap": False,
        "expect_fallback": True,
        "expected_total_windows": 17,
        "expected_control_indices": [0, 5, 11, 16],
        "expected_selected_indices": [0, 5, 11, 16],
        "expected_control_target_coverage_ratio": 1.0,
        "expected_selected_target_coverage_ratio": 1.0,
        "expected_control_retained_ratio": 0.44148115494820367,
        "expected_retained_ratio": 0.44148115494820367,
    },
    {
        "purpose": "long_train_batch_improvement",
        "source": "dive",
        "contract_id": "83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d",
        "expect_under_cap": False,
        "expect_fallback": False,
        "expected_total_windows": 62,
        "expected_control_indices": [0, 20, 41, 61],
        "expected_selected_indices": [1, 4, 7, 10],
        "expected_control_target_coverage_ratio": 0.11139967195188627,
        "expected_selected_target_coverage_ratio": 0.1394204483324221,
        "expected_control_retained_ratio": 0.11345311408799441,
        "expected_retained_ratio": 0.12951558631198018,
    },
    {
        "purpose": "worst_case_cuda_forward_probe",
        "source": "dive",
        "contract_id": "f50cd5d7df9ab644a02eb760ceab56548d327984db313015a66bca85513fa3c5",
        "expect_under_cap": False,
        "expect_fallback": False,
        "expected_total_windows": 353,
        "expected_control_indices": [0, 117, 235, 352],
        "expected_selected_indices": [96, 99, 102, 105],
        "expected_control_target_coverage_ratio": 0.016383089382569285,
        "expected_selected_target_coverage_ratio": 0.03789285979641875,
        "expected_control_retained_ratio": 0.020759741923981233,
        "expected_retained_ratio": 0.02273208455444,
    },
    {
        "purpose": "train_weak_improvement",
        "source": "dive",
        "contract_id": "087f69b560460734f646e30aa9be314c7f9085289ba394677905d008cf3a7ae0",
        "expect_under_cap": False,
        "expect_fallback": False,
        "expected_total_windows": 17,
        "expected_control_indices": [0, 5, 11, 16],
        "expected_selected_indices": [3, 6, 9, 12],
        "expected_control_target_coverage_ratio": 0.36030374443571617,
        "expected_selected_target_coverage_ratio": 0.5341712490180676,
        "expected_control_retained_ratio": 0.42682650983940285,
        "expected_retained_ratio": 0.4614340646912463,
    },
    {
        "purpose": "train_strong_target_shape_improvement",
        "source": "solidifi",
        "contract_id": "d4b90b62c2ab33ce14d403f1d995b132e8178a09ca243db3be58b610c30cd297",
        "expect_under_cap": False,
        "expect_fallback": False,
        "expected_total_windows": 12,
        "expected_control_indices": [0, 4, 7, 11],
        "expected_selected_indices": [1, 4, 7, 10],
        "expected_control_target_coverage_ratio": 0.6095149830089589,
        "expected_selected_target_coverage_ratio": 0.6302131603336423,
        "expected_control_retained_ratio": 0.6166211707612982,
        "expected_retained_ratio": 0.618744313011829,
    },
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _max_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB. Sentinel's supported local runtime is Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _resolve_runtime_exception_source(parent_root: Path) -> str:
    matches = sorted(parent_root.glob(f"*/{RUNTIME_EXCEPTION_CONTRACT_ID}.rep.json"))
    if len(matches) != 1:
        raise GuardedTokenCandidateError(
            "D4 requires exactly one accepted parent identity for the declared "
            f"Slither runtime exception; found {len(matches)}"
        )
    return matches[0].parent.name


def _candidate_root(work_root: Path, repeat: str) -> Path:
    return work_root / repeat / GUARDED_REPRESENTATION_ROOT_NAME


def _load_token(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"token payload is not a mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload is not an object: {path}")
    return value


def _tensor_digest(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("input_ids", "attention_mask"):
        tensor = payload[key].detach().cpu().contiguous()
        digest.update(key.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(-1).numpy().tobytes())
    return digest.hexdigest()


def _validate_identity(
    *,
    case: dict[str, Any],
    parent_root: Path,
    first_root: Path,
    second_root: Path,
) -> dict[str, Any]:
    source = str(case["source"])
    contract_id = str(case["contract_id"])
    parent_dir = parent_root / source
    first_dir = first_root / source
    second_dir = second_root / source

    parent_graph = parent_dir / f"{contract_id}.pt"
    parent_tokens = parent_dir / f"{contract_id}.tokens.pt"
    parent_sidecar = parent_dir / f"{contract_id}.rep.json"
    first_graph = first_dir / f"{contract_id}.pt"
    first_tokens = first_dir / f"{contract_id}.tokens.pt"
    first_sidecar = first_dir / f"{contract_id}.rep.json"
    second_graph = second_dir / f"{contract_id}.pt"
    second_tokens = second_dir / f"{contract_id}.tokens.pt"
    second_sidecar = second_dir / f"{contract_id}.rep.json"

    required = (
        parent_graph,
        parent_tokens,
        parent_sidecar,
        first_graph,
        first_tokens,
        first_sidecar,
        second_graph,
        second_tokens,
        second_sidecar,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"D4 identity is incomplete: {missing}")

    historical = _load_token(parent_tokens)
    first = _load_token(first_tokens)
    second = _load_token(second_tokens)
    historical_meta = _load_json(parent_sidecar)
    first_meta = _load_json(first_sidecar)
    second_meta = _load_json(second_sidecar)

    decision = first.get("selector_decision")
    second_decision = second.get("selector_decision")
    if not isinstance(decision, dict) or not isinstance(second_decision, dict):
        raise ValueError(f"selector decision missing for {source}/{contract_id}")

    errors: list[str] = []
    first_shape = tuple(first["input_ids"].shape)
    first_mask_shape = tuple(first["attention_mask"].shape)
    second_shape = tuple(second["input_ids"].shape)
    historical_shape = tuple(historical["input_ids"].shape)
    expected_shape = tuple(TOKEN_TENSOR_SHAPE)

    if first_shape != expected_shape or first_mask_shape != expected_shape:
        errors.append(f"candidate shape changed: ids={first_shape} mask={first_mask_shape}")
    if second_shape != expected_shape or historical_shape != expected_shape:
        errors.append(
            f"repeat/control shape changed: repeat={second_shape} control={historical_shape}"
        )
    if first["input_ids"].dtype != torch.long or first["attention_mask"].dtype != torch.long:
        errors.append("candidate tensor dtype is not torch.int64")
    if not torch.equal(first["input_ids"], second["input_ids"]):
        errors.append("repeat input_ids are not deterministic")
    if not torch.equal(first["attention_mask"], second["attention_mask"]):
        errors.append("repeat attention_mask is not deterministic")
    if decision != second_decision:
        errors.append("repeat selector decisions differ")
    if first_meta != second_meta:
        errors.append("repeat sidecar metadata differs")

    parent_graph_sha = _sha256_file(parent_graph)
    first_graph_sha = _sha256_file(first_graph)
    second_graph_sha = _sha256_file(second_graph)
    if not (parent_graph_sha == first_graph_sha == second_graph_sha):
        errors.append("guarded graph bytes differ from immutable R4-D-011 parent")

    if first.get("token_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
        errors.append("fresh token lineage identity missing from token payload")
    if first_meta.get("token_lineage") != GUARDED_TOKEN_LINEAGE_VERSION:
        errors.append("fresh token lineage identity missing from sidecar")
    if first.get("token_selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
        errors.append("guarded selector policy missing from token payload")
    if first_meta.get("token_selector_policy") != GUARDED_TOKEN_SELECTOR_VERSION:
        errors.append("guarded selector policy missing from sidecar")
    if decision.get("requested_selector") != GUARDED_TOKEN_SELECTOR_VERSION:
        errors.append("selector decision does not request target_aware_guarded_v1")
    if decision.get("control_selector") != HISTORICAL_TOKEN_SELECTOR_VERSION:
        errors.append("selector decision control identity changed")

    if first.get("transformers_version") != GUARDED_TOKEN_TRANSFORMERS_VERSION:
        errors.append("token payload transformers runtime identity changed")
    if first_meta.get("transformers_version") != GUARDED_TOKEN_TRANSFORMERS_VERSION:
        errors.append("sidecar transformers runtime identity changed")
    if decision.get("transformers_version") != GUARDED_TOKEN_TRANSFORMERS_VERSION:
        errors.append("selector decision transformers runtime identity changed")

    historical_indices = [int(value) for value in historical["selected_window_indices"]]
    control_indices = [int(value) for value in decision["control_indices"]]
    selected_indices = [int(value) for value in decision["selected_indices"]]
    candidate_indices = [int(value) for value in decision["candidate_indices"]]
    if historical_indices != control_indices:
        errors.append("dynamic control indices differ from R4-D-011 bound indices")
    if selected_indices != sorted(set(selected_indices)):
        errors.append("selected indices are not sorted unique values")
    total_windows = int(decision["pre_subsampling_window_count"])
    if any(index < 0 or index >= total_windows for index in selected_indices):
        errors.append("selected index is outside pre-subsampling window bounds")

    selected_target = int(decision["selected_target_coverage_tokens"])
    control_target = int(decision["control_target_coverage_tokens"])
    candidate_target = int(decision["candidate_target_coverage_tokens"])
    control_target_ratio = float(decision["control_target_coverage_ratio"])
    selected_target_ratio = float(decision["selected_target_coverage_ratio"])
    used_fallback = bool(decision["used_control_fallback"])
    effective = str(decision["effective_selector"])

    if selected_target < control_target:
        errors.append("guarded output regressed accepted target-token coverage")
    if used_fallback:
        if selected_indices != control_indices:
            errors.append("control fallback did not select historical control indices")
        if effective != HISTORICAL_TOKEN_SELECTOR_VERSION:
            errors.append("fallback effective selector identity is incorrect")
        if not torch.equal(first["input_ids"], historical["input_ids"]):
            errors.append("control-fallback input_ids differ from R4-D-011 control tensor")
        if not torch.equal(first["attention_mask"], historical["attention_mask"]):
            errors.append("control-fallback mask differs from R4-D-011 control tensor")
    else:
        if effective != GUARDED_TOKEN_SELECTOR_VERSION:
            errors.append("strictly improved output is not labeled guarded")
        if selected_indices != candidate_indices:
            errors.append("guarded output does not use greedy candidate indices")
        if candidate_target <= control_target:
            errors.append("guarded candidate lacks strict target-coverage improvement")

    if "expect_under_cap" in case:
        actual_under_cap = total_windows <= TOKEN_TENSOR_SHAPE[0]
        if bool(case["expect_under_cap"]) != actual_under_cap:
            errors.append(
                f"retained evidence under-cap expectation changed: "
                f"expected={case['expect_under_cap']} actual={actual_under_cap}"
            )
    if "expect_fallback" in case and bool(case["expect_fallback"]) != used_fallback:
        errors.append(
            f"retained evidence fallback expectation changed: "
            f"expected={case['expect_fallback']} actual={used_fallback}"
        )

    if "expected_total_windows" in case and total_windows != int(
        case["expected_total_windows"]
    ):
        errors.append(
            "retained evidence window count changed: "
            f"expected={case['expected_total_windows']} actual={total_windows}"
        )
    if "expected_control_indices" in case and control_indices != [
        int(value) for value in case["expected_control_indices"]
    ]:
        errors.append(
            "retained evidence control indices changed: "
            f"expected={case['expected_control_indices']} actual={control_indices}"
        )
    if "expected_selected_indices" in case and selected_indices != [
        int(value) for value in case["expected_selected_indices"]
    ]:
        errors.append(
            "retained evidence selected indices changed: "
            f"expected={case['expected_selected_indices']} actual={selected_indices}"
        )
    if "expected_control_target_coverage_ratio" in case and not math.isclose(
        control_target_ratio,
        float(case["expected_control_target_coverage_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(
            "retained evidence control target coverage changed: "
            f"expected={case['expected_control_target_coverage_ratio']} "
            f"actual={control_target_ratio}"
        )
    if "expected_selected_target_coverage_ratio" in case and not math.isclose(
        selected_target_ratio,
        float(case["expected_selected_target_coverage_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(
            "retained evidence selected target coverage changed: "
            f"expected={case['expected_selected_target_coverage_ratio']} "
            f"actual={selected_target_ratio}"
        )

    parent_binding = first_meta.get("graph_parent") or {}
    if parent_binding.get("decision_id") != "R4-D-011":
        errors.append("candidate sidecar does not bind graph parent to R4-D-011")
    if parent_binding.get("graph_sha256") != parent_graph_sha:
        errors.append("candidate sidecar graph-parent hash is incorrect")
    if historical_meta.get("requested_contract_names") != decision.get(
        "requested_contract_names"
    ):
        errors.append("candidate target names differ from accepted graph target names")

    return {
        "purpose": str(case["purpose"]),
        "source": source,
        "contract_id": contract_id,
        "passed": not errors,
        "errors": errors,
        "total_windows": total_windows,
        "historical_indices": historical_indices,
        "candidate_indices": candidate_indices,
        "selected_indices": selected_indices,
        "used_control_fallback": used_fallback,
        "effective_selector": effective,
        "target_tokens": int(decision["target_tokens"]),
        "control_target_coverage_tokens": control_target,
        "candidate_target_coverage_tokens": candidate_target,
        "selected_target_coverage_tokens": selected_target,
        "control_target_coverage_ratio": control_target_ratio,
        "candidate_target_coverage_ratio": float(
            decision["candidate_target_coverage_ratio"]
        ),
        "selected_target_coverage_ratio": selected_target_ratio,
        "retained_unique_code_tokens": int(decision["retained_unique_code_tokens"]),
        "retained_token_ratio": float(decision["retained_token_ratio"]),
        "input_ids_shape": list(first_shape),
        "attention_mask_shape": list(first_mask_shape),
        "input_ids_dtype": str(first["input_ids"].dtype),
        "attention_mask_dtype": str(first["attention_mask"].dtype),
        "first_tensor_digest_sha256": _tensor_digest(first),
        "second_tensor_digest_sha256": _tensor_digest(second),
        "parent_graph_sha256": parent_graph_sha,
        "first_graph_sha256": first_graph_sha,
        "second_graph_sha256": second_graph_sha,
        "sidecar_repeat_equal": first_meta == second_meta,
    }


def _build_once(
    *,
    repeat: str,
    acceptance: Path,
    preprocessed_root: Path,
    parent_root: Path,
    work_root: Path,
    identities: list[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = _candidate_root(work_root, repeat)
    before_rss = _max_rss_mb()
    started = time.monotonic()
    manifest = build_guarded_token_candidate(
        acceptance_path=acceptance,
        repo_root=REPO_ROOT,
        preprocessed_root=preprocessed_root,
        parent_root=parent_root,
        output_root=output_root,
        identities=identities,
    )
    elapsed = time.monotonic() - started
    after_rss = _max_rss_mb()
    return manifest, {
        "repeat": repeat,
        "elapsed_seconds": elapsed,
        "max_rss_mb_before": before_rss,
        "max_rss_mb_after": after_rss,
        "candidate_root": str(output_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument(
        "--include-extra-stress",
        action="store_true",
        help="Also include the retained 403-window sensitivity identity.",
    )
    args = parser.parse_args()

    acceptance = args.acceptance.resolve()
    preprocessed_root = args.preprocessed_root.resolve()
    parent_root = args.parent_root.resolve()
    work_root = args.work_root.resolve()
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(
            f"D4 work root is not empty; use a fresh path: {work_root}"
        )
    work_root.mkdir(parents=True, exist_ok=True)

    cases = [dict(case) for case in EVIDENCE_CASES]
    runtime_source = _resolve_runtime_exception_source(parent_root)
    cases.append(
        {
            "purpose": "accepted_v10_slither_runtime_exception_graph_parent",
            "source": runtime_source,
            "contract_id": RUNTIME_EXCEPTION_CONTRACT_ID,
        }
    )
    if args.include_extra_stress:
        source, contract_id = OPTIONAL_EXTRA_STRESS
        cases.append(
            {
                "purpose": "optional_403_window_sensitivity_stress",
                "source": source,
                "contract_id": contract_id,
            }
        )

    identities = [
        (str(case["source"]), str(case["contract_id"]))
        for case in cases
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("D4 case set contains duplicate identities")

    first_manifest, first_runtime = _build_once(
        repeat="repeat-a",
        acceptance=acceptance,
        preprocessed_root=preprocessed_root,
        parent_root=parent_root,
        work_root=work_root,
        identities=identities,
    )
    second_manifest, second_runtime = _build_once(
        repeat="repeat-b",
        acceptance=acceptance,
        preprocessed_root=preprocessed_root,
        parent_root=parent_root,
        work_root=work_root,
        identities=identities,
    )

    first_root = _candidate_root(work_root, "repeat-a")
    second_root = _candidate_root(work_root, "repeat-b")
    results = [
        _validate_identity(
            case=case,
            parent_root=parent_root,
            first_root=first_root,
            second_root=second_root,
        )
        for case in cases
    ]
    failures = [result for result in results if not result["passed"]]

    if first_manifest["contracts_written"] != len(cases):
        failures.append(
            {
                "purpose": "first_manifest_population",
                "passed": False,
                "errors": [
                    f"first manifest wrote {first_manifest['contracts_written']} "
                    f"of {len(cases)} D4 identities"
                ],
            }
        )
    if second_manifest["contracts_written"] != len(cases):
        failures.append(
            {
                "purpose": "second_manifest_population",
                "passed": False,
                "errors": [
                    f"second manifest wrote {second_manifest['contracts_written']} "
                    f"of {len(cases)} D4 identities"
                ],
            }
        )

    status = "PASS_BOUNDED_D4_REVIEW_REQUIRED" if not failures else "FAIL"
    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "source_commit": _source_commit(),
        "implementation_sha256": _sha256_file(Path(__file__)),
        "acceptance_manifest_sha256": _sha256_file(acceptance),
        "parent_root": str(parent_root),
        "preprocessed_root": str(preprocessed_root),
        "work_root": str(work_root),
        "selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
        "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
        "guarded_token_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
        "transformers_version": transformers.__version__,
        "required_transformers_version": GUARDED_TOKEN_TRANSFORMERS_VERSION,
        "frozen_token_shape": list(TOKEN_TENSOR_SHAPE),
        "identities_requested": len(cases),
        "identities_passed": sum(result.get("passed", False) for result in results),
        "identity_failures": len(
            [result for result in results if not result.get("passed", False)]
        ),
        "runtime_exception_identity": {
            "source": runtime_source,
            "contract_id": RUNTIME_EXCEPTION_CONTRACT_ID,
        },
        "optional_extra_stress_included": bool(args.include_extra_stress),
        "runtime": {
            "repeat_a": first_runtime,
            "repeat_b": second_runtime,
        },
        "manifest_summary": {
            "repeat_a_effective_selector_counts": first_manifest[
                "effective_selector_counts"
            ],
            "repeat_b_effective_selector_counts": second_manifest[
                "effective_selector_counts"
            ],
        },
        "results": results,
        "failures": failures,
        "physical_acceptance": False,
        "d5_authorized": False,
        "training_authorized": False,
        "review_required": True,
        "decision_boundary": (
            "A bounded PASS satisfies executable D4 evidence only after review. "
            "It does not accept the guarded physical lineage, does not itself "
            "authorize D5 full-population generation, and does not authorize training."
        ),
    }
    report_path = work_root / "d4_guarded_token_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
