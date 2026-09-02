#!/usr/bin/env python3
"""Verify full-population historical-control token equivalence for R4-D-011.

This read-only verifier dynamically retokenizes accepted preprocessed sources
through the research selector's historical control path and compares the result
with the exact token payloads bound to the accepted V10 V2.6 representation.
It never rewrites representation artifacts and grants no selector or training
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json"
)
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_REPRESENTATIONS = (
    DATA_ROOT
    / "v10-v26-full-candidate-attempt-2026-09-01-a/representations-r4-v3-candidate"
)
DEFAULT_OUTPUT = (
    DATA_ROOT
    / "v10-v26-full-candidate-attempt-2026-09-01-a/selector-control-equivalence-v1.json"
)
REPORT_SCHEMA = "sentinel-r4-v10-selector-control-equivalence-v1"
MAX_FAILURE_SAMPLES = 100

_TOKENIZER: Any | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _init_worker() -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer
    from ml.src.data_extraction.windowed_tokenizer import TOKENIZER_MODEL

    _TOKENIZER = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        use_fast=True,
        local_files_only=True,
    )


def _tensor_digest(input_ids: Any, attention_mask: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in (
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(-1).numpy().tobytes())
    return digest.hexdigest()


def _check_identity(task: tuple[str, str, str, str]) -> dict[str, Any]:
    source, contract_id, representations_root, preprocessed_root = task
    try:
        import torch

        from ml.src.data_extraction.bounded_window_selector import (
            CONTROL_STRATEGY,
            tokenize_with_selector,
        )
        from sentinel_data.representation.r4_target_spans import (
            target_contract_char_spans,
        )

        if _TOKENIZER is None:
            raise RuntimeError("worker tokenizer was not initialized")

        rep_dir = Path(representations_root) / source
        token_path = rep_dir / f"{contract_id}.tokens.pt"
        sidecar_path = rep_dir / f"{contract_id}.rep.json"
        source_path = Path(preprocessed_root) / source / f"{contract_id}.sol"

        source_text = source_path.read_text(encoding="utf-8")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        targets = [
            str(value)
            for value in (sidecar.get("requested_contract_names") or ())
        ]
        if not targets:
            raise ValueError("sidecar has no requested_contract_names")
        char_spans = target_contract_char_spans(source_text, targets)
        dynamic = tokenize_with_selector(
            source_text,
            target_char_spans=char_spans,
            tokenizer=_TOKENIZER,
            strategy=CONTROL_STRATEGY,
        )
        bound = torch.load(token_path, map_location="cpu", weights_only=True)

        input_equal = torch.equal(dynamic["input_ids"], bound["input_ids"])
        mask_equal = torch.equal(
            dynamic["attention_mask"], bound["attention_mask"]
        )
        dynamic_indices = list(dynamic["selector"]["selected_indices"])
        bound_indices = [int(value) for value in bound["selected_window_indices"]]
        sidecar_indices = [
            int(value) for value in sidecar.get("selected_window_indices", ())
        ]
        indices_equal = (
            dynamic_indices == bound_indices
            and (not sidecar_indices or dynamic_indices == sidecar_indices)
        )

        bound_digest = _tensor_digest(bound["input_ids"], bound["attention_mask"])
        dynamic_digest = _tensor_digest(
            dynamic["input_ids"], dynamic["attention_mask"]
        )
        passed = input_equal and mask_equal and indices_equal
        return {
            "contract_id": contract_id,
            "source": source,
            "passed": passed,
            "input_ids_equal": input_equal,
            "attention_mask_equal": mask_equal,
            "selected_window_indices_equal": indices_equal,
            "bound_tensor_digest_sha256": bound_digest,
            "dynamic_tensor_digest_sha256": dynamic_digest,
            "selected_window_indices": dynamic_indices,
            "total_windows": int(dynamic["total_windows"]),
        }
    except Exception as exc:
        return {
            "contract_id": contract_id,
            "source": source,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _population_digest(results: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for result in sorted(
        results, key=lambda value: (str(value["source"]), str(value["contract_id"]))
    ):
        fields = (
            str(result["source"]),
            str(result["contract_id"]),
            str(result.get("bound_tensor_digest_sha256") or ""),
            json.dumps(result.get("selected_window_indices") or [], separators=(",", ":")),
            "1" if result.get("passed") else "0",
        )
        digest.update("\0".join(fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument(
        "--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED
    )
    parser.add_argument(
        "--representations-root", type=Path, default=DEFAULT_REPRESENTATIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")

    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    lineage = acceptance["accepted_lineage"]
    expected_root = (REPO_ROOT / str(lineage["physical_root"])).resolve()
    actual_root = args.representations_root.resolve()
    if expected_root != actual_root:
        raise ValueError(
            f"representation root is not the R4-D-011 root: {actual_root} != {expected_root}"
        )
    if acceptance.get("decision_id") != "R4-D-011":
        raise ValueError("acceptance manifest is not R4-D-011")
    if acceptance.get("physical_acceptance") is not True:
        raise ValueError("acceptance manifest does not grant physical acceptance")

    token_paths = sorted(actual_root.glob("*/*.tokens.pt"))
    expected_contracts = int(lineage["contracts"])
    if len(token_paths) != expected_contracts:
        raise ValueError(
            f"accepted population mismatch: {len(token_paths)} != {expected_contracts}"
        )
    identities = [
        (path.parent.name, path.name[: -len(".tokens.pt")]) for path in token_paths
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("accepted population contains duplicate source/contract identities")
    selected = identities[: args.limit] if args.limit is not None else identities
    tasks = [
        (source, contract_id, str(actual_root), str(args.preprocessed_root.resolve()))
        for source, contract_id in selected
    ]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker
    ) as executor:
        for index, result in enumerate(executor.map(_check_identity, tasks, chunksize=16), 1):
            results.append(result)
            if args.progress_every and index % args.progress_every == 0:
                failures = sum(not item.get("passed", False) for item in results)
                print(f"checked={index}/{len(tasks)} failures={failures}", flush=True)

    failures = [result for result in results if not result.get("passed", False)]
    full_population = args.limit is None and len(results) == expected_contracts
    passed = full_population and not failures
    partial_pass = not full_population and not failures
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if passed else ("PASS_PARTIAL_DIAGNOSTIC" if partial_pass else "FAIL"),
        "decision_boundary": (
            "Control equivalence is prerequisite evidence only; selector promotion and "
            "training remain unauthorized."
        ),
        "source_commit": _source_commit(),
        "implementation_sha256": _sha256_file(Path(__file__)),
        "acceptance_manifest_sha256": _sha256_file(args.acceptance),
        "decision_id": acceptance["decision_id"],
        "physical_root": str(lineage["physical_root"]),
        "binding_digest_sha256": str(lineage["binding_digest_sha256"]),
        "extractor_version": str(lineage["extractor_version"]),
        "tokenizer": "microsoft/graphcodebert-base",
        "strategy": "historical_linspace_v1",
        "workers": args.workers,
        "expected_contracts": expected_contracts,
        "enumerated_contracts": len(identities),
        "checked_contracts": len(results),
        "full_population": full_population,
        "matching_contracts": len(results) - len(failures),
        "mismatching_or_failed_contracts": len(failures),
        "population_result_digest_sha256": _population_digest(results),
        "failure_samples": failures[:MAX_FAILURE_SAMPLES],
        "failure_samples_truncated": max(0, len(failures) - MAX_FAILURE_SAMPLES),
        "selector_promotion_authorized": False,
        "training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed or partial_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
