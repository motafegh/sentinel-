#!/usr/bin/env python3
"""Audit the complete accepted-v9 to candidate-v10 representation transition.

This is a protected-local diagnostic. It independently rechecks population
identity, token byte equality, graph/sidecar versioning, call IR-to-edge
reconciliation, and aggregate call-kind transitions. It never accepts the
candidate or authorizes training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from sentinel_data.preprocessing.r4_versions import (
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.graph_schema_versions import get_graph_schema


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[tuple[str, str], Path]:
    return {
        (path.parent.name, path.name.removesuffix(".rep.json")): path
        for path in sorted(root.glob("*/*.rep.json"))
    }


def _graph_call_counts(graph: Any, edge_types: dict[str, int]) -> dict[str, int]:
    return {
        name: int((graph.edge_attr == edge_id).sum())
        for name, edge_id in edge_types.items()
        if edge_id >= 11
    }


def _edge_topology_equal_through(
    left: Any, right: Any, max_edge_type: int
) -> bool:
    """Compare edge multisets through an unchanged type ID, ignoring order."""

    def topology(graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
        mask = graph.edge_attr <= max_edge_type
        triples = torch.stack(
            (
                graph.edge_attr[mask],
                graph.edge_index[0, mask],
                graph.edge_index[1, mask],
            ),
            dim=1,
        )
        return torch.unique(triples, dim=0, sorted=True, return_counts=True)

    left_unique, left_counts = topology(left)
    right_unique, right_counts = topology(right)
    return torch.equal(left_unique, right_unique) and torch.equal(
        left_counts, right_counts
    )


_PARSE_ONLY_SOURCE_PATTERNS = {
    "raw_low_level": re.compile(
        r"\.(?:call|callcode|delegatecall|staticcall)\b"
        r"(?:\s*\.\s*(?:value|gas)\s*\([^)]*\))*\s*\("
    ),
    "ether_transfer": re.compile(r"\.transfer\s*\("),
    "ether_send": re.compile(r"\.send\s*\("),
    "contract_creation": re.compile(r"\bnew\s+[A-Za-z_]\w*"),
}


def _parse_only_source_syntax_hits(path: Path) -> dict[str, int]:
    """Return diagnostic raw-text hits, never semantic call counts."""

    source = path.read_text(encoding="utf-8")
    return {
        name: len(pattern.findall(source))
        for name, pattern in _PARSE_ONLY_SOURCE_PATTERNS.items()
    }


def _historical_v9_extraction_mode(sidecar: dict[str, Any]) -> str:
    """Resolve the implicit normal mode used by old accepted-v9 sidecars."""

    return str(sidecar.get("graph_extraction_mode") or "slither_full_analysis")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_root.name != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError("candidate root has the wrong versioned name")
    binding_path = args.candidate_root / "v10_candidate_binding_report.json"
    if not binding_path.is_file():
        raise FileNotFoundError(binding_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("passed") is not True:
        raise ValueError("transition audit requires a passing diagnostic binding")
    if binding.get("physical_acceptance") is not False:
        raise ValueError("candidate binding unexpectedly claims physical acceptance")
    if binding.get("training_authorized") is not False:
        raise ValueError("candidate binding unexpectedly authorizes training")
    reference_binding_path = args.reference_v10_root / "v10_candidate_binding_report.json"
    if not reference_binding_path.is_file():
        raise FileNotFoundError(reference_binding_path)
    reference_binding = json.loads(reference_binding_path.read_text(encoding="utf-8"))
    if reference_binding.get("passed") is not True:
        raise ValueError("transition audit requires a passing structural reference")

    accepted = _inventory(args.accepted_v9_root)
    candidate = _inventory(args.candidate_root)
    reference = _inventory(args.reference_v10_root)
    accepted_keys = set(accepted)
    candidate_keys = set(candidate)
    reference_keys = set(reference)
    schema = get_graph_schema(V10_GRAPH_SCHEMA_VERSION)
    errors: list[dict[str, str]] = []
    totals: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    compatibility_contracts: list[dict[str, Any]] = []
    structural_drift_contracts: list[dict[str, Any]] = []

    for source, contract_id in sorted(accepted_keys - candidate_keys):
        errors.append({"contract": f"{source}/{contract_id}", "detail": "missing v10 identity"})
    for source, contract_id in sorted(candidate_keys - accepted_keys):
        errors.append({"contract": f"{source}/{contract_id}", "detail": "extra v10 identity"})
    for source, contract_id in sorted(accepted_keys - reference_keys):
        errors.append(
            {
                "contract": f"{source}/{contract_id}",
                "detail": "missing structural-reference identity",
            }
        )
    for source, contract_id in sorted(reference_keys - accepted_keys):
        errors.append(
            {
                "contract": f"{source}/{contract_id}",
                "detail": "extra structural-reference identity",
            }
        )

    for ordinal, (source, contract_id) in enumerate(
        sorted(accepted_keys & candidate_keys), start=1
    ):
        logical = f"{source}/{contract_id}"
        try:
            v9_dir = accepted[(source, contract_id)].parent
            v10_dir = candidate[(source, contract_id)].parent
            v9_graph = torch.load(
                v9_dir / f"{contract_id}.pt", map_location="cpu", weights_only=False
            )
            v10_graph = torch.load(
                v10_dir / f"{contract_id}.pt", map_location="cpu", weights_only=False
            )
            reference_dir = reference[(source, contract_id)].parent
            reference_graph = torch.load(
                reference_dir / f"{contract_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            sidecar = json.loads(
                (v10_dir / f"{contract_id}.rep.json").read_text(encoding="utf-8")
            )
            v9_sidecar = json.loads(
                (v9_dir / f"{contract_id}.rep.json").read_text(encoding="utf-8")
            )
            reference_sidecar = json.loads(
                (reference_dir / f"{contract_id}.rep.json").read_text(
                    encoding="utf-8"
                )
            )
            if reference_sidecar.get("schema_version") != V10_GRAPH_SCHEMA_VERSION:
                raise ValueError("structural-reference graph schema mismatch")
            if reference_sidecar.get("extractor_version") != "v2.3-r4-call-semantics":
                raise ValueError("structural-reference extractor mismatch")
            if sidecar.get("schema_version") != V10_GRAPH_SCHEMA_VERSION:
                raise ValueError("sidecar schema mismatch")
            if sidecar.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
                raise ValueError("sidecar extractor mismatch")
            if sidecar.get("token_lineage") != "accepted_v9_byte_copy":
                raise ValueError("sidecar token lineage mismatch")
            if list(sidecar.get("unclassified_call_ir") or []):
                raise ValueError("unclassified call IR remains")
            if list(sidecar.get("call_mapping_errors") or []):
                raise ValueError("call-to-graph mapping error remains")
            classified = dict(sidecar.get("classified_call_ir_counts") or {})
            emitted = dict(sidecar.get("emitted_call_edge_counts") or {})
            observed = _graph_call_counts(v10_graph, dict(schema.edge_types))
            if classified != emitted or emitted != observed:
                raise ValueError("classified/emitted/observed call counts differ")
            if getattr(v10_graph, "graph_schema_version", None) != V10_GRAPH_SCHEMA_VERSION:
                raise ValueError("graph payload schema mismatch")
            if (
                getattr(v10_graph, "representation_extractor_version", None)
                != V10_REPRESENTATION_EXTRACTOR_VERSION
            ):
                raise ValueError("graph payload extractor mismatch")
            if _sha256(v9_dir / f"{contract_id}.tokens.pt") != _sha256(
                v10_dir / f"{contract_id}.tokens.pt"
            ):
                raise ValueError("token bytes changed")

            v9_external = int((v9_graph.edge_attr == 11).sum())
            totals["graphs_checked"] += 1
            totals["v9_external_call_edges"] += v9_external
            source_totals = by_source.setdefault(source, Counter())
            source_totals["graphs_checked"] += 1
            source_totals["v9_external_call_edges"] += v9_external
            # Normal accepted-v9 sidecars predate this explicit field; only
            # compatibility cases persisted it. Historical absence therefore
            # means the normal full-analysis path, not missing provenance.
            v9_mode = _historical_v9_extraction_mode(v9_sidecar)
            v10_mode = str(sidecar.get("graph_extraction_mode") or "MISSING")
            v9_parse_only = v9_mode.startswith("slither_parse_only")

            v9_call_entry = int((v9_graph.edge_attr == 8).sum())
            v10_call_entry = int((v10_graph.edge_attr == 8).sum())
            v9_return_to = int((v9_graph.edge_attr == 9).sum())
            v10_return_to = int((v10_graph.edge_attr == 9).sum())
            for label, value in (
                ("v9_call_entry_edges", v9_call_entry),
                ("v10_call_entry_edges", v10_call_entry),
                ("v9_return_to_edges", v9_return_to),
                ("v10_return_to_edges", v10_return_to),
            ):
                totals[label] += value
                source_totals[label] += value
            for label, value in (
                ("graphs_with_v9_call_entry", v9_call_entry),
                ("graphs_with_v10_call_entry", v10_call_entry),
                ("graphs_with_v9_return_to", v9_return_to),
                ("graphs_with_v10_return_to", v10_return_to),
            ):
                totals[label] += bool(value)
                source_totals[label] += bool(value)

            node_features_equal = torch.equal(reference_graph.x, v10_graph.x)
            node_metadata_equal = getattr(
                reference_graph, "node_metadata", None
            ) == getattr(
                v10_graph, "node_metadata", None
            )
            non_call_topology_equal = _edge_topology_equal_through(
                reference_graph, v10_graph, 10
            )
            structural_drift = not (
                node_features_equal
                and node_metadata_equal
                and non_call_topology_equal
            )
            if structural_drift:
                totals[
                    "graphs_with_structural_drift_from_reference_through_edge_10"
                ] += 1
                source_totals[
                    "graphs_with_structural_drift_from_reference_through_edge_10"
                ] += 1
                if not v9_parse_only:
                    totals[
                        "graphs_with_unexpected_structural_drift_from_reference_through_edge_10"
                    ] += 1
                    source_totals[
                        "graphs_with_unexpected_structural_drift_from_reference_through_edge_10"
                    ] += 1
                if len(structural_drift_contracts) < args.max_errors:
                    structural_drift_contracts.append(
                        {
                            "contract": logical,
                            "v9_parse_only": v9_parse_only,
                            "node_features_equal": node_features_equal,
                            "node_metadata_equal": node_metadata_equal,
                            "non_call_edge_topology_equal": non_call_topology_equal,
                            "v9_call_entry_edges": v9_call_entry,
                            "v10_call_entry_edges": v10_call_entry,
                            "v9_return_to_edges": v9_return_to,
                            "v10_return_to_edges": v10_return_to,
                        }
                    )
            totals[f"v9_extraction_mode_{v9_mode}"] += 1
            totals[f"v10_extraction_mode_{v10_mode}"] += 1
            source_totals[f"v9_extraction_mode_{v9_mode}"] += 1
            source_totals[f"v10_extraction_mode_{v10_mode}"] += 1
            if v9_mode != v10_mode:
                totals["graphs_with_changed_extraction_mode"] += 1
                source_totals["graphs_with_changed_extraction_mode"] += 1
            if v10_mode != "slither_full_analysis":
                if v10_mode.startswith("slither_parse_only"):
                    semantic_completeness = "IR_CALL_EDGES_NOT_COMPLETE"
                elif sidecar.get("graph_analyzer_repair") is not None:
                    semantic_completeness = (
                        "FULL_ANALYSIS_WITH_RECORDED_ANALYZER_REPAIR"
                    )
                elif sidecar.get("graph_source_transform") is not None:
                    semantic_completeness = (
                        "FULL_ANALYSIS_WITH_RECORDED_SOURCE_TRANSFORM"
                    )
                else:
                    semantic_completeness = "FULL_ANALYSIS_COMPATIBILITY_MODE"
                compatibility_record = {
                    "contract": logical,
                    "v9_extraction_mode": v9_mode,
                    "v10_extraction_mode": v10_mode,
                    "semantic_completeness": semantic_completeness,
                    "graph_analyzer_repair": sidecar.get(
                        "graph_analyzer_repair"
                    ),
                    "graph_source_transform": sidecar.get(
                        "graph_source_transform"
                    ),
                }
                if v10_mode.startswith("slither_parse_only"):
                    source_path = args.preprocessed_root / source / f"{contract_id}.sol"
                    if not source_path.is_file():
                        raise FileNotFoundError(source_path)
                    syntax_hits = _parse_only_source_syntax_hits(source_path)
                    compatibility_record["source_call_syntax_hits"] = syntax_hits
                    for name, count in syntax_hits.items():
                        totals[f"v10_parse_only_source_{name}_hits"] += count
                        source_totals[f"v10_parse_only_source_{name}_hits"] += count
                compatibility_contracts.append(compatibility_record)
            if v10_mode.startswith("slither_parse_only"):
                totals["v10_parse_only_contracts"] += 1
                source_totals["v10_parse_only_contracts"] += 1
            if v9_external != sum(observed.values()):
                totals["graphs_with_changed_total_call_edges"] += 1
                source_totals["graphs_with_changed_total_call_edges"] += 1
            for name, count in observed.items():
                totals[f"v10_{name.lower()}_edges"] += count
                source_totals[f"v10_{name.lower()}_edges"] += count
                if count:
                    totals[f"graphs_with_v10_{name.lower()}"] += 1
                    source_totals[f"graphs_with_v10_{name.lower()}"] += 1
        except Exception as exc:
            errors.append({"contract": logical, "detail": str(exc)})
        if args.progress_every and ordinal % args.progress_every == 0:
            print(
                f"audited {ordinal}/{len(accepted_keys & candidate_keys)} transitions",
                file=sys.stderr,
                flush=True,
            )

    passed = (
        not errors
        and totals["graphs_checked"] == len(accepted)
        and int(binding.get("checked_contracts", -1)) == len(accepted)
    )
    structural_blocked = bool(
        totals[
            "graphs_with_unexpected_structural_drift_from_reference_through_edge_10"
        ]
    )
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]
    implementation_paths = (
        Path("data_module/sentinel_data/preprocessing/r4_versions.py"),
        Path("data_module/sentinel_data/representation/_schema_version_registry.json"),
        Path("data_module/sentinel_data/representation/graph_schema_versions.py"),
        Path("data_module/sentinel_data/representation/graph_extractor.py"),
        Path("data_module/sentinel_data/representation/r4_compatibility.py"),
        Path("data_module/sentinel_data/representation/r4_orchestrator.py"),
        Path("data_module/sentinel_data/vnext/r4_v10_binding.py"),
        Path("ml/src/models/gnn_encoder.py"),
        Path("ml/src/models/sentinel_model.py"),
        Path("ml/src/datasets/vnext_logical_v3_v10_dataset.py"),
        Path("ml/src/training/vnext_binding.py"),
        Path("docs/plan/ml-R4/scripts/p8_generate_v10_candidate.py"),
        Path("docs/plan/ml-R4/scripts/p8_audit_v10_transition.py"),
    )
    return {
        "schema": "sentinel-r4-v9-to-v10-transition-audit-v2",
        "passed": passed,
        "status": (
            "PASS_DIAGNOSTIC_WITH_COMPATIBILITY_BLOCKER"
            if passed and totals["v10_parse_only_contracts"]
            else "PASS_DIAGNOSTIC_WITH_STRUCTURAL_BLOCKER"
            if passed and structural_blocked
            else "PASS_DIAGNOSTIC_ONLY" if passed else "FAIL"
        ),
        "accepted_v9_contracts": len(accepted),
        "candidate_v10_contracts": len(candidate),
        "structural_reference_v10_contracts": len(reference),
        "structural_reference_binding_report_sha256": _sha256(
            reference_binding_path
        ),
        "structural_reference_binding_digest_sha256": reference_binding.get(
            "binding_digest_sha256"
        ),
        "candidate_binding_report_sha256": _sha256(binding_path),
        "candidate_binding_digest_sha256": binding.get("binding_digest_sha256"),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "repository_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True
            ).strip()
        ),
        "implementation_sha256": {
            path.as_posix(): _sha256(repo_root / path) for path in implementation_paths
        },
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "totals": dict(sorted(totals.items())),
        "by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source.items())
        },
        "errors_total": len(errors),
        "errors": errors[: args.max_errors],
        "errors_truncated": max(0, len(errors) - args.max_errors),
        "compatibility_contracts": compatibility_contracts,
        "structural_drift_contracts": structural_drift_contracts,
        "structural_drift_records_truncated": max(
            0,
            totals["graphs_with_structural_drift_from_reference_through_edge_10"]
            - len(structural_drift_contracts),
        ),
        "physical_acceptance_blockers": (
            [
                "parse-only compatibility graphs lack complete Slither IR call edges; "
                "resolve through versioned extraction repair, explicit exclusion/role "
                "decision, or complete source-level reconciliation"
            ]
            if totals["v10_parse_only_contracts"]
            else []
        )
        + (
            [
                "node features, metadata, or edge topology through unchanged edge type 10 drifted from the passing Slither-0.10 V10 structural reference outside historical v9 parse-only contracts"
            ]
            if structural_blocked
            else []
        ),
        "physical_acceptance": False,
        "training_authorized": False,
        "limitations": [
            "This proves representation mechanics and Slither-IR-to-edge reconciliation, not vulnerability labels.",
            *(
                [
                    "Parse-only compatibility graphs cannot prove complete IR call semantics and block physical acceptance until explicitly resolved."
                ]
                if totals["v10_parse_only_contracts"]
                else []
            ),
            *(
                [
                    "Structural drift from the passing Slither-0.10 V10 reference outside historical parse-only contracts blocks physical acceptance even when call-kind reconciliation passes."
                ]
                if structural_blocked
                else []
            ),
            "Physical acceptance still requires review of this report, bounded source regressions, tests, and an explicit decision record.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-v9-root", type=Path, default=Path("data_module/data/representations-r4-v2"))
    parser.add_argument("--candidate-root", type=Path, default=Path("data_module/data/representations-r4-v3-candidate"))
    parser.add_argument("--reference-v10-root", type=Path, required=True)
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data_module/data/sentinel-preprocessed-r4-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-errors", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
