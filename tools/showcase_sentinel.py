#!/usr/bin/env python3
"""Fresh-clone SENTINEL portfolio showcase.

This script is intentionally dependency-light. It inspects committed source and
configuration to demonstrate a few high-value SENTINEL boundaries without
pretending that local ML artifacts, analyzers, proving material, or transaction
signing are available.

It exits non-zero if a checked repository invariant no longer matches the
current portfolio contract. Live/runtime capabilities that are not exercised
are emitted explicitly as NOT_RUN rather than being reported as clean.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_GRAPH_NODES = [
    "ml_assessment",
    "quick_screen",
    "evidence_router",
    "rag_research",
    "static_analysis",
    "graph_explain",
    "formal_verification",
    "audit_check",
    "consensus_engine",
    "cross_validator",
    "synthesizer",
    "reflection",
    "explainer",
    "visualizer",
]
EXPECTED_AUDIT_TOOLS = {
    "check_audit_exists",
    "get_audit_history",
    "get_latest_audit",
}
EXPECTED_PROXY_DIMS = [128, 64, 32, 10]
EXPECTED_PUBLIC_SIGNALS = 138


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _record(name: str, status: str, detail: str, evidence: str) -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "evidence": evidence,
    }


def _graph_observation() -> dict[str, str]:
    path = "agents/src/orchestration/graph.py"
    source = _read(path)
    nodes = re.findall(r'graph\.add_node\(\s*"([a-z_]+)"', source)
    entry_match = re.search(r'graph\.set_entry_point\(\s*"([a-z_]+)"\s*\)', source)
    exit_match = re.search(r'graph\.add_edge\(\s*"([a-z_]+)"\s*,\s*END\s*\)', source)

    entry = entry_match.group(1) if entry_match else None
    exit_node = exit_match.group(1) if exit_match else None
    passed = nodes == EXPECTED_GRAPH_NODES and entry == "ml_assessment" and exit_node == "visualizer"
    return _record(
        "orchestration_topology",
        "PASS" if passed else "FAIL",
        f"nodes={len(nodes)} entry={entry} exit={exit_node}",
        path,
    )


def _audit_mcp_observation() -> dict[str, str]:
    path = "agents/src/mcp/servers/audit/_readonly_handlers.py"
    source = _read(path)
    block = re.search(
        r"READ_ONLY_TOOLS\s*=\s*frozenset\(\s*\{(?P<body>.*?)\}\s*\)",
        source,
        flags=re.DOTALL,
    )
    tools = set(re.findall(r'"([a-z_]+)"', block.group("body"))) if block else set()
    has_rejection_gate = "if name not in READ_ONLY_TOOLS" in source
    passed = tools == EXPECTED_AUDIT_TOOLS and has_rejection_gate
    return _record(
        "audit_mcp_surface",
        "PASS" if passed else "FAIL",
        "read_only=true tools=" + ",".join(sorted(tools)),
        path,
    )


def _proxy_observation() -> dict[str, str]:
    model_path = "zkml/src/distillation/proxy_model.py"
    settings_path = "zkml/ezkl/settings.json"
    model_source = _read(model_path)
    settings = json.loads(_read(settings_path))

    def const(name: str) -> int | None:
        match = re.search(rf"\b{name}\s*=\s*(\d+)", model_source)
        return int(match.group(1)) if match else None

    dims = [
        const("FROZEN_INPUT_DIM"),
        const("FROZEN_HIDDEN1"),
        const("FROZEN_HIDDEN2"),
        const("FROZEN_NUM_CLASSES"),
    ]
    public_signals = sum(int(shape[-1]) for shape in settings.get("model_instance_shapes", []))
    check_mode = settings.get("run_args", {}).get("check_mode", settings.get("check_mode"))
    passed = dims == EXPECTED_PROXY_DIMS and public_signals == EXPECTED_PUBLIC_SIGNALS and check_mode == "UNSAFE"
    status = "PASS_WITH_LIMITATION" if passed else "FAIL"
    return _record(
        "zkml_proxy_boundary",
        status,
        f"proxy={'→'.join(map(str, dims))} public_signals={public_signals} check_mode={check_mode}",
        f"{model_path}; {settings_path}",
    )


def _r4_observation() -> dict[str, str]:
    path = "docs/plan/ml-R4/PLAN_STATUS_MATRIX.md"
    source = _read(path)
    phase8 = re.search(
        r"^\|\s*8\s*\|[^\n]*\|\s*(IN_PROGRESS)\s*\|",
        source,
        flags=re.MULTILINE,
    )
    digest = re.search(r"R4-D-011[^\n]*?`([0-9a-f]{64})`", source)
    full_training_held = (
        "full training is unauthorized" in source.lower()
        or re.search(r"^\| Full training / G8 \| HOLD \|", source, flags=re.MULTILINE) is not None
    )
    confirmed_negative_zero = "confirmed negatives remain zero" in source.lower()

    passed = bool(phase8 and digest and full_training_held and confirmed_negative_zero)
    short_digest = f"{digest.group(1)[:12]}…" if digest else "missing"
    return _record(
        "r4_authority",
        "PASS" if passed else "FAIL",
        (
            f"phase8={phase8.group(1) if phase8 else 'missing'} "
            f"d011_digest={short_digest} full_training_authorized=false "
            f"confirmed_negatives=0"
        ),
        path,
    )


def _not_run_boundaries() -> list[dict[str, str]]:
    return [
        _record(
            "live_ml_inference",
            "NOT_RUN",
            "Run12 checkpoint/service availability is local and not guaranteed in a fresh clone.",
            "ml runtime",
        ),
        _record(
            "external_analyzers_and_formal_tools",
            "NOT_RUN",
            "Slither/Aderyn/Halmos-style live execution requires module/toolchain prerequisites.",
            "agents runtime",
        ),
        _record(
            "live_langgraph_audit",
            "NOT_RUN",
            "The showcase inspects committed orchestration topology; it does not claim a live audit completed.",
            "agents runtime",
        ),
        _record(
            "zk_proof_generation",
            "NOT_RUN",
            "Tracked proof metadata is inspected, but witness/proving prerequisites are not assumed.",
            "zkml runtime",
        ),
        _record(
            "v3_signing_and_broadcast",
            "NOT_RUN",
            "No production signer/broadcaster is claimed by the current analysis service.",
            "V3 transaction authority",
        ),
    ]


def build_report() -> dict[str, Any]:
    checked = [
        _graph_observation(),
        _audit_mcp_observation(),
        _proxy_observation(),
        _r4_observation(),
    ]
    failures = [item for item in checked if item["status"] == "FAIL"]
    return {
        "showcase": "SENTINEL fresh-clone boundary showcase",
        "overall": "PASS" if not failures else "FAIL",
        "checked": checked,
        "not_run": _not_run_boundaries(),
        "claim_boundary": (
            "PASS means committed source/config matches the showcased architecture and governance boundaries. "
            "It does not establish live-service availability, model quality, vulnerability correctness, "
            "proof generation, or transaction submission."
        ),
    }


def _print_human(report: dict[str, Any]) -> None:
    print("SENTINEL fresh-clone boundary showcase")
    print("=" * 42)
    for item in report["checked"]:
        print(f"[{item['status']}] {item['name']}: {item['detail']}")
    print("\nExplicitly not exercised:")
    for item in report["not_run"]:
        print(f"[{item['status']}] {item['name']}: {item['detail']}")
    print(f"\nOverall: {report['overall']}")
    print(report["claim_boundary"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    try:
        report = build_report()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SHOWCASE_ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
