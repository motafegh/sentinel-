#!/usr/bin/env python3
"""Validate and inventory the current SENTINEL developer handbook.

Standard-library only. Executable source plus committed R4 policy/manifests are
behavioral truth; handbook.toml is the machine-readable documentation contract.
The validator deliberately distinguishes current live entry points from
historical compatibility code.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    raise SystemExit("Python 3.11+ is required (tomllib missing).")

ROOT = Path(__file__).resolve().parents[3]
HANDBOOK = ROOT / "docs" / "handbook"
META_PATH = HANDBOOK / "_meta" / "handbook.toml"
R4 = ROOT / "docs" / "plan" / "ml-R4"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _meta() -> dict[str, Any]:
    with META_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _text(path: str | Path) -> str:
    p = path if isinstance(path, Path) and path.is_absolute() else ROOT / path
    return p.read_text(encoding="utf-8")


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(_text(path))


def _git_files() -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return {line for line in proc.stdout.splitlines() if line}


def _assignment(path: str, name: str) -> Any:
    tree = ast.parse(_text(path), filename=path)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise KeyError(f"{path}::{name}")


def _python_symbol_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()

    def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                names.add(qualified)
                visit(node.body, qualified)

    visit(tree.body)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _solidity_symbol_names(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    names = set(
        re.findall(
            r"\b(?:contract|interface|library|struct|event|error)\s+([A-Za-z_][A-Za-z0-9_]*)",
            source,
        )
    )
    names.update(re.findall(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source))
    names.update(
        re.findall(
            r"\b(?:uint\d*|bytes\d*|address|bool|string)\s+public\s+constant\s+([A-Za-z_][A-Za-z0-9_]*)",
            source,
        )
    )
    return names


def _symbol_exists(anchor: str) -> tuple[bool, str]:
    if "::" not in anchor:
        return False, "anchor must use path::symbol"
    raw_path, symbol = anchor.split("::", 1)
    path = ROOT / raw_path
    if not path.is_file():
        return False, f"missing source path {raw_path}"
    try:
        if path.suffix == ".py":
            names = _python_symbol_names(path)
        elif path.suffix == ".sol":
            names = _solidity_symbol_names(path)
        else:
            return False, f"unsupported source type {path.suffix}"
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return False, f"cannot parse {raw_path}: {exc}"
    return (symbol in names, "found" if symbol in names else f"missing symbol {symbol}")


def _missing_sections(path: Path, required: Iterable[str]) -> list[str]:
    """Accept canonical intro labels as bold fields and normal sections as H2s."""
    body = path.read_text(encoding="utf-8")
    missing: list[str] = []
    intro_labels = {"Read this when", "Skip this if", "Estimated reading time"}
    for section in required:
        if f"## {section}" in body:
            continue
        if section in intro_labels and re.search(rf"^\*\*{re.escape(section)}:\*\*", body, re.MULTILINE):
            continue
        missing.append(section)
    return missing


def _secret_leaks(text: str) -> list[str]:
    patterns = {
        "private-key assignment": r"(?i)(?:private|operator)[_-]?key\s*[=:]\s*[`'\"]?0x[0-9a-f]{64}",
        "credentialed RPC URL": r"https?://[^\s)]+(?:infura|alchemy)[^\s)]*/[A-Za-z0-9_-]{16,}",
        "mnemonic phrase": r"(?i)mnemonic\s*[=:]\s*[`'\"][a-z]+(?:\s+[a-z]+){11,}",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def _volatile_count_pages(pages: Iterable[Path]) -> list[str]:
    volatile: list[str] = []
    for page in pages:
        if page.name == "16_current_status.md" or not page.exists():
            continue
        if re.search(r"\b\d+\s+passed\b|\b\d+\s+failed\b|\b\d+\s+skipped\b", page.read_text(encoding="utf-8")):
            volatile.append(page.name)
    return volatile


def _artifact_classification_ok(item: dict[str, Any]) -> bool:
    allowed = {"tracked", "dvc-managed-local", "regenerated", "ignored-private", "ignored-local"}
    if item["classification"] not in allowed:
        return False
    if item["classification"] == "tracked":
        return bool(item.get("tracked")) and bool(item.get("fresh_clone"))
    return not bool(item.get("fresh_clone"))


def _const_int(source: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*(?::[^=\n]+)?=\s*(\d+)", source)
    if not match:
        raise ValueError(f"constant not found: {name}")
    return int(match.group(1))


def _port(path: str, env_name: str) -> int:
    match = re.search(
        rf'os\.getenv\(\s*"{re.escape(env_name)}"\s*,\s*"(\d+)"\s*\)', _text(path)
    )
    if not match:
        raise ValueError(f"port default not found: {path}::{env_name}")
    return int(match.group(1))


def _routes(path: str) -> list[str]:
    source = _text(path)
    return [
        f"{method.upper()} {route}"
        for method, route in re.findall(r'@app\.(get|post|put|delete)\(\s*[rf]?"([^"]+)"', source)
    ]


def _graph_nodes() -> list[str]:
    return re.findall(r'graph\.add_node\(\s*"([a-z_]+)"', _text("agents/src/orchestration/graph.py"))


def _tool_names(path: str) -> list[str]:
    return re.findall(r'Tool\(\s*name="([a-z_]+)"', _text(path))


def _class_names() -> list[str]:
    return list(_assignment("data_module/sentinel_data/representation/graph_schema.py", "CLASS_NAMES"))


def _test_definitions(path: Path) -> int:
    total = 0
    for test_file in path.rglob("test_*.py"):
        try:
            tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        total += sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        )
    return total


def _r4_facts() -> dict[str, Any]:
    policy = _json("docs/plan/ml-R4/specs/data_vnext_policy_v1.json")
    partition = _json("docs/plan/ml-R4/manifests/p6_partition_manifest.json")
    acceptance = _json("docs/plan/ml-R4/manifests/p6_untouched_acceptance_manifest.json")
    support = _json("docs/plan/ml-R4/manifests/p6_role_support_table.json")
    status = _text("docs/plan/ml-R4/PLAN_STATUS_MATRIX.md")
    g7_manifest = _json("data_module/data/exports/sentinel-r4-vnext-v1/manifest.json")
    g7_representation = _json("data_module/data/exports/sentinel-r4-vnext-v1/representation_binding_report.json")
    g7_validation = _json("data_module/data/exports/sentinel-r4-vnext-v1/g7_validation_report.json")
    return {
        "policy": policy,
        "partition": partition,
        "acceptance": acceptance,
        "support": support,
        "status_text": status,
        "g7_manifest": g7_manifest,
        "g7_representation": g7_representation,
        "g7_validation": g7_validation,
    }


def _discover() -> dict[str, Any]:
    meta = _meta()
    schema_path = "data_module/sentinel_data/representation/graph_schema.py"
    schema = _text(schema_path)
    proxy = _text("zkml/src/distillation/proxy_model.py")
    graph = _text("agents/src/orchestration/graph.py")
    registry = _text("contracts/src/AuditRegistry.sol")
    settings = _json("zkml/ezkl/settings.json")
    run_args = settings["run_args"]
    input_dim = _const_int(proxy, "FROZEN_INPUT_DIM")
    hidden1 = _const_int(proxy, "FROZEN_HIDDEN1")
    hidden2 = _const_int(proxy, "FROZEN_HIDDEN2")
    output_dim = _const_int(proxy, "FROZEN_NUM_CLASSES")
    params = input_dim * hidden1 + hidden1 + hidden1 * hidden2 + hidden2 + hidden2 * output_dim + output_dim
    stages = list(_assignment("data_module/sentinel_data/cli.py", "STAGES")) + ["freshness"]
    tracked = _git_files()
    methods = re.findall(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", registry)
    r4 = _r4_facts()
    return {
        "commit": subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "verified_runtime_commit": meta["verified_commit"],
        "ports": {
            "gateway": _port("agents/src/api/gateway.py", "GATEWAY_PORT"),
            "ml": 8001,
            "mcp_inference": _port("agents/src/mcp/servers/inference_server.py", "MCP_INFERENCE_PORT"),
            "mcp_rag": _port("agents/src/mcp/servers/rag_server.py", "MCP_RAG_PORT"),
            "mcp_audit": _port("agents/src/mcp/servers/audit/_config.py", "MCP_AUDIT_PORT"),
            "mcp_graph_inspector": _port("agents/src/mcp/servers/graph_inspector_server.py", "MCP_GRAPH_INSPECTOR_PORT"),
            "mcp_representation": _port("agents/src/mcp/servers/representation_server.py", "MCP_REPRESENTATION_PORT"),
            "anvil": 8545,
        },
        "routes": {"gateway": _routes("agents/src/api/gateway.py"), "ml": _routes("ml/src/inference/api.py")},
        "mcp_tools": {
            "mcp_inference": _tool_names("agents/src/mcp/servers/inference_server.py"),
            "mcp_rag": _tool_names("agents/src/mcp/servers/rag_server.py"),
            "mcp_audit": _tool_names("agents/src/mcp/servers/audit/_readonly_handlers.py"),
            "mcp_graph_inspector": _tool_names("agents/src/mcp/servers/graph_inspector_server.py"),
            "mcp_representation": _tool_names("agents/src/mcp/servers/representation_server.py"),
        },
        "nodes": _graph_nodes(),
        "entry": re.search(r'graph\.set_entry_point\("([a-z_]+)"\)', graph).group(1),
        "exit": re.search(r'graph\.add_edge\("([a-z_]+)",\s*END\)', graph).group(1),
        "data_stages": stages,
        "schema": {
            "version": _assignment(schema_path, "FEATURE_SCHEMA_VERSION"),
            "node_feature_dim": _const_int(schema, "NODE_FEATURE_DIM"),
            "node_types": _const_int(schema, "NUM_NODE_TYPES"),
            "edge_types": _const_int(schema, "EDGE_TYPES"),
            "classes": _class_names(),
        },
        "proxy": {
            "dimensions": [input_dim, hidden1, hidden2, output_dim],
            "parameters": params,
            "circuit_version": re.search(r'CIRCUIT_VERSION\s*=\s*"([^"]+)"', proxy).group(1),
            "public_signals": sum(shape[-1] for shape in settings["model_instance_shapes"]),
            "input_visibility": run_args["input_visibility"],
            "output_visibility": run_args["output_visibility"],
            "parameter_visibility": run_args["param_visibility"],
            "check_mode": run_args["check_mode"],
            "ezkl_version": settings["version"],
        },
        "registry": {"num_classes": _const_int(registry, "NUM_CLASSES"), "input_offset": _const_int(registry, "INPUT_OFFSET"), "methods": methods},
        "r4": {
            "policy_version": r4["policy"]["policy_version"],
            "policy_status": r4["policy"]["status"],
            "partition_version": r4["partition"]["partition_version"],
            "partition_status": r4["partition"]["status"],
            "population_contracts": r4["partition"]["population_contracts"],
            "population_groups": r4["partition"]["population_groups"],
            "role_contract_counts": r4["partition"]["role_contract_counts"],
            "acceptance_status": r4["acceptance"]["status"],
            "acceptance_contracts": len(r4["acceptance"]["contract_ids"]),
            "g7_status": r4["g7_manifest"]["status"],
            "g7_binding_digest": r4["g7_representation"]["binding_digest_sha256"],
            "g7_checked_contracts": r4["g7_representation"]["checked_contracts"],
        },
        "artifacts": [{**item, "exists": (ROOT / item["path"]).exists(), "tracked": item["path"] in tracked or any(p.startswith(item["path"].rstrip("/") + "/") for p in tracked)} for item in meta["artifact"]],
        "test_files": {module: len(list((ROOT / path).rglob("test_*.py"))) for module, path in {"agents": "agents/tests", "ml": "ml/tests", "data": "data_module/tests", "zkml": "zkml/tests"}.items()},
        "static_test_definitions": {module: _test_definitions(ROOT / path) for module, path in {"agents": "agents/tests", "ml": "ml/tests", "data": "data_module/tests", "zkml": "zkml/tests"}.items()},
    }


def _check_links(pages: Iterable[Path]) -> list[Check]:
    checks: list[Check] = []
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for page in pages:
        for raw_target in pattern.findall(page.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^(?:https?://|mailto:)", target):
                continue
            resolved = (page.parent / target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                checks.append(Check("links", False, f"{page.relative_to(ROOT)} escapes repository: {target}"))
                continue
            if not resolved.exists():
                checks.append(Check("links", False, f"{page.relative_to(ROOT)} -> missing {target}"))
    if not checks:
        checks.append(Check("links", True, "all local Markdown links resolve"))
    return checks


def _static_checks() -> list[Check]:
    meta = _meta()
    discovered = _discover()
    checks: list[Check] = []
    pages = [HANDBOOK / page for page in meta["canonical_pages"]]
    guides = [HANDBOOK / item["path"] for item in meta.get("technical_guide", [])]
    labs = [HANDBOOK / item["path"] for item in meta.get("lab", [])]

    missing_pages = [str(p.relative_to(ROOT)) for p in pages if not p.is_file()]
    checks.append(Check("pages", not missing_pages, "18 canonical pages present" if not missing_pages else f"missing {missing_pages}"))
    for path in pages:
        if path.exists():
            missing = _missing_sections(path, meta["required_sections"])
            checks.append(Check("canonical sections", not missing, f"{path.name}: " + ("ok" if not missing else f"missing {missing}")))
    for path in guides:
        missing = [] if not path.exists() else _missing_sections(path, meta["technical_required_sections"])
        checks.append(Check("supplementary guide structure", path.exists() and not missing, f"{path.name}: " + ("ok" if path.exists() and not missing else f"missing {missing}")))
    for path in labs:
        missing = [] if not path.exists() else _missing_sections(path, meta["lab_required_sections"])
        checks.append(Check("supplementary lab structure", path.exists() and not missing, f"{path.name}: " + ("ok" if path.exists() and not missing else f"missing {missing}")))

    checks.extend(_check_links([p for p in [*pages, *guides, *labs] if p.exists()]))
    canonical_text = "\n".join(p.read_text(encoding="utf-8") for p in pages if p.exists())
    leaks = _secret_leaks(canonical_text)
    checks.append(Check("secrets", not leaks, "no secret-shaped values" if not leaks else f"possible leaks: {leaks}"))
    fragile = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|sol|ts|sh):\d+", canonical_text)
    checks.append(Check("source anchors", not fragile, "no fragile file:line citations" if not fragile else f"fragile citations: {fragile[:5]}"))
    volatile = _volatile_count_pages([*pages, *guides, *labs])
    checks.append(Check("volatile counts", not volatile, "counts confined to current status" if not volatile else f"counts outside status: {volatile}"))

    for item in discovered["artifacts"]:
        checks.append(Check("artifact classification", _artifact_classification_ok(item), f"{item['name']}: class={item['classification']}, tracked={item['tracked']}, fresh_clone={item['fresh_clone']}"))
    for owner in meta.get("source_ownership", []):
        missing = [p for p in owner["paths"] if not (ROOT / p).exists()]
        checks.append(Check("source ownership", not missing, f"{owner['page']}: " + ("ok" if not missing else f"missing {missing}")))
    for guide in meta.get("technical_guide", []):
        bad = [a for a in guide["source_anchors"] if not _symbol_exists(a)[0]]
        checks.append(Check("guide anchors", not bad, f"{guide['id']}: " + ("ok" if not bad else f"invalid {bad}")))

    critical = meta["critical"]
    expected_equal = {
        "DATA stages": (discovered["data_stages"], critical["data_stages"]),
        "class count": (len(discovered["schema"]["classes"]), critical["class_count"]),
        "class order": (discovered["schema"]["classes"], critical["class_order"]),
        "schema version": (discovered["schema"]["version"], critical["data_schema_version"]),
        "node feature dim": (discovered["schema"]["node_feature_dim"], critical["node_feature_dim"]),
        "node types": (discovered["schema"]["node_types"], critical["node_types"]),
        "edge types": (discovered["schema"]["edge_types"], critical["edge_types"]),
        "LangGraph nodes": (discovered["nodes"], critical["langgraph_nodes"]),
        "LangGraph entry": (discovered["entry"], critical["langgraph_entry"]),
        "LangGraph exit": (discovered["exit"], critical["langgraph_exit"]),
        "proxy dimensions": (discovered["proxy"]["dimensions"], [critical["proxy_input_dim"], *critical["proxy_hidden_dims"], critical["proxy_output_dim"]]),
        "proxy parameters": (discovered["proxy"]["parameters"], critical["proxy_parameter_count"]),
        "circuit version": (discovered["proxy"]["circuit_version"], critical["circuit_version"]),
        "public signals": (discovered["proxy"]["public_signals"], critical["circuit_public_signals"]),
        "input visibility": (discovered["proxy"]["input_visibility"], critical["input_visibility"]),
        "output visibility": (discovered["proxy"]["output_visibility"], critical["output_visibility"]),
        "parameter visibility": (discovered["proxy"]["parameter_visibility"], critical["parameter_visibility"]),
        "check mode": (discovered["proxy"]["check_mode"], critical["check_mode"]),
        "registry classes": (discovered["registry"]["num_classes"], critical["registry_num_classes"]),
        "registry offset": (discovered["registry"]["input_offset"], critical["registry_input_offset"]),
        "R4 policy version": (discovered["r4"]["policy_version"], critical["r4_policy_version"]),
        "R4 partition version": (discovered["r4"]["partition_version"], critical["r4_partition_version"]),
    }
    for name, (actual, expected) in expected_equal.items():
        checks.append(Check("critical fact", actual == expected, f"{name}: source={actual!r}, metadata={expected!r}"))

    for name, service in meta["services"].items():
        actual = discovered["ports"].get(name)
        checks.append(Check("ports", actual == service["port"], f"{name}: source={actual}, metadata={service['port']}"))
        if "routes" in service:
            actual_routes = discovered["routes"][name]
            # Metadata routes are the documented public/core contract; source may
            # expose extra health probes such as /health/live and /health/ready.
            checks.append(Check("routes", set(service["routes"]).issubset(set(actual_routes)), f"{name}: source={actual_routes}, metadata_core={service['routes']}"))
        if "tools" in service:
            actual_tools = discovered["mcp_tools"][name]
            checks.append(Check("MCP tools", actual_tools == service["tools"], f"{name}: source={actual_tools}, metadata={service['tools']}"))

    audit_tools = discovered["mcp_tools"]["mcp_audit"]
    checks.append(Check("audit MCP read-only", audit_tools == ["get_latest_audit", "get_audit_history", "check_audit_exists"] and "submit_audit" not in audit_tools, f"live_tools={audit_tools}"))
    server_text = _text("agents/src/mcp/servers/audit/_server.py")
    checks.append(Check("audit live handler", "_readonly_handlers" in server_text and "read-only" in server_text, "live server imports read-only handlers"))

    required_registry = {"submitAudit", "submitAuditV2", "initializeV3", "submitAuditV3", "hasAuditV3", "getLatestAuditV3", "getAuditHistoryV3", "getAuditCountV3", "setZkmlVerifierV3", "setAuditPolicySignerV3", "pause", "unpause"}
    checks.append(Check("registry V3 methods", required_registry.issubset(discovered["registry"]["methods"]), f"required present={required_registry.issubset(discovered['registry']['methods'])}"))

    r4 = _r4_facts()
    policy = r4["policy"]
    partition = r4["partition"]
    acceptance = r4["acceptance"]
    checks.append(Check("R4 policy accepted", policy["status"] == "ACCEPTED_G5", f"status={policy['status']}"))
    checks.append(Check("R4 no blanket negatives", policy["negative_authority"]["first_baseline_blanket_negative_sources"] == [], "blanket negative sources empty"))
    disabled = {name for name, cfg in policy["class_supervision"].items() if cfg["status"] == "SUPERVISION_DISABLED_PENDING_EVIDENCE"}
    checks.append(Check("R4 disabled classes", disabled == {"GasException", "UnusedReturn"}, f"disabled={sorted(disabled)}"))
    checks.append(Check("R4 partition frozen", partition["status"] == "FROZEN_G6" and partition.get("gate") == "G6_PASS", f"status={partition['status']}, gate={partition.get('gate')}"))
    checks.append(Check("R4 population", partition["population_contracts"] == 22493 and partition["population_groups"] == 13509, f"contracts={partition['population_contracts']}, groups={partition['population_groups']}"))
    checks.append(Check("R4 acceptance unsupported", acceptance["status"] == "UNSUPPORTED_EMPTY_FROZEN" and acceptance["contract_ids"] == [] and acceptance["group_ids"] == [], f"status={acceptance['status']}"))
    checks.append(Check("R4 phase status", "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED |" in r4["status_text"] and "| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS |" in r4["status_text"], "Phase 7 PASSED / Phase 8 IN_PROGRESS on canonical main"))
    g7_manifest = r4["g7_manifest"]
    g7_rep = r4["g7_representation"]
    g7_report = r4["g7_validation"]
    checks.append(Check("R4 G7 publication", g7_manifest.get("status") == "VALIDATED_G7_CANDIDATE" and g7_manifest.get("export_schema_version") == "v2" and g7_manifest.get("population", {}).get("contracts") == 22493, f"status={g7_manifest.get('status')}, schema={g7_manifest.get('export_schema_version')}"))
    checks.append(Check("R4 G7 representation binding", g7_rep.get("passed") is True and g7_rep.get("checked_contracts") == 21657 and g7_rep.get("checked_files") == 64971 and g7_rep.get("missing_files_total") == 0 and g7_rep.get("mismatch_total") == 0 and g7_rep.get("physical_root_recorded") is False and g7_rep.get("binding_digest_sha256") == "7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420", f"contracts={g7_rep.get('checked_contracts')}, files={g7_rep.get('checked_files')}, missing={g7_rep.get('missing_files_total')}, mismatches={g7_rep.get('mismatch_total')}"))
    checks.append(Check("R4 G7 final validation", g7_report.get("passed") is True and g7_report.get("require_representation_binding") is True and g7_report.get("target_counts") == {"1": 1007, "None": 223923} and g7_report.get("training_strength_counts") == {"NONE": 223923, "STRONG": 403, "WEAK": 604}, f"passed={g7_report.get('passed')}, targets={g7_report.get('target_counts')}"))

    verified_commit = meta["verified_commit"]
    commit_ok = subprocess.run(["git", "cat-file", "-e", f"{verified_commit}^{{commit}}"], cwd=ROOT, capture_output=True).returncode == 0
    checks.append(Check("verified source/runtime commit", commit_ok and verified_commit == "81d9c547d", f"metadata={verified_commit}, exists={commit_ok}"))
    checks.append(Check("verified date", meta["verified_date"] == "2026-08-12", f"metadata={meta['verified_date']}"))

    required_truth = {
        "00_README.md": ["V3", "R4", "read-only", "supplementary"],
        "02_runtime_flows.md": ["read-only", "submitAuditV3", "policy signer", "do not automatically promote"],
        "03_data_pipeline.md": ["224,930", "Historical zero", "DATA vNext"],
        "04_data_artifacts.md": ["historical v1", "DATA vNext v2", "target `0`"],
        "06_ml_training_quality.md": ["UNSUPPORTED_EMPTY", "UNSUPPORTED_EMPTY_FROZEN", "WEAK"],
        "07_zkml.md": ["legacy_proxy_only_unbound", "context_attested_v3", "check_mode=\"UNSAFE\""],
        "08_contracts.md": ["submitAuditV3", "initializeV3", "V1 is historical", "V2 is the historical"],
        "10_agents_services.md": ["read-only", "get_latest_audit", "automatic V3→RAG promotion remains disabled"],
        "12_security_and_trust.md": ["comment", "string", "role-swap", "extraction", "identifier", "NatSpec", "multi", "import", "policy signer"],
        "13_evaluation.md": ["positive-only limited", "UNSUPPORTED_EMPTY", "UNSUPPORTED_EMPTY_FROZEN"],
        "16_current_status.md": ["81d9c547d", "G7", "VALIDATED_G7_CANDIDATE", "7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420", "UNSUPPORTED_EMPTY_FROZEN"],
        "17_reference.md": ["supplementary learning guides", "V3 registry", "DATA vNext"],
    }
    for page, phrases in required_truth.items():
        body = (HANDBOOK / page).read_text(encoding="utf-8")
        absent = [phrase for phrase in phrases if phrase not in body]
        checks.append(Check("documented truth", not absent, f"{page}: " + ("ok" if not absent else f"missing {absent}")))
    return checks


def static() -> int:
    checks = _static_checks()
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    failures = [c for c in checks if not c.passed]
    print(f"\nstatic: {len(checks) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


def inventory(as_json: bool) -> int:
    data = _discover()
    meta = _meta()
    data["technical_guides"] = [{"id": i["id"], "path": i["path"], "classification": "supplementary", "owner_pages": i["owner_pages"]} for i in meta.get("technical_guide", [])]
    data["labs"] = [{"id": i["id"], "path": i["path"], "classification": "supplementary", "guide": i["guide"], "tier": i["tier"], "safe_preflight": i["safe_preflight"]} for i in meta.get("lab", [])]
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    print(f"checkout commit: {data['commit']}")
    print(f"verified runtime baseline: {data['verified_runtime_commit']}")
    print("ports:", ", ".join(f"{k}={v}" for k, v in data["ports"].items()))
    print("gateway routes:", ", ".join(data["routes"]["gateway"]))
    print("ML routes:", ", ".join(data["routes"]["ml"]))
    print("MCP tools:", json.dumps(data["mcp_tools"], sort_keys=True))
    print(f"LangGraph ({len(data['nodes'])}):", " -> ".join(data["nodes"]))
    print("DATA stages:", " -> ".join(data["data_stages"]))
    print("schema:", json.dumps(data["schema"], sort_keys=True))
    print("proxy:", json.dumps(data["proxy"], sort_keys=True))
    print("registry methods include V3:", "submitAuditV3" in data["registry"]["methods"])
    print("R4:", json.dumps(data["r4"], sort_keys=True))
    print("artifacts:")
    for item in data["artifacts"]:
        print(f"  - {item['classification']:17} exists={str(item['exists']).lower():5} tracked={str(item['tracked']).lower():5} {item['path']}")
    print("technical guides: 10 supplementary")
    print("labs: 10 supplementary")
    return 0


def _lab_preflight(item: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> list[Check]:
    checks: list[Check] = []
    tracked = _git_files()
    for raw_path in item.get("required_paths", []):
        path = ROOT / raw_path
        is_tracked = raw_path in tracked or any(p.startswith(raw_path.rstrip("/") + "/") for p in tracked)
        checks.append(Check("required path", path.exists() and is_tracked, f"{raw_path}: exists={path.exists()}, tracked={is_tracked}"))
    for name in item.get("required_artifacts", []):
        artifact = artifacts.get(name)
        if artifact is None:
            checks.append(Check("required artifact", False, f"{name}: not registered"))
        else:
            checks.append(Check("required artifact", bool(artifact["exists"]), f"{name}: exists={artifact['exists']}, class={artifact['classification']}"))
    for executable in item.get("required_executables", []):
        found = shutil.which(executable)
        checks.append(Check("required executable", found is not None, f"{executable}: {found or 'not found'}"))
    return checks


def lab(args: argparse.Namespace) -> int:
    meta = _meta()
    labs = {item["id"]: item for item in meta.get("lab", [])}
    artifacts = {item["name"]: item for item in _discover()["artifacts"]}
    if args.list_labs:
        for item in labs.values():
            print(f"{item['id']}  supplementary tier={item['tier']:6} safe={str(item['safe_preflight']).lower():5} guide={item['guide']}  {item['path']}")
            print(f"      prerequisites: {'; '.join(item.get('prerequisites', []))}")
            print(f"      artifacts: {'; '.join(item.get('artifact_requirements', []))}")
        return 0
    if args.check_all_safe:
        selected = [item for item in labs.values() if item.get("safe_preflight")]
    else:
        if args.check not in labs:
            print(f"Unknown lab id: {args.check}. Available: {', '.join(labs)}", file=sys.stderr)
            return 2
        selected = [labs[args.check]]
    failures = 0
    for item in selected:
        print(f"\n{item['id']} — {item['path']} [supplementary/{item['tier']}]" )
        checks = _lab_preflight(item, artifacts) or [Check("preflight", True, "no external preflight requirements")]
        for check in checks:
            print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        failures += sum(not c.passed for c in checks)
    print(f"\nlab preflight: {len(selected)} lab(s), {failures} failed requirement(s)")
    return 1 if failures else 0


def _run(command: list[str], cwd: Path, timeout: int = 1800) -> bool:
    print(f"\n$ (cd {cwd}) {' '.join(command)}", flush=True)
    env = {**os.environ, "TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp"}
    try:
        return subprocess.run(command, cwd=cwd, env=env, timeout=timeout).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = response.read()
            status = None
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    status = decoded.get("status")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            passed = 200 <= response.status < 300
            print(f"[{'PASS' if passed else 'FAIL'}] {url}: HTTP {response.status}, status={status!r}")
            return passed
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[FAIL] {url}: {exc}")
        return False


def live(args: argparse.Namespace) -> int:
    results: list[bool] = []
    if args.services:
        for port in (8000, 8001, 8010, 8011, 8012, 8013, 8014):
            results.append(_probe(f"http://127.0.0.1:{port}/health"))
    commands = {
        "agents": (["poetry", "run", "pytest", "-q"], ROOT / "agents"),
        "ml": ([str(ROOT / "ml/.venv/bin/python"), "-m", "pytest", "ml/tests", "-q"], ROOT),
        "data": ([str(ROOT / "data_module/.venv/bin/python"), "-m", "pytest", "data_module/tests", "-q"], ROOT),
        "zkml": ([str(ROOT / "ml/.venv/bin/python"), "-m", "pytest", "zkml/tests", "-q"], ROOT),
        "contracts": (["forge", "test"], ROOT / "contracts"),
    }
    if args.module:
        results.append(_run(*commands[args.module]))
    if args.anvil:
        results.append(_probe("http://127.0.0.1:8545"))
    if args.ezkl:
        print("EZKL live proof is an explicit historical/retained-proof exercise; prerequisites are not auto-acquired.")
        results.append((ROOT / "zkml/ezkl/proving_key.pk").exists() and (ROOT / "zkml/ezkl/srs.params").exists())
    return 0 if all(results) else (1 if results else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    inv = sub.add_parser("inventory")
    inv.add_argument("--json", action="store_true")
    labp = sub.add_parser("lab")
    labp.add_argument("--list", dest="list_labs", action="store_true")
    labp.add_argument("--check")
    labp.add_argument("--check-all-safe", action="store_true")
    livep = sub.add_parser("live")
    livep.add_argument("--services", action="store_true")
    livep.add_argument("--module", choices=["agents", "ml", "data", "zkml", "contracts"])
    livep.add_argument("--anvil", action="store_true")
    livep.add_argument("--ezkl", action="store_true")
    args = parser.parse_args()
    if args.command == "static":
        return static()
    if args.command == "inventory":
        return inventory(args.json)
    if args.command == "lab":
        if not (args.list_labs or args.check_all_safe or args.check):
            parser.error("lab requires --list, --check ID, or --check-all-safe")
        return lab(args)
    if args.command == "live":
        return live(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
