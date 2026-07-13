#!/usr/bin/env python3
"""Validate and inventory the SENTINEL developer handbook.

Standard-library only. Source files are authoritative; handbook.toml is the
declared documentation contract checked against them.
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
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 is unsupported
    raise SystemExit("Python 3.11+ is required (tomllib missing).")


ROOT = Path(__file__).resolve().parents[3]
HANDBOOK = ROOT / "docs" / "handbook"
META_PATH = HANDBOOK / "_meta" / "handbook.toml"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _meta() -> dict[str, Any]:
    with META_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _text(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


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
    """Return stable top-level and Class.member names from a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    def visit_definitions(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                names.add(qualified)
                visit_definitions(node.body, qualified)

    visit_definitions(tree.body)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _solidity_symbol_names(path: Path) -> set[str]:
    """Resolve public documentation anchors without requiring a Solidity parser."""
    source = path.read_text(encoding="utf-8")
    names = set(re.findall(
        r"\b(?:contract|interface|library|struct|event|error)\s+([A-Za-z_][A-Za-z0-9_]*)",
        source,
    ))
    names.update(re.findall(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source))
    names.update(re.findall(
        r"\b(?:uint\d*|bytes\d*|address|bool|string)\s+public\s+constant\s+([A-Za-z_][A-Za-z0-9_]*)",
        source,
    ))
    return names


def _symbol_exists(anchor: str) -> tuple[bool, str]:
    """Validate `relative/path::symbol` against source, never a line number."""
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
    body = path.read_text(encoding="utf-8")
    return [section for section in required if f"## {section}" not in body]


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
        return bool(item["tracked"]) and bool(item["fresh_clone"])
    return not bool(item["fresh_clone"])


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
            1 for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        )
    return total


def _discover() -> dict[str, Any]:
    meta = _meta()
    schema_path = "data_module/sentinel_data/representation/graph_schema.py"
    schema = _text(schema_path)
    proxy = _text("zkml/src/distillation/proxy_model.py")
    graph = _text("agents/src/orchestration/graph.py")
    registry = _text("contracts/src/AuditRegistry.sol")
    settings = json.loads(_text("zkml/ezkl/settings.json"))
    run_args = settings["run_args"]
    input_dim = _const_int(proxy, "FROZEN_INPUT_DIM")
    hidden1 = _const_int(proxy, "FROZEN_HIDDEN1")
    hidden2 = _const_int(proxy, "FROZEN_HIDDEN2")
    output_dim = _const_int(proxy, "FROZEN_NUM_CLASSES")
    params = input_dim * hidden1 + hidden1 + hidden1 * hidden2 + hidden2 + hidden2 * output_dim + output_dim
    stages = list(_assignment("data_module/sentinel_data/cli.py", "STAGES")) + ["freshness"]
    ports = {
        "gateway": _port("agents/src/api/gateway.py", "GATEWAY_PORT"),
        "ml": 8001,
        "mcp_inference": _port("agents/src/mcp/servers/inference_server.py", "MCP_INFERENCE_PORT"),
        "mcp_rag": _port("agents/src/mcp/servers/rag_server.py", "MCP_RAG_PORT"),
        "mcp_audit": _port("agents/src/mcp/servers/audit/_config.py", "MCP_AUDIT_PORT"),
        "mcp_graph_inspector": _port("agents/src/mcp/servers/graph_inspector_server.py", "MCP_GRAPH_INSPECTOR_PORT"),
        "mcp_representation": _port("agents/src/mcp/servers/representation_server.py", "MCP_REPRESENTATION_PORT"),
        "anvil": 8545,
    }
    methods = re.findall(r"function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", registry)
    return {
        "commit": subprocess.run(
            ["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True
        ).stdout.strip(),
        "ports": ports,
        "routes": {
            "gateway": _routes("agents/src/api/gateway.py"),
            "ml": _routes("ml/src/inference/api.py"),
        },
        "mcp_tools": {
            "mcp_inference": _tool_names("agents/src/mcp/servers/inference_server.py"),
            "mcp_rag": _tool_names("agents/src/mcp/servers/rag_server.py"),
            "mcp_audit": _tool_names("agents/src/mcp/servers/audit/_handlers.py"),
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
            "edge_types": _const_int(schema, "NUM_EDGE_TYPES"),
            "classes": _class_names(),
        },
        "proxy": {
            "dimensions": [input_dim, hidden1, hidden2, output_dim],
            "parameters": params,
            "circuit_version": re.search(r'CIRCUIT_VERSION\s*=\s*"([^"]+)"', proxy).group(1),
            "signal_shapes": settings["model_instance_shapes"],
            "public_signals": sum(shape[-1] for shape in settings["model_instance_shapes"]),
            "input_visibility": run_args["input_visibility"],
            "output_visibility": run_args["output_visibility"],
            "parameter_visibility": run_args["param_visibility"],
            "check_mode": run_args["check_mode"],
            "ezkl_version": settings["version"],
        },
        "registry": {
            "num_classes": _const_int(registry, "NUM_CLASSES"),
            "input_offset": _const_int(registry, "INPUT_OFFSET"),
            "methods": methods,
        },
        "artifacts": [
            {
                **item,
                "exists": (ROOT / item["path"]).exists(),
                "tracked": item["path"] in _git_files()
                or any(p.startswith(item["path"].rstrip("/") + "/") for p in _git_files()),
            }
            for item in meta["artifact"]
        ],
        "test_files": {
            module: len(list((ROOT / path).rglob("test_*.py")))
            for module, path in {
                "agents": "agents/tests",
                "ml": "ml/tests",
                "data": "data_module/tests",
                "zkml": "zkml/tests",
            }.items()
        },
        "static_test_definitions": {
            module: _test_definitions(ROOT / path)
            for module, path in {
                "agents": "agents/tests",
                "ml": "ml/tests",
                "data": "data_module/tests",
                "zkml": "zkml/tests",
            }.items()
        },
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
    checks.append(Check("pages", not missing_pages, "missing: " + ", ".join(missing_pages) if missing_pages else "18 canonical pages present"))

    for page in pages:
        if not page.exists():
            continue
        body = page.read_text(encoding="utf-8")
        missing_sections = [section for section in meta["required_sections"] if section not in body]
        checks.append(Check("template", not missing_sections, f"{page.name}: " + ("ok" if not missing_sections else f"missing {missing_sections}")))

    for kind, documents, required in (
        ("technical template", guides, meta.get("technical_required_sections", [])),
        ("lab template", labs, meta.get("lab_required_sections", [])),
    ):
        for document in documents:
            if not document.is_file():
                checks.append(Check(kind, False, f"missing {document.relative_to(ROOT)}"))
                continue
            missing = _missing_sections(document, required)
            checks.append(Check(kind, not missing, f"{document.name}: " + ("ok" if not missing else f"missing {missing}")))

    plan_pages = [ROOT / "docs/plan/system-finalization/D1_developer_handbook.md"]
    plan_pages.extend((ROOT / "docs/plan/system-finalization/handbook").glob("*.md"))
    checks.extend(_check_links(
        [p for p in [*pages, *guides, *labs] if p.exists()]
        + [ROOT / "README.md", *plan_pages]
    ))

    tracked = _git_files()
    for owner in meta["source_ownership"]:
        for path in owner["paths"]:
            exists = (ROOT / path).exists()
            has_tracked = path in tracked or any(item.startswith(path.rstrip("/") + "/") for item in tracked)
            checks.append(Check("source ownership", exists and has_tracked, f"{owner['page']} -> {path}: exists={exists}, tracked={has_tracked}"))

    guide_ids = [item["id"] for item in meta.get("technical_guide", [])]
    lab_ids = [item["id"] for item in meta.get("lab", [])]
    checks.append(Check("guide registry", len(guide_ids) == 10 and len(set(guide_ids)) == 10, f"guides={guide_ids}"))
    checks.append(Check("lab registry", len(lab_ids) == 10 and len(set(lab_ids)) == 10, f"labs={lab_ids}"))
    for guide in meta.get("technical_guide", []):
        unknown_pages = sorted(set(guide["owner_pages"]) - set(meta["canonical_pages"]))
        checks.append(Check("guide ownership", not unknown_pages, f"{guide['id']}: " + ("ok" if not unknown_pages else f"unknown pages {unknown_pages}")))
        for anchor in guide.get("source_anchors", []):
            exists, detail = _symbol_exists(anchor)
            checks.append(Check("source symbol", exists, f"{guide['id']} {anchor}: {detail}"))
        for path in guide.get("source_paths", []):
            exists = (ROOT / path).exists()
            has_tracked = path in tracked or any(item.startswith(path.rstrip("/") + "/") for item in tracked)
            checks.append(Check("guide source path", exists and has_tracked, f"{guide['id']} -> {path}: exists={exists}, tracked={has_tracked}"))
    for lab in meta.get("lab", []):
        checks.append(Check("lab guide", lab["guide"] in guide_ids, f"{lab['id']} -> {lab['guide']}"))
        for path in lab.get("required_paths", []):
            source = ROOT / path
            has_tracked = path in tracked or any(item.startswith(path.rstrip("/") + "/") for item in tracked)
            checks.append(Check("lab source", source.exists() and has_tracked, f"{lab['id']} -> {path}: exists={source.exists()}, tracked={has_tracked}"))

    owned_pages = {page for guide in meta.get("technical_guide", []) for page in guide["owner_pages"]}
    required_owned = set(meta["canonical_pages"]) - {"00_README.md", "16_current_status.md", "17_reference.md"}
    uncovered = sorted(required_owned - owned_pages)
    checks.append(Check("guide coverage", not uncovered, "all subsystem chapters owned" if not uncovered else f"uncovered: {uncovered}"))

    declared_source_paths = [
        path.rstrip("/")
        for guide in meta.get("technical_guide", [])
        for path in guide.get("source_paths", [])
    ]
    production_files: list[str] = []
    for root in meta.get("coverage", {}).get("production_roots", []):
        root_path = ROOT / root
        if root_path.is_file():
            production_files.append(root)
            continue
        for source in root_path.rglob("*"):
            if source.is_file() and source.suffix in {".py", ".sol", ".sh"}:
                production_files.append(str(source.relative_to(ROOT)))
    uncovered_source = sorted(
        source for source in production_files
        if not any(source == prefix or source.startswith(prefix + "/") for prefix in declared_source_paths)
    )
    checks.append(Check("production coverage", not uncovered_source, f"{len(production_files)} active source files covered" if not uncovered_source else f"uncovered: {uncovered_source[:20]}"))

    index = (HANDBOOK / "00_README.md").read_text(encoding="utf-8")
    missing_nav = [page for page in meta["canonical_pages"][1:] if f"({page})" not in index]
    checks.append(Check("navigation", not missing_nav, "all pages indexed" if not missing_nav else f"not indexed: {missing_nav}"))

    all_handbook_docs = [p for p in [*pages, *guides, *labs] if p.exists()]
    all_docs = "\n".join(p.read_text(encoding="utf-8") for p in all_handbook_docs)
    leaks = _secret_leaks(all_docs)
    checks.append(Check("secrets", not leaks, "no secret-shaped values" if not leaks else f"possible leaks: {leaks}"))

    fragile = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|sol|ts|sh):\d+", all_docs)
    checks.append(Check("source anchors", not fragile, "no fragile file:line citations" if not fragile else f"fragile citations: {fragile[:5]}"))

    volatile = _volatile_count_pages(all_handbook_docs)
    checks.append(Check("volatile counts", not volatile, "counts confined to status" if not volatile else f"counts outside status: {volatile}"))

    artifact_names = {item["name"] for item in meta["artifact"]}
    for item in discovered["artifacts"]:
        checks.append(Check("artifact classification", _artifact_classification_ok(item), f"{item['name']}: class={item['classification']}, tracked={item['tracked']}, fresh_clone={item['fresh_clone']}"))
    for lab in meta.get("lab", []):
        unknown = sorted(set(lab.get("required_artifacts", [])) - artifact_names)
        checks.append(Check("lab artifacts", not unknown, f"{lab['id']}: " + ("ok" if not unknown else f"unknown {unknown}")))

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
    }
    for name, (actual, expected) in expected_equal.items():
        checks.append(Check("critical fact", actual == expected, f"{name}: source={actual!r}, metadata={expected!r}"))

    for name, service in meta["services"].items():
        actual = discovered["ports"].get(name)
        checks.append(Check("ports", actual == service["port"], f"{name}: source={actual}, metadata={service['port']}"))
        if "routes" in service:
            actual_routes = discovered["routes"][name]
            checks.append(Check("routes", set(actual_routes) == set(service["routes"]), f"{name}: source={actual_routes}, metadata={service['routes']}"))
        if "tools" in service:
            actual_tools = discovered["mcp_tools"][name]
            checks.append(Check("MCP tools", actual_tools == service["tools"], f"{name}: source={actual_tools}, metadata={service['tools']}"))

    mcp_services = [name for name in meta["services"] if name.startswith("mcp_")]
    checks.append(Check("MCP services", len(mcp_services) == 5, f"declared={mcp_services}"))

    required_registry = {"submitAudit", "submitAuditV2", "hasAudit", "getLatestAudit", "getAuditHistory", "getAuditCount", "hasAuditV2", "getLatestAuditV2", "getAuditHistoryV2", "getAuditCountV2", "pause", "unpause"}
    checks.append(Check("registry methods", required_registry.issubset(discovered["registry"]["methods"]), f"methods={discovered['registry']['methods']}"))

    required_truth = {
        "02_runtime_flows.md": ["does not invoke", "submit_audit", "unsubmitted placeholder"],
        "07_zkml.md": ["does not prove", "check_mode=\"UNSAFE\"", "138 total"],
        "09_agents_orchestration.md": ["14 nodes", "visualizer", "verdict_provable"],
        "12_security_and_trust.md": ["comment", "string", "role-swap", "extraction", "identifier", "NatSpec", "multi", "import"],
        "13_evaluation.md": ["measured precision", "tp/(tp+fp)", "alpha = 5"],
        "16_current_status.md": ["ac78c057b", "631 passed, 3 failed", "198 passed, 19 failed", "569 passed, 9 failed, 47 skipped", "37 passed", "66 passed"],
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
    failures = [check for check in checks if not check.passed]
    print(f"\nstatic: {len(checks) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


def inventory(as_json: bool) -> int:
    data = _discover()
    meta = _meta()
    data["technical_guides"] = [
        {
            "id": item["id"],
            "path": item["path"],
            "owner_pages": item["owner_pages"],
            "source_paths": item.get("source_paths", []),
            "source_anchors": item["source_anchors"],
        }
        for item in meta.get("technical_guide", [])
    ]
    data["labs"] = [
        {
            "id": item["id"],
            "path": item["path"],
            "guide": item["guide"],
            "tier": item["tier"],
            "safe_preflight": item["safe_preflight"],
        }
        for item in meta.get("lab", [])
    ]
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"commit: {data['commit']}")
        print("ports:", ", ".join(f"{k}={v}" for k, v in data["ports"].items()))
        print("gateway routes:", ", ".join(data["routes"]["gateway"]))
        print("ML routes:", ", ".join(data["routes"]["ml"]))
        print("MCP tools:", json.dumps(data["mcp_tools"], sort_keys=True))
        print(f"LangGraph ({len(data['nodes'])}):", " -> ".join(data["nodes"]))
        print(f"entry/exit: {data['entry']} / {data['exit']}")
        print("DATA stages:", " -> ".join(data["data_stages"]))
        print("schema:", json.dumps(data["schema"], sort_keys=True))
        print("proxy:", json.dumps(data["proxy"], sort_keys=True))
        print("registry:", json.dumps(data["registry"], sort_keys=True))
        print("test files:", json.dumps(data["test_files"], sort_keys=True))
        print("static test definitions (parametrization expands only in pytest collection):", json.dumps(data["static_test_definitions"], sort_keys=True))
        print("artifacts:")
        for item in data["artifacts"]:
            print(f"  - {item['classification']:17} exists={str(item['exists']).lower():5} tracked={str(item['tracked']).lower():5} {item['path']}")
        print("technical guide coverage:")
        for item in data["technical_guides"]:
            print(f"  - {item['id']} {item['path']} -> {', '.join(item['owner_pages'])} | sources: {', '.join(item['source_paths'])}")
        print("lab readiness registry:")
        for item in data["labs"]:
            print(f"  - {item['id']} tier={item['tier']:6} safe={str(item['safe_preflight']).lower():5} guide={item['guide']} {item['path']}")
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
            checks.append(Check(
                "required artifact",
                bool(artifact["exists"]),
                f"{name}: exists={artifact['exists']}, class={artifact['classification']}, path={artifact['path']}",
            ))
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
            print(f"{item['id']}  tier={item['tier']:6} safe={str(item['safe_preflight']).lower():5} guide={item['guide']}  {item['path']}")
            print(f"      prerequisites: {'; '.join(item.get('prerequisites', []))}")
            print(f"      artifacts: {'; '.join(item.get('artifact_requirements', []))}")
        return 0

    selected: list[dict[str, Any]]
    if args.check_all_safe:
        selected = [item for item in labs.values() if item.get("safe_preflight")]
    else:
        if args.check not in labs:
            print(f"Unknown lab id: {args.check}. Available: {', '.join(labs)}", file=sys.stderr)
            return 2
        selected = [labs[args.check]]

    failures = 0
    for item in selected:
        print(f"\n{item['id']} — {item['path']} [{item['tier']}]")
        print("prerequisites:", "; ".join(item.get("prerequisites", [])))
        print("artifact requirements:", "; ".join(item.get("artifact_requirements", [])))
        checks = _lab_preflight(item, artifacts)
        if not checks:
            checks = [Check("preflight", True, "no external preflight requirements")]
        for check in checks:
            print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        failures += sum(not check.passed for check in checks)
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


def _probe(url: str, body: bytes | None = None, *, require_healthy: bool = False) -> bool:
    request = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = response.read()
            http_ok = 200 <= response.status < 300
            status: str | None = None
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    status = decoded.get("status")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            semantic_ok = not require_healthy or status in {"ok", "healthy"}
            passed = http_ok and semantic_ok
            suffix = f", status={status!r}" if status is not None else ""
            print(f"[{'PASS' if passed else 'FAIL'}] {url}: HTTP {response.status}{suffix}")
            return passed
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[FAIL] {url}: {exc}")
        return False


def live(args: argparse.Namespace) -> int:
    results: list[bool] = []
    if args.services:
        for port in (8000, 8001, 8010, 8011, 8012, 8013, 8014):
            results.append(_probe(f"http://127.0.0.1:{port}/health", require_healthy=True))
    commands = {
        "agents": (["poetry", "run", "pytest", "-q"], ROOT / "agents"),
        "ml": ([str(ROOT / "ml/.venv/bin/python"), "-m", "pytest", "ml/tests", "-q"], ROOT),
        "data": ([str(ROOT / "data_module/.venv/bin/python"), "-m", "pytest", "data_module/tests", "-q"], ROOT),
        "zkml": ([str(ROOT / "ml/.venv/bin/python"), "-m", "pytest", "zkml/tests", "-q"], ROOT),
        "contracts": (["forge", "test"], ROOT / "contracts"),
    }
    for module in args.module:
        command, cwd = commands[module]
        results.append(_run(command, cwd))
    if args.gpu:
        results.append(_run([
            str(ROOT / "ml/.venv/bin/python"), "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
        ], ROOT, timeout=60))
    if args.ezkl:
        results.append(_run([
            str(ROOT / "ml/.venv/bin/python"), "-m", "zkml.src.ezkl.run_proof"
        ], ROOT, timeout=1800))
    if args.anvil:
        payload = json.dumps({"jsonrpc": "2.0", "method": "eth_chainId", "params": [], "id": 1}).encode()
        results.append(_probe("http://127.0.0.1:8545", payload))
    if not results:
        print("No live checks selected. Use --services, --module, --gpu, --ezkl, or --anvil.", file=sys.stderr)
        return 2
    return 0 if all(results) else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="mode", required=True)
    sub.add_parser("static", help="Validate links, tracked paths, template, navigation, secrets, and source facts.")
    inv = sub.add_parser("inventory", help="Report source-derived ports, routes, nodes, stages, schemas, tests, and artifacts.")
    inv.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    labs = sub.add_parser("lab", help="List labs or preflight their declared source, artifact, and executable requirements.")
    lab_mode = labs.add_mutually_exclusive_group(required=True)
    lab_mode.add_argument("--list", dest="list_labs", action="store_true", help="List labs, tiers, prerequisites, and artifacts.")
    lab_mode.add_argument("--check", metavar="ID", help="Preflight one lab by id, for example L05.")
    lab_mode.add_argument("--check-all-safe", action="store_true", help="Preflight all labs declared non-mutating and fresh-clone-safe.")
    run = sub.add_parser("live", help="Run explicit module, service, GPU, EZKL, or Anvil checks without hiding failures.")
    run.add_argument("--services", action="store_true", help="Probe gateway, ML, and five MCP health endpoints.")
    run.add_argument("--module", action="append", choices=["agents", "ml", "data", "zkml", "contracts"], default=[], help="Run a full module suite; repeat as needed.")
    run.add_argument("--gpu", action="store_true", help="Require and identify a CUDA device.")
    run.add_argument("--ezkl", action="store_true", help="Run the real proof workflow; requires all proving artifacts.")
    run.add_argument("--anvil", action="store_true", help="Require a live JSON-RPC node on 127.0.0.1:8545.")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.mode == "static":
        return static()
    if args.mode == "inventory":
        return inventory(args.json)
    if args.mode == "lab":
        return lab(args)
    return live(args)


if __name__ == "__main__":
    raise SystemExit(main())
