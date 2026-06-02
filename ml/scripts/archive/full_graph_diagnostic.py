"""
full_graph_diagnostic.py — Complete graph extraction audit for all 20 test contracts.

For every contract:
  • Lists every node: index, type name, decoded feature values
  • Lists every edge: (src_idx, src_type) → (dst_idx, dst_type) with edge type name
  • Summarises: node-type distribution, edge-type distribution
  • Cross-checks: what vulnerability signals are present vs what the expected label needs
  • Flags: missing signals, suspicious features, CEI ordering, truncation

Usage:
  PYTHONPATH=. PATH="/home/motafeq/projects/sentinel/.local-bin:$PATH" \
    ml/.venv/bin/python ml/scripts/full_graph_diagnostic.py
"""

import math
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ml.src.preprocessing.graph_extractor import (
    GraphExtractionConfig,
    extract_contract_graph,
)
from ml.src.preprocessing.graph_schema import (
    EDGE_TYPES, NODE_TYPES, NODE_FEATURE_DIM,
)

# ── Constants ────────────────────────────────────────────────────────────────

CONTRACTS_DIR = Path(__file__).parent / "test_contracts"

FEATURE_NAMES = [
    "type_id_norm",      # 0
    "visibility",        # 1  (0=pub, 0.5=internal, 1=private)
    "uses_block_globals",# 2
    "view",              # 3
    "payable",           # 4
    "complexity",        # 5
    "loc",               # 6
    "return_ignored",    # 7
    "call_target_typed", # 8  (0=raw-addr/unsafe, 1=typed/safe, -1=sentinel)
    "has_loop",          # 9
    "external_call_count",# 10
]

# Reverse-map id→name for both node types and edge types
ID_TO_NODE = {v: k for k, v in NODE_TYPES.items()}
ID_TO_EDGE = {v: k for k, v in EDGE_TYPES.items()}
MAX_TYPE_ID = float(max(NODE_TYPES.values()))

def decode_node_type(val: float) -> str:
    tid = int(round(val * MAX_TYPE_ID))
    return ID_TO_NODE.get(tid, f"UNKNOWN({tid})")

def fmt(v: float) -> str:
    if v == 0.0:   return "  0    "
    if v == 1.0:   return "  1    "
    if v == -1.0:  return " -1    "
    return f"{v:+.4f}"

# ── Expected signals per vulnerability class ──────────────────────────────────
# What graph features/structure the model MUST see to detect each class.
EXPECTED_SIGNALS = {
    "Reentrancy": [
        "FUNCTION node with call_target_typed=0.0 (raw addr external call)",
        "FUNCTION node with external_call_count>0",
        "CFG_NODE_CALL before CFG_NODE_WRITE in CONTROL_FLOW (CEI violation)",
        "return_ignored=0 is OK (require(ok) captures it) — NOT the reentrancy signal",
    ],
    "IntegerUO": [
        "FUNCTION node — complexity>0 (has CFG nodes)",
        "NOTE: no 'unchecked' feature in v7 schema (dropped BUG-L2) — model must rely on CodeBERT tokens",
        "CFG_NODE_WRITE nodes present (state mutation)",
        "call_target_typed=1.0 (no external calls expected in pure overflow)",
    ],
    "Timestamp": [
        "FUNCTION node with uses_block_globals=1.0",
        "CFG_NODE_WRITE after uses_block_globals read (decision/unlock from timestamp)",
    ],
    "DenialOfService": [
        "FUNCTION node with has_loop=1.0",
        "FUNCTION node with external_call_count>0 (transfer inside loop)",
        "Multiple CFG_NODE_CALL nodes under loop CFG structure",
    ],
    "MishandledException": [
        "FUNCTION node with return_ignored=1.0",
        "FUNCTION node with external_call_count>0",
        "CFG_NODE_CALL present",
    ],
    "TransactionOrderDependence": [
        "FUNCTION nodes writing shared state (CFG_NODE_WRITE)",
        "FUNCTION nodes reading shared state (CFG_NODE_READ)",
        "NOTE: no explicit TOD structural signal in graph — model relies on CodeBERT",
        "state_variable READS/WRITES edges visible in declaration graph",
    ],
    "GasException": [
        "FUNCTION node with has_loop=1.0 OR large struct push",
        "FUNCTION node with external_call_count>0 (transfer() inside loop)",
        "CFG_NODE_CALL present",
    ],
    "ExternalBug": [
        "FUNCTION node with call_target_typed=0.0 OR high_level call to interface",
        "CFG_NODE_CALL present",
        "NOTE: oracle trust issues mostly detectable via CodeBERT (interface name patterns)",
    ],
    "CallToUnknown": [
        "FUNCTION node with call_target_typed=0.0 (raw addr or delegatecall)",
        "FUNCTION node with external_call_count>0",
        "CFG_NODE_CALL present",
    ],
    "UnusedReturn": [
        "FUNCTION node with return_ignored=1.0",
        "FUNCTION node with external_call_count>0 (high-level call to interface)",
    ],
}


def analyse_contract(sol_path: Path) -> dict:
    """Extract graph and compute all diagnostic data for one contract."""
    config = GraphExtractionConfig()
    try:
        g = extract_contract_graph(sol_path, config)
    except Exception as e:
        return {"error": str(e), "path": str(sol_path)}

    x = g.x  # [N, 11]
    ei = g.edge_index  # [2, E]
    ea = g.edge_attr   # [E]
    meta = g.node_metadata

    N = g.num_nodes
    E = g.num_edges

    # --- Per-node decoded features ---
    nodes = []
    for i in range(N):
        feat = x[i].tolist()
        ntype = decode_node_type(feat[0])
        nodes.append({
            "idx":   i,
            "type":  ntype,
            "name":  meta[i]["name"] if i < len(meta) else "?",
            "src":   meta[i].get("source_lines", []) if i < len(meta) else [],
            "feat":  feat,
        })

    # --- Per-edge decoded ---
    edges = []
    for e in range(E):
        src = int(ei[0, e])
        dst = int(ei[1, e])
        etype_id = int(ea[e])
        etype_name = ID_TO_EDGE.get(etype_id, f"UNKNOWN({etype_id})")
        src_type = nodes[src]["type"] if src < N else "?"
        dst_type = nodes[dst]["type"] if dst < N else "?"
        edges.append({
            "src": src, "src_type": src_type, "src_name": nodes[src]["name"] if src < N else "?",
            "dst": dst, "dst_type": dst_type, "dst_name": nodes[dst]["name"] if dst < N else "?",
            "etype": etype_name, "etype_id": etype_id,
        })

    # --- Distribution summaries ---
    node_type_dist = Counter(n["type"] for n in nodes)
    edge_type_dist = Counter(e["etype"] for e in edges)

    # --- Vulnerability signal extraction ---
    func_nodes = [n for n in nodes if n["type"] in (
        "FUNCTION", "CONSTRUCTOR", "FALLBACK", "RECEIVE"
    )]
    cfg_call_nodes  = [n for n in nodes if n["type"] == "CFG_NODE_CALL"]
    cfg_write_nodes = [n for n in nodes if n["type"] == "CFG_NODE_WRITE"]
    cfg_read_nodes  = [n for n in nodes if n["type"] == "CFG_NODE_READ"]
    cfg_check_nodes = [n for n in nodes if n["type"] == "CFG_NODE_CHECK"]
    cfg_other_nodes = [n for n in nodes if n["type"] == "CFG_NODE_OTHER"]

    # Key signal flags
    any_raw_call     = any(n["feat"][8] < 0.5 for n in func_nodes)  # call_target_typed=0 or -1
    any_extcall      = any(n["feat"][10] > 0 for n in func_nodes)
    any_return_ign   = any(n["feat"][7] == 1.0 for n in func_nodes)
    any_block_global = any(n["feat"][2] == 1.0 for n in func_nodes)
    any_loop         = any(n["feat"][9] == 1.0 for n in func_nodes)

    # CEI check: for each function, check if CFG_NODE_CALL appears before CFG_NODE_WRITE
    # in CONTROL_FLOW order. Use edge connectivity.
    cf_edges = [(e["src"], e["dst"]) for e in edges if e["etype"] == "CONTROL_FLOW"]
    # Build adjacency for reachability
    cf_succ = defaultdict(list)
    for s, d in cf_edges:
        cf_succ[s].append(d)

    def reachable(start: int, targets: set, visited=None) -> bool:
        if visited is None:
            visited = set()
        if start in visited:
            return False
        visited.add(start)
        if start in targets:
            return True
        return any(reachable(nb, targets, visited) for nb in cf_succ[start])

    call_to_write_paths = []  # (call_idx, write_idx) where write is reachable from call
    call_idxs  = set(n["idx"] for n in cfg_call_nodes)
    write_idxs = set(n["idx"] for n in cfg_write_nodes)
    for ci in call_idxs:
        for wi in write_idxs:
            if reachable(ci, {wi}):
                call_to_write_paths.append((ci, wi))

    # ICFG edges
    icfg_edges = [(e["src"], e["dst"]) for e in edges if e["etype"] in ("CALL_ENTRY", "RETURN_TO")]

    # Tokenizer truncation check
    source_text = sol_path.read_text(encoding="utf-8", errors="ignore")

    return {
        "path": str(sol_path),
        "name": sol_path.name,
        "contract_name": g.contract_name,
        "N": N,
        "E": E,
        "nodes": nodes,
        "edges": edges,
        "node_type_dist": dict(node_type_dist),
        "edge_type_dist": dict(edge_type_dist),
        "func_nodes": func_nodes,
        "cfg_call_nodes": cfg_call_nodes,
        "cfg_write_nodes": cfg_write_nodes,
        "cfg_read_nodes": cfg_read_nodes,
        "cfg_check_nodes": cfg_check_nodes,
        "cfg_other_nodes": cfg_other_nodes,
        "any_raw_call":     any_raw_call,
        "any_extcall":      any_extcall,
        "any_return_ign":   any_return_ign,
        "any_block_global": any_block_global,
        "any_loop":         any_loop,
        "call_to_write_paths": call_to_write_paths,
        "icfg_count": len(icfg_edges),
        "source_len": len(source_text),
        "source_lines": source_text.count("\n") + 1,
    }


def print_separator(char="─", width=100):
    print(char * width)

def print_contract_report(result: dict, expected_labels: list[str]):
    name = result["name"]
    print()
    print("=" * 100)
    print(f"  CONTRACT: {name}  (Slither name: {result.get('contract_name','?')})")
    print(f"  Expected: {expected_labels if expected_labels else ['SAFE (no vulnerability)']}")
    print(f"  Source: {result['source_lines']} lines, {result['source_len']} chars")
    print(f"  Graph:  {result['N']} nodes, {result['E']} edges")
    print("=" * 100)

    if "error" in result:
        print(f"  !! EXTRACTION ERROR: {result['error']}")
        return

    # ── Node type distribution ────────────────────────────────────────────────
    print("\n  NODE TYPE DISTRIBUTION:")
    for ntype, cnt in sorted(result["node_type_dist"].items(), key=lambda x: -x[1]):
        print(f"    {ntype:<25} {cnt:>3}")

    # ── Edge type distribution ────────────────────────────────────────────────
    print("\n  EDGE TYPE DISTRIBUTION:")
    for etype, cnt in sorted(result["edge_type_dist"].items(), key=lambda x: -x[1]):
        print(f"    {etype:<25} {cnt:>3}")

    # ── Full node table ───────────────────────────────────────────────────────
    print(f"\n  NODE FEATURE MATRIX  [N={result['N']}]")
    hdr = (f"  {'idx':>3} {'type':<20} {'name':<35}  "
           f"{'vis':>5} {'blk':>5} {'vw':>5} {'pay':>5} {'cpx':>6} {'loc':>6} "
           f"{'ret_ign':>7} {'ctt':>6} {'loop':>5} {'extcall':>7}  src_lines")
    print(hdr)
    print_separator("-", 120)
    for n in result["nodes"]:
        f = n["feat"]
        src_str = str(n["src"][:3])[1:-1] if n["src"] else "-"
        print(
            f"  {n['idx']:>3} {n['type']:<20} {n['name'][:34]:<35}  "
            f"{f[1]:>5.2f} {f[2]:>5.1f} {f[3]:>5.1f} {f[4]:>5.1f} {f[5]:>6.3f} {f[6]:>6.3f} "
            f"{f[7]:>7.1f} {f[8]:>6.1f} {f[9]:>5.1f} {f[10]:>7.3f}  {src_str}"
        )

    # ── Full edge list ────────────────────────────────────────────────────────
    print(f"\n  EDGE LIST  [E={result['E']}]")
    print_separator("-", 120)
    for e in result["edges"]:
        print(
            f"  [{e['src']:>3}]{e['src_type']:<20} "
            f"──{e['etype']:<18}──► "
            f"[{e['dst']:>3}]{e['dst_type']:<20} "
            f"  ({e['src_name'][:25]} → {e['dst_name'][:25]})"
        )

    # ── Vulnerability signal analysis ─────────────────────────────────────────
    print("\n  VULNERABILITY SIGNAL ANALYSIS:")
    print(f"    any_raw_call     (call_target_typed=0): {result['any_raw_call']}")
    print(f"    any_extcall      (external_call_count>0): {result['any_extcall']}")
    print(f"    any_return_ign   (return_ignored=1.0): {result['any_return_ign']}")
    print(f"    any_block_global (uses_block_globals=1): {result['any_block_global']}")
    print(f"    any_loop         (has_loop=1.0): {result['any_loop']}")
    print(f"    CFG_NODE_CALL nodes:  {len(result['cfg_call_nodes'])}")
    print(f"    CFG_NODE_WRITE nodes: {len(result['cfg_write_nodes'])}")
    print(f"    CFG_NODE_READ nodes:  {len(result['cfg_read_nodes'])}")
    print(f"    CFG_NODE_CHECK nodes: {len(result['cfg_check_nodes'])}")
    print(f"    ICFG edges (CALL_ENTRY+RETURN_TO): {result['icfg_count']}")
    if result["call_to_write_paths"]:
        print(f"    CEI VIOLATION PATHS (CALL→WRITE reachable): {len(result['call_to_write_paths'])}")
        for (ci, wi) in result["call_to_write_paths"][:5]:
            cn = result["nodes"][ci]["name"][:30] if ci < result["N"] else "?"
            wn = result["nodes"][wi]["name"][:30] if wi < result["N"] else "?"
            print(f"       CFG_NODE_CALL[{ci}] ({cn}) → CFG_NODE_WRITE[{wi}] ({wn})")
    else:
        print(f"    CEI paths: NONE (no CFG_NODE_CALL → CFG_NODE_WRITE reachable)")

    # Per expected-label: signal checklist
    print("\n  EXPECTED SIGNAL CHECKLIST:")
    for label in expected_labels:
        sigs = EXPECTED_SIGNALS.get(label, [])
        print(f"    [{label}]")
        for s in sigs:
            print(f"      • {s}")

    # ── Function-level deep dive ──────────────────────────────────────────────
    print("\n  FUNCTION NODE DETAILS:")
    for fn in result["func_nodes"]:
        f = fn["feat"]
        print(f"    [{fn['idx']:>3}] {fn['type']:<12} '{fn['name'][:50]}'")
        print(f"          vis={f[1]:.2f}  blk_glob={f[2]:.0f}  view={f[3]:.0f}  payable={f[4]:.0f}")
        print(f"          complexity={f[5]:.4f}  loc={f[6]:.4f}  return_ign={f[7]:.0f}")
        print(f"          call_typed={f[8]:.0f}  loop={f[9]:.0f}  extcall={f[10]:.4f}")
        print(f"          src_lines={fn['src'][:4]}")

    # ── Issues / red flags ────────────────────────────────────────────────────
    print("\n  ISSUES / FLAGS:")
    issues = []

    # Check if schema version is correct (11 dims)
    if result["N"] > 0 and len(result["nodes"][0]["feat"]) != NODE_FEATURE_DIM:
        issues.append(f"!! WRONG FEATURE DIM: got {len(result['nodes'][0]['feat'])}, expected {NODE_FEATURE_DIM}")

    # Reentrancy-specific
    if "Reentrancy" in expected_labels:
        if not result["any_raw_call"]:
            issues.append("MISS: Reentrancy expected but call_target_typed=0 not on any FUNCTION")
        if not result["any_extcall"]:
            issues.append("MISS: Reentrancy expected but external_call_count=0 on all FUNCTIONs")
        if not result["call_to_write_paths"]:
            issues.append("MISS: No CFG_NODE_CALL → CFG_NODE_WRITE CONTROL_FLOW path (CEI not visible)")
        if result["icfg_count"] == 0:
            issues.append("INFO: No CALL_ENTRY/RETURN_TO edges (low-level calls don't generate ICFG)")

    # Timestamp-specific
    if "Timestamp" in expected_labels:
        if not result["any_block_global"]:
            issues.append("MISS: Timestamp expected but uses_block_globals=0 on all FUNCTIONs")

    # DoS-specific
    if "DenialOfService" in expected_labels:
        if not result["any_loop"]:
            issues.append("MISS: DoS expected but has_loop=0 on all FUNCTIONs")
        if not result["any_extcall"]:
            issues.append("MISS: DoS expected but external_call_count=0 (loop with transfer not counted?)")

    # MishandledException-specific
    if "MishandledException" in expected_labels:
        if not result["any_return_ign"]:
            issues.append("MISS: MishandledException expected but return_ignored=0 on all FUNCTIONs")

    # UnusedReturn-specific
    if "UnusedReturn" in expected_labels:
        if not result["any_return_ign"]:
            issues.append("MISS: UnusedReturn expected but return_ignored=0 on all FUNCTIONs")

    # IntegerUO-specific
    if "IntegerUO" in expected_labels:
        issues.append("NOTE: No 'unchecked' feature in v7 schema — model can only detect via CodeBERT tokens + complexity/loc signals")
        if not result["cfg_write_nodes"]:
            issues.append("MISS: IntegerUO expected but no CFG_NODE_WRITE nodes")

    # CallToUnknown-specific
    if "CallToUnknown" in expected_labels:
        if not result["any_raw_call"]:
            issues.append("MISS: CallToUnknown expected but call_target_typed != 0 on all FUNCTIONs")
        if not result["any_extcall"]:
            issues.append("MISS: CallToUnknown expected but external_call_count=0 on all FUNCTIONs")

    # Safe contract checks
    if not expected_labels:
        if result["any_raw_call"]:
            issues.append("WARN: Safe contract but has call_target_typed=0 FUNCTION node — may cause false positive")
        if result["any_return_ign"]:
            issues.append("WARN: Safe contract but return_ignored=1.0 FUNCTION node — may cause false positive")
        if result["call_to_write_paths"]:
            issues.append(f"WARN: Safe contract but has {len(result['call_to_write_paths'])} CEI-looking paths — may cause false positive Reentrancy")

    # Graph size
    if result["N"] < 15:
        issues.append(f"WARN: Very small graph (N={result['N']}) — likely far below training distribution median")
    if result["N"] < 8:
        issues.append(f"!! CRITICAL: Extremely small graph (N={result['N']}) — GNN has almost no neighborhood")

    # Missing CFG
    if result["func_nodes"] and not result["cfg_call_nodes"] and not result["cfg_write_nodes"] and not result["cfg_read_nodes"]:
        issues.append("WARN: No CFG node types at all — CONTAINS/CONTROL_FLOW edge extraction may have failed")

    if not issues:
        issues.append("OK: No major issues detected")

    for issue in issues:
        print(f"    {issue}")

    print()


# ── Parse expected labels from contract comments ───────────────────────────────
def parse_expected(sol_path: Path) -> list[str]:
    text = sol_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("// expect:"):
            raw = line.split("//")[1].strip().removeprefix("expect:").strip()
            if not raw:
                return []
            return [s.strip() for s in raw.split(",") if s.strip()]
    return []


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sol_files = sorted(CONTRACTS_DIR.glob("*.sol"))
    if not sol_files:
        print("No .sol files found in", CONTRACTS_DIR)
        sys.exit(1)

    print(f"\nSENTINEL FULL GRAPH DIAGNOSTIC — {len(sol_files)} contracts")
    print(f"Schema: NODE_FEATURE_DIM={NODE_FEATURE_DIM}, {len(NODE_TYPES)} node types, {len(EDGE_TYPES)} edge types")
    print(f"Feature dims: {FEATURE_NAMES}")

    # ── Global stats accumulator ──────────────────────────────────────────────
    global_node_counts = []
    global_edge_counts = []
    global_issues = {}

    for sol in sol_files:
        expected = parse_expected(sol)
        result = analyse_contract(sol)
        print_contract_report(result, expected)
        if "error" not in result:
            global_node_counts.append(result["N"])
            global_edge_counts.append(result["E"])

    # ── Summary across all contracts ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  GLOBAL SUMMARY ACROSS ALL 20 CONTRACTS")
    print("=" * 100)
    if global_node_counts:
        print(f"  Node counts: min={min(global_node_counts)}  max={max(global_node_counts)}  "
              f"mean={sum(global_node_counts)/len(global_node_counts):.1f}  "
              f"median={sorted(global_node_counts)[len(global_node_counts)//2]}")
        print(f"  Edge counts: min={min(global_edge_counts)}  max={max(global_edge_counts)}  "
              f"mean={sum(global_edge_counts)/len(global_edge_counts):.1f}  "
              f"median={sorted(global_edge_counts)[len(global_edge_counts)//2]}")
    print()


if __name__ == "__main__":
    main()
