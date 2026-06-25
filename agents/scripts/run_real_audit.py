#!/usr/bin/env python3
"""
Real E2E audit harness for the SENTINEL agents module.

Runs the 9-node LangGraph against a real .sol contract with REAL services:
  - LM Studio (LLM calls in cross_validator + synthesizer narrative)
  - ML API at :8001 (Run 12 inference — ml_assessment, graph_explain)
  - 4 MCP servers at :8010, :8011, :8012, :8013

No mocks (unless --no-llm is passed for a fast smoke run).

USAGE
    cd ~/projects/sentinel/agents
    poetry run python scripts/run_real_audit.py [OPTIONS] [CONTRACT.sol ...]

    # Default: runs both test_contracts fixtures with full LLM calls
    poetry run python scripts/run_real_audit.py

    # Single contract, fast path (no LLM calls)
    poetry run python scripts/run_real_audit.py --no-llm test_contracts/vulnerable_reentrant.sol

    # Custom LM Studio port, profile mode
    poetry run python scripts/run_real_audit.py --lm-studio-url http://localhost:9999/v1 --profile

    # See TRUE per-step timing with nothing artificially truncated
    poetry run python scripts/run_real_audit.py --unbounded-timeouts test_contracts/safe_storage.sol

CLI FLAGS (all have env-var fallbacks)
    --lm-studio-url URL       default: $LM_STUDIO_BASE_URL
    --ml-api-url URL          default: $MODULE1_INFERENCE_URL (http://localhost:8001)
    --mcp-inference-url URL   default: http://localhost:$MCP_INFERENCE_PORT (8010)
    --mcp-rag-url URL         default: http://localhost:$MCP_RAG_PORT (8011)
    --mcp-audit-url URL       default: http://localhost:$MCP_AUDIT_PORT (8012)
    --mcp-graph-url URL       default: http://localhost:$MCP_GRAPH_INSPECTOR_PORT (8013)
    --mcp-representation-url URL  default: http://localhost:$MCP_REPRESENTATION_PORT (8014)
    --output-dir DIR          default: ./test_audit_reports
    --timeout-s N             default: 300 (per-audit wall-clock limit)
    --no-llm                  skip cross_validator + synthesizer LLM calls (fast path)
    --profile                 raise console log level to DEBUG (timing is always-on)
    --check-services          probe all 5 services at startup (default: true)
    --no-check-services       skip startup health check
    --log-file PATH           default: <output-dir>/run_<timestamp>.log

TIMEOUT FLAGS (2026-06-21 — defaults centralized in src/orchestration/timeouts.py)
    --lm-studio-timeout-s S        LM Studio HTTP client floor (env LM_STUDIO_TIMEOUT)
    --cross-validator-timeout-s S  single-pass mode, DEBATE_MODE=off (env CROSS_VALIDATOR_TIMEOUT_S)
    --debate-timeout-s S           entire 3-role debate, ONE budget (env DEBATE_TIMEOUT_S)
    --synthesizer-timeout-s S      narrative LLM call (env SYNTHESIZER_TIMEOUT_S)
    --reflection-timeout-s S       self-critique LLM call (env REFLECTION_TIMEOUT_S)
    --aderyn-timeout-s S           Aderyn subprocess (env ADERYN_TIMEOUT_S)
    --unbounded-timeouts           set every timeout above + --timeout-s to 3600s
                                    (an explicit --<x>-timeout-s flag still wins per-value)
    Every node logs its own START/DONE+elapsed time automatically (see
    src/orchestration/timing.py) — no flag needed to see per-step timing,
    only to change how long a step is ALLOWED to run before being cut off.

OUTPUT
    JSON report per contract:   <output-dir>/<contract_stem>_report.json
    Combined run log:           <output-dir>/run_<timestamp>.log
    Console output:             real-time status

EXIT CODES
    0  all audits succeeded
    1  one or more audits failed or timed out
    2  startup health check failed (no point running audits)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
from dataclasses import asdict
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── CRITICAL: load .env BEFORE reading any env vars ──────────────────────
# python-dotenv is already a dep of the agents module, so it's available.
# We do this BEFORE _DEFAULT_ENV so that LM_STUDIO_BASE_URL etc. are populated.
from dotenv import load_dotenv

_THIS_DIR     = Path(__file__).resolve().parent
_AGENTS_DIR   = _THIS_DIR.parent
_PROJECT_ROOT = _AGENTS_DIR.parent
load_dotenv(_AGENTS_DIR / ".env", override=True)

# ── CRITICAL: set env vars BEFORE importing the agents module ─────────────
# client.py reads LM_STUDIO_BASE_URL etc. at IMPORT time (module-level).
# CLI args must override .env values, so we set os.environ first.
# This block runs before any `from src...` import below.

# We parse env-var fallbacks here too, just to know what we'd use.
_DEFAULT_ENV = {
    "LM_STUDIO_BASE_URL":     os.getenv("LM_STUDIO_BASE_URL", ""),
    "MODULE1_INFERENCE_URL":  os.getenv("MODULE1_INFERENCE_URL", "http://localhost:8001"),
    "MCP_INFERENCE_PORT":     os.getenv("MCP_INFERENCE_PORT", "8010"),
    "MCP_RAG_PORT":           os.getenv("MCP_RAG_PORT", "8011"),
    "MCP_AUDIT_PORT":         os.getenv("MCP_AUDIT_PORT", "8012"),
    "MCP_GRAPH_INSPECTOR_PORT": os.getenv("MCP_GRAPH_INSPECTOR_PORT", "8013"),
    "MCP_REPRESENTATION_PORT":  os.getenv("MCP_REPRESENTATION_PORT", "8014"),
}

# Make agents/ importable for src.* modules — set BEFORE any src import
sys.path.insert(0, str(_AGENTS_DIR))

from src.orchestration import timeouts  # noqa: E402 — needs sys.path set above

# ── Logger setup (loguru) ────────────────────────────────────────────────
# We add TWO sinks:
#   1. Console — colored, real-time feedback
#   2. File    — full log including DEBUG level, persistent record
# loguru's logger is the same one used by the agents module, so their
# logs flow into our file too.

from loguru import logger as _loguru
_loguru.remove()  # strip default stderr sink

_RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ══════════════════════════════════════════════════════════════════════════
# CLI parsing
# ══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_real_audit",
        description="SENTINEL E2E audit harness — real LLM + real MCP + real ML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "contracts", nargs="*", type=Path,
        help="Path(s) to .sol contract(s). Default: test_contracts/{vulnerable_reentrant,safe_storage}.sol. "
             "Mutually exclusive with --corpus (if --corpus is given, positional contracts are ignored).",
    )
    p.add_argument("--corpus", type=Path, default=None, metavar="DIR",
                   help="Run every *.sol under DIR (recursive) as a corpus batch. "
                        "Used by the WS0 gate infra (see docs/plan/agents/2026-06-21-agents-redesign/"
                        "03_GATE_INFRASTRUCTURE_PLAN.md). Per-contract reports land in --output-dir. "
                        "Intended use: the 66-contract benchmark at "
                        "data_module/benchmarks/benchmark_v0.1_quickstart/contracts/by_class/ "
                        "or agents/test_contracts/edge/. Combine with --no-llm for a fast deterministic gate.")
    p.add_argument("--lm-studio-url",     metavar="URL", help="LM Studio OpenAI-compatible base URL")
    p.add_argument("--ml-api-url",        metavar="URL", help="SENTINEL ML inference API URL")
    p.add_argument("--mcp-inference-url", metavar="URL", help="MCP inference server URL")
    p.add_argument("--mcp-rag-url",       metavar="URL", help="MCP RAG server URL")
    p.add_argument("--mcp-audit-url",     metavar="URL", help="MCP audit server URL")
    p.add_argument("--mcp-graph-url",     metavar="URL", help="MCP graph inspector server URL")
    p.add_argument("--mcp-representation-url", metavar="URL", help="MCP representation server URL (data_module CFG wrapper, port 8014)")
    p.add_argument("--output-dir",        type=Path, default=Path("test_audit_reports"),
                   help="Where to save JSON reports and run log")
    p.add_argument("--timeout-s",         type=float, metavar="S", default=None,
                   help="Per-audit wall-clock timeout in seconds (default: 300, or "
                        f"{int(timeouts.UNBOUNDED_TIMEOUT_S)} with --unbounded-timeouts)")
    p.add_argument("--no-llm",            action="store_true",
                   help="Skip LLM-dependent nodes (cross_validator, synthesizer narrative). "
                        "Falls back to rule-based verdicts. ~5-10x faster.")
    p.add_argument("--profile",           action="store_true",
                   help="Raise console log level to DEBUG (per-node timing is always-on; "
                        "see src/orchestration/timing.py)")
    p.add_argument("--check-services",    action="store_true", default=True,
                   help="Probe all 5 services at startup")
    p.add_argument("--no-check-services", dest="check_services", action="store_false",
                   help="Skip startup health check (risky)")

    # ── Timeouts (2026-06-21) — every default lives in src/orchestration/
    # timeouts.py; these flags just set the matching env var before any
    # agents module is imported. Omit a flag to use that module's default.
    p.add_argument("--lm-studio-timeout-s",        type=float, metavar="S",
                   help=f"LM Studio HTTP client timeout (env LM_STUDIO_TIMEOUT, "
                        f"default {timeouts.DEFAULT_LM_STUDIO_TIMEOUT_S}s)")
    p.add_argument("--cross-validator-timeout-s",  type=float, metavar="S",
                   help=f"Single-pass cross_validator call, DEBATE_MODE=off "
                        f"(env CROSS_VALIDATOR_TIMEOUT_S, default "
                        f"{timeouts.DEFAULT_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S}s)")
    p.add_argument("--debate-timeout-s",           type=float, metavar="S",
                   help=f"Entire 3-role debate as one budget, DEBATE_MODE=on "
                        f"(env DEBATE_TIMEOUT_S, default {timeouts.DEFAULT_DEBATE_TIMEOUT_S}s)")
    p.add_argument("--synthesizer-timeout-s",      type=float, metavar="S",
                   help=f"synthesizer's LLM narrative call (env SYNTHESIZER_TIMEOUT_S, "
                        f"default {timeouts.DEFAULT_SYNTHESIZER_NARRATIVE_TIMEOUT_S}s)")
    p.add_argument("--reflection-timeout-s",       type=float, metavar="S",
                   help=f"reflection's LLM self-critique call (env REFLECTION_TIMEOUT_S, "
                        f"default {timeouts.DEFAULT_REFLECTION_TIMEOUT_S}s)")
    p.add_argument("--aderyn-timeout-s",           type=float, metavar="S",
                   help=f"Aderyn subprocess invocation (env ADERYN_TIMEOUT_S, "
                        f"default {timeouts.DEFAULT_ADERYN_TIMEOUT_S}s)")
    p.add_argument("--unbounded-timeouts",         action="store_true",
                   help=f"Set every timeout above (and --timeout-s, unless explicitly "
                        f"given) to {timeouts.UNBOUNDED_TIMEOUT_S:.0f}s, so each step only "
                        f"stops at natural completion — use to observe true per-step "
                        f"timing with nothing artificially truncated. An explicit "
                        f"--<x>-timeout-s flag still overrides this for that one value.")
    return p


def _resolve_urls(args: argparse.Namespace) -> dict[str, str]:
    """Build final URL map: CLI > env > hard default."""
    urls = {
        "lm_studio":     args.lm_studio_url     or _DEFAULT_ENV["LM_STUDIO_BASE_URL"]     or "http://localhost:1234/v1",
        "ml_api":        args.ml_api_url        or _DEFAULT_ENV["MODULE1_INFERENCE_URL"],
        "mcp_inference": args.mcp_inference_url or f"http://localhost:{_DEFAULT_ENV['MCP_INFERENCE_PORT']}",
        "mcp_rag":       args.mcp_rag_url       or f"http://localhost:{_DEFAULT_ENV['MCP_RAG_PORT']}",
        "mcp_audit":     args.mcp_audit_url     or f"http://localhost:{_DEFAULT_ENV['MCP_AUDIT_PORT']}",
        "mcp_graph":     args.mcp_graph_url     or f"http://localhost:{_DEFAULT_ENV['MCP_GRAPH_INSPECTOR_PORT']}",
        "mcp_representation": args.mcp_representation_url or f"http://localhost:{_DEFAULT_ENV['MCP_REPRESENTATION_PORT']}",
    }
    # Push into os.environ so the agents module sees them at import time
    os.environ["LM_STUDIO_BASE_URL"]    = urls["lm_studio"]
    os.environ["MODULE1_INFERENCE_URL"] = urls["ml_api"]
    return urls


def _resolve_timeouts(args: argparse.Namespace) -> dict[str, float]:
    """
    Build the final timeout map: explicit CLI flag > --unbounded-timeouts >
    src/orchestration/timeouts.py default. Pushes each into os.environ BEFORE
    any agents module is imported, so every node picks it up on first use
    (and so src.llm.client — the one import-time read — sees it too).

    Also resolves args.timeout_s (the script's own overall per-audit limit,
    default None) the same way, since it doesn't have a fixed default either
    when --unbounded-timeouts is in play.
    """
    pairs = [
        ("--lm-studio-timeout-s",       args.lm_studio_timeout_s,
         timeouts.ENV_LM_STUDIO_TIMEOUT_S,       timeouts.DEFAULT_LM_STUDIO_TIMEOUT_S),
        ("--cross-validator-timeout-s", args.cross_validator_timeout_s,
         timeouts.ENV_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
         timeouts.DEFAULT_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S),
        ("--debate-timeout-s",          args.debate_timeout_s,
         timeouts.ENV_DEBATE_TIMEOUT_S,          timeouts.DEFAULT_DEBATE_TIMEOUT_S),
        ("--synthesizer-timeout-s",     args.synthesizer_timeout_s,
         timeouts.ENV_SYNTHESIZER_NARRATIVE_TIMEOUT_S,
         timeouts.DEFAULT_SYNTHESIZER_NARRATIVE_TIMEOUT_S),
        ("--reflection-timeout-s",      args.reflection_timeout_s,
         timeouts.ENV_REFLECTION_TIMEOUT_S,      timeouts.DEFAULT_REFLECTION_TIMEOUT_S),
        ("--aderyn-timeout-s",          args.aderyn_timeout_s,
         timeouts.ENV_ADERYN_TIMEOUT_S,          timeouts.DEFAULT_ADERYN_TIMEOUT_S),
    ]
    resolved: dict[str, float] = {}
    for flag_name, explicit_value, env_var, default in pairs:
        if explicit_value is not None:
            value = explicit_value
        elif args.unbounded_timeouts:
            value = timeouts.UNBOUNDED_TIMEOUT_S
        else:
            value = default
        os.environ[env_var] = str(value)
        resolved[env_var] = value

    # The script's own overall per-audit wall-clock limit follows the same
    # precedence, with its own historical default (300s) when neither an
    # explicit value nor --unbounded-timeouts is given.
    if args.timeout_s is not None:
        args.timeout_s = args.timeout_s
    elif args.unbounded_timeouts:
        args.timeout_s = timeouts.UNBOUNDED_TIMEOUT_S
    else:
        args.timeout_s = 300.0
    resolved["--timeout-s"] = args.timeout_s

    return resolved


# ══════════════════════════════════════════════════════════════════════════
# Logging helpers
# ══════════════════════════════════════════════════════════════════════════

def _setup_logging(log_file: Path, profile: bool) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    console_level = "DEBUG" if profile else "INFO"

    # Console sink — colored, level filtered
    _loguru.add(
        sys.stderr,
        level=console_level,
        format=("<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <7}</level> | "
                "<cyan>{name}:{function}:{line}</cyan> — <level>{message}</level>"),
        colorize=True,
    )
    # File sink — full DEBUG, persistent
    _loguru.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS Z} | {level: <7} | {name}:{function}:{line} — {message}",
        rotation=None,  # one file per run
        enqueue=True,   # thread-safe
    )
    _loguru.info(f"logging initialised — console={console_level} | file={log_file}")


def _section(title: str) -> None:
    """Visual section break in the log."""
    bar = "═" * 72
    _loguru.info(f"\n{bar}\n{title}\n{bar}")


def _kv(d: dict, indent: int = 2) -> None:
    """Log a dict as key=value lines."""
    pad = " " * indent
    width = max(len(str(k)) for k in d)
    for k, v in d.items():
        _loguru.info(f"{pad}{str(k):<{width}}  {v}")


# ══════════════════════════════════════════════════════════════════════════
# Service health + version logging
# ══════════════════════════════════════════════════════════════════════════

def _probe(url: str, label: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Return (ok, detail). Logs the probe."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")[:200]
            _loguru.debug(f"PROBE {label} {url} → {r.status} | {body[:80]}")
            return True, f"HTTP {r.status} | {body[:120]}"
    except urllib.error.URLError as e:
        _loguru.warning(f"PROBE {label} {url} → DOWN | {e}")
        return False, f"UNREACHABLE | {e}"
    except Exception as e:
        _loguru.warning(f"PROBE {label} {url} → ERROR | {e}")
        return False, f"ERROR | {e}"


def _check_services(urls: dict[str, str]) -> dict[str, tuple[bool, str]]:
    """Probe all 5 services. Returns status dict."""
    _section("SERVICE HEALTH CHECK")
    targets = [
        ("ml_api",        urls["ml_api"]        + "/health"),
        ("mcp_inference", urls["mcp_inference"] + "/health"),
        ("mcp_rag",       urls["mcp_rag"]       + "/health"),
        ("mcp_audit",     urls["mcp_audit"]     + "/health"),
        ("mcp_graph",     urls["mcp_graph"]     + "/health"),
        ("mcp_representation", urls["mcp_representation"] + "/health"),
        ("lm_studio",     urls["lm_studio"]     + "/models"),
    ]
    results = {}
    for label, url in targets:
        ok, detail = _probe(url, label)
        results[label] = (ok, detail)
        status = "✓" if ok else "✗"
        _loguru.info(f"  {status} {label:14} {url:55}  {detail}")
    return results


def _log_service_versions(urls: dict[str, str]) -> None:
    """Capture and log every service version we can identify (per analysis plan §Step 0)."""
    _section("SERVICE VERSIONS")
    versions = {}

    # ML checkpoint (project-root relative)
    ckpt = _PROJECT_ROOT / "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
    if ckpt.exists():
        md5 = hashlib.md5()
        with ckpt.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                md5.update(chunk)
        versions["checkpoint"] = f"{ckpt.name} | {ckpt.stat().st_size / 1024 / 1024:.1f} MB | md5={md5.hexdigest()[:12]}"
    else:
        versions["checkpoint"] = f"MISSING at {ckpt}"

    # LM Studio models
    try:
        with urllib.request.urlopen(urls["lm_studio"] + "/models", timeout=5) as r:
            data = json.loads(r.read())
            model_ids = [m["id"] for m in data.get("data", [])]
            versions["lm_studio_models"] = ", ".join(model_ids) if model_ids else "EMPTY"
    except Exception as e:
        versions["lm_studio_models"] = f"PROBE FAILED | {e}"

    # RAG index
    rag_dir = _AGENTS_DIR / "data/index"
    if rag_dir.exists():
        for f in ("faiss.index", "bm25.pkl", "chunks.pkl", "index_metadata.json"):
            fp = rag_dir / f
            if fp.exists():
                h = hashlib.sha256(fp.read_bytes()).hexdigest()[:12]
                versions[f"rag_{f}"] = f"{f} | {fp.stat().st_size / 1024:.1f} KB | sha256={h}"
        meta = rag_dir / "index_metadata.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                versions["rag_meta"] = f"build_date={m.get('build_date', '?')} | num_chunks={m.get('num_chunks', '?')} | emb={m.get('embedding_model', '?')}"
            except Exception as e:
                versions["rag_meta"] = f"parse error: {e}"
    else:
        versions["rag_index"] = f"MISSING at {rag_dir}"

    # Slither version (run from agents venv)
    try:
        out = subprocess.run(
            ["poetry", "run", "slither", "--version"],
            cwd=str(_AGENTS_DIR), capture_output=True, text=True, timeout=15,
        )
        versions["slither"] = (out.stdout or out.stderr).strip().split("\n")[0][:80]
    except Exception as e:
        versions["slither"] = f"NOT INSTALLED | {e}"

    # aderyn (non-fatal)
    try:
        out = subprocess.run(
            ["poetry", "run", "aderyn", "--version"],
            cwd=str(_AGENTS_DIR), capture_output=True, text=True, timeout=15,
        )
        versions["aderyn"] = (out.stdout or out.stderr).strip().split("\n")[0][:80]
    except Exception:
        versions["aderyn"] = "NOT INSTALLED (non-fatal per nodes.py:265)"

    # Mock flags
    versions["AUDIT_MOCK"]        = os.getenv("AUDIT_MOCK", "false")
    versions["MODULE1_MOCK"]      = os.getenv("MODULE1_MOCK", "false")
    versions["LM_STUDIO_TIMEOUT"] = os.getenv("LM_STUDIO_TIMEOUT", "60")

    _kv(versions, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# Node timing (2026-06-21: now built into the module itself)
# ══════════════════════════════════════════════════════════════════════════
# This script used to monkeypatch every node with its OWN ad-hoc timing
# wrapper (`_wrap_node`/`_build_instrumented_graph`) — meaning timing
# visibility only existed when a contract happened to be audited through THIS
# script. `graph.py`'s `build_graph()` now wraps every node with
# `timed_node()` (src/orchestration/timing.py) at registration time, so the
# identical START/DONE+elapsed log pair appears for every node in EVERY
# context (production MCP-driven server, this script, ad-hoc REPL use) with
# no per-caller wrapper needed. `--profile` now just raises the console log
# level to DEBUG (see `_setup_logging`) to surface the extra detail already
# emitted by individual nodes/sub-steps (e.g. cross_validator's 3 debate roles
# are each individually timed, not just the aggregate).


# ══════════════════════════════════════════════════════════════════════════
# --no-llm patch (skip LLM calls inside nodes)
# ══════════════════════════════════════════════════════════════════════════

def _patch_no_llm() -> None:
    """
    Make cross_validator + synthesizer narrative return their non-LLM fallback
    immediately. cross_validator already returns {} on LLM failure, so the
    synthesizer uses rule-based verdicts. The synthesizer's LLM narrative
    branch is wrapped in try/except and sets narrative=None on failure.

    We can't easily inject failures, but we CAN replace the LLM call sites
    with stubs that raise immediately. The except blocks already handle
    the failure gracefully.
    """
    from src.llm import client as llm_client

    _loguru.warning("=" * 60)
    _loguru.warning(" --no-llm MODE ACTIVE")
    _loguru.warning(" Replacing get_strong_llm() with a stub that raises immediately.")
    _loguru.warning(" cross_validator → returns {} (rule-based verdicts via synthesizer)")
    _loguru.warning(" synthesizer narrative → falls back to None")
    _loguru.warning(" Expected speedup: 5-10x. Expected cost: lower verdict quality.")
    _loguru.warning("=" * 60)

    class _StubLLM:
        def invoke(self, *a, **kw):
            raise RuntimeError("LLM disabled by --no-llm flag")

    _original_get_strong_llm = llm_client.get_strong_llm

    def _stub_strong_llm(*a, **kw):
        return _StubLLM()

    llm_client.get_strong_llm = _stub_strong_llm
    # Also patch the import in graph.py's namespace (it was imported at graph import time)
    import src.orchestration.graph as graph_mod
    import src.orchestration.nodes as nodes_mod
    for mod in (graph_mod, nodes_mod):
        if hasattr(mod, "get_strong_llm"):
            mod.get_strong_llm = _stub_strong_llm
    # Keep reference so we can restore
    _patch_no_llm._original = _original_get_strong_llm


# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
# Main audit runner
# ══════════════════════════════════════════════════════════════════════════

def _serialize_evidence(evidence_list: list) -> list[dict]:
    """Convert Evidence dataclass objects to plain dicts for JSON serialization."""
    return [asdict(e) for e in evidence_list]


async def run_audit(contract_path: Path, urls: dict[str, str], args: argparse.Namespace) -> dict:
    contract_code = contract_path.read_text()
    # Generate a valid 20-byte (40 hex chars) Ethereum address for E2E.
    # AuditRegistry requires real address format. We derive it deterministically
    # from the contract filename so the same contract → same address across runs.
    import hashlib
    address_hash = hashlib.sha256(contract_path.name.encode()).hexdigest()[:40]
    contract_address = "0x" + address_hash

    _section(f"AUDIT START — {contract_path.name}")
    _kv({
        "contract":         str(contract_path),
        "contract_address": contract_address,
        "contract_chars":   len(contract_code),
        "contract_lines":   contract_code.count("\n") + 1,
        "started_at_utc":   datetime.now(timezone.utc).isoformat(),
        "timeout_s":        args.timeout_s,
        "no_llm_mode":      args.no_llm,
        "profile_mode":     args.profile,
    })

    from src.orchestration.graph import build_graph
    graph = build_graph(use_checkpointer=False)
    initial_state = {
        "contract_code": contract_code,
        "contract_address": contract_address,
    }

    start_wall = time.time()
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=args.timeout_s,
        )
    except asyncio.TimeoutError:
        dt = time.time() - start_wall
        _loguru.error(f"AUDIT TIMEOUT after {dt:.1f}s (limit={args.timeout_s}s)")
        return {
            "contract": contract_path.name, "success": False,
            "error": f"timeout after {dt:.1f}s",
            "total_wall_s": dt,
        }
    except Exception as e:
        dt = time.time() - start_wall
        _loguru.error(f"AUDIT EXCEPTION after {dt:.1f}s: {type(e).__name__}: {e}")
        _loguru.debug(f"traceback:\n{traceback.format_exc()}")
        return {
            "contract": contract_path.name, "success": False,
            "error": f"{type(e).__name__}: {e}",
            "total_wall_s": dt,
        }

    total_wall = time.time() - start_wall
    if total_wall > 180:
        _loguru.warning(f"AUDIT SLOW: total wall {total_wall:.1f}s > 180s")
    elif total_wall > 120:
        _loguru.warning(f"AUDIT BORDERLINE: total wall {total_wall:.1f}s > 120s")

    # ── Extract + summarise ───────────────────────────────────────────────
    final_report = result.get("final_report", {}) or {}
    routing      = result.get("routing_decisions", []) or []
    verdicts     = result.get("verdicts", {}) or {}
    ml_result    = result.get("ml_result", {}) or {}
    # narrative is nested inside final_report (synthesizer puts it there at nodes.py:1393)
    narrative    = final_report.get("narrative") or result.get("narrative")

    _section(f"AUDIT COMPLETE — {contract_path.name}  ({total_wall:.2f}s)")
    _kv({
        "overall_label":     final_report.get("overall_label", "N/A"),
        "overall_verdict":   final_report.get("overall_verdict", "N/A"),
        "risk_probability":  final_report.get("risk_probability", "N/A"),
        "top_vulnerability": final_report.get("top_vulnerability", "N/A"),
        "path_taken":        final_report.get("path_taken", "N/A"),
        "truncated":         ml_result.get("truncated", "N/A"),
        "num_nodes_AST":     ml_result.get("num_nodes", "N/A"),
        "num_edges_AST":     ml_result.get("num_edges", "N/A"),
        "verdict_count":     len(verdicts),
        "static_findings":   len(result.get("static_findings", []) or []),
        "rag_results":       len(result.get("rag_results", []) or []),
        "audit_history":     len(result.get("audit_history", []) or []),
        "narrative_chars":   len(result.get("narrative") or ""),
    })

    if verdicts:
        _loguru.info("verdicts:")
        for cls, v in verdicts.items():
            _loguru.info(f"    {cls:25} {v}")
    if narrative:
        _loguru.info(f"\n  ── NARRATIVE ({len(narrative)} chars) ──")
        for line in narrative.split("\n"):
            _loguru.info(f"    {line}")
    if routing:
        _loguru.info(f"\n  ── routing decisions ({len(routing)}) ──")
        for r in routing[:8]:
            _loguru.info(f"    {r}")

    # ── Persist report ────────────────────────────────────────────────────
    out_path = args.output_dir / f"{contract_path.stem}_report.json"
    report = {
        "contract":         contract_path.name,
        "contract_address": contract_address,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "total_wall_s":     total_wall,
        "config": {
            "lm_studio_url":     urls["lm_studio"],
            "ml_api_url":        urls["ml_api"],
            "mcp_inference_url": urls["mcp_inference"],
            "mcp_rag_url":       urls["mcp_rag"],
            "mcp_audit_url":     urls["mcp_audit"],
            "mcp_graph_url":     urls["mcp_graph"],
            "no_llm":            args.no_llm,
            "profile":           args.profile,
            "timeout_s":         args.timeout_s,
        },
        "routing_decisions":        routing,
        "ml_result":                ml_result,
        "verdicts":                 verdicts,
        "quick_screen_hits":        result.get("quick_screen_hits", {}),
        "static_findings_count":    len(result.get("static_findings", []) or []),
        "static_findings":          result.get("static_findings", []) or [],
        "rag_results_count":        len(result.get("rag_results", []) or []),
        "audit_history_count":      len(result.get("audit_history", []) or []),
        "graph_explanations_classes": list((result.get("graph_explanations", {}) or {}).keys()),
        "consensus_verdict":         result.get("consensus_verdict", {}),
        "debate_transcript":         result.get("debate_transcript", {}),
        "confirmations":            result.get("confirmations", {}),
        "contradictions":           result.get("contradictions", {}),
        "final_report":             final_report,
        "narrative":                narrative,
        "error":                    result.get("error"),
        # ── P2 dual-write (2026-06-24) ────────────────────────────────────
        "evidence_list":            _serialize_evidence(result.get("evidence_list", []) or []),
        "verdict_provable":         result.get("verdict_provable", {}),
        "verdict_full":             result.get("verdict_full", {}),
    }
    out_path.write_text(json.dumps(report, indent=2, default=str))
    _loguru.info(f"report saved → {out_path}")

    return {
        "contract":         contract_path.name,
        "success":          True,
        "total_wall_s":     total_wall,
        "verdicts":         verdicts,
        "overall_label":    final_report.get("overall_label"),
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

async def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Resolve URLs + timeouts BEFORE any agents import (so client.py and
    # every node's `os.getenv(...)` call sees them on first use).
    urls = _resolve_urls(args)
    resolved_timeouts = _resolve_timeouts(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / f"run_{_RUN_TIMESTAMP}.log"

    _setup_logging(log_file, args.profile)

    _section("CONFIGURATION")
    _kv({
        "cwd":                str(Path.cwd()),
        "pid":                os.getpid(),
        "args":               " ".join(sys.argv[1:]),
        "log_file":           str(log_file),
        "output_dir":         str(args.output_dir),
        "timeout_s":          args.timeout_s,
        "no_llm":             args.no_llm,
        "profile":            args.profile,
        "unbounded_timeouts": args.unbounded_timeouts,
    }, indent=2)

    _section("RESOLVED URLs (CLI > env > default)")
    _kv(urls, indent=2)

    _section("RESOLVED TIMEOUTS (CLI > --unbounded-timeouts > default)")
    _kv(resolved_timeouts, indent=2)

    # Service health check
    if args.check_services:
        results = _check_services(urls)
        down = [k for k, (ok, _) in results.items() if not ok]
        if down:
            _loguru.error(f"SERVICES DOWN: {down}")
            if not args.no_llm and "lm_studio" in down:
                _loguru.error("Cannot run LLM-dependent mode without LM Studio. Use --no-llm to skip LLM calls.")
                sys.exit(2)
            if "ml_api" in down or "mcp_inference" in down:
                _loguru.error("ML API and inference MCP are required even in --no-llm mode.")
                sys.exit(2)
    else:
        _loguru.warning("Skipping service health check (--no-check-services)")

    # Service versions
    _log_service_versions(urls)

    # --no-llm patch (MUST happen after env-var injection, before graph build)
    if args.no_llm:
        _patch_no_llm()

    # Contracts
    if args.corpus:
        # --corpus mode (WS0 gate infra): walk every *.sol under the dir,
        # recursively (following symlinks — the combined corpus uses
        # symlinked class dirs), sorted for deterministic ordering. The
        # by_class/<CLASS>/ directory structure is the label source for the
        # comparator (eval_benchmark.py); the per-contract .json sidecar
        # alongside each .sol carries labels + ground_truth (or `// expect:`
        # header in the .sol itself for the manual_hand_written_contracts
        # corpus).
        if not args.corpus.is_dir():
            _loguru.error(f"--corpus dir does not exist: {args.corpus}")
            sys.exit(1)
        import os as _os
        _sol_paths = [
            Path(root) / f
            for root, _dirs, files in _os.walk(args.corpus, followlinks=True)
            for f in files
            if f.endswith(".sol")
        ]
        contracts = sorted(_sol_paths)
        if not contracts:
            _loguru.error(f"--corpus dir contains no .sol files: {args.corpus}")
            sys.exit(1)
        # Class breakdown from the parent dir name (by_class/<CLASS>/<file>.sol).
        from collections import Counter
        class_counts = Counter(c.parent.name for c in contracts)
        _loguru.info(
            f"--corpus mode: {len(contracts)} contract(s) under {args.corpus} | "
            f"class breakdown: {dict(class_counts)}"
        )
    elif args.contracts:
        contracts = args.contracts
    else:
        contracts = [
            Path("test_contracts/vulnerable_reentrant.sol"),
            Path("test_contracts/safe_storage.sol"),
        ]
    missing = [c for c in contracts if not c.exists()]
    if missing:
        _loguru.error(f"Missing contract files: {missing}")
        sys.exit(1)

    _section("RUN PLAN")
    _kv({
        "contract_count": len(contracts),
        "contracts":      [str(c) for c in contracts],
    }, indent=2)

    # ── Run audits ────────────────────────────────────────────────────────
    _section("EXECUTING AUDITS")
    results = []
    for c in contracts:
        r = await run_audit(c, urls, args)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────
    _section("E2E TEST SUMMARY")
    total_s = 0.0
    ok_count = 0
    for r in results:
        if r.get("success"):
            ok_count += 1
            total_s += r["total_wall_s"]
            _loguru.info(f"  ✓ {r['contract']:35} wall={r['total_wall_s']:5.1f}s  label={r.get('overall_label', 'N/A')}")
        else:
            _loguru.info(f"  ✗ {r['contract']:35} ERROR={r.get('error', '?')[:80]}")
    _loguru.info(f"  pass={ok_count}/{len(results)}  total_wall={total_s:.1f}s")
    _loguru.info(f"  log: {log_file}")
    sys.exit(0 if ok_count == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
