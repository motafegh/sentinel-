"""Tests for routing isolation — regression guards (P4).

Asserts that routing.py and evidence_router.py:
  1. Do NOT import any LLM client
  2. Do NOT read contract_code from state
"""

from __future__ import annotations

import ast
from pathlib import Path


_AGENTS_ROOT = Path(__file__).resolve().parent.parent
_ROUTING_PATH = _AGENTS_ROOT / "src" / "orchestration" / "routing.py"
_EVIDENCE_ROUTER_PATH = _AGENTS_ROOT / "src" / "orchestration" / "nodes" / "evidence_router.py"

_LLM_IMPORT_PATTERNS = {
    "src.llm", "src.llm.client", "langchain", "ChatOpenAI", "OpenAI",
    "get_fast_llm", "get_strong_llm", "get_llm",
}


def _get_imports(filepath: Path) -> set[str]:
    tree = ast.parse(filepath.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                for alias in node.names:
                    imports.add(f"{node.module}.{alias.name}")
    return imports


def _get_source_text(filepath: Path) -> str:
    return filepath.read_text()


class TestRoutingIsolation:
    def test_routing_no_llm_import(self):
        imports = _get_imports(_ROUTING_PATH)
        llm_hits = imports & _LLM_IMPORT_PATTERNS
        assert not llm_hits, f"routing.py imports LLM-related modules: {llm_hits}"

    def test_evidence_router_no_llm_import(self):
        imports = _get_imports(_EVIDENCE_ROUTER_PATH)
        llm_hits = imports & _LLM_IMPORT_PATTERNS
        assert not llm_hits, f"evidence_router.py imports LLM-related modules: {llm_hits}"

    def test_routing_no_contract_code_access(self):
        src = _get_source_text(_ROUTING_PATH)
        assert "contract_code" not in src

    def test_evidence_router_no_contract_code_access(self):
        src = _get_source_text(_EVIDENCE_ROUTER_PATH)
        assert "contract_code" not in src
