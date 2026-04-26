# agents/src/orchestration/__init__.py
#
# M5 — LangGraph orchestration layer.
# Wires the three MCP servers (inference, rag, audit) into a
# single stateful audit graph.
#
# Usage:
#   from src.orchestration.graph import build_graph
#   graph = build_graph()
#   result = await graph.ainvoke({"contract_code": ..., "contract_address": ...})
