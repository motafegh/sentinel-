"""Inference transport implementation behind the public MCP server shim."""

from src.mcp.servers.inference.runtime import call_inference_api, mock_prediction

__all__ = ["call_inference_api", "mock_prediction"]
