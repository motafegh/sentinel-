"""Tests for P7 RAG query construction fix."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import src.orchestration.nodes._helpers as _helpers_mod
from src.orchestration.nodes.rag_research import rag_research, _VULN_CLASS_TO_RAG_KEYWORDS


class TestRagQueryConstruction:
    """Test that RAG queries are constructed correctly for each vulnerability class."""

    @pytest.mark.asyncio
    async def test_skip_when_no_flagged_classes(self):
        """RAG should skip when no classes are flagged (topic=unknown)."""
        state = {
            "ml_result": {"label": "safe", "confirmed": [], "suspicious": []},
            "contract_code": "pragma solidity ^0.8.0;",
        }

        result = await rag_research(state)
        assert result["rag_results"] == []

    @pytest.mark.asyncio
    async def test_skip_when_topic_unknown(self):
        """RAG should skip when topic is 'unknown'."""
        state = {
            "ml_result": {
                "label": "unknown",
                "confirmed": [],
                "suspicious": [],
            },
            "contract_code": "pragma solidity ^0.8.0;",
        }

        result = await rag_research(state)
        assert result["rag_results"] == []

    @pytest.mark.asyncio
    async def test_reentrancy_query_uses_keywords(self):
        """Reentrancy query should use RAG-friendly keywords, not raw class name."""
        mock_fn = AsyncMock(return_value={"results": [{"id": "1", "content": "test"}]})

        state = {
            "ml_result": {
                "label": "confirmed_vulnerable",
                "confirmed": [{"vulnerability_class": "Reentrancy", "probability": 0.85}],
                "suspicious": [],
            },
            "contract_code": "pragma solidity ^0.8.0; contract Test {}",
        }

        with patch.object(_helpers_mod, "_call_mcp_tool", mock_fn):
            await rag_research(state)

        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        query = call_args.kwargs["arguments"]["query"]
        assert "reentrancy" in query.lower()
        assert "reentrant" in query.lower()

    @pytest.mark.asyncio
    async def test_integer_uo_query_uses_keywords(self):
        """IntegerUO query should use 'integer overflow underflow arithmetic'."""
        mock_fn = AsyncMock(return_value={"results": [{"id": "1", "content": "test"}]})

        state = {
            "ml_result": {
                "label": "confirmed_vulnerable",
                "confirmed": [{"vulnerability_class": "IntegerUO", "probability": 0.65}],
                "suspicious": [],
            },
            "contract_code": "pragma solidity ^0.8.0;",
        }

        with patch.object(_helpers_mod, "_call_mcp_tool", mock_fn):
            await rag_research(state)

        query = mock_fn.call_args.kwargs["arguments"]["query"]
        assert "integer overflow" in query.lower()
        assert "underflow" in query.lower()

    @pytest.mark.asyncio
    async def test_no_solidity_code_in_query(self):
        """Query should NOT contain Solidity code (confuses text embedder)."""
        mock_fn = AsyncMock(return_value={"results": [{"id": "1", "content": "test"}]})

        state = {
            "ml_result": {
                "label": "confirmed_vulnerable",
                "confirmed": [{"vulnerability_class": "Reentrancy", "probability": 0.85}],
                "suspicious": [],
            },
            "contract_code": "pragma solidity ^0.8.0; contract Vault { mapping(address => uint) balances; }",
        }

        with patch.object(_helpers_mod, "_call_mcp_tool", mock_fn):
            await rag_research(state)

        query = mock_fn.call_args.kwargs["arguments"]["query"]
        assert "pragma solidity" not in query
        assert "mapping(address" not in query
        assert "balances" not in query

    @pytest.mark.asyncio
    async def test_fallback_query_on_zero_results(self):
        """If first query returns 0 results, should try a fallback query."""
        mock_fn = AsyncMock(return_value={"results": []})

        state = {
            "ml_result": {
                "label": "confirmed_vulnerable",
                "confirmed": [{"vulnerability_class": "Reentrancy", "probability": 0.85}],
                "suspicious": [],
            },
            "contract_code": "pragma solidity ^0.8.0;",
        }

        with patch.object(_helpers_mod, "_call_mcp_tool", mock_fn):
            await rag_research(state)

        assert mock_fn.call_count == 2

        fallback_query = mock_fn.call_args_list[1].kwargs["arguments"]["query"]
        assert "reentrancy" in fallback_query.lower()

    @pytest.mark.asyncio
    async def test_external_bug_query_uses_call_summary(self):
        """ExternalBug query should include external call summary."""
        mock_fn = AsyncMock(return_value={"results": [{"id": "1", "content": "test"}]})

        state = {
            "ml_result": {
                "label": "confirmed_vulnerable",
                "confirmed": [{"vulnerability_class": "ExternalBug", "probability": 0.7}],
                "suspicious": [],
            },
            "contract_code": "pragma solidity ^0.8.0;",
            "external_call_summary": [
                {
                    "caller_function": "getPrice",
                    "caller_contract": "Oracle",
                    "callee_contract": "Chainlink",
                    "callee_function": "latestRoundData",
                    "callee_is_interface": True,
                },
            ],
        }

        with patch.object(_helpers_mod, "_call_mcp_tool", mock_fn):
            await rag_research(state)

        query = mock_fn.call_args.kwargs["arguments"]["query"]
        assert "oracle" in query.lower()
        assert "getPrice" in query or "Chainlink" in query


class TestVulnClassMapping:
    """Test that all ML vulnerability classes have RAG keyword mappings."""

    def test_all_classes_mapped(self):
        """Every ML vulnerability class should have a RAG keyword mapping."""
        ml_classes = [
            "Reentrancy", "IntegerUO", "GasException", "Timestamp",
            "TransactionOrderDependence", "ExternalBug", "CallToUnknown",
            "MishandledException", "UnusedReturn", "DenialOfService",
        ]

        for cls in ml_classes:
            assert cls in _VULN_CLASS_TO_RAG_KEYWORDS, f"Missing RAG keyword mapping for {cls}"
            assert len(_VULN_CLASS_TO_RAG_KEYWORDS[cls]) > 0, f"Empty mapping for {cls}"

    def test_mappings_are_descriptive(self):
        """Each mapping should contain meaningful keywords, not just the class name."""
        for cls, keywords in _VULN_CLASS_TO_RAG_KEYWORDS.items():
            assert len(keywords.split()) >= 2, f"Mapping for {cls} too short: '{keywords}'"
