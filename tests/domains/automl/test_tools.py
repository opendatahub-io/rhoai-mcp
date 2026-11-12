"""Tests for the AutoML MCP tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.mcpserver import MCPServer

from rhoai_mcp.domains.automl.client import AutoMLClient
from rhoai_mcp.domains.automl.tools import register_tools


async def test_registers_discovery_and_prediction_tools() -> None:
    mcp = MCPServer("test")
    register_tools(mcp, MagicMock())

    names = {tool.name for tool in await mcp.list_tools()}
    assert {
        "list_automl_deployments",
        "get_automl_deployment",
        "predict_automl_deployment",
    } <= names


async def test_predict_deployment_uses_automl_client(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp = MCPServer("test")
    server = MagicMock()
    predict = AsyncMock(
        return_value={
            "task_type": "classification",
            "predictions": [{"class": "yes", "probabilities": {"no": 0.1, "yes": 0.9}}],
            "probabilities_available": True,
        }
    )
    server.get_k8s = MagicMock(return_value=server.k8s)
    monkeypatch.setattr(AutoMLClient, "predict_deployment", predict)
    register_tools(mcp, server)

    result = await mcp.call_tool(
        "predict_automl_deployment",
        {"namespace": "demo", "name": "churn", "records": [{"age": 42, "country": "PL"}]},
    )

    assert result.structured_content["predictions"][0]["class"] == "yes"
    predict.assert_awaited_once_with("churn", "demo", [{"age": 42, "country": "PL"}])
