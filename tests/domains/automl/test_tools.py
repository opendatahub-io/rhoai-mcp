"""Tests for the AutoML MCP tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rhoai_mcp.domains.automl.client import AutoMLClient
from rhoai_mcp.domains.automl.tools import register_tools
from rhoai_mcp.utils.errors import ValidationError


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
    server.config.read_only_mode = False
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


async def test_predict_validation_error_is_visible_to_mcp_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp = MCPServer("test")
    server = MagicMock()
    server.config.read_only_mode = False
    server.get_k8s = MagicMock(return_value=server.k8s)
    monkeypatch.setattr(
        AutoMLClient,
        "predict_deployment",
        AsyncMock(side_effect=ValidationError("records[0] is missing required feature 'age'")),
    )
    register_tools(mcp, server)

    with pytest.raises(ToolError, match=r"records\[0\] is missing required feature 'age'"):
        await mcp.call_tool(
            "predict_automl_deployment",
            {"namespace": "demo", "name": "churn", "records": [{}]},
        )


async def test_predict_is_blocked_in_read_only_mode() -> None:
    mcp = MCPServer("test")
    server = MagicMock()
    server.config.read_only_mode = True
    register_tools(mcp, server)

    with pytest.raises(ToolError, match="Operation 'prediction' is not allowed"):
        await mcp.call_tool(
            "predict_automl_deployment",
            {"namespace": "demo", "name": "churn", "records": [{}]},
        )
