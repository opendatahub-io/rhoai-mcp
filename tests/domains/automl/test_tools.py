"""Tests for the AutoML MCP proof-of-concept tools."""

from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.automl.tools import register_tools
from tests.domains.automl.test_models import _resource


async def test_registers_discovery_and_payload_tools() -> None:
    mcp = FastMCP("test")
    register_tools(mcp, MagicMock())

    names = {tool.name for tool in mcp._tool_manager.list_tools()}
    assert {
        "list_automl_deployments",
        "get_automl_deployment",
        "prepare_automl_prediction",
    } <= names


async def test_prepare_prediction_returns_kserve_v2_payload() -> None:
    mcp = FastMCP("test")
    server = MagicMock()
    server.k8s.get.return_value = _resource()
    register_tools(mcp, server)

    result = await mcp._tool_manager.call_tool(
        "prepare_automl_prediction",
        {"namespace": "demo", "name": "churn", "records": [{"age": 42, "country": "PL"}]},
    )

    assert result["inputs"][0] == {
        "name": "age",
        "shape": [1],
        "datatype": "INT64",
        "data": [42],
    }
