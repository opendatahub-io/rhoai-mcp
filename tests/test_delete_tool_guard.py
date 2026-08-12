"""Tests for delete tool managed-by guard integration (inference domain).

Verifies that delete tools:
- Return error when read_only_mode=True (before confirm check)
- Return "set confirm=True" when confirm=False
- Succeed when confirm=True and K8sClient allows the delete
- Return error when confirm=True and K8sClient raises NotManagedByMCPError
"""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.inference.tools import register_tools
from rhoai_mcp.utils.errors import NotManagedByMCPError


def _capture_tools(mock_server: MagicMock) -> dict:  # type: ignore[type-arg]
    """Register tools with a mock MCP and return captured tool functions."""
    mock_mcp = MagicMock()
    tools: dict = {}

    def capture_tool():
        def decorator(f):
            tools[f.__name__] = f
            return f
        return decorator

    mock_mcp.tool = capture_tool
    register_tools(mock_mcp, mock_server)
    return tools


class TestDeleteInferenceServiceTool:
    """Test delete_inference_service tool guard behavior."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        server = MagicMock()
        server.config.read_only_mode = False
        server.config.is_operation_allowed.return_value = (True, None)
        server.config.max_list_limit = 100
        server.config.default_list_limit = None
        return server

    @pytest.fixture
    def setup(self, mock_server: MagicMock):
        tools = _capture_tools(mock_server)
        return tools["delete_inference_service"], mock_server

    def test_read_only_returns_error(self, setup) -> None:
        delete_fn, server = setup
        server.config.read_only_mode = True
        result = delete_fn(name="my-model", namespace="ns", confirm=True)
        assert "error" in result
        assert "Read-only" in result["error"]

    def test_confirm_false_returns_error(self, setup) -> None:
        delete_fn, _server = setup
        result = delete_fn(name="my-model", namespace="ns")
        assert "error" in result
        assert "confirm" in result["message"].lower()

    def test_confirm_true_succeeds(self, setup) -> None:
        delete_fn, _server = setup
        result = delete_fn(name="my-model", namespace="ns", confirm=True)
        assert result["deleted"] is True
        assert result["name"] == "my-model"

    def test_not_managed_returns_error(self, setup) -> None:
        delete_fn, server = setup
        server.k8s.delete.side_effect = NotManagedByMCPError(
            "InferenceService", "my-model", "ns"
        )
        result = delete_fn(name="my-model", namespace="ns", confirm=True)
        assert "error" in result
        assert "not created by this MCP server" in result["error"]
