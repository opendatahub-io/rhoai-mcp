"""Tests for pipeline MCP tools."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register pipeline tools and return captured tool functions."""
    from rhoai_mcp.domains.pipelines.tools import register_tools

    mcp = MagicMock()
    registered_tools: dict[str, Any] = {}

    def capture_tool() -> Any:
        def decorator(func: Any) -> Any:
            registered_tools[func.__name__] = func
            return func

        return decorator

    mcp.tool = capture_tool
    register_tools(mcp, mock_server)
    return registered_tools


@pytest.fixture()
def mock_server() -> MagicMock:
    """Create a mock server with default allowed operations."""
    server = MagicMock()
    server.config.is_operation_allowed.return_value = (True, None)
    return server


class TestToolRegistration:
    """Test that pipeline tools are properly registered."""

    def test_tools_registered(self, mock_server: MagicMock) -> None:
        """All 3 pipeline tools are registered."""
        tools = _register_tools(mock_server)

        assert set(tools.keys()) == {
            "get_pipeline_server",
            "create_pipeline_server",
            "delete_pipeline_server",
        }


class TestGetPipelineServerTool:
    """Test get_pipeline_server tool."""

    @patch("rhoai_mcp.domains.pipelines.tools.PipelineClient")
    def test_get_pipeline_server_exists(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Returns exists=True when a DSPA is found."""
        mock_client = mock_client_cls.return_value
        mock_client.get_pipeline_server.return_value = {
            "name": "dspa",
            "status": "Ready",
            "api_server_ready": True,
            "persistence_agent_ready": True,
            "scheduled_workflow_ready": True,
            "database_available": True,
            "object_store_available": True,
        }

        tools = _register_tools(mock_server)
        result = tools["get_pipeline_server"](namespace="my-project")

        assert result["exists"] is True
        assert result["name"] == "dspa"
        assert result["status"] == "Ready"
        mock_client.get_pipeline_server.assert_called_once_with("my-project")

    @patch("rhoai_mcp.domains.pipelines.tools.PipelineClient")
    def test_get_pipeline_server_not_exists(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Returns exists=False when no DSPA is found."""
        mock_client = mock_client_cls.return_value
        mock_client.get_pipeline_server.return_value = None

        tools = _register_tools(mock_server)
        result = tools["get_pipeline_server"](namespace="empty-ns")

        assert result["exists"] is False
        assert "message" in result


class TestCreatePipelineServerTool:
    """Test create_pipeline_server tool."""

    @patch("rhoai_mcp.domains.pipelines.tools.PipelineClient")
    def test_create_pipeline_server_read_only_blocked(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Create is blocked when read-only mode is active."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Operation 'create' is not allowed in read-only mode",
        )

        tools = _register_tools(mock_server)
        result = tools["create_pipeline_server"](
            namespace="my-project",
            object_storage_secret="aws-secret",
            object_storage_bucket="my-bucket",
            object_storage_endpoint="https://s3.amazonaws.com",
        )

        assert "error" in result
        assert "read-only" in result["error"]
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.pipelines.tools.PipelineClient")
    def test_create_pipeline_server_success(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Create succeeds and returns server info."""
        mock_dspa = MagicMock()
        mock_dspa.metadata.name = "dspa"
        mock_dspa.metadata.namespace = "my-project"
        mock_dspa.status.value = "Creating"
        mock_client = mock_client_cls.return_value
        mock_client.create_pipeline_server.return_value = mock_dspa

        tools = _register_tools(mock_server)
        result = tools["create_pipeline_server"](
            namespace="my-project",
            object_storage_secret="aws-secret",
            object_storage_bucket="pipeline-artifacts",
            object_storage_endpoint="https://s3.amazonaws.com",
            object_storage_region="us-west-2",
        )

        assert result["name"] == "dspa"
        assert result["namespace"] == "my-project"
        assert result["status"] == "Creating"
        assert "message" in result
        mock_client.create_pipeline_server.assert_called_once()


class TestDeletePipelineServerTool:
    """Test delete_pipeline_server tool."""

    @patch("rhoai_mcp.domains.pipelines.tools.PipelineClient")
    def test_delete_pipeline_server_read_only_blocked(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Delete is blocked when read-only mode is active."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Operation 'delete' is not allowed in read-only mode",
        )

        tools = _register_tools(mock_server)
        result = tools["delete_pipeline_server"](namespace="my-project", confirm=True)

        assert "error" in result
        assert "read-only" in result["error"]
        mock_client_cls.assert_not_called()

    def test_delete_pipeline_server_dangerous_ops_disabled(
        self,
        mock_server: MagicMock,
    ) -> None:
        """Delete is blocked when dangerous operations are disabled."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Dangerous operations are disabled",
        )

        tools = _register_tools(mock_server)
        result = tools["delete_pipeline_server"](namespace="my-project", confirm=True)

        assert "error" in result
        assert "Dangerous" in result["error"]

    def test_delete_pipeline_server_no_confirm(
        self,
        mock_server: MagicMock,
    ) -> None:
        """Delete without confirm=True is rejected."""
        tools = _register_tools(mock_server)
        result = tools["delete_pipeline_server"](namespace="my-project", confirm=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()

    @patch("rhoai_mcp.domains.pipelines.tools.PipelineClient")
    def test_delete_pipeline_server_success(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Delete succeeds when DSPA exists and confirm=True."""
        mock_client = mock_client_cls.return_value
        mock_client.get_pipeline_server.return_value = {
            "name": "dspa",
            "status": "Ready",
            "api_server_ready": True,
            "persistence_agent_ready": True,
            "scheduled_workflow_ready": True,
            "database_available": True,
            "object_store_available": True,
        }

        tools = _register_tools(mock_server)
        result = tools["delete_pipeline_server"](namespace="my-project", confirm=True)

        assert result["deleted"] is True
        assert result["namespace"] == "my-project"
        mock_client.delete_pipeline_server.assert_called_once_with("dspa", "my-project")

    @patch("rhoai_mcp.domains.pipelines.tools.PipelineClient")
    def test_delete_pipeline_server_not_exists(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Delete returns error when no DSPA exists."""
        mock_client = mock_client_cls.return_value
        mock_client.get_pipeline_server.return_value = None

        tools = _register_tools(mock_server)
        result = tools["delete_pipeline_server"](namespace="empty-ns", confirm=True)

        assert "error" in result
        assert "empty-ns" in result["error"]
        mock_client.delete_pipeline_server.assert_not_called()
