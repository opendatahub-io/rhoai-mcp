"""Tests for data connection MCP tools."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.utils.errors import NotManagedByMCPError


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register connection tools and return captured tool functions."""
    from rhoai_mcp.domains.connections.tools import register_tools

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


@pytest.fixture
def mock_server() -> MagicMock:
    """Mock RHOAIServer with default config."""
    server = MagicMock()
    server.config.is_operation_allowed.return_value = (True, None)
    server.config.read_only_mode = False
    server.config.max_list_limit = 100
    server.config.default_list_limit = None
    return server


class TestToolRegistration:
    """Verify that all expected tools are registered."""

    def test_tools_registered(self, mock_server: MagicMock) -> None:
        """Four connection tools should be registered."""
        tools = _register_tools(mock_server)

        assert set(tools.keys()) == {
            "list_data_connections",
            "get_data_connection",
            "create_s3_data_connection",
            "delete_data_connection",
        }


class TestListDataConnectionsTool:
    """Tests for the list_data_connections tool."""

    @patch("rhoai_mcp.domains.connections.tools.ConnectionClient")
    def test_list_data_connections_calls_client(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Tool delegates to ConnectionClient and returns paginated response."""
        mock_client = mock_client_cls.return_value
        mock_client.list_data_connections.return_value = [
            {
                "name": "conn-1",
                "display_name": "Connection 1",
                "type": "s3",
                "endpoint": "https://s3.amazonaws.com",
                "bucket": "bucket-1",
                "region": "us-east-1",
                "_source": {
                    "kind": "Secret",
                    "api_version": "v1",
                    "name": "conn-1",
                    "namespace": "ns",
                    "uid": "uid-1",
                },
            },
        ]

        tools = _register_tools(mock_server)
        result = tools["list_data_connections"](namespace="ns")

        mock_client.list_data_connections.assert_called_once_with("ns")
        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "conn-1"


class TestGetDataConnectionTool:
    """Tests for the get_data_connection tool."""

    @patch("rhoai_mcp.domains.connections.tools.ConnectionClient")
    def test_get_data_connection_calls_client(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Tool delegates to ConnectionClient.get_data_connection."""
        mock_conn = MagicMock()
        mock_conn.metadata.name = "my-conn"
        mock_conn.metadata.namespace = "ns"
        mock_conn.display_name = "My Connection"
        mock_conn.connection_type = "s3"
        mock_conn.aws_access_key_id = "AKIA****MPLE"
        mock_conn.aws_s3_endpoint = "https://s3.amazonaws.com"
        mock_conn.aws_s3_bucket = "my-bucket"
        mock_conn.aws_default_region = "us-east-1"
        mock_conn.metadata.creation_timestamp = None
        mock_conn.metadata.to_source_dict.return_value = {
            "kind": "Secret",
            "api_version": "v1",
            "name": "my-conn",
            "namespace": "ns",
            "uid": "uid-1",
        }

        mock_client = mock_client_cls.return_value
        mock_client.get_data_connection.return_value = mock_conn

        tools = _register_tools(mock_server)
        result = tools["get_data_connection"](name="my-conn", namespace="ns")

        mock_client.get_data_connection.assert_called_once_with("my-conn", "ns", mask_secrets=True)
        assert result["name"] == "my-conn"
        assert result["type"] == "s3"
        assert result["aws_access_key_id"] == "AKIA****MPLE"
        assert result["created"] is None


class TestCreateS3DataConnectionTool:
    """Tests for the create_s3_data_connection tool."""

    @patch("rhoai_mcp.domains.connections.tools.ConnectionClient")
    def test_create_s3_data_connection_read_only_blocked(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Create is blocked when config disallows the operation."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Read-only mode is enabled",
        )

        tools = _register_tools(mock_server)
        result = tools["create_s3_data_connection"](
            name="new-conn",
            namespace="ns",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            aws_s3_endpoint="https://s3.amazonaws.com",
            aws_s3_bucket="bucket",
        )

        assert result["error"] == "Read-only mode is enabled"
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.connections.tools.ConnectionClient")
    def test_create_s3_data_connection_success(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Create succeeds when operation is allowed."""
        mock_conn = MagicMock()
        mock_conn.metadata.name = "new-conn"
        mock_conn.metadata.namespace = "ns"
        mock_conn.connection_type = "s3"
        mock_conn.aws_s3_bucket = "bucket"
        mock_conn.metadata.to_source_dict.return_value = {
            "kind": "Secret",
            "api_version": "v1",
            "name": "new-conn",
            "namespace": "ns",
            "uid": "new-uid",
        }

        mock_client = mock_client_cls.return_value
        mock_client.create_s3_data_connection.return_value = mock_conn

        tools = _register_tools(mock_server)
        result = tools["create_s3_data_connection"](
            name="new-conn",
            namespace="ns",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            aws_s3_endpoint="https://s3.amazonaws.com",
            aws_s3_bucket="bucket",
        )

        assert result["name"] == "new-conn"
        assert result["bucket"] == "bucket"
        assert "created successfully" in result["message"]
        mock_client.create_s3_data_connection.assert_called_once()


class TestDeleteDataConnectionTool:
    """Tests for the delete_data_connection tool."""

    @patch("rhoai_mcp.domains.connections.tools.ConnectionClient")
    def test_delete_data_connection_read_only_blocked(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Delete is blocked when read-only mode is enabled."""
        mock_server.config.read_only_mode = True

        tools = _register_tools(mock_server)
        result = tools["delete_data_connection"](name="conn", namespace="ns", confirm=True)

        assert result["error"] == "Read-only mode is enabled"
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.connections.tools.ConnectionClient")
    def test_delete_data_connection_not_managed_by_mcp(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Delete returns error when resource is not managed by MCP."""
        mock_client = mock_client_cls.return_value
        mock_client.delete_data_connection.side_effect = NotManagedByMCPError(
            "Secret", "conn", "ns"
        )

        tools = _register_tools(mock_server)
        result = tools["delete_data_connection"](name="conn", namespace="ns", confirm=True)

        assert "error" in result
        assert "conn" in result["error"]

    def test_delete_data_connection_no_confirm(
        self,
        mock_server: MagicMock,
    ) -> None:
        """Delete requires confirm=True to proceed."""
        tools = _register_tools(mock_server)
        result = tools["delete_data_connection"](name="conn", namespace="ns", confirm=False)

        assert result["error"] == "Deletion not confirmed"
        assert "set confirm=True" in result["message"]

    @patch("rhoai_mcp.domains.connections.tools.ConnectionClient")
    def test_delete_data_connection_success(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """Delete succeeds with confirm=True and allowed operation."""
        mock_client = mock_client_cls.return_value

        tools = _register_tools(mock_server)
        result = tools["delete_data_connection"](name="conn", namespace="ns", confirm=True)

        mock_client.delete_data_connection.assert_called_once_with("conn", "ns")
        assert result["deleted"] is True
        assert result["name"] == "conn"
        assert result["namespace"] == "ns"
        assert "deleted" in result["message"]
