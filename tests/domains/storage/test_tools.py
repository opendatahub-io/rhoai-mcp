"""Tests for storage MCP tools."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register storage tools and return captured tool functions."""
    from rhoai_mcp.domains.storage.tools import register_tools

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
    """Create a mock server with default config."""
    server = MagicMock()
    server.config.is_operation_allowed.return_value = (True, None)
    server.config.max_list_limit = 100
    server.config.default_list_limit = None
    return server


class TestToolRegistration:
    """Tests for tool registration."""

    def test_tools_registered(self, mock_server: MagicMock) -> None:
        """All three storage tools are registered."""
        tools = _register_tools(mock_server)

        assert set(tools.keys()) == {
            "list_storage",
            "create_storage",
            "delete_storage",
        }


class TestListStorageTool:
    """Tests for the list_storage tool."""

    @patch("rhoai_mcp.domains.storage.tools.StorageClient")
    def test_list_storage_calls_client(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """list_storage invokes StorageClient.list_storage."""
        mock_client = MagicMock()
        mock_client.list_storage.return_value = [
            {
                "name": "pvc-1",
                "display_name": "PVC One",
                "size": "10Gi",
                "access_modes": ["ReadWriteOnce"],
                "storage_class": "gp3",
                "status": "Bound",
                "_source": {
                    "kind": "PersistentVolumeClaim",
                    "api_version": "v1",
                    "name": "pvc-1",
                    "namespace": "test-project",
                    "uid": "uid-1",
                },
            }
        ]
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["list_storage"](namespace="test-project")

        mock_client_cls.assert_called_once_with(mock_server.k8s)
        mock_client.list_storage.assert_called_once_with("test-project")
        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "pvc-1"


class TestCreateStorageTool:
    """Tests for the create_storage tool."""

    @patch("rhoai_mcp.domains.storage.tools.StorageClient")
    def test_create_storage_read_only_blocked(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """create_storage returns error and does not instantiate client."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Read-only mode enabled",
        )

        tools = _register_tools(mock_server)
        result = tools["create_storage"](name="new-pvc", namespace="test-project")

        assert result == {"error": "Read-only mode enabled"}
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.storage.tools.StorageClient")
    def test_create_storage_success(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """create_storage creates PVC and returns success response."""
        mock_storage = MagicMock()
        mock_storage.metadata.name = "new-pvc"
        mock_storage.metadata.namespace = "test-project"
        mock_storage.size = "10Gi"
        mock_storage.status.value = "Bound"
        mock_storage.metadata.to_source_dict.return_value = {
            "kind": "PersistentVolumeClaim",
            "api_version": "v1",
            "name": "new-pvc",
            "namespace": "test-project",
            "uid": "new-uid",
        }
        mock_client = MagicMock()
        mock_client.create_storage.return_value = mock_storage
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["create_storage"](
            name="new-pvc",
            namespace="test-project",
            size="10Gi",
            display_name="New PVC",
        )

        assert result["name"] == "new-pvc"
        assert result["namespace"] == "test-project"
        assert result["size"] == "10Gi"
        assert result["status"] == "Bound"
        assert "created successfully" in result["message"]
        assert "_source" in result

        # Verify StorageCreate was passed with correct fields
        call_args = mock_client.create_storage.call_args[0][0]
        assert call_args.name == "new-pvc"
        assert call_args.namespace == "test-project"
        assert call_args.size == "10Gi"
        assert call_args.display_name == "New PVC"


class TestDeleteStorageTool:
    """Tests for the delete_storage tool."""

    @patch("rhoai_mcp.domains.storage.tools.StorageClient")
    def test_delete_storage_read_only_blocked(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """delete_storage returns error and does not instantiate client."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Read-only mode enabled",
        )

        tools = _register_tools(mock_server)
        result = tools["delete_storage"](name="my-pvc", namespace="test-project", confirm=True)

        assert result == {"error": "Read-only mode enabled"}
        mock_client_cls.assert_not_called()

    def test_delete_storage_dangerous_ops_disabled(self, mock_server: MagicMock) -> None:
        """delete_storage returns error when dangerous ops are disabled."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Dangerous operations are disabled",
        )

        tools = _register_tools(mock_server)
        result = tools["delete_storage"](name="my-pvc", namespace="test-project", confirm=True)

        assert result == {"error": "Dangerous operations are disabled"}

    def test_delete_storage_no_confirm(self, mock_server: MagicMock) -> None:
        """delete_storage requires confirm=True."""
        tools = _register_tools(mock_server)
        result = tools["delete_storage"](name="my-pvc", namespace="test-project", confirm=False)

        assert "error" in result
        assert result["error"] == "Deletion not confirmed"
        assert "confirm=True" in result["message"]

    @patch("rhoai_mcp.domains.storage.tools.StorageClient")
    def test_delete_storage_success(
        self,
        mock_client_cls: MagicMock,
        mock_server: MagicMock,
    ) -> None:
        """delete_storage deletes PVC and returns confirmation."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["delete_storage"](name="my-pvc", namespace="test-project", confirm=True)

        mock_client.delete_storage.assert_called_once_with("my-pvc", "test-project")
        assert result["name"] == "my-pvc"
        assert result["namespace"] == "test-project"
        assert result["deleted"] is True
        assert "deleted" in result["message"]
        assert result["_source"]["kind"] == "PersistentVolumeClaim"
        assert result["_source"]["uid"] is None
