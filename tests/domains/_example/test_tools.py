"""Tests for _example domain tools."""

from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains._example.tools import register_tools


class TestRegisterTools:
    """Test that register_tools registers the expected tools."""

    def test_registers_list_and_get(self) -> None:
        """register_tools should add list_example_items and get_example_item."""
        mcp = FastMCP("test")
        server = MagicMock()
        register_tools(mcp, server)

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "list_example_items" in tool_names
        assert "get_example_item" in tool_names


class TestListExampleItems:
    """Test list_example_items tool."""

    async def test_returns_paginated_response(self) -> None:
        """Tool returns paginated response with items."""
        mcp = FastMCP("test")
        server = MagicMock()
        server.config.max_list_limit = 100
        server.config.default_list_limit = None
        server.k8s = MagicMock()

        # Mock ConfigMap list
        cm = MagicMock()
        cm.metadata.name = "item-1"
        cm.metadata.namespace = "ns"
        cm.metadata.uid = "uid-1"
        cm.metadata.creation_timestamp = None
        cm.metadata.labels = {"rhoai.io/example": "true"}
        cm.metadata.annotations = {"openshift.io/display-name": "Item One"}
        cm.data = {"key": "val"}
        server.k8s.core_v1.list_namespaced_config_map.return_value.items = [cm]

        register_tools(mcp, server)

        # Call the tool via the MCP tool manager
        result = await mcp._tool_manager.call_tool("list_example_items", {"namespace": "ns"})
        # call_tool returns the dict directly in this FastMCP version
        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "item-1"


class TestGetExampleItem:
    """Test get_example_item tool."""

    async def test_returns_single_item(self) -> None:
        """Tool returns single example item details."""
        mcp = FastMCP("test")
        server = MagicMock()
        server.k8s = MagicMock()

        cm = MagicMock()
        cm.metadata.name = "item-1"
        cm.metadata.namespace = "ns"
        cm.metadata.uid = "uid-1"
        cm.metadata.creation_timestamp = None
        cm.metadata.labels = {"rhoai.io/example": "true"}
        cm.metadata.annotations = {"openshift.io/display-name": "Item One"}
        cm.data = {"key": "val"}
        server.k8s.core_v1.read_namespaced_config_map.return_value = cm

        register_tools(mcp, server)

        result = await mcp._tool_manager.call_tool(
            "get_example_item", {"name": "item-1", "namespace": "ns"}
        )
        assert result["name"] == "item-1"
        assert result["namespace"] == "ns"
        assert result["display_name"] == "Item One"
        assert result["data"] == {"key": "val"}
        assert "_source" in result
