"""Tests for {{DOMAIN_NAME}} domain tools."""

from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.{{DOMAIN_NAME}}.tools import register_tools


class TestRegisterTools:
    """Test that register_tools registers the expected tools."""

    def test_registers_expected_tools(self) -> None:
        """register_tools should add list and get tools."""
        mcp = FastMCP("test")
        server = MagicMock()
        register_tools(mcp, server)

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "list_{{RESOURCE_NAME}}s" in tool_names
        assert "get_{{RESOURCE_NAME}}" in tool_names
