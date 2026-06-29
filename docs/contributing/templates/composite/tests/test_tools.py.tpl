"""Tests for {{DOMAIN_NAME}} composite tools."""

from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.composites.{{DOMAIN_NAME}}.tools import register_tools


class TestRegisterTools:
    """Test that register_tools registers the expected tools."""

    def test_registers_expected_tools(self) -> None:
        """register_tools should add the summary tool."""
        mcp = FastMCP("test")
        server = MagicMock()
        register_tools(mcp, server)

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "{{DOMAIN_NAME}}_summary" in tool_names
