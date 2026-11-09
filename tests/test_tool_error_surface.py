"""Tool error surfaces under mcp 2.x error scrubbing (RHAI-14623).

mcp 2.x scrubs unhandled tool exceptions down to "Error executing tool
<name>" — only ToolError messages cross the wire. Our user-facing
exception hierarchies (RHOAIError and friends) therefore inherit from
the SDK's ToolError so their guidance text survives. These tests pin
both sides of that contract by driving the SDK's real tool-call path.
"""

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from rhoai_mcp.utils.errors import NotFoundError


def _make_mcp() -> MCPServer:
    mcp = MCPServer(name="tool-error-surface")

    @mcp.tool()
    def raise_not_found() -> dict[str, str]:
        """Simulate a user-facing domain error."""
        raise NotFoundError("Data Science Project", "demo")

    @mcp.tool()
    def raise_internal() -> dict[str, str]:
        """Simulate an unexpected internal failure (must be scrubbed)."""
        raise ValueError("secret internal detail: token abc123")

    return mcp


class TestToolErrorSurface:
    async def test_user_facing_error_message_survives(self) -> None:
        """A RHOAIError raised in a tool crosses the SDK wrapper with its text.

        The SDK re-raises it as ToolError("Error executing tool <name>:
        <message>"), which _handle_call_tool converts to an is_error
        CallToolResult whose content the model reads.
        """
        mcp = _make_mcp()
        with pytest.raises(ToolError) as exc_info:
            await mcp.call_tool("raise_not_found", {})
        message = str(exc_info.value)
        assert "Error executing tool raise_not_found" in message
        assert "Data Science Project 'demo' not found" in message

    async def test_internal_error_message_is_scrubbed(self) -> None:
        """A plain exception never leaks its text to the client.

        This is the failure mode RHAI-14623 guards against: without the
        ToolError inheritance, every user-facing message would be scrubbed
        exactly like this internal one.
        """
        mcp = _make_mcp()
        with pytest.raises(UnexpectedToolError) as exc_info:
            await mcp.call_tool("raise_internal", {})
        message = str(exc_info.value)
        assert message == "Error executing tool raise_internal"
        assert "secret internal detail" not in message
