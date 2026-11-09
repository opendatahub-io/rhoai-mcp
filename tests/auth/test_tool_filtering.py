"""Tests for per-user tool list filtering."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types as mcp_types
from mcp.server.mcpserver import MCPServer
from mcp.types import Tool as MCPTool

from rhoai_mcp.auth.user_context import UserContext
from rhoai_mcp.config import RHOAIConfig, TransportMode
from rhoai_mcp.server import RHOAIServer


class TestToolFiltering:
    def test_get_allowed_tools_returns_all_when_oidc_disabled(self) -> None:
        config = RHOAIConfig(oidc_enabled=False, mock_cluster=True)
        server = RHOAIServer(config)
        server.create_mcp()
        # When OIDC disabled, all tools should be allowed
        assert server.get_allowed_tools() is None  # None means "all"

    def test_get_allowed_tools_checks_rbac_when_oidc_enabled(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_k8s = MagicMock()
        mock_k8s.is_connected = True
        server._k8s_client = mock_k8s

        # Set up a user context
        ctx = UserContext(username="alice", groups=["team-a"])
        token = UserContext.set_current(ctx)

        try:
            with patch("rhoai_mcp.auth.rbac.RBACChecker") as MockChecker:
                checker_instance = MagicMock()
                checker_instance.filter_tools.return_value = {"list_data_science_projects"}
                MockChecker.return_value = checker_instance

                with patch.object(server, "_plugin_manager") as mock_pm:
                    mock_pm.collect_tool_permissions.return_value = {
                        "list_data_science_projects": [
                            {
                                "apiGroup": "project.openshift.io",
                                "resource": "projects",
                                "verb": "list",
                            }
                        ],
                        "delete_data_science_project": [
                            {
                                "apiGroup": "",
                                "resource": "namespaces",
                                "verb": "delete",
                            }
                        ],
                    }
                    result = server.get_allowed_tools()
                    assert result is not None
                    allowed, governed = result
                    assert allowed == {"list_data_science_projects"}
                    assert governed == {
                        "list_data_science_projects",
                        "delete_data_science_project",
                    }
        finally:
            UserContext.reset_current(token)

    def test_get_allowed_tools_returns_empty_set_when_no_user_context(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        server._k8s_client = MagicMock()
        server._plugin_manager = MagicMock()
        # No user context set
        result = server.get_allowed_tools()
        assert result == (set(), set())

    def test_get_allowed_tools_raises_when_no_plugin_manager(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        server._k8s_client = MagicMock()

        ctx = UserContext(username="alice", groups=[])
        token = UserContext.set_current(ctx)
        try:
            with pytest.raises(RuntimeError, match="plugin_manager not initialized"):
                server.get_allowed_tools()
        finally:
            UserContext.reset_current(token)


def _make_tool(name: str) -> MCPTool:
    return MCPTool(name=name, description=f"Tool {name}", input_schema={"type": "object"})


def _make_list_tools_result(names: list[str]) -> mcp_types.ListToolsResult:
    """Build a ListToolsResult (v2 handlers return concrete result types)."""
    tools = [_make_tool(n) for n in names]
    return mcp_types.ListToolsResult(tools=tools)


class TestToolFilteringEnforcement:
    """Tests that request_handlers are patched for RBAC filtering."""

    def _create_server_with_filtering(self) -> tuple["RHOAIServer", Any]:
        """Create an RHOAIServer with tool filtering installed, return (server, mcp)."""
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            oidc_kube_auth_strategy="impersonation",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mcp = server.create_mcp()
        server._k8s_client = MagicMock()
        server._k8s_client.is_connected = True
        server._install_tool_filtering(mcp)
        return server, mcp

    async def test_list_tools_filtered_by_allowed_set(self) -> None:
        """list_tools handler only returns governed tools the user is allowed to use."""
        server, mcp = self._create_server_with_filtering()
        lowlevel = mcp._mcp_server

        async def mock_original(_req: mcp_types.ListToolsRequest) -> mcp_types.ListToolsResult:
            return _make_list_tools_result(["tool_a", "tool_b", "tool_c"])

        # Reinstall filtering over our mock
        lowlevel.request_handlers[mcp_types.ListToolsRequest] = mock_original
        server._install_tool_filtering(mcp)

        handler = lowlevel.request_handlers[mcp_types.ListToolsRequest]
        # tool_a allowed, tool_b and tool_c governed but denied
        governed = {"tool_a", "tool_b", "tool_c"}
        with patch.object(server, "get_allowed_tools", return_value=({"tool_a"}, governed)):
            result = await handler(mcp_types.ListToolsRequest(method="tools/list"))

        names = [t.name for t in result.tools]
        assert names == ["tool_a"]

    async def test_list_tools_passes_ungoverned_tools(self) -> None:
        """list_tools handler allows tools without permission mappings."""
        server, mcp = self._create_server_with_filtering()
        lowlevel = mcp._mcp_server

        async def mock_original(_req: mcp_types.ListToolsRequest) -> mcp_types.ListToolsResult:
            return _make_list_tools_result(["governed_ok", "governed_deny", "ungoverned"])

        lowlevel.request_handlers[mcp_types.ListToolsRequest] = mock_original
        server._install_tool_filtering(mcp)

        handler = lowlevel.request_handlers[mcp_types.ListToolsRequest]
        governed = {"governed_ok", "governed_deny"}
        with patch.object(server, "get_allowed_tools", return_value=({"governed_ok"}, governed)):
            result = await handler(mcp_types.ListToolsRequest(method="tools/list"))

        names = [t.name for t in result.tools]
        assert sorted(names) == ["governed_ok", "ungoverned"]

    async def test_list_tools_returns_all_when_no_filtering(self) -> None:
        """list_tools handler returns all tools when get_allowed_tools returns None."""
        server, mcp = self._create_server_with_filtering()
        lowlevel = mcp._mcp_server

        async def mock_original(_req: mcp_types.ListToolsRequest) -> mcp_types.ListToolsResult:
            return _make_list_tools_result(["tool_a", "tool_b"])

        lowlevel.request_handlers[mcp_types.ListToolsRequest] = mock_original
        server._install_tool_filtering(mcp)

        handler = lowlevel.request_handlers[mcp_types.ListToolsRequest]
        with patch.object(server, "get_allowed_tools", return_value=None):
            result = await handler(mcp_types.ListToolsRequest(method="tools/list"))

        names = [t.name for t in result.tools]
        assert "tool_a" in names
        assert "tool_b" in names

    async def test_list_tools_returns_empty_on_rbac_failure(self) -> None:
        """list_tools handler returns empty list when RBAC check raises."""
        server, mcp = self._create_server_with_filtering()
        lowlevel = mcp._mcp_server

        async def mock_original(_req: mcp_types.ListToolsRequest) -> mcp_types.ListToolsResult:
            return _make_list_tools_result(["tool_a"])

        lowlevel.request_handlers[mcp_types.ListToolsRequest] = mock_original
        server._install_tool_filtering(mcp)

        handler = lowlevel.request_handlers[mcp_types.ListToolsRequest]
        with patch.object(server, "get_allowed_tools", side_effect=RuntimeError("fail")):
            result = await handler(mcp_types.ListToolsRequest(method="tools/list"))

        assert result.tools == []

    async def test_call_tool_denied_returns_error_result(self) -> None:
        """call_tool handler returns CallToolResult(is_error=True) for denied tools."""
        server, mcp = self._create_server_with_filtering()
        lowlevel = mcp._mcp_server

        handler = lowlevel.request_handlers[mcp_types.CallToolRequest]
        req = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(name="tool_b", arguments={}),
        )

        governed = {"tool_a", "tool_b"}
        with patch.object(server, "get_allowed_tools", return_value=({"tool_a"}, governed)):
            result = await handler(req)

        assert isinstance(result, mcp_types.CallToolResult)
        assert result.is_error is True
        assert "not permitted" in result.content[0].text

    async def test_call_tool_allowed_passes_through(self) -> None:
        """call_tool handler passes through to original for allowed tools."""
        server, mcp = self._create_server_with_filtering()
        lowlevel = mcp._mcp_server

        # Replace original call handler with a mock
        mock_result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="ok")],
            is_error=False,
        )
        original_call_handler = AsyncMock(return_value=mock_result)
        lowlevel.request_handlers[mcp_types.CallToolRequest] = original_call_handler
        server._install_tool_filtering(mcp)

        handler = lowlevel.request_handlers[mcp_types.CallToolRequest]
        req = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(name="tool_a", arguments={"x": "1"}),
        )

        governed = {"tool_a", "tool_b"}
        with patch.object(server, "get_allowed_tools", return_value=({"tool_a"}, governed)):
            result = await handler(req)

        assert result == mock_result
        original_call_handler.assert_called_once_with(req)

    async def test_call_tool_ungoverned_passes_through(self) -> None:
        """call_tool handler allows tools without permission mappings."""
        server, mcp = self._create_server_with_filtering()
        lowlevel = mcp._mcp_server

        mock_result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="ok")],
            is_error=False,
        )
        original_call_handler = AsyncMock(return_value=mock_result)
        lowlevel.request_handlers[mcp_types.CallToolRequest] = original_call_handler
        server._install_tool_filtering(mcp)

        handler = lowlevel.request_handlers[mcp_types.CallToolRequest]
        req = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(name="ungoverned_tool", arguments={}),
        )

        governed = {"tool_a"}
        with patch.object(server, "get_allowed_tools", return_value=({"tool_a"}, governed)):
            result = await handler(req)

        assert result == mock_result
        original_call_handler.assert_called_once_with(req)


class TestToolFilteringInstallation:
    """Tripwire for the *installation* of RBAC tool filtering (RHAI-14595).

    Silent-bypass failure mode (RHOAIENG-95548, Risk B): if the MCP SDK
    changes internals so that ``_install_tool_filtering()`` no longer
    replaces the lowlevel request handlers — the ``request_handlers`` dict
    renamed, populated lazily, keyed by different request classes, or no
    longer read by dispatch — the patch silently no-ops. Nothing raises:
    every authenticated user simply sees and can call every tool. These
    tests convert that silent bypass into a loud CI failure: they must
    FAIL (never skip) when installation does not take effect.
    """

    def _make_server(self) -> RHOAIServer:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_token_mode="token-review",
            oidc_kube_auth_strategy="impersonation",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_k8s = MagicMock()
        mock_k8s.is_connected = True
        mock_k8s._api_client = MagicMock()
        server._k8s_client = mock_k8s
        return server

    @staticmethod
    def _bare_mcp() -> MCPServer:
        """A bare MCPServer whose lowlevel server holds the SDK's own handlers."""
        return MCPServer(name="rbac-install-tripwire")

    def test_setup_auth_replaces_sdk_handlers(self) -> None:
        """_setup_auth() must swap the SDK's own handlers for RBAC wrappers.

        If _install_tool_filtering() no-ops (early return, dispatch moved
        elsewhere), the dict still holds the SDK originals and this test
        fails — the silent RBAC bypass becomes visible in CI.
        """
        server = self._make_server()
        mcp = self._bare_mcp()
        lowlevel = mcp._mcp_server
        sdk_list_handler = lowlevel.request_handlers[mcp_types.ListToolsRequest]
        sdk_call_handler = lowlevel.request_handlers[mcp_types.CallToolRequest]

        server._setup_auth(mcp)

        assert lowlevel.request_handlers[mcp_types.ListToolsRequest] is not sdk_list_handler
        assert lowlevel.request_handlers[mcp_types.CallToolRequest] is not sdk_call_handler

    def test_install_raises_when_sdk_handlers_missing(self) -> None:
        """A missing SDK handler must raise, not warn-and-skip.

        Simulates SDK drift: if the lowlevel server no longer exposes the
        CallToolRequest handler, warn-and-return would leave every tool
        callable for every user. Installation must fail loudly instead.
        """
        server = self._make_server()
        mcp = self._bare_mcp()
        del mcp._mcp_server.request_handlers[mcp_types.CallToolRequest]

        with pytest.raises(RuntimeError, match="RBAC tool filtering"):
            server._install_tool_filtering(mcp)
