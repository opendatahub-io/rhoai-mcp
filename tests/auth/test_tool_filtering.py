"""Tests for per-user RBAC enforcement (RHAI-14627).

Covers the RBACMiddleware (the ratified 1.6 design: one ServerMiddleware
for both enforcement points, installed via the public
MCPServer(middleware=[...]) constructor parameter), the get_allowed_tools
logic it delegates to, and both enforcement points driven end-to-end
through the real dispatch path.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer

from rhoai_mcp.auth.rbac_middleware import RBACMiddleware
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


# --- RBACMiddleware unit tests -------------------------------------------------
#
# The middleware operates on the JSON-RPC wire surface: ctx.params is the raw
# params dict and call_next returns the wire-shaped result dict.


def _ctx(method: str, params: Any = None) -> SimpleNamespace:
    return SimpleNamespace(method=method, params=params)


def _list_result(names: list[str]) -> dict[str, Any]:
    return {"tools": [{"name": n, "description": f"Tool {n}"} for n in names]}


def _middleware(
    allowed: set[str] | None, governed: set[str] | None = None
) -> tuple[RBACMiddleware, MagicMock]:
    server = MagicMock()
    if allowed is None:
        server.get_allowed_tools.return_value = None  # OIDC disabled
    else:
        server.get_allowed_tools.return_value = (allowed, governed or set())
    return RBACMiddleware(server), server


class TestRBACMiddlewareCall:
    async def test_denied_call_does_not_dispatch(self) -> None:
        """A governed, disallowed tool is denied before the handler runs."""
        mw, _ = _middleware(allowed={"tool_a"}, governed={"tool_a", "tool_b"})
        call_next = AsyncMock(return_value={"content": [], "isError": False})

        result = await mw(_ctx("tools/call", {"name": "tool_b", "arguments": {}}), call_next)

        call_next.assert_not_awaited()
        assert result["isError"] is True
        assert "not permitted for the current user" in result["content"][0]["text"]
        assert "tool_b" in result["content"][0]["text"]

    async def test_allowed_call_passes_through(self) -> None:
        mw, _ = _middleware(allowed={"tool_a"}, governed={"tool_a", "tool_b"})
        call_next = AsyncMock(
            return_value={"content": [{"type": "text", "text": "ok"}], "isError": False}
        )

        result = await mw(_ctx("tools/call", {"name": "tool_a", "arguments": {}}), call_next)

        call_next.assert_awaited_once()
        assert result["isError"] is False

    async def test_ungoverned_call_passes_through(self) -> None:
        """Tools without permission mappings are allowed."""
        mw, _ = _middleware(allowed=set(), governed={"governed_tool"})
        call_next = AsyncMock(return_value={"content": [], "isError": False})

        result = await mw(
            _ctx("tools/call", {"name": "ungoverned_tool", "arguments": {}}), call_next
        )

        call_next.assert_awaited_once()
        assert result["isError"] is False

    async def test_denied_on_rbac_failure(self) -> None:
        """Fail-closed: an RBAC check failure denies the call."""
        server = MagicMock()
        server.get_allowed_tools.side_effect = RuntimeError("SAR unavailable")
        mw = RBACMiddleware(server)
        call_next = AsyncMock()

        result = await mw(_ctx("tools/call", {"name": "tool_a", "arguments": {}}), call_next)

        call_next.assert_not_awaited()
        assert result["isError"] is True
        assert "RBAC check failed" in result["content"][0]["text"]

    async def test_no_filtering_when_oidc_disabled(self) -> None:
        """get_allowed_tools() None means no enforcement."""
        mw, _ = _middleware(allowed=None)
        call_next = AsyncMock(return_value={"content": [], "isError": False})

        result = await mw(_ctx("tools/call", {"name": "anything", "arguments": {}}), call_next)

        call_next.assert_awaited_once()
        assert result["isError"] is False

    async def test_malformed_params_pass_through(self) -> None:
        """Malformed requests fall through to the SDK's own validation."""
        mw, _ = _middleware(allowed={"tool_a"}, governed={"tool_a", "tool_b"})
        call_next = AsyncMock(return_value={"content": [], "isError": False})

        await mw(_ctx("tools/call", {"arguments": {}}), call_next)
        assert call_next.await_count == 1

        result = await mw(_ctx("tools/call", "not-a-dict"), call_next)
        assert call_next.await_count == 2
        assert result["isError"] is False


class TestRBACMiddlewareList:
    async def test_list_filters_governed_denied(self) -> None:
        mw, _ = _middleware(allowed={"tool_a"}, governed={"tool_a", "tool_b"})
        call_next = AsyncMock(return_value=_list_result(["tool_a", "tool_b", "tool_c"]))

        result = await mw(_ctx("tools/list", {}), call_next)

        assert [t["name"] for t in result["tools"]] == ["tool_a", "tool_c"]
        # A fresh result dict — the original is not mutated
        assert call_next.return_value["tools"][1]["name"] == "tool_b"

    async def test_list_passes_ungoverned(self) -> None:
        mw, _ = _middleware(allowed={"governed_ok"}, governed={"governed_ok", "governed_deny"})
        call_next = AsyncMock(
            return_value=_list_result(["governed_ok", "governed_deny", "ungoverned"])
        )

        result = await mw(_ctx("tools/list", {}), call_next)

        assert [t["name"] for t in result["tools"]] == ["governed_ok", "ungoverned"]

    async def test_list_passthrough_when_no_filtering(self) -> None:
        mw, _ = _middleware(allowed=None)
        original = _list_result(["tool_a", "tool_b"])
        call_next = AsyncMock(return_value=original)

        result = await mw(_ctx("tools/list", {}), call_next)

        assert result is original

    async def test_list_empty_on_rbac_failure(self) -> None:
        """Fail-closed: an RBAC check failure yields an empty tool list."""
        server = MagicMock()
        server.get_allowed_tools.side_effect = RuntimeError("SAR unavailable")
        mw = RBACMiddleware(server)
        call_next = AsyncMock(return_value=_list_result(["tool_a"]))

        result = await mw(_ctx("tools/list", {}), call_next)

        assert result["tools"] == []

    async def test_non_tool_method_passes_through(self) -> None:
        mw, _ = _middleware(allowed={"tool_a"}, governed={"tool_a"})
        call_next = AsyncMock(return_value={"serverInfo": {}})

        result = await mw(_ctx("initialize", {"protocolVersion": "2025-03-26"}), call_next)

        call_next.assert_awaited_once()
        assert result == {"serverInfo": {}}


class TestRBACEnforcementEndToEnd:
    """Both enforcement points through the real dispatch path.

    Drives a real MCPServer (with RBACMiddleware installed via the public
    constructor parameter) over streamable-http with raw JSON-RPC — the
    permanent regression coverage RHAI-14629 promotes from the RHAI-14595
    tripwire: if the middleware is not installed or not enforcing, these
    fail rather than skip.
    """

    @staticmethod
    def _build_mcp() -> MCPServer:
        server = MagicMock()
        server.get_allowed_tools.return_value = ({"visible_tool"}, {"visible_tool", "hidden_tool"})
        mcp = MCPServer(name="rbac-e2e", middleware=[RBACMiddleware(server)])

        @mcp.tool()
        def hidden_tool() -> dict[str, str]:
            """Governed tool the caller is denied."""
            return {"hidden": "never runs"}

        @mcp.tool()
        def visible_tool() -> dict[str, str]:
            """Governed tool the caller may use."""
            return {"visible": "yes"}

        return mcp

    async def test_enforcement_through_dispatch(self) -> None:
        mcp = self._build_mcp()
        app = mcp.streamable_http_app(json_response=True, stateless_http=True)

        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        try:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            async with httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "clientInfo": {"name": "rbac-e2e", "version": "0"},
                        },
                    },
                )

                # tools/list: the governed-denied tool is filtered from view
                resp = await client.post(
                    "/mcp",
                    headers=headers,
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                )
                names = [t["name"] for t in resp.json()["result"]["tools"]]
                assert "visible_tool" in names
                assert "hidden_tool" not in names, "RBAC list filtering not enforced"

                # tools/call: the governed-denied tool is denied before dispatch
                resp = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "hidden_tool", "arguments": {}},
                    },
                )
                result = resp.json()["result"]
                assert result["isError"] is True, "RBAC call denial not enforced"
                assert "not permitted" in result["content"][0]["text"]

                # tools/call: the allowed tool runs
                resp = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {"name": "visible_tool", "arguments": {}},
                    },
                )
                assert resp.json()["result"]["isError"] is False
        finally:
            server.should_exit = True
            await task
