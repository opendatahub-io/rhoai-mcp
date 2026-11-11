"""RBAC enforcement middleware for MCP SDK v2 (RHAI-14627).

Implements the ratified 1.6 design (RHAI-14612): one ServerMiddleware
covers both enforcement points — tools/list filtering (UX) and tools/call
denial (security) — installed via the public ``MCPServer(middleware=[...])``
constructor parameter with zero private API access.

The middleware operates on the JSON-RPC wire surface: ``ctx.params`` is the
raw params dict (no validation has happened yet) and ``call_next`` returns
the wire-shaped result dict (camelCase keys — ``isError``, ``inputSchema``).

Fail-closed semantics, carried over from the v1 request_handlers patch
exactly: an RBAC check failure denies all tools (empty list / denied
call), and a missing UserContext under OIDC denies all. RBAC is
re-evaluated on every request — never a closure-captured permitted set.
The in-band denial message is a wire-shaped ``isError`` result per the
RHAI-14611 decision (no HTTP 403 for per-tool denial).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

_DENIED_MESSAGE = "Tool '{name}' is not permitted for the current user"
_DENIED_RBAC_FAILURE_MESSAGE = "Tool '{name}' is not permitted: RBAC check failed"


class RBACMiddleware:
    """Dual-point RBAC enforcement: tools/list filtering + tools/call denial.

    Matches the SDK's ``ServerMiddleware`` protocol shape
    (``(ctx, call_next) -> result``) without inheriting from it.
    """

    def __init__(self, server: RHOAIServer) -> None:
        self._server = server

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        if ctx.method == "tools/call":
            return await self._enforce_call(ctx, call_next)
        if ctx.method == "tools/list":
            return await self._filter_list(ctx, call_next)
        return await call_next(ctx)

    async def _check_permissions(self) -> tuple[set[str], set[str]] | None:
        """Run the per-request RBAC check off the event loop.

        Returns None when OIDC is disabled (no filtering); otherwise the
        (allowed, governed) tool-name sets. Raises on check failure —
        callers convert that to a fail-closed denial.
        """
        return await asyncio.to_thread(self._server.get_allowed_tools)

    async def _enforce_call(self, ctx: Any, call_next: Any) -> Any:
        params = ctx.params if isinstance(ctx.params, dict) else {}
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            # Malformed request — let the SDK's own validation answer it
            return await call_next(ctx)

        try:
            check = await self._check_permissions()
        except Exception:
            logger.error("RBAC check failed for tool '%s'", tool_name, exc_info=True)
            return self._denied_result(tool_name, rbac_failure=True)
        if check is None:
            return await call_next(ctx)

        allowed, governed = check
        # Only enforce for tools with permission mappings
        if tool_name in governed and tool_name not in allowed:
            return self._denied_result(tool_name)
        return await call_next(ctx)

    async def _filter_list(self, ctx: Any, call_next: Any) -> Any:
        result = await call_next(ctx)
        if not isinstance(result, dict):
            return result

        try:
            check = await self._check_permissions()
        except Exception:
            logger.error("RBAC check failed, denying all tools", exc_info=True)
            return {**result, "tools": []}
        if check is None:
            return result

        allowed, governed = check
        tools = result.get("tools", [])
        # Tools with permission mappings must pass RBAC; unmapped tools are allowed
        filtered = [
            tool
            for tool in tools
            if not isinstance(tool, dict)
            or tool.get("name") not in governed
            or tool.get("name") in allowed
        ]
        logger.info("Tool filtering: %d/%d tools allowed", len(filtered), len(tools))
        return {**result, "tools": filtered}

    @staticmethod
    def _denied_result(tool_name: str, rbac_failure: bool = False) -> dict[str, Any]:
        """Wire-shaped in-band denial (RHAI-14611: no HTTP 403 for per-tool denial)."""
        message = (
            _DENIED_RBAC_FAILURE_MESSAGE.format(name=tool_name)
            if rbac_failure
            else _DENIED_MESSAGE.format(name=tool_name)
        )
        return {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        }
