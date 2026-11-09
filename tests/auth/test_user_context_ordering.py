"""UserContext ordering under mcp 2.x Streamable HTTP (RHAI-14608).

Verifies the ordering guarantee the 3.1 RBAC middleware will rely on:
OIDCAuthMiddleware (ASGI layer) must have validated the token and
populated UserContext before message dispatch reaches a ServerMiddleware.
Driven end-to-end — real uvicorn server, raw JSON-RPC client — under both
the stateless and stateful Streamable HTTP stacks.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx2
import uvicorn
from mcp.server.mcpserver import MCPServer

from rhoai_mcp.auth.middleware import OIDCAuthMiddleware
from rhoai_mcp.auth.oidc import ValidatedIdentity
from rhoai_mcp.auth.user_context import UserContext


@dataclass
class _StubValidator:
    """Validates any token as alice — ordering is the subject, not validation."""

    async def validate_token(self, _token: str) -> ValidatedIdentity:
        return ValidatedIdentity(username="alice", groups=["team-a"], uid="u-1")


class _OrderingProbe:
    """ServerMiddleware stand-in for the 3.1 RBAC middleware.

    Records, per dispatched message, the username visible via
    UserContext.current() — None means auth middleware had not populated
    the context before dispatch (the silent-failure mode this guards).
    """

    def __init__(self) -> None:
        self.captured: list[tuple[str, str | None]] = []

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        uc = UserContext.current()
        self.captured.append((ctx.method, uc.username if uc else None))
        return await call_next(ctx)


def _build_app(stateless: bool) -> tuple[OIDCAuthMiddleware, _OrderingProbe]:
    probe = _OrderingProbe()
    mcp = MCPServer(name="ordering-test", middleware=[probe])

    @mcp.tool()
    def ping_tool() -> dict[str, Any]:
        """Trivial tool so tools/call exercises the full dispatch path."""
        return {"pong": "yes"}

    kwargs: dict[str, Any] = {"json_response": True}
    if stateless:
        kwargs["stateless_http"] = True
    app = OIDCAuthMiddleware(
        mcp.streamable_http_app(**kwargs),
        validator=_StubValidator(),
        exclude_paths=["/health", "/.well-known/oauth-protected-resource"],
    )
    return app, probe


async def _drive(app: OIDCAuthMiddleware, authorized: bool) -> None:
    """Serve the app on an ephemeral port and drive a JSON-RPC conversation."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        async with httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if authorized:
                headers["Authorization"] = "Bearer alice-token"

            resp = await client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "ordering-test", "version": "0"},
                    },
                },
            )
            if not authorized:
                assert resp.status_code == 401
                return
            assert resp.status_code == 200

            session_id = resp.headers.get("mcp-session-id")
            if session_id:
                headers["Mcp-Session-Id"] = session_id
                await client.post(
                    "/mcp",
                    headers=headers,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )

            resp = await client.post(
                "/mcp",
                headers=headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            assert resp.status_code == 200

            resp = await client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "ping_tool", "arguments": {}},
                },
            )
            assert resp.status_code == 200
            assert '"isError":false' in resp.text
    finally:
        server.should_exit = True
        await task


class TestUserContextOrderingStreamableHttp:
    async def test_stateless_user_context_populated_before_dispatch(self) -> None:
        """Stateless Streamable HTTP: every dispatched message sees the user."""
        app, probe = _build_app(stateless=True)
        await _drive(app, authorized=True)

        methods = dict(probe.captured)
        assert "initialize" in methods
        assert "tools/list" in methods
        assert "tools/call" in methods
        assert all(user == "alice" for user in methods.values()), probe.captured

    async def test_stateful_user_context_populated_before_dispatch(self) -> None:
        """Stateful Streamable HTTP: per-POST context reaches the session dispatch."""
        app, probe = _build_app(stateless=False)
        await _drive(app, authorized=True)

        methods = dict(probe.captured)
        assert "initialize" in methods
        assert "tools/list" in methods
        assert "tools/call" in methods
        assert all(user == "alice" for user in methods.values()), probe.captured

    async def test_unauthenticated_request_never_reaches_dispatch(self) -> None:
        """Negative control: no Bearer header means 401 and zero dispatched messages."""
        app, probe = _build_app(stateless=True)
        await _drive(app, authorized=False)

        assert probe.captured == []
