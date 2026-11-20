"""Streamable-HTTP (stateful) session-owner pinning (RHAI-14626 security fix).

The streamable companion to test_sse_session_pinning.py: the SDK's
stateful Streamable HTTP session manager records the creating
credential's principal (from the ASGI ``scope["user"]`` slot, populated
by OIDCAuthMiddleware after validation) and rejects messages posted to
a session with a different credential (404). The SSE test covers the
legacy transport; this one closes the coverage asymmetry on the
transport the product deploys by default.
"""

import asyncio
from typing import Any

import httpx2
import uvicorn
from mcp.server.mcpserver import MCPServer

from rhoai_mcp.auth.middleware import OIDCAuthMiddleware
from rhoai_mcp.auth.oidc import ValidatedIdentity
from tests.auth.test_sse_session_pinning import ALICE_TOKEN, BOB_TOKEN

# Same two-principal validator as the SSE pinning test, redeclared locally
# so each module reads as a self-contained proof.


class _TwoUserValidator:
    """Validates the two fixture tokens as distinct principals."""

    _identities: dict[str, ValidatedIdentity] = {
        ALICE_TOKEN: ValidatedIdentity(username="alice", groups=["team-a"], uid="u-alice"),
        BOB_TOKEN: ValidatedIdentity(username="bob", groups=["team-b"], uid="u-bob"),
    }

    async def validate_token(self, token: str) -> ValidatedIdentity:
        try:
            return self._identities[token]
        except KeyError:
            raise ValueError("unknown token") from None


def _build_app() -> OIDCAuthMiddleware:
    mcp = MCPServer(name="streamable-pinning-test")

    @mcp.tool()
    def ping_tool() -> dict[str, Any]:
        """Trivial tool."""
        return {"pong": "yes"}

    return OIDCAuthMiddleware(
        mcp.streamable_http_app(json_response=True),  # stateful (default)
        validator=_TwoUserValidator(),
        exclude_paths=["/health", "/.well-known/oauth-protected-resource"],
    )


async def _serve(app: OIDCAuthMiddleware) -> tuple[int, Any, Any]:
    """Serve the app; return (port, server, task)."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    deadline = asyncio.get_running_loop().time() + 30
    while not server.started:
        if task.done() or asyncio.get_running_loop().time() > deadline:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise RuntimeError("uvicorn server failed to start")
        await asyncio.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return port, server, task


async def _initialize(port: int, token: str) -> str:
    """Initialize a stateful session with `token`; return the session id."""
    async with httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        resp = await client.post(
            "/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pinning-test", "version": "0"},
                },
            },
        )
        assert resp.status_code == 200, f"initialize rejected: HTTP {resp.status_code}"
        session_id = resp.headers.get("mcp-session-id")
        assert session_id, "no session id on the initialize response"
        return session_id


async def _post_with_session(port: int, session_id: str, token: str, msg_id: int) -> int:
    """POST one JSON-RPC message to an existing session; return the status."""
    async with httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        resp = await client.post(
            "/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "Mcp-Session-Id": session_id,
            },
            json={"jsonrpc": "2.0", "id": msg_id, "method": "tools/list", "params": {}},
        )
        return resp.status_code


class TestStreamableSessionOwnerPinning:
    async def test_same_credential_continues_its_session(self) -> None:
        """The creating credential's subsequent POSTs are accepted (not 404)."""
        port, server, task = await _serve(_build_app())
        try:
            session_id = await _initialize(port, ALICE_TOKEN)
            status = await _post_with_session(port, session_id, ALICE_TOKEN, 2)
            assert status == 200, f"creating credential's own POST rejected: HTTP {status}"
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(task, timeout=10)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_second_credential_rejected_with_404(self) -> None:
        """A different valid credential posting to the session gets 404, not dispatch.

        The streamable twin of the SSE test's core assertion: the pinning
        engages on both session-based transports once the validated
        principal reaches scope["user"].
        """
        port, server, task = await _serve(_build_app())
        try:
            session_id = await _initialize(port, ALICE_TOKEN)
            status = await _post_with_session(port, session_id, BOB_TOKEN, 3)
            assert status == 404, f"second credential not rejected: HTTP {status}"
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(task, timeout=10)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_each_credential_can_drive_its_own_session(self) -> None:
        """The pinning does not lock out the second credential entirely —
        bob initializing his own session and posting to it works."""
        port, server, task = await _serve(_build_app())
        try:
            session_id = await _initialize(port, BOB_TOKEN)
            status = await _post_with_session(port, session_id, BOB_TOKEN, 2)
            assert status == 200, f"bob's own session POST rejected: HTTP {status}"
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(task, timeout=10)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
