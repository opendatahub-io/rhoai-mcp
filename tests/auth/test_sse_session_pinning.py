"""SSE session-owner pinning (RHAI-14626 security fix, 2026-11-18).

The SDK's SSE transport rejects messages posted to a session with a
credential other than the one that created it (404) — but only when the
validated principal reaches the ASGI ``scope["user"]`` slot as an
``AuthenticatedUser``. OIDCAuthMiddleware now writes it after validation
(closing the gap where the comparison was vacuously None == None and a
second valid credential posting to a victim's session was evaluated as
the connection identity).

The harness serves the SSE transport directly rather than through
``MCPServer.sse_app()``: mcp 2.3.0's ``handle_sse`` wrapper returns a
``Response()`` after the ``EventSourceResponse`` has already completed —
double-sending ``http.response.start`` — which reliably crashes once our
OIDCAuthMiddleware is in the stack (the route is ``# pragma: no cover``
upstream). Everything else mirrors ``sse_app``'s construction exactly.
The client speaks raw HTTP over asyncio sockets for the same reason:
the SDK's streaming client trips the same teardown defect.
"""

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from rhoai_mcp.auth.middleware import OIDCAuthMiddleware
from rhoai_mcp.auth.oidc import ValidatedIdentity

# Credential fixtures: two distinct valid principals
ALICE_TOKEN = "alice-session-token"
BOB_TOKEN = "bob-session-token"


@dataclass
class _TwoUserValidator:
    """Validates the two fixture tokens as distinct principals."""

    identities: dict[str, ValidatedIdentity] = field(
        default_factory=lambda: {
            ALICE_TOKEN: ValidatedIdentity(username="alice", groups=["team-a"], uid="u-alice"),
            BOB_TOKEN: ValidatedIdentity(username="bob", groups=["team-b"], uid="u-bob"),
        }
    )

    async def validate_token(self, token: str) -> ValidatedIdentity:
        try:
            return self.identities[token]
        except KeyError:
            raise ValueError("unknown token") from None


class _SSEHandler:
    """Raw ASGI SSE endpoint mirroring ``MCPServer.sse_app``'s wiring.

    Starlette's ``Route`` wraps plain-function endpoints with
    ``request_response`` (a single ``Request`` argument); a class
    instance is invoked directly as an ASGI app, which is what
    ``connect_sse`` needs.

    Unlike the SDK's wrapper, this handler does not return a
    ``Response()`` after the stream completes — that is the mcp 2.3.0
    teardown defect this harness exists to avoid (see module docstring).
    """

    def __init__(self, sse: SseServerTransport, mcp: MCPServer) -> None:
        self._sse = sse
        self._mcp = mcp

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        async with self._sse.connect_sse(scope, receive, send) as streams:
            # Same dispatch wiring sse_app uses (private accessor is the
            # only path to the lowlevel server; the SDK does the same
            # internally).
            await self._mcp._lowlevel_server.run(
                streams[0], streams[1], self._mcp._lowlevel_server.create_initialization_options()
            )


def _build_app() -> OIDCAuthMiddleware:
    mcp = MCPServer(name="sse-pinning-test")

    @mcp.tool()
    def ping_tool() -> dict[str, Any]:
        """Trivial tool."""
        return {"pong": "yes"}

    # The same localhost transport security sse_app auto-enables
    sse = SseServerTransport(
        "/messages/",
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=[
                "http://127.0.0.1:*",
                "http://localhost:*",
                "http://[::1]:*",
            ],
        ),
    )
    inner = Starlette(
        routes=[
            Route("/sse", _SSEHandler(sse, mcp), methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )
    return OIDCAuthMiddleware(
        inner,
        validator=_TwoUserValidator(),
        exclude_paths=["/health", "/.well-known/oauth-protected-resource"],
    )


# Open SSE streams, kept for the session's lifetime (the session's writer
# lives in the GET's task; closing it terminates the session)
_SSE_STREAMS: list[tuple[Any, Any]] = []


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


async def _teardown(server: Any, task: Any) -> None:
    await _close_sse_sessions()
    # should_exit alone waits for the long-lived SSE connections; force_exit
    # after a beat
    server.should_exit = True
    await asyncio.sleep(0.2)
    server.force_exit = True
    try:
        await asyncio.wait_for(task, timeout=10)
    except TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _close_sse_sessions() -> None:
    while _SSE_STREAMS:
        _reader, writer = _SSE_STREAMS.pop()
        writer.close()
        with contextlib.suppress(Exception):  # best-effort teardown
            await writer.wait_closed()


def _extract_endpoint(total: str) -> str | None:
    """The message endpoint from the SSE handshake bytes.

    mcp 2.3.0 sends the endpoint event's data as a plain URI string
    (``data: /messages/?session_id=<hex>``) — not JSON — so scan for a
    data line carrying the session id. The check is on the data line,
    not just the ``event: endpoint`` line, because the event name can
    arrive in an earlier chunk than its payload.
    """
    for line in total.splitlines():
        if line.startswith("data:") and "session_id=" in line:
            return line.removeprefix("data:").strip()
    return None


async def _open_sse_session(port: int, token: str) -> str:
    """Open an SSE connection with `token`; return the message endpoint."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    req = (
        f"GET /sse HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Authorization: Bearer {token}\r\nAccept: text/event-stream\r\n\r\n"
    )
    writer.write(req.encode())
    await writer.drain()

    # Read the response incrementally and scan for the endpoint event's
    # data line (read()-based: readline() consumption trips the same SDK
    # teardown defect this harness avoids elsewhere).
    total = b""
    endpoint = None
    while endpoint is None:
        chunk = await asyncio.wait_for(reader.read(4096), timeout=30)
        if not chunk:
            break
        total += chunk
        endpoint = _extract_endpoint(total.decode("latin-1"))
    assert total.startswith(b"HTTP/1.1 200"), f"SSE rejected: {total[:40]!r}"
    assert endpoint, f"no endpoint event on the SSE stream: {total[:200]!r}"

    _SSE_STREAMS.append((reader, writer))
    return endpoint


async def _post_message(port: int, endpoint: str, token: str, msg_id: int) -> int:
    """POST one JSON-RPC message to the endpoint; return the HTTP status."""
    body = json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": "tools/list", "params": {}})
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    req = (
        f"POST {endpoint} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Authorization: Bearer {token}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n{body}"
    )
    writer.write(req.encode())
    await writer.drain()

    status_line = await asyncio.wait_for(reader.readline(), timeout=30)
    status = int(status_line.split()[1])
    # drain headers + body so the response completes cleanly
    content_length: int | None = None
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=30)
        if line in (b"\r\n", b"\n", b""):
            break
        header = line.decode("latin-1").lower()
        if header.startswith("content-length:"):
            content_length = int(header.split(":")[1])
    if content_length:
        await asyncio.wait_for(reader.readexactly(content_length), timeout=30)
    writer.close()
    return status


class TestSSESessionOwnerPinning:
    async def test_same_credential_continues_its_session(self) -> None:
        """The creating credential's subsequent POSTs are accepted (not 404)."""
        port, server, task = await _serve(_build_app())
        try:
            endpoint = await _open_sse_session(port, ALICE_TOKEN)
            status = await _post_message(port, endpoint, ALICE_TOKEN, 1)
            assert status == 202, f"creating credential's own POST rejected: HTTP {status}"
        finally:
            await _teardown(server, task)

    async def test_second_credential_rejected_with_404(self) -> None:
        """A different valid credential posting to the session gets 404, not dispatch.

        This is the security fix's core assertion: before it, the SDK's
        pinning comparison was vacuous (scope["user"] unset) and this POST
        was accepted — dispatching with the connection identity (the victim).
        """
        port, server, task = await _serve(_build_app())
        try:
            endpoint = await _open_sse_session(port, ALICE_TOKEN)
            status = await _post_message(port, endpoint, BOB_TOKEN, 2)
            assert status == 404, f"second credential not rejected: HTTP {status}"
        finally:
            await _teardown(server, task)

    async def test_each_credential_can_drive_its_own_session(self) -> None:
        """The pinning does not lock out the second credential entirely —
        bob opening his own session and posting to it works."""
        port, server, task = await _serve(_build_app())
        try:
            endpoint = await _open_sse_session(port, BOB_TOKEN)
            status = await _post_message(port, endpoint, BOB_TOKEN, 1)
            assert status == 202, f"bob's own session POST rejected: HTTP {status}"
        finally:
            await _teardown(server, task)
