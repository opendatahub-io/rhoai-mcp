"""Port forwards survive HTTP sessions and close when the transport stops."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.applications import Starlette

from rhoai_mcp.config import RHOAIConfig, TransportMode
from rhoai_mcp.server import RHOAIServer


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [TransportMode.SSE, TransportMode.STREAMABLE_HTTP])
async def test_http_session_end_preserves_shared_port_forward(transport: TransportMode) -> None:
    server = RHOAIServer(RHOAIConfig(transport=transport))
    manager = AsyncMock()

    with patch(
        "rhoai_mcp.utils.port_forward.PortForwardManager.get_instance",
        return_value=manager,
    ):
        async with server._create_lifespan()(None):
            async with server._create_lifespan()(None):
                pass
            manager.close_all.assert_not_awaited()

    manager.close_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_stdio_session_end_closes_port_forwards() -> None:
    server = RHOAIServer(RHOAIConfig(transport=TransportMode.STDIO))
    manager = AsyncMock()

    with patch(
        "rhoai_mcp.utils.port_forward.PortForwardManager.get_instance",
        return_value=manager,
    ):
        async with server._create_lifespan()(None):
            manager.close_all.assert_not_awaited()

    manager.close_all.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "method"),
    [
        (TransportMode.SSE, "sse_app"),
        (TransportMode.STREAMABLE_HTTP, "streamable_http_app"),
    ],
)
@pytest.mark.parametrize("fails", [False, True])
async def test_http_app_shutdown_closes_port_forwards(
    transport: TransportMode, method: str, fails: bool
) -> None:
    events: list[str] = []
    manager = MagicMock(close_all=AsyncMock())

    @asynccontextmanager
    async def original_lifespan(_app: Starlette) -> AsyncGenerator[None]:
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")
            if fails:
                raise RuntimeError("app shutdown failed")

    async def close_all() -> None:
        events.append("close")

    app = Starlette(lifespan=original_lifespan)
    mcp = MagicMock(sse_app=MagicMock(return_value=app), streamable_http_app=MagicMock(return_value=app))
    server = RHOAIServer(RHOAIConfig(transport=transport))
    server._install_http_shutdown(mcp)
    manager.close_all.side_effect = close_all

    with patch(
        "rhoai_mcp.utils.port_forward.PortForwardManager.get_instance",
        return_value=manager,
    ):
        wrapped_app = getattr(mcp, method)()
        if fails:
            with pytest.raises(RuntimeError, match="app shutdown failed"):
                async with wrapped_app.router.lifespan_context(wrapped_app):
                    manager.close_all.assert_not_awaited()
        else:
            async with wrapped_app.router.lifespan_context(wrapped_app):
                manager.close_all.assert_not_awaited()

    assert events == ["start", "stop", "close"]
    manager.close_all.assert_awaited_once()
