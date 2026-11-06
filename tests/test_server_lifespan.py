"""Port forwards must survive unrelated HTTP session endings,
but are cleaned up on server shutdown."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def test_shutdown_terminates_port_forward_subprocesses() -> None:
    server = RHOAIServer(RHOAIConfig())
    proc_alive = MagicMock(returncode=None)
    proc_done = MagicMock(returncode=0)

    fake_connections = {
        ("ns", "svc-a", 8080): MagicMock(process=proc_alive),
        ("ns", "svc-b", 9090): MagicMock(process=proc_done),
    }
    manager = MagicMock(_connections=fake_connections)

    with patch(
        "rhoai_mcp.utils.port_forward.PortForwardManager.get_instance",
        return_value=manager,
    ):
        server.shutdown()

    proc_alive.terminate.assert_called_once()
    proc_done.terminate.assert_not_called()
    assert fake_connections == {}
