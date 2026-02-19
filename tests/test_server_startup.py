"""Tests for RHOAIServer.startup() idempotency."""

from unittest.mock import MagicMock, Mock, patch

from rhoai_mcp.server import RHOAIServer


def test_startup_preserves_pre_injected_connected_client() -> None:
    """startup() should not replace an already-connected K8s client."""
    server = RHOAIServer()
    mock_client = Mock()
    mock_client.is_connected = True
    server._k8s_client = mock_client

    with patch("rhoai_mcp.server.K8sClient") as k8s_cls:
        server.startup()
        k8s_cls.assert_not_called()

    assert server._k8s_client is mock_client


def test_startup_creates_client_when_none_exists() -> None:
    """startup() should create and connect a K8s client when none is set."""
    server = RHOAIServer()
    assert server._k8s_client is None

    mock_client = MagicMock()
    mock_client.is_connected = True
    with patch("rhoai_mcp.server.K8sClient", return_value=mock_client) as k8s_cls:
        server.startup()
        k8s_cls.assert_called_once_with(server._config)
        mock_client.connect.assert_called_once()

    assert server._k8s_client is mock_client


def test_startup_replaces_disconnected_client() -> None:
    """startup() should replace a client that exists but is not connected."""
    server = RHOAIServer()
    old_client = Mock()
    old_client.is_connected = False
    server._k8s_client = old_client

    new_client = MagicMock()
    new_client.is_connected = True
    with patch("rhoai_mcp.server.K8sClient", return_value=new_client) as k8s_cls:
        server.startup()
        k8s_cls.assert_called_once_with(server._config)
        new_client.connect.assert_called_once()

    assert server._k8s_client is new_client


def test_startup_runs_health_checks_with_pre_injected_client() -> None:
    """Health checks should run even when the K8s client was pre-injected."""
    server = RHOAIServer()
    mock_client = Mock()
    mock_client.is_connected = True
    server._k8s_client = mock_client

    mock_pm = Mock()
    mock_pm.registered_plugins = {"p1": Mock()}
    mock_pm.healthy_plugins = {"p1": Mock()}
    server._plugin_manager = mock_pm

    with patch("rhoai_mcp.server.K8sClient"):
        server.startup()

    mock_pm.run_health_checks.assert_called_once_with(server)


def test_startup_skips_health_checks_without_plugin_manager() -> None:
    """startup() should not fail when plugin_manager is None."""
    server = RHOAIServer()
    assert server._plugin_manager is None

    mock_client = MagicMock()
    mock_client.is_connected = True
    with patch("rhoai_mcp.server.K8sClient", return_value=mock_client):
        server.startup()  # Should not raise
