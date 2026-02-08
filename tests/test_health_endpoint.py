"""Tests for the /health endpoint."""

from collections.abc import Generator
from http import HTTPStatus
from typing import Any
from unittest.mock import Mock

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from rhoai_mcp.server import RHOAIServer


@pytest.fixture
def health_client() -> Generator[tuple[RHOAIServer, TestClient], Any, None]:
    """Create a server + MCP wired for health-endpoint testing."""
    server = RHOAIServer()
    mcp = FastMCP(name="test-server")
    server._mcp = mcp
    server._register_health_endpoint(mcp)
    app = mcp.streamable_http_app()
    with TestClient(app) as client:
        yield server, client


def test_health_endpoint_returns_200(
    health_client: tuple[RHOAIServer, TestClient],
) -> None:
    """Test that /health endpoint returns 200 when K8s is connected."""
    server, client = health_client

    mock_k8s = Mock()
    mock_k8s.is_connected = True
    server._k8s_client = mock_k8s

    mock_pm = Mock()
    mock_pm.registered_plugins = {"plugin1": Mock(), "plugin2": Mock()}
    mock_pm.healthy_plugins = {"plugin1": Mock()}
    server._plugin_manager = mock_pm

    response = client.get("/health")

    assert response.status_code == HTTPStatus.OK  # 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["connected"] is True
    assert data["plugins"]["total"] == 2
    assert data["plugins"]["healthy"] == 1


def test_health_endpoint_before_k8s_connection(
    health_client: tuple[RHOAIServer, TestClient],
) -> None:
    """Test health endpoint when K8s client exists but is not connected."""
    server, client = health_client

    mock_k8s = Mock()
    mock_k8s.is_connected = False
    server._k8s_client = mock_k8s

    mock_pm = Mock()
    mock_pm.registered_plugins = {}
    mock_pm.healthy_plugins = {}
    server._plugin_manager = mock_pm

    response = client.get("/health")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE  # 503
    data = response.json()

    assert data["status"] == "unhealthy"
    assert data["connected"] is False
    assert data["plugins"]["total"] == 0
    assert data["plugins"]["healthy"] == 0


def test_health_endpoint_before_startup(
    health_client: tuple[RHOAIServer, TestClient],
) -> None:
    """Test health endpoint when server has not started (no k8s client, no plugin manager)."""
    _server, client = health_client
    # _k8s_client and _plugin_manager remain None from fixture

    response = client.get("/health")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE  # 503
    data = response.json()

    assert data["status"] == "unhealthy"
    assert data["connected"] is False
    assert data["plugins"]["total"] == 0
    assert data["plugins"]["healthy"] == 0
