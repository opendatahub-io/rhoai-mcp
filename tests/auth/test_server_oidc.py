"""Tests for OIDC integration in server.py."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.config import RHOAIConfig, TransportMode
from rhoai_mcp.server import RHOAIServer


class TestServerOIDCIntegration:
    def test_k8s_property_returns_shared_client_when_oidc_disabled(self) -> None:
        config = RHOAIConfig(oidc_enabled=False, mock_cluster=True)
        server = RHOAIServer(config)
        server.create_mcp()
        # Should return the shared mock client
        assert server.k8s is not None
        assert server.k8s.is_connected

    def test_k8s_property_raises_when_no_user_context_under_oidc(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        server._k8s_client = MagicMock()
        server._k8s_client.is_connected = True
        # Without user context under OIDC, should fail closed
        with pytest.raises(RuntimeError, match="no UserContext is set"):
            _ = server.k8s

    def test_k8s_property_returns_impersonating_client_with_user_context(self) -> None:
        from rhoai_mcp.auth.user_context import UserContext

        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_issuer_url="https://idp.example.com",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_client = MagicMock()
        mock_client.is_connected = True
        mock_imp_client = MagicMock()
        mock_client.create_impersonating_client.return_value = mock_imp_client
        server._k8s_client = mock_client

        ctx = UserContext(username="alice", groups=["team-a"])
        token = UserContext.set_current(ctx)
        try:
            result = server.k8s
            mock_client.create_impersonating_client.assert_called_once_with(
                "alice", ["team-a"]
            )
            assert result is mock_imp_client
        finally:
            UserContext.reset_current(token)

    def test_oidc_validation_called_at_startup(self) -> None:
        config = RHOAIConfig(
            oidc_enabled=True, mock_cluster=True, transport=TransportMode.SSE
        )
        with pytest.raises(ValueError, match="oidc_issuer_url is required"):
            server = RHOAIServer(config)
            server.create_mcp()
