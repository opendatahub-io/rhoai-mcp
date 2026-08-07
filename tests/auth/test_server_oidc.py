"""Tests for OIDC integration in server.py."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

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
            oidc_kube_auth_strategy="impersonation",
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

    def test_k8s_user_token_strategy_uses_session_token(self) -> None:
        """user-token strategy with no request_ctx falls back to UserContext.token."""
        from rhoai_mcp.auth.user_context import UserContext

        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_token_mode="token-review",
            oidc_kube_auth_strategy="user-token",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_client = MagicMock()
        mock_client.is_connected = True
        mock_user_client = MagicMock()
        mock_client.create_user_token_client.return_value = mock_user_client
        server._k8s_client = mock_client

        ctx = UserContext(
            username="alice", groups=["team-a"], token=SecretStr("alice-token")
        )
        token = UserContext.set_current(ctx)
        try:
            result = server.k8s
            mock_client.create_user_token_client.assert_called_once()
            call_token = mock_client.create_user_token_client.call_args[0][0]
            assert call_token.get_secret_value() == "alice-token"
            assert result is mock_user_client
        finally:
            UserContext.reset_current(token)

    def test_k8s_user_token_no_token_raises(self) -> None:
        """user-token strategy with no token available raises RuntimeError."""
        from rhoai_mcp.auth.user_context import UserContext

        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_token_mode="token-review",
            oidc_kube_auth_strategy="user-token",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_client = MagicMock()
        mock_client.is_connected = True
        server._k8s_client = mock_client

        ctx = UserContext(username="alice", groups=["team-a"])
        # No token set
        token = UserContext.set_current(ctx)
        try:
            with pytest.raises(RuntimeError, match="no token is available"):
                _ = server.k8s
        finally:
            UserContext.reset_current(token)

    def test_k8s_impersonation_strategy_calls_impersonating_client(self) -> None:
        """impersonation strategy calls create_impersonating_client."""
        from rhoai_mcp.auth.user_context import UserContext

        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_token_mode="token-review",
            oidc_kube_auth_strategy="impersonation",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_client = MagicMock()
        mock_client.is_connected = True
        mock_imp_client = MagicMock()
        mock_client.create_impersonating_client.return_value = mock_imp_client
        server._k8s_client = mock_client

        ctx = UserContext(
            username="bob", groups=["team-b"], token=SecretStr("bob-token")
        )
        token = UserContext.set_current(ctx)
        try:
            result = server.k8s
            mock_client.create_impersonating_client.assert_called_once_with(
                "bob", ["team-b"]
            )
            mock_client.create_user_token_client.assert_not_called()
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


class TestGetPerMessageToken:
    def test_returns_none_when_request_ctx_not_set(self) -> None:
        from rhoai_mcp.server import _get_per_message_token

        result = _get_per_message_token(SecretStr("session-token"))
        assert result is None

    def test_returns_none_on_import_error(self) -> None:
        from rhoai_mcp.server import _get_per_message_token

        with patch.dict("sys.modules", {"mcp.server.lowlevel.server": None}):
            result = _get_per_message_token(SecretStr("token"))
            assert result is None

    def test_returns_token_from_request_ctx(self) -> None:
        from mcp.server.lowlevel.server import request_ctx

        from rhoai_mcp.server import _get_per_message_token

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer per-msg-token"}
        mock_ctx = MagicMock()
        mock_ctx.request = mock_request

        ctx_token = request_ctx.set(mock_ctx)
        try:
            result = _get_per_message_token(SecretStr("session-token"))
            assert result is not None
            assert result.get_secret_value() == "per-msg-token"
        finally:
            request_ctx.reset(ctx_token)

    def test_warns_when_tokens_differ(self) -> None:
        from mcp.server.lowlevel.server import request_ctx

        from rhoai_mcp.server import _get_per_message_token

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer new-token"}
        mock_ctx = MagicMock()
        mock_ctx.request = mock_request

        ctx_token = request_ctx.set(mock_ctx)
        try:
            with patch("rhoai_mcp.server.logger") as mock_logger:
                result = _get_per_message_token(SecretStr("old-token"))
                mock_logger.warning.assert_called_once()
                assert "changed mid-session" in mock_logger.warning.call_args[0][0]
                assert result.get_secret_value() == "new-token"
        finally:
            request_ctx.reset(ctx_token)

    def test_no_warning_when_tokens_match(self) -> None:
        from mcp.server.lowlevel.server import request_ctx

        from rhoai_mcp.server import _get_per_message_token

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer same-token"}
        mock_ctx = MagicMock()
        mock_ctx.request = mock_request

        ctx_token = request_ctx.set(mock_ctx)
        try:
            with patch("rhoai_mcp.server.logger") as mock_logger:
                result = _get_per_message_token(SecretStr("same-token"))
                mock_logger.warning.assert_not_called()
                assert result.get_secret_value() == "same-token"
        finally:
            request_ctx.reset(ctx_token)
