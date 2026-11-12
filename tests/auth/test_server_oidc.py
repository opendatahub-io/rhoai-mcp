"""Tests for OIDC integration in server.py."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from rhoai_mcp.config import RHOAIConfig, TransportMode
from rhoai_mcp.server import RHOAIServer
from rhoai_mcp.utils.errors import RHOAIError


class TestServerOIDCIntegration:
    def test_k8s_property_returns_shared_client_when_oidc_disabled(self) -> None:
        config = RHOAIConfig(oidc_enabled=False, mock_cluster=True)
        server = RHOAIServer(config)
        server.create_mcp()
        # Should return the shared mock client
        assert server.get_k8s(None) is not None
        assert server.get_k8s(None).is_connected

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
        with pytest.raises(RHOAIError, match="no UserContext is set"):
            server.get_k8s(None)

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
            result = server.get_k8s(None)
            mock_client.create_impersonating_client.assert_called_once_with("alice", ["team-a"])
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

        ctx = UserContext(username="alice", groups=["team-a"], token=SecretStr("alice-token"))
        token = UserContext.set_current(ctx)
        try:
            result = server.get_k8s(None)
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
            with pytest.raises(RHOAIError, match="no token is available"):
                server.get_k8s(None)
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

        ctx = UserContext(username="bob", groups=["team-b"], token=SecretStr("bob-token"))
        token = UserContext.set_current(ctx)
        try:
            result = server.get_k8s(None)
            mock_client.create_impersonating_client.assert_called_once_with("bob", ["team-b"])
            mock_client.create_user_token_client.assert_not_called()
            assert result is mock_imp_client
        finally:
            UserContext.reset_current(token)

    def test_oidc_validation_called_at_startup(self) -> None:
        config = RHOAIConfig(oidc_enabled=True, mock_cluster=True, transport=TransportMode.SSE)
        with pytest.raises(ValueError, match="oidc_issuer_url is required"):
            server = RHOAIServer(config)
            server.create_mcp()


class TestAuthAppWrapping:
    """Transport app builders must be wrapped by OIDCAuthMiddleware (RHAI-14618).

    Under mcp 2.x the app builders take keyword-only arguments and no
    ``mount_path``; the wrappers installed by ``_setup_auth()`` must forward
    whatever the SDK passes and still apply the auth middleware. Tool-filtering
    installation is stubbed out here — its v2 port is RHAI-14627 (story 3.1).
    """

    def _make_server(self) -> RHOAIServer:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_token_mode="token-review",
            oidc_kube_auth_strategy="impersonation",
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_k8s = MagicMock()
        mock_k8s.is_connected = True
        mock_k8s._api_client = MagicMock()
        server._k8s_client = mock_k8s
        return server

    @staticmethod
    def _setup_auth(mcp: Any, server: RHOAIServer) -> None:
        """Run _setup_auth (RBAC enforcement is constructor-installed now)."""
        server._setup_auth(mcp)

    def test_sse_app_wrapped_in_oidc_middleware(self) -> None:
        from mcp.server.mcpserver import MCPServer

        from rhoai_mcp.auth.middleware import OIDCAuthMiddleware

        server = self._make_server()
        mcp = MCPServer(name="wrap-test")
        self._setup_auth(mcp, server)

        app = mcp.sse_app()
        assert isinstance(app, OIDCAuthMiddleware)

    def test_sse_app_forwards_v2_kwargs(self) -> None:
        """The wrapper forwards v2's keyword-only sse_app arguments (no mount_path)."""
        from mcp.server.mcpserver import MCPServer

        from rhoai_mcp.auth.middleware import OIDCAuthMiddleware

        server = self._make_server()
        mcp = MCPServer(name="wrap-test")
        self._setup_auth(mcp, server)

        app = mcp.sse_app(sse_path="/sse", message_path="/messages/")
        assert isinstance(app, OIDCAuthMiddleware)

    def test_streamable_http_app_wrapped_in_oidc_middleware(self) -> None:
        from mcp.server.mcpserver import MCPServer

        from rhoai_mcp.auth.middleware import OIDCAuthMiddleware

        server = self._make_server()
        mcp = MCPServer(name="wrap-test")
        self._setup_auth(mcp, server)

        app = mcp.streamable_http_app()
        assert isinstance(app, OIDCAuthMiddleware)


class TestGetK8sTokenPrecedence:
    """Token sourcing for get_k8s(ctx) (RHAI-14631, the 1.6 design's precedence).

    Per-message request headers (SSE carries the POST's Request on the
    message metadata) win over UserContext.token; UserContext covers
    Streamable HTTP, where the ASGI middleware sets it per POST.
    """

    def _make_server(self, strategy: str = "user-token") -> RHOAIServer:
        config = RHOAIConfig(
            oidc_enabled=True,
            oidc_token_mode="token-review",
            oidc_kube_auth_strategy=strategy,
            mock_cluster=True,
            transport=TransportMode.SSE,
        )
        server = RHOAIServer(config)
        mock_k8s = MagicMock()
        mock_k8s.is_connected = True
        server._k8s_client = mock_k8s
        return server

    @staticmethod
    def _ctx_with_headers(headers: dict[str, str]) -> Any:
        """A tool-side Context whose message metadata carries a Request."""
        from types import SimpleNamespace

        request = SimpleNamespace(headers=headers)
        meta = SimpleNamespace(request_context=request)
        return SimpleNamespace(request_context=SimpleNamespace(meta=meta))

    def test_per_message_token_wins_over_user_context(self) -> None:
        from rhoai_mcp.auth.user_context import UserContext

        server = self._make_server()
        ctx = UserContext(username="alice", groups=[], token=SecretStr("session-token"))
        token = UserContext.set_current(ctx)
        try:
            server.get_k8s(self._ctx_with_headers({"authorization": "Bearer fresh-token"}))
            call_token = server._k8s_client.create_user_token_client.call_args[0][0]
            assert call_token.get_secret_value() == "fresh-token"
        finally:
            UserContext.reset_current(token)

    def test_falls_back_to_user_context_token(self) -> None:
        from rhoai_mcp.auth.user_context import UserContext

        server = self._make_server()
        ctx = UserContext(username="alice", groups=[], token=SecretStr("session-token"))
        token = UserContext.set_current(ctx)
        try:
            # No ctx at all (the deprecated k8s property path)
            server.get_k8s(None)
            call_token = server._k8s_client.create_user_token_client.call_args[0][0]
            assert call_token.get_secret_value() == "session-token"

            # ctx present but meta None (Streamable HTTP)
            from types import SimpleNamespace

            server.get_k8s(SimpleNamespace(request_context=SimpleNamespace(meta=None)))
            call_token = server._k8s_client.create_user_token_client.call_args[0][0]
            assert call_token.get_secret_value() == "session-token"
        finally:
            UserContext.reset_current(token)

    def test_non_bearer_header_is_ignored(self) -> None:
        from rhoai_mcp.auth.user_context import UserContext

        server = self._make_server()
        ctx = UserContext(username="alice", groups=[], token=SecretStr("session-token"))
        token = UserContext.set_current(ctx)
        try:
            server.get_k8s(self._ctx_with_headers({"authorization": "Basic something"}))
            call_token = server._k8s_client.create_user_token_client.call_args[0][0]
            assert call_token.get_secret_value() == "session-token"
        finally:
            UserContext.reset_current(token)

    def test_impersonation_strategy_uses_identity_not_token(self) -> None:
        from rhoai_mcp.auth.user_context import UserContext

        server = self._make_server(strategy="impersonation")
        ctx = UserContext(username="alice", groups=["team-a"], token=SecretStr("ignored"))
        token = UserContext.set_current(ctx)
        try:
            server.get_k8s(self._ctx_with_headers({"authorization": "Bearer fresh-token"}))
            server._k8s_client.create_impersonating_client.assert_called_once_with(
                "alice", ["team-a"]
            )
            server._k8s_client.create_user_token_client.assert_not_called()
        finally:
            UserContext.reset_current(token)

